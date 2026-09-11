# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Scaffolding shared by the gfx950 GEMM kernels.

Everything here is denominated in storage bytes rather than logical elements.
That is what lets one tile schedule serve BF16/FP16 (2 bytes per element),
MXFP8 (1 byte) and MXFP4 (2 elements per byte): the DMA width, the LDS
footprint and the vmcnt budget all depend on bytes moved, never on how many
values those bytes encode.
"""

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch


GFX950_DMA_BYTES = 16
GFX950_WAVE_SIZE = 64
GFX950_MAX_BLOCK_THREADS = 1024
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3
# gfx950 LDS is banked on a 64-dword period; the XOR swizzle only needs to
# permute within that period.
LDS_BANK_PERIOD_LOG2 = 6
# ds_read_b64_tr_* moves 8 bytes per lane regardless of element width.
LDS_READ_TRANS_BASE = 4
_LDS_CAPACITY = {"gfx942": 65536, "gfx950": 163840}


def lds_capacity() -> int:
    return _LDS_CAPACITY.get(get_rocm_arch(), 65536)


def barrier(vmcnt=0):
    """Drain to `vmcnt` outstanding VMEM loads, then join the workgroup.

    Matches FlyDSL's own gfx950 GEMM (`wait_vmcnt_and_barrier`): the typed
    intrinsics carry the same side effects as the equivalent inline-asm block,
    so nothing can be scheduled between them.
    """
    rocdl.s_waitcnt(vmcnt=vmcnt)
    rocdl.s_barrier()


def waitcnt(vmcnt=0):
    rocdl.s_waitcnt(vmcnt=vmcnt)


def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, dma_bytes):
    buffer_load_asm_dict = {
        16: "buffer_load_dwordx4",
        8: "buffer_load_dwordx2",
        4: "buffer_load_dword",
    }
    # Match LLVM's gfx950 buffer_load_lds lowering: VMEM needs one wait state
    # after the SALU write to M0 (llvm-project#116681).
    llvm.InlineAsmOp(
        None,
        [
            llvm.IntToPtrOp(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.ptrtoint(lds_ptr)),
            ).result,
            fx.as_ir_value(global_offset),
            fx.as_ir_value(rsrc),
        ],
        f"s_mov_b32 m0, $0\n\ts_nop 0\n\t{buffer_load_asm_dict[dma_bytes]} $1, $2, 0 offen sc0 lds",
        "s,v,s",
        has_side_effects=True,
    )


def get_wave_lds_offset(tid, async_load_bytes):
    return rocdl.readfirstlane(
        fx.Int64.ir_type,
        fx.Int64(tid // GFX950_WAVE_SIZE * GFX950_WAVE_SIZE * async_load_bytes),
    )


def make_wave_lds_ptr(ptr, wave_offset):
    return fx.recast_iter(fx.Int8, ptr) + fx.Int32(wave_offset)


def swizzled_contiguous_idx(idx0, idx1, layout, extent):
    # The XOR swizzle is self-inverse. Map each physical contiguous position
    # written by direct-to-LDS DMA back to its logical global vector.
    elem_offset = fx.get_scalar(fx.crd2idx((idx0, idx1), layout))
    return elem_offset % extent


class BlockSwizzle:
    def __init__(self, NUM_XCDS, NUM_PIDS_THRESHOLD, GROUP_M, N_MAJOR_FALLBACK=False):
        self.NUM_XCDS = NUM_XCDS
        self.NUM_PIDS_THRESHOLD = NUM_PIDS_THRESHOLD
        self.GROUP_M = GROUP_M
        self.N_MAJOR_FALLBACK = N_MAJOR_FALLBACK

    @flyc.jit
    def swizzle(self, num_pid_m, num_pid_n, pid):
        if const_expr(self.N_MAJOR_FALLBACK):
            simple_m = pid % num_pid_m
            simple_n = pid // num_pid_m
        else:
            simple_m = pid // num_pid_n
            simple_n = pid % num_pid_n
        if const_expr(self.GROUP_M <= 0):
            return simple_m, simple_n
        num_xcds = self.NUM_XCDS
        swizzle_threshold = self.NUM_PIDS_THRESHOLD
        num_wg = num_pid_m * num_pid_n
        linear_id = pid
        intra_xcd = linear_id // num_xcds
        xcd = linear_id % num_xcds
        wgid = xcd * (num_wg // num_xcds) + intra_xcd
        group_m = self.GROUP_M
        wgid_per_group = group_m * num_pid_n
        group_id = wgid // wgid_per_group
        intra_group = wgid % wgid_per_group
        first_pid_m = group_id * group_m
        remaining_m = num_pid_m - first_pid_m
        group_size_m = (remaining_m < group_m).select(remaining_m, group_m)
        swizzled_n = intra_group // group_size_m
        swizzled_m = first_pid_m + (intra_group % group_size_m)
        use_simple = (num_wg < swizzle_threshold) | ((num_wg % num_xcds) != 0)
        if const_expr(isinstance(use_simple, bool)):
            if const_expr(use_simple):
                return simple_m, simple_n
            return swizzled_m, swizzled_n
        return (
            use_simple.select(simple_m, swizzled_m),
            use_simple.select(simple_n, swizzled_n),
        )


def make_lds_layout(rows, inner_extent, unit_bytes, is_transposed, full_mask=False):
    """XOR-swizzled LDS layout for one (rows, inner_extent) operand tile.

    inner_extent is counted in whatever unit the caller views the tile through
    and unit_bytes is that unit's width, so a BF16 kernel passes elements with
    unit_bytes=2 and a packed MXFP kernel passes bytes with unit_bytes=1.

    full_mask widens the XOR beyond the bank period, and applies to the
    row-major orientation only. The MXFP row-major path needs it because its
    LDS reader recomputes the same permutation by hand as
    `granule ^ (row & (granules_per_row - 1))`; changing one without the other
    silently mismatches the write and read addresses. The transposed
    orientation is read back through crd2idx on this layout, so it always
    follows the bank-period rule.
    """
    if const_expr(is_transposed):
        contiguous_extent = rows
        base = LDS_READ_TRANS_BASE
        order = (0, 1)
        full_mask = False
    else:
        contiguous_extent = inner_extent
        base = (GFX950_DMA_BYTES // unit_bytes).bit_length() - 1
        order = (1, 0)

    base_layout = fx.make_ordered_layout((rows, inner_extent), order)
    extent_log2 = contiguous_extent.bit_length() - 1
    shift = extent_log2 - base
    mask = shift if full_mask else LDS_BANK_PERIOD_LOG2 - base
    is_power_of_two = contiguous_extent == 1 << extent_log2
    if const_expr(not is_power_of_two or shift < mask):
        return base_layout
    return fx.make_composed_layout(
        fx.static(fx.SwizzleType.get(mask, base, shift)),
        base_layout,
    )


def make_ab_lds_layouts(
    rows_a, rows_b, inner_extent, unit_bytes, a_is_transposed, b_is_transposed, **kwargs
):
    return (
        make_lds_layout(rows_a, inner_extent, unit_bytes, a_is_transposed, **kwargs),
        make_lds_layout(
            rows_b, inner_extent, unit_bytes, not b_is_transposed, **kwargs
        ),
    )


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadContext:
    wave_offset: Any
    tid: Any
    inner_bound: Any
    block_threads: Any
    async_load_bytes: Any
    in_data_bytes: Any
    ldg_x_threads: Any
    block_k: Any
    has_k_tail: Any


@dataclass(slots=True, kw_only=True, eq=False)
class AsyncLoadOperand:
    context: AsyncLoadContext
    rsrc: Any
    lds_layout: Any
    outer_tile_size: Any
    outer_bound: Any
    leading_stride: Any
    load_iters: Any
    is_k_major: Any
    has_outer_tail: Any


def async_load_operand(
    operand: AsyncLoadOperand,
    lds_base,
    global_outer_offset,
    k_tile,
):
    context = operand.context
    tid = context.tid
    block_threads = context.block_threads
    async_load_bytes = context.async_load_bytes
    async_load_vec_size = async_load_bytes // context.in_data_bytes
    ldg_x_threads = context.ldg_x_threads
    block_k = context.block_k
    inner_bound = context.inner_bound
    lds_ptr = make_wave_lds_ptr(lds_base, context.wave_offset)
    for i in range_constexpr(operand.load_iters):
        global_tid = block_threads * i + tid
        if const_expr(operand.is_k_major):
            outer_x_threads = operand.outer_tile_size // async_load_vec_size
            outer_lds_idx = global_tid % outer_x_threads * async_load_vec_size
            k_local_idx = global_tid // outer_x_threads
            outer_local_idx = swizzled_contiguous_idx(
                outer_lds_idx,
                k_local_idx,
                operand.lds_layout,
                operand.outer_tile_size,
            )
            global_k_idx = k_tile * block_k + k_local_idx
        else:
            outer_local_idx = global_tid // ldg_x_threads
            k_local_idx = global_tid % ldg_x_threads * async_load_vec_size
            global_k_idx = k_tile * block_k + swizzled_contiguous_idx(
                outer_local_idx,
                k_local_idx,
                operand.lds_layout,
                block_k,
            )
        if const_expr(context.has_k_tail):
            safe_global_k_idx = (global_k_idx < inner_bound).select(global_k_idx, 0)
        else:
            safe_global_k_idx = global_k_idx
        global_outer_idx = global_outer_offset + outer_local_idx
        if const_expr(operand.has_outer_tail):
            safe_global_outer_idx = (global_outer_idx < operand.outer_bound).select(
                global_outer_idx, 0
            )
        else:
            safe_global_outer_idx = global_outer_idx
        if const_expr(operand.is_k_major):
            global_offset = (
                safe_global_k_idx * operand.leading_stride + safe_global_outer_idx
            ) * context.in_data_bytes
        else:
            global_offset = (
                safe_global_outer_idx * operand.leading_stride + safe_global_k_idx
            ) * context.in_data_bytes
        buffer_load_lds_inline(operand.rsrc, lds_ptr, global_offset, async_load_bytes)
        if i < operand.load_iters - 1:
            lds_ptr = lds_ptr + block_threads * async_load_bytes


@dataclass(frozen=True)
class TileSchedule:
    """Derived tile quantities shared by every gfx950 GEMM kernel."""

    block_threads: int
    ldg_x_threads: int
    ldg_a_iters: int
    ldg_b_iters: int
    ldg_wait_count: int
    a_stage_bytes: int
    b_stage_bytes: int
    mma_m_repeat: int
    mma_n_repeat: int


def make_tile_schedule(
    *,
    block_m: int,
    block_n: int,
    block_k_bytes: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    mma_m: int,
    mma_n: int,
    extra_wait_iters: int = 0,
    extra_stage_bytes: int = 0,
    epilogue_bytes: int = 0,
) -> TileSchedule:
    """Validate one tile config and return its direct-to-LDS load schedule.

    block_k_bytes is the packed K-row width, so the caller has already folded
    in its element width. Raises ValueError for any config the staged pipeline
    cannot express.
    """
    if block_m <= 0 or block_n <= 0 or block_k_bytes <= 0:
        raise ValueError("block_m, block_n, and block_k must be positive")
    if stages < 2:
        raise ValueError("stages must be at least 2 for the staged LDS pipeline")
    if m_waves <= 0 or n_waves <= 0:
        raise ValueError("m_waves and n_waves must be positive")
    if group_m < 0:
        raise ValueError("group_m must be non-negative")

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if block_threads > GFX950_MAX_BLOCK_THREADS:
        raise ValueError(f"block exceeds {GFX950_MAX_BLOCK_THREADS} threads")

    ldg_x_threads, remainder = divmod(block_k_bytes, GFX950_DMA_BYTES)
    if remainder:
        raise ValueError(
            "the packed K-row byte count must be divisible by the DMA width: "
            f"block_k_bytes={block_k_bytes}, dma_bytes={GFX950_DMA_BYTES}"
        )

    dma_bytes_per_pass = block_threads * GFX950_DMA_BYTES
    a_stage_bytes = block_m * block_k_bytes
    b_stage_bytes = block_n * block_k_bytes
    if a_stage_bytes % dma_bytes_per_pass:
        raise ValueError(
            "A tile load schedule must exactly cover the LDS tile: "
            f"block_m={block_m}, block_k_bytes={block_k_bytes}, "
            f"block_threads={block_threads}"
        )
    if b_stage_bytes % dma_bytes_per_pass:
        raise ValueError(
            "B tile load schedule must exactly cover the LDS tile: "
            f"block_n={block_n}, block_k_bytes={block_k_bytes}, "
            f"block_threads={block_threads}"
        )
    ldg_a_iters = a_stage_bytes // dma_bytes_per_pass
    ldg_b_iters = b_stage_bytes // dma_bytes_per_pass

    mma_m_repeat, rem_m = divmod(block_m, m_waves * mma_m)
    mma_n_repeat, rem_n = divmod(block_n, n_waves * mma_n)
    if rem_m or mma_m_repeat == 0:
        raise ValueError(
            f"block_m must be a positive multiple of m_waves * mma_m: "
            f"block_m={block_m}, m_waves={m_waves}, mma_m={mma_m}"
        )
    if rem_n or mma_n_repeat == 0:
        raise ValueError(
            f"block_n must be a positive multiple of n_waves * mma_n: "
            f"block_n={block_n}, n_waves={n_waves}, mma_n={mma_n}"
        )

    smem_bytes = stages * (a_stage_bytes + b_stage_bytes + extra_stage_bytes)
    capacity = lds_capacity()
    if max(smem_bytes, epilogue_bytes) > capacity:
        raise ValueError(
            "staged LDS buffers exceed the device shared-memory capacity: "
            f"stages={stages}, block_m={block_m}, block_n={block_n}, "
            f"block_k_bytes={block_k_bytes}, smem_bytes={smem_bytes}, "
            f"epilogue_bytes={epilogue_bytes}, capacity={capacity}"
        )

    ldg_wait_count = ldg_a_iters + ldg_b_iters + extra_wait_iters
    if (stages - 2) * ldg_wait_count >= 63:
        raise ValueError("staged pipeline wait count exceeds supported range")

    return TileSchedule(
        block_threads=block_threads,
        ldg_x_threads=ldg_x_threads,
        ldg_a_iters=ldg_a_iters,
        ldg_b_iters=ldg_b_iters,
        ldg_wait_count=ldg_wait_count,
        a_stage_bytes=a_stage_bytes,
        b_stage_bytes=b_stage_bytes,
        mma_m_repeat=mma_m_repeat,
        mma_n_repeat=mma_n_repeat,
    )


@dataclass(slots=True, kw_only=True, eq=False)
class ABCopyAtoms:
    uni_copy_atom: Any
    buffer_copy_atom: Any
    a_s2r_copy_atom: Any
    b_s2r_copy_atom: Any
    thr_copy_a: Any
    thr_copy_b: Any


def make_ab_s2r_atoms(
    elem_dtype, trans_op, tiled_mma, tid, a_is_transposed, b_is_transposed
) -> ABCopyAtoms:
    """Pick LDS-to-register atoms for A and B.

    A K-major operand reads straight back through a 128-bit copy; the other
    orientation needs the transposing ds_read, whose width depends on the
    element and so arrives as trans_op from the caller.
    """
    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    trans_copy_atom = fx.make_copy_atom(trans_op, elem_dtype)

    if const_expr(a_is_transposed):
        a_s2r_copy_atom = a_tiled_copy_atom = trans_copy_atom
    else:
        a_s2r_copy_atom, a_tiled_copy_atom = uni_copy_atom, buffer_copy_atom
    if const_expr(not b_is_transposed):
        b_s2r_copy_atom = b_tiled_copy_atom = trans_copy_atom
    else:
        b_s2r_copy_atom, b_tiled_copy_atom = uni_copy_atom, buffer_copy_atom

    return ABCopyAtoms(
        uni_copy_atom=uni_copy_atom,
        buffer_copy_atom=buffer_copy_atom,
        a_s2r_copy_atom=a_s2r_copy_atom,
        b_s2r_copy_atom=b_s2r_copy_atom,
        thr_copy_a=fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid),
        thr_copy_b=fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid),
    )


@dataclass(slots=True, kw_only=True, eq=False)
class CShufflePlan:
    sC: Any
    thr_sC: Any
    thr_gC: Any
    thr_mma_cRow: Any
    thr_mma_cCol: Any
    s2r_atom: Any
    r2g_atom: Any
    frag_C_cshuffle: Any
    pred_C: Any


def make_cshuffle_plan(
    *,
    block_m,
    block_n,
    block_threads,
    out_data_bytes,
    tid,
    thr_mma,
    smem_c,
    gC,
    s2r_atom,
    r2g_atom,
    want_pred=False,
):
    """Stage the accumulator through LDS so the global store is contiguous.

    The MMA fragment layout scatters each lane's accumulator across the tile;
    writing it straight out would produce 2-byte stores. Bouncing through LDS
    lets every lane read back a 16-byte run of one row.
    """
    c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
    sC = fx.make_view(smem_c, c_lds_layout)
    row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))

    cshuffle_vec_size = GFX950_DMA_BYTES // out_data_bytes
    cshuffle_x_threads = block_n // cshuffle_vec_size
    cshuffle_tile, cshuffle_tv_layout = fx.make_layout_tv(
        fx.make_layout(
            (block_threads // cshuffle_x_threads, cshuffle_x_threads),
            (cshuffle_x_threads, 1),
        ),
        fx.make_layout((1, cshuffle_vec_size), (1, 1)),
    )
    thr_copy = fx.make_tiled_copy(
        r2g_atom, cshuffle_tv_layout, cshuffle_tile
    ).get_slice(tid)
    thr_sC = thr_copy.partition_S(sC)

    if const_expr(want_pred):
        thr_cRow = thr_copy.partition_S(row_coords)[(0, None), None, None]
        thr_cCol = thr_copy.partition_S(col_coords)[(0, None), None, None]
        pred_C = (fx.make_fragment_like(thr_cRow, dtype=fx.Boolean), thr_cRow, thr_cCol)
    else:
        pred_C = None

    return CShufflePlan(
        sC=sC,
        thr_sC=thr_sC,
        thr_gC=thr_copy.partition_D(gC),
        thr_mma_cRow=thr_mma.partition_C(row_coords),
        thr_mma_cCol=thr_mma.partition_C(col_coords),
        s2r_atom=s2r_atom,
        r2g_atom=r2g_atom,
        frag_C_cshuffle=fx.make_fragment_like(thr_sC),
        pred_C=pred_C,
    )


def store_c_tile(plan: CShufflePlan, frag_C_out, pred=None):
    fx.gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(plan.thr_mma_cRow[i])
        col = fx.get_scalar(plan.thr_mma_cCol[i])
        plan.sC[row, col] = frag_C_out[i]
    fx.gpu.barrier()
    fx.copy(plan.s2r_atom, plan.thr_sC, plan.frag_C_cshuffle)
    if const_expr(pred is None):
        fx.copy(plan.r2g_atom, plan.frag_C_cshuffle, plan.thr_gC)
    else:
        fx.copy(plan.r2g_atom, plan.frag_C_cshuffle, plan.thr_gC, pred=pred)


def run_staged_pipeline(
    *, stages, main_loop_end, ldg_wait_count, load_a, load_b, compute
):
    """Drive the multi-buffered LDS pipeline.

    Stage s is filled while stage s-1 is consumed, so each iteration issues the
    DMA for the tile stages-1 ahead and then computes on the tile whose DMA has
    already landed. The barrier's vmcnt leaves exactly the in-flight loads of
    the intervening stages outstanding.
    """
    for stage in range_constexpr(stages - 1):
        load_b(stage, stage)
        load_a(stage, stage)
    rocdl.sched_barrier(0)

    for k_tile in range(0, main_loop_end, 1):
        current_stage = k_tile % stages
        write_stage = (current_stage + stages - 1) % stages
        barrier((stages - 2) * ldg_wait_count)
        load_b(k_tile + (stages - 1), write_stage)
        load_a(k_tile + (stages - 1), write_stage)
        compute(current_stage, k_tile)

    current_stage = main_loop_end % stages
    for s in range_constexpr(0, stages - 1):
        barrier((stages - 2 - s) * ldg_wait_count)
        compute(current_stage, main_loop_end + s)
        current_stage = (current_stage + 1) % stages


def make_kernel_name(prefix: str, *, block_m, block_n, block_k, stages, m_waves, n_waves, group_m, a_is_transposed, b_is_transposed, **flags) -> str:
    name = f"{prefix}_t{block_m}x{block_n}x{block_k}x{stages}"
    name += f"_w{m_waves}x{n_waves}_gm{group_m}"
    for key, value in flags.items():
        name += f"_{key}{int(value)}"
    name += f"_l{'t' if a_is_transposed else 'n'}{'t' if b_is_transposed else 'n'}"
    return name


def make_gemm_tiled_mma(mma_op, m_waves, n_waves, permutation=None):
    """Tile one MMA atom over an (m_waves, n_waves) wave grid.

    Returns the atom alongside the tiled MMA because the scaled path issues
    per-16x16 `fx.gemm` calls against the atom directly, while the unscaled
    path drives the whole tile through the tiled form.
    """
    mma_atom = fx.make_mma_atom(mma_op)
    wave_layout = fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0))
    if const_expr(permutation is None):
        return mma_atom, fx.make_tiled_mma(mma_atom, wave_layout)
    return mma_atom, fx.make_tiled_mma(mma_atom, wave_layout, permutation)
