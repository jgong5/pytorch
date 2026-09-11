# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl

from .gfx950_common import (
    ABCopyAtoms,
    AsyncLoadContext,
    AsyncLoadOperand,
    async_load_operand,
    barrier,
    BlockSwizzle,
    GEMM_DTYPE_BF16,
    GEMM_DTYPE_FP16,
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    get_wave_lds_offset,
    make_ab_lds_layouts,
    make_ab_s2r_atoms,
    make_cshuffle_plan,
    make_gemm_tiled_mma,
    make_kernel_name,
    make_tile_schedule,
    run_staged_pipeline,
    store_c_tile,
    waitcnt,
)


IN_DATA_BYTES = 2
OUT_DATA_BYTES = 2


@fx.struct
class GemmGfx950Param:
    dtype_id: fx.Constexpr[int]
    block_m: fx.Constexpr[int]
    block_n: fx.Constexpr[int]
    block_k: fx.Constexpr[int]
    stages: fx.Constexpr[int]
    m_waves: fx.Constexpr[int]
    n_waves: fx.Constexpr[int]
    group_m: fx.Constexpr[int]
    use_half_tile_interleaved: fx.Constexpr[bool]
    a_is_transposed: fx.Constexpr[bool]
    b_is_transposed: fx.Constexpr[bool]
    has_bias: fx.Constexpr[bool]
    has_k_tail: fx.Constexpr[bool]
    async_load_bytes: fx.Constexpr[int]
    in_data_bytes: fx.Constexpr[int]
    out_data_bytes: fx.Constexpr[int]
    ldg_x_threads: fx.Constexpr[int]
    block_threads: fx.Constexpr[int]
    ldg_a_iters: fx.Constexpr[int]
    ldg_b_iters: fx.Constexpr[int]
    mma_m: fx.Constexpr[int]
    mma_n: fx.Constexpr[int]
    mma_k: fx.Constexpr[int]


@dataclass(slots=True, kw_only=True, eq=False)
class GemmABLoadContext(AsyncLoadContext, ABCopyAtoms):
    pass


def make_gemm_gfx950_param(
    dtype_id: int = GEMM_DTYPE_BF16,
    tile_m: int = 256,
    tile_n: int = 256,
    tile_k: int = 64,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 4,
    group_m: int = 0,
    use_half_tile_interleaved: bool = False,
    *,
    a_is_transposed: bool,
    b_is_transposed: bool,
    has_bias: bool = False,
    has_k_tail: bool = False,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
) -> GemmGfx950Param:
    # Keep the kernel implementation's internal block terminology unchanged.
    block_m, block_n, block_k = tile_m, tile_n, tile_k
    if dtype_id not in (GEMM_DTYPE_BF16, GEMM_DTYPE_FP16):
        raise ValueError(f"unsupported dtype_id={dtype_id}")
    if (mma_m, mma_n, mma_k) != (16, 16, 32):
        raise ValueError("the gfx950 layout kernel currently requires mma=16x16x32")

    in_dbytes, out_dbytes = IN_DATA_BYTES, OUT_DATA_BYTES
    schedule = make_tile_schedule(
        block_m=block_m,
        block_n=block_n,
        block_k_bytes=block_k * in_dbytes,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        mma_m=mma_m,
        mma_n=mma_n,
        epilogue_bytes=block_m * block_n * out_dbytes,
    )

    cshuffle_vec_size = GFX950_DMA_BYTES // out_dbytes
    if use_half_tile_interleaved:
        half_block_m = block_m // 2
        half_block_n = block_n // 2
        if stages != 2:
            raise ValueError("half-tile interleaved kernel requires stages=2")
        if m_waves != 2 or n_waves < 2:
            raise ValueError(
                "half-tile interleaved kernel requires m_waves=2 and n_waves>=2"
            )
        if half_block_m * 2 != block_m or half_block_n * 2 != block_n:
            raise ValueError(
                "half-tile interleaved kernel requires even block_m and block_n"
            )
        mma_m_half_repeat = half_block_m // m_waves // mma_m
        mma_n_half_repeat = half_block_n // n_waves // mma_n
        if mma_m_half_repeat * m_waves * mma_m != half_block_m:
            raise ValueError("half block_m must be divisible by m_waves * mma_m")
        if mma_n_half_repeat * n_waves * mma_n != half_block_n:
            raise ValueError("half block_n must be divisible by n_waves * mma_n")
        if mma_n_half_repeat != 2:
            raise ValueError(
                "half-tile interleaved kernel requires "
                "half_block_n / n_waves / mma_n == 2"
            )
        if half_block_n % cshuffle_vec_size != 0:
            raise ValueError(
                "half block_n must be divisible by the c-shuffle vector size"
            )
    elif block_n % cshuffle_vec_size != 0:
        raise ValueError("block_n must be divisible by the c-shuffle vector size")

    if use_half_tile_interleaved:
        load_elems_per_iter = schedule.block_threads * (GFX950_DMA_BYTES // in_dbytes)
        if ((block_m // 2) * block_k) % load_elems_per_iter:
            raise ValueError(
                "half-tile A load schedule must exactly cover the LDS tile"
            )
        if ((block_n // 2) * block_k) % load_elems_per_iter:
            raise ValueError(
                "half-tile B load schedule must exactly cover the LDS tile"
            )

    return GemmGfx950Param(
        dtype_id=dtype_id,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        group_m=group_m,
        use_half_tile_interleaved=use_half_tile_interleaved,
        a_is_transposed=a_is_transposed,
        b_is_transposed=b_is_transposed,
        has_bias=has_bias,
        has_k_tail=has_k_tail,
        async_load_bytes=GFX950_DMA_BYTES,
        in_data_bytes=in_dbytes,
        out_data_bytes=out_dbytes,
        ldg_x_threads=schedule.ldg_x_threads,
        block_threads=schedule.block_threads,
        ldg_a_iters=schedule.ldg_a_iters,
        ldg_b_iters=schedule.ldg_b_iters,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
    )


def make_gemm_gfx950_kernel_name(param: GemmGfx950Param) -> str:
    dtype_str = "fp16" if param.dtype_id == GEMM_DTYPE_FP16 else "bf16"
    return make_kernel_name(
        f"gemm_{dtype_str}",
        block_m=param.block_m,
        block_n=param.block_n,
        block_k=param.block_k,
        stages=param.stages,
        m_waves=param.m_waves,
        n_waves=param.n_waves,
        group_m=param.group_m,
        a_is_transposed=param.a_is_transposed,
        b_is_transposed=param.b_is_transposed,
        bias=param.has_bias,
        ktail=param.has_k_tail,
        hti=param.use_half_tile_interleaved,
    )



def _elem_dtype(param: GemmGfx950Param):
    return fx.Float16 if const_expr(param.dtype_id == GEMM_DTYPE_FP16) else fx.BFloat16


def make_gemm_ab_load_context(elem_dtype, tiled_mma, tid, k, param: GemmGfx950Param):
    atoms = make_ab_s2r_atoms(
        elem_dtype,
        fx.rocdl.cdna4.LDSReadTrans16_64b(),
        tiled_mma,
        tid,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    return GemmABLoadContext(
        wave_offset=get_wave_lds_offset(tid, param.async_load_bytes),
        tid=tid,
        inner_bound=k,
        block_threads=param.block_threads,
        async_load_bytes=param.async_load_bytes,
        in_data_bytes=param.in_data_bytes,
        ldg_x_threads=param.ldg_x_threads,
        block_k=param.block_k,
        has_k_tail=param.has_k_tail,
        uni_copy_atom=atoms.uni_copy_atom,
        buffer_copy_atom=atoms.buffer_copy_atom,
        a_s2r_copy_atom=atoms.a_s2r_copy_atom,
        b_s2r_copy_atom=atoms.b_s2r_copy_atom,
        thr_copy_a=atoms.thr_copy_a,
        thr_copy_b=atoms.thr_copy_b,
    )



def _make_gemm_gfx950_tiled_mma(param: GemmGfx950Param):
    k_per_mfma_group = param.mma_k // 4
    _, tiled_mma = make_gemm_tiled_mma(
        fx.rocdl.MFMA(param.mma_m, param.mma_n, param.mma_k, _elem_dtype(param)),
        param.m_waves,
        param.n_waves,
        fx.make_tile(
            None,
            None,
            fx.make_layout((k_per_mfma_group, 4), (1, k_per_mfma_group)),
        ),
    )
    return tiled_mma


@flyc.kernel
def gemm_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    stages = param.stages
    has_k_tail = param.has_k_tail
    block_threads = param.block_threads
    ldg_a_iters = param.ldg_a_iters
    ldg_b_iters = param.ldg_b_iters
    ldg_wait_count = ldg_a_iters + ldg_b_iters
    elem_dtype = _elem_dtype(param)

    tid = fx.thread_idx.x
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out_buf = fx.rocdl.make_buffer_tensor(out, max_size=True)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))

    gC = fx.flat_divide(out_buf, (block_m, block_n))[None, None, bid_m, bid_n]
    thr_mma = tiled_mma.thr_slice(tid)
    ab_load_context = make_gemm_ab_load_context(elem_dtype, tiled_mma, tid, k, param)
    uni_copy_atom = ab_load_context.uni_copy_atom
    buffer_copy_atom = ab_load_context.buffer_copy_atom
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    a_lds_layout, b_lds_layout = make_ab_lds_layouts(
        block_m,
        block_n,
        block_k,
        param.in_data_bytes,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=a_rsrc,
        lds_layout=a_lds_layout,
        outer_tile_size=block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=ldg_a_iters,
        is_k_major=param.a_is_transposed,
        has_outer_tail=True,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=b_rsrc,
        lds_layout=b_lds_layout,
        outer_tile_size=block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=ldg_b_iters,
        is_k_major=not param.b_is_transposed,
        has_outer_tail=True,
    )
    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)

    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)
    frag_C = thr_mma.make_fragment_C(gC)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)

    cshuffle = make_cshuffle_plan(
        block_m=block_m,
        block_n=block_n,
        block_threads=block_threads,
        out_data_bytes=param.out_data_bytes,
        tid=tid,
        thr_mma=thr_mma,
        smem_c=smem_c,
        gC=gC,
        s2r_atom=uni_copy_atom,
        r2g_atom=buffer_copy_atom,
        want_pred=True,
    )
    thr_mma_cCol = cshuffle.thr_mma_cCol
    pred_C, thr_cRow, thr_cCol = cshuffle.pred_C

    frag_C.fill(0.0)
    if const_expr(param.has_bias):
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            col_idx = fx.get_scalar(thr_mma_cCol[i])
            global_n_idx = bid_n * block_n + col_idx
            safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
            frag_C[i] = bias_buf[safe_global_n_idx].to(fx.Float32)

    for i in range_constexpr(fx.size(pred_C.shape).unpack()):
        local_row = fx.get_scalar(thr_cRow[i])
        local_col = fx.get_scalar(thr_cCol[i])
        row_idx = bid_m * block_m + local_row
        col_idx = bid_n * block_n + local_col
        pred_C[i] = (
            (local_row < block_m)
            & (local_col < block_n)
            & (row_idx < m)
            & (col_idx < n)
        )

    def async_load_a_to_lds(k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=smem_a + stage * block_m * block_k,
            global_outer_offset=bid_m * block_m,
            k_tile=k_tile,
        )

    def async_load_b_to_lds(k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=smem_b + stage * block_n * block_k,
            global_outer_offset=bid_n * block_n,
            k_tile=k_tile,
        )

    def compute_stage(read_stage, k_tile):
        sA_stage = fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
        sB_stage = fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
        thr_sA_s2r = thr_copy_A.partition_S(sA_stage)
        thr_sB_s2r = thr_copy_B.partition_S(sB_stage)

        def compute_k_chunk(block_k_iter):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, block_k_iter],
                frag_B_retile[None, None, block_k_iter],
            )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, block_k_iter],
                frag_A_retile[None, None, block_k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, block_k_iter],
                frag_B[None, None, block_k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    compute_k_chunk(block_k_iter)
            else:
                compute_k_chunk(block_k_iter)

    if const_expr(has_k_tail):
        main_loop_end = (k_tiles > stages - 1).select(k_tiles - (stages - 1), 0)
    else:
        main_loop_end = k_tiles - (stages - 1)
    run_staged_pipeline(
        stages=stages,
        main_loop_end=main_loop_end,
        ldg_wait_count=ldg_wait_count,
        load_a=async_load_a_to_lds,
        load_b=async_load_b_to_lds,
        compute=compute_stage,
    )

    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))
    store_c_tile(cshuffle, frag_C_out, pred=pred_C)


@flyc.kernel
def gemm_hti_gfx950_kernel(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    bias: fx.Tensor,
    m: fx.Int32,
    n: fx.Int32,
    k: fx.Int32,
    a_leading_stride: fx.Int32,
    b_leading_stride: fx.Int32,
    tiled_mma: fx.TiledMma,
    param: GemmGfx950Param,
):
    block_m = param.block_m
    block_n = param.block_n
    block_k = param.block_k
    half_block_m = block_m // 2
    half_block_n = block_n // 2
    stages = param.stages
    has_k_tail = param.has_k_tail
    block_threads = param.block_threads
    n_waves = param.n_waves
    half_ldg_a_iters = param.ldg_a_iters // 2
    half_ldg_b_iters = param.ldg_b_iters // 2
    elem_dtype = _elem_dtype(param)

    tid = fx.thread_idx.x
    wid = tid // GFX950_WAVE_SIZE
    num_pid_m = (m - 1) // block_m + 1
    num_pid_n = (n - 1) // block_n + 1
    block_swizzle = BlockSwizzle(
        NUM_XCDS=8, NUM_PIDS_THRESHOLD=256, GROUP_M=param.group_m
    )
    bid_m, bid_n = block_swizzle.swizzle(num_pid_m, num_pid_n, fx.block_idx.x)
    k_tiles = (k - 1) // block_k + 1

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    storage = fx.SharedAllocator().allocate(SharedStorage)
    smem_a = storage.ab.a.peek().ptr
    smem_b = storage.ab.b.peek().ptr
    smem_c = storage.c.peek().ptr

    a_buf = fx.rocdl.make_buffer_tensor(a, max_size=True)
    b_buf = fx.rocdl.make_buffer_tensor(b, max_size=True)
    out = fx.rocdl.make_buffer_tensor(out, max_size=False)
    if const_expr(param.has_bias):
        bias_buf = fx.rocdl.make_buffer_tensor(bias, max_size=True)
    else:
        bias_buf = None

    a_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(a_buf))
    b_rsrc = fx.rocdl.get_buffer_rsrc(fx.get_iter(b_buf))

    thr_mma = tiled_mma.thr_slice(tid)
    ab_load_context = make_gemm_ab_load_context(elem_dtype, tiled_mma, tid, k, param)
    uni_copy_atom = ab_load_context.uni_copy_atom
    buffer_copy_atom = ab_load_context.buffer_copy_atom
    a_s2r_copy_atom = ab_load_context.a_s2r_copy_atom
    b_s2r_copy_atom = ab_load_context.b_s2r_copy_atom
    thr_copy_A = ab_load_context.thr_copy_a
    thr_copy_B = ab_load_context.thr_copy_b
    a_lds_layout, b_lds_layout = make_ab_lds_layouts(
        half_block_m,
        half_block_n,
        block_k,
        param.in_data_bytes,
        param.a_is_transposed,
        param.b_is_transposed,
    )
    a_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=a_rsrc,
        lds_layout=a_lds_layout,
        outer_tile_size=half_block_m,
        outer_bound=m,
        leading_stride=a_leading_stride,
        load_iters=half_ldg_a_iters,
        is_k_major=param.a_is_transposed,
        has_outer_tail=True,
    )
    b_load_operand = AsyncLoadOperand(
        context=ab_load_context,
        rsrc=b_rsrc,
        lds_layout=b_lds_layout,
        outer_tile_size=half_block_n,
        outer_bound=n,
        leading_stride=b_leading_stride,
        load_iters=half_ldg_b_iters,
        is_k_major=not param.b_is_transposed,
        has_outer_tail=True,
    )

    def half_a_base(stage, m_part):
        return smem_a + (stage * block_m + m_part * half_block_m) * block_k

    def half_b_base(stage, n_part):
        return smem_b + (stage * block_n + n_part * half_block_n) * block_k

    def async_load_a_to_lds(m_part, k_tile, stage):
        async_load_operand(
            a_load_operand,
            lds_base=half_a_base(stage, m_part),
            global_outer_offset=bid_m * block_m + m_part * half_block_m,
            k_tile=k_tile,
        )

    def async_load_b_to_lds(n_part, k_tile, stage):
        async_load_operand(
            b_load_operand,
            lds_base=half_b_base(stage, n_part),
            global_outer_offset=bid_n * block_n + n_part * half_block_n,
            k_tile=k_tile,
        )

    def make_gC(m_part, n_part):
        return fx.flat_divide(out, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]

    def make_c_fragment(m_part, n_part):
        gC = make_gC(m_part, n_part)
        frag_C = thr_mma.make_fragment_C(gC)
        frag_C.fill(0.0)
        return frag_C

    def load_a_fragment(m_part, read_stage, k_tile):
        sA = fx.make_view(half_a_base(read_stage, m_part), a_lds_layout)
        frag_A = thr_mma.make_fragment_A(sA)
        frag_A_retile = thr_copy_A.retile(frag_A)
        thr_sA_s2r = thr_copy_A.partition_S(sA)

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        a_s2r_copy_atom,
                        thr_sA_s2r[None, None, block_k_iter],
                        frag_A_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    a_s2r_copy_atom,
                    thr_sA_s2r[None, None, block_k_iter],
                    frag_A_retile[None, None, block_k_iter],
                )
        return frag_A

    def load_b_fragment(n_part, read_stage, k_tile):
        sB = fx.make_view(half_b_base(read_stage, n_part), b_lds_layout)
        frag_B = thr_mma.make_fragment_B(sB)
        frag_B_retile = thr_copy_B.retile(frag_B)
        thr_sB_s2r = thr_copy_B.partition_S(sB)

        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.copy(
                        b_s2r_copy_atom,
                        thr_sB_s2r[None, None, block_k_iter],
                        frag_B_retile[None, None, block_k_iter],
                    )
            else:
                fx.copy(
                    b_s2r_copy_atom,
                    thr_sB_s2r[None, None, block_k_iter],
                    frag_B_retile[None, None, block_k_iter],
                )
        return frag_B

    def consume(k_tile, frag_C, frag_A, frag_B, emit_sched_barrier):
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)
        for block_k_iter in range_constexpr(block_k // param.mma_k):
            if const_expr(has_k_tail):
                global_k_iter = k_tile * block_k + block_k_iter * param.mma_k
                if global_k_iter < k:
                    fx.gemm(
                        tiled_mma,
                        frag_C,
                        frag_A[None, None, block_k_iter],
                        frag_B[None, None, block_k_iter],
                        frag_C,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )
            else:
                fx.gemm(
                    tiled_mma,
                    frag_C,
                    frag_A[None, None, block_k_iter],
                    frag_B[None, None, block_k_iter],
                    frag_C,
                    traversal_order=fx.GemmTraversalOrder.KNM,
                )
        if const_expr(emit_sched_barrier):
            rocdl.sched_barrier(0)

    def store_half_tile(m_part, n_part, frag_C):
        gC = fx.flat_divide(out, (half_block_m, half_block_n))[
            None, None, bid_m * 2 + m_part, bid_n * 2 + n_part
        ]
        cshuffle = make_cshuffle_plan(
            block_m=half_block_m,
            block_n=half_block_n,
            block_threads=block_threads,
            out_data_bytes=param.out_data_bytes,
            tid=tid,
            thr_mma=thr_mma,
            smem_c=smem_c,
            gC=gC,
            s2r_atom=uni_copy_atom,
            r2g_atom=buffer_copy_atom,
            want_pred=True,
        )
        pred_C, thr_cRow, thr_cCol = cshuffle.pred_C

        for i in range_constexpr(fx.size(pred_C.shape).unpack()):
            local_row = fx.get_scalar(thr_cRow[i])
            local_col = fx.get_scalar(thr_cCol[i])
            row_idx = bid_m * block_m + m_part * half_block_m + local_row
            col_idx = bid_n * block_n + n_part * half_block_n + local_col
            pred_C[i] = (
                (local_row < half_block_m)
                & (local_col < half_block_n)
                & (row_idx < m)
                & (col_idx < n)
            )

        frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
        for i in range_constexpr(fx.size(frag_C.shape).unpack()):
            val = frag_C[i]
            if const_expr(param.has_bias):
                col = fx.get_scalar(cshuffle.thr_mma_cCol[i])
                global_n_idx = bid_n * block_n + n_part * half_block_n + col
                safe_global_n_idx = (global_n_idx < n).select(global_n_idx, 0)
                val = val + bias_buf[safe_global_n_idx].to(fx.Float32)
            frag_C_out[i] = val.to(elem_dtype)

        store_c_tile(cshuffle, frag_C_out, pred=pred_C)
        fx.gpu.barrier()

    c00 = make_c_fragment(0, 0)
    c01 = make_c_fragment(0, 1)
    c10 = make_c_fragment(1, 0)
    c11 = make_c_fragment(1, 1)

    async_load_b_to_lds(0, 0, 0)
    async_load_a_to_lds(0, 0, 0)
    async_load_b_to_lds(1, 0, 0)
    async_load_a_to_lds(1, 0, 0)
    rocdl.sched_barrier(0)
    if wid // n_waves == 1:
        rocdl.s_barrier()
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)
    async_load_b_to_lds(0, 1, 1)
    async_load_a_to_lds(0, 1, 1)
    async_load_b_to_lds(1, 1, 1)
    barrier(half_ldg_b_iters + half_ldg_a_iters)

    def compute_double_tile(k_tile, prefetch_next):
        next_k_tile = k_tile + 2

        b0 = load_b_fragment(0, 0, k_tile)
        a0 = load_a_fragment(0, 0, k_tile)
        async_load_a_to_lds(1, k_tile + 1, 1)
        rocdl.s_barrier()
        consume(k_tile, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 0, k_tile)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile, c10, a1, b0, True)
        rocdl.s_barrier()

        b0 = load_b_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile, 0)
            barrier(2 * half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile, c11, a1, b1, True)
        if const_expr(not prefetch_next):
            waitcnt(0)
        rocdl.s_barrier()

        a0 = load_a_fragment(0, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(1, next_k_tile, 0)
            rocdl.s_barrier()
        consume(k_tile + 1, c00, a0, b0, True)
        rocdl.s_barrier()

        b1 = load_b_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_b_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c01, a0, b1, True)
        rocdl.s_barrier()

        a1 = load_a_fragment(1, 1, k_tile + 1)
        if const_expr(prefetch_next):
            async_load_a_to_lds(0, next_k_tile + 1, 1)
            rocdl.s_barrier()
        consume(k_tile + 1, c10, a1, b0, True)
        rocdl.s_barrier()

        if const_expr(prefetch_next):
            async_load_b_to_lds(1, next_k_tile + 1, 1)
            barrier(half_ldg_b_iters + half_ldg_a_iters)
        consume(k_tile + 1, c11, a1, b1, True)
        rocdl.s_barrier()

    final_double_tile = ((k_tiles % 2) == 0).select(k_tiles - 2, k_tiles - 1)
    main_loop_end = (k_tiles > 2).select(final_double_tile, 0)
    for k_tile in range(0, main_loop_end, 2):
        compute_double_tile(k_tile, True)

    compute_double_tile(main_loop_end, False)

    store_half_tile(0, 0, c00)
    store_half_tile(0, 1, c01)
    store_half_tile(1, 0, c10)
    store_half_tile(1, 1, c11)


@flyc.jit
def gemm_gfx950(
    out: fx.Tensor,
    a: fx.Tensor,
    b: fx.Tensor,
    param: GemmGfx950Param,
    stream: fx.Stream = fx.Stream(None),
):
    r"""gemm_gfx950(out, a, b, param, stream=fx.Stream(None))

    Compute ``out[M, N] = a[M, K] @ b[K, N]``. The physical input layouts are
    specified by ``param.a_is_transposed`` and ``param.b_is_transposed``.
    """
    m = fx.Int32(fx.get_scalar(a.shape[0]))
    n = fx.Int32(fx.get_scalar(b.shape[1]))
    k = fx.Int32(fx.get_scalar(a.shape[1]))
    a_leading_stride = fx.Int32(
        fx.get_scalar(a.stride[1] if const_expr(param.a_is_transposed) else a.stride[0])
    )
    b_leading_stride = fx.Int32(
        fx.get_scalar(b.stride[1] if const_expr(param.b_is_transposed) else b.stride[0])
    )
    tiled_mma = _make_gemm_gfx950_tiled_mma(param)
    num_pid_m = (m - 1) // param.block_m + 1
    num_pid_n = (n - 1) // param.block_n + 1
    kernel_impl = (
        gemm_hti_gfx950_kernel
        if param.use_half_tile_interleaved
        else gemm_gfx950_kernel
    )
    kernel_impl._known_block_size = [param.block_threads, 1, 1]
    kernel_impl._func.__name__ = make_gemm_gfx950_kernel_name(param)
    kernel_impl(
        out,
        a,
        b,
        out,
        m,
        n,
        k,
        a_leading_stride,
        b_leading_stride,
        tiled_mma,
        param,
    ).launch(
        grid=(num_pid_m * num_pid_n, 1, 1),
        block=(param.block_threads, 1, 1),
        stream=stream,
    )


def infer_has_k_tail(k: int, tile_k: int, stages: int):
    k_tiles = (k + tile_k - 1) // tile_k
    return (k % tile_k != 0) or (k_tiles < stages - 1)


def make_gemm_param_and_validate(m, n, k, kwargs):
    result = None
    try:
        result = make_gemm_gfx950_param(**kwargs)
    except Exception:
        return None
    output_vec_size = GFX950_DMA_BYTES // result.out_data_bytes
    if n % output_vec_size != 0 or k % result.mma_k != 0:
        return None
    async_load_vec_size = GFX950_DMA_BYTES // result.in_data_bytes
    if result.a_is_transposed and m % async_load_vec_size != 0:
        return None
    if result.b_is_transposed and k % async_load_vec_size != 0:
        return None
    if result.use_half_tile_interleaved:
        k_tiles = (k + result.block_k - 1) // result.block_k
        if k_tiles < 2:
            return None
    return result
