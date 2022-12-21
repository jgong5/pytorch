from collections import namedtuple
import contextlib
import dataclasses
from enum import Enum
import functools
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import sympy

import torch
from torch._prims_common import is_float_dtype

from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..utils import sympy_product, sympy_subs, sympy_symbol
from ..virtualized import ops, V
from .common import (
    BracesBuffer,
    CppWrapperKernelArgs,
    DeferredIndentedBuffer,
    ExprPrinter,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
)

DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "long",
    torch.int32: "int",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
}

DTYPE_TO_ATEN = {
    torch.float32: "at::ScalarType::Float",
    torch.float64: "at::ScalarType::Double",
    torch.float16: "at::ScalarType::Half",
    torch.int64: "at::ScalarType::Long",
    torch.int32: "at::ScalarType::Int",
    torch.int16: "at::ScalarType::Short",
    torch.int8: "at::ScalarType::Char",
    torch.uint8: "at::ScalarType::Byte",
    torch.bool: "at::ScalarType::Bool",
    torch.bfloat16: "at::ScalarType::BFloat16",
}

INDEX_TYPE = "long"

RTYPE_TO_CPP = {
    "sum": "+",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
}

class CppKernelType(Enum):
    SCALAR = 0
    VECTOR = 1
    TILE2D = 2
    TILE2D_TAIL = 3

def reduction_init(reduction_type, dtype):
    if reduction_type in ("sum", "any"):
        return 0
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    raise AssertionError(reduction_type)


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} += {next_value}"
    if reduction_type == "any":
        return f"{var} = {var} || {next_value}"
    return f"{var} = std::{reduction_type}({var}, {next_value})"


def reduction_combine_vec(reduction_type, var, next_value):
    if reduction_type == "max":
        return f"{var} = at::vec::maximum({var}, {next_value})"
    elif reduction_type == "min":
        return f"{var} = at::vec::minimum({var}, {next_value})"
    elif reduction_type == "sum":
        return f"{var} += {next_value}"
    else:
        raise NotImplementedError()


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    if reduction_type == "argmax":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmax : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    elif reduction_type == "argmin":
        prefix.extend(
            [
                f"#pragma omp declare reduction(argmin : struct {struct_name} :\\",
                "    omp_out.value = omp_in.value > omp_out.value ? omp_out.value : omp_in.value,\\",
                "    omp_out.index = omp_in.value > omp_out.value ? omp_out.index : omp_in.index)\\",
                f"\tinitializer(omp_priv = {{0, {reduction_init(reduction_type, src_dtype)}}})",
            ]
        )
    return prefix


def float16_reduction_prefix(rtype):
    # TODO: This user-defined reduction uses float16 accumulation for sum. To reduce numerical
    # errors, float32 accumulation should be used instead.
    assert rtype in (
        "sum",
        "any",
    ), f"float16 user-defined reduction only supports 'sum' and 'any' but got {rtype}"
    prefix = [
        f"#pragma omp declare reduction({RTYPE_TO_CPP[rtype]}:{DTYPE_TO_CPP[torch.float16]}:"
        + f"omp_out = omp_out {RTYPE_TO_CPP[rtype]} omp_in)"
    ]
    return prefix


def parallel_num_threads():
    threads = config.cpp.threads
    if threads < 1:
        threads = torch.get_num_threads()
    return threads


@functools.lru_cache()
def cpp_prefix():
    path = Path(__file__).parent / "cpp_prefix.h"
    with path.open() as f:
        _, filename = codecache.write(
            f.read(),
            "h",
        )
    return f'#include "{filename}"'


class CppPrinter(ExprPrinter):
    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        mod = self.paren(self.doprint(mod))
        if div != "1":
            x = f"({x} / {div})"
        return f"{x} % {mod}"

    def _print_IndexingDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        return f"({x} / {div})"


cexpr = CppPrinter().doprint


class CppVecOverrides(OpOverrides):
    """Map element-wise ops to aten vectorization C++"""

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def div(a, b):
        return f"{a} / {b}"

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def erf(x):
        return f"{x}.erf()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def tanh(a):
        return f"{a}.tanh()"

    @staticmethod
    def reciprocal(a):
        return f"{a}.reciprocal()"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            quote = f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif math.isnan(val):
            quote = f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            quote = f"static_cast<{DTYPE_TO_CPP[dtype]}>({str(val).lower()})"
        else:
            quote = f"static_cast<{DTYPE_TO_CPP[dtype]}>({repr(val)})"
        return f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>({quote})"

    @staticmethod
    def relu(x):
        return f"at::vec::clamp_min({x}, decltype({x})(0))"

    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())"

    @staticmethod
    def neg(x):
        return f"{x}.neg()"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        _t = f"decltype({a})"
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < {_t}(0)) != ({b} < {_t}(0)) ? ({rem} != {_t}(0) ? {quot} - {_t}(1) : {quot}) : {quot})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def minimum(a, b):
        return f"at::vec::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"at::vec::maximum({a}, {b})"

    @staticmethod
    def square(a):
        return f"{a}.pow(2)"

    @staticmethod
    def where(a, b, c):
        return f"decltype({b})::blendv({c}, {b}, {a})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        vec_zero = f"decltype({x})(0)"
        vec_one = f"decltype({x})(1)"
        blendv = f"decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})"
        left = V.kernel.cse.newvar()
        code.writeline(f"auto {left} = {blendv};")

        # auto tmp6 = tmp4 == 0 ? 0 : tmp5;
        blendv = f"decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})"
        right = V.kernel.cse.newvar()
        code.writeline(f"auto {right} = {blendv};")
        result = V.kernel.cse.newvar()
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in [torch.bool], f"{__name__} does not support {dtype}"
        return f"({x})"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def to_dtype(x, dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        return f"std::log1p({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

    @staticmethod
    def signbit(x):
        return f"std::signbit({x})"

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def relu(x):
        return f"{x} * ({x}>0)"

    @staticmethod
    def minimum(a, b):
        return f"({b} != {b}) ? {b} : std::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"({b} != {b}) ? {b} : std::max({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif val == float("-inf"):
            return f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
        elif math.isnan(val):
            return f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::quiet_NaN()"
        elif val is True or val is False:
            return ops.to_dtype(str(val).lower(), dtype)
        return ops.to_dtype(repr(val), dtype)

    @staticmethod
    def index_expr(expr, dtype):
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        if other == float("-inf"):
            code.writeline(f"float {var} = -std::numeric_limits<float>::infinity();")
        elif other == float("inf"):
            code.writeline(f"float {var} = std::numeric_limits<float>::infinity();")
        elif isinstance(other, float):
            code.writeline(f"float {var} = {other};")
        else:
            code.writeline(f"auto {var} = {other!r};")
        code.writeline(f"if({mask})")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"{var} = {result};")
        V.kernel.compute.splice(code)
        return var

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(normalized_rand_cpu({seed}, {offset}));"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr, dtype):
        return f"static_cast<{DTYPE_TO_CPP[dtype]}>(randn_cpu({seed}, {offset}));"

    @staticmethod
    def sigmoid(x):
        x = ops.exp(f"-{x}")
        return f"1 / (1 + {x})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        # auto tmp5 = tmp4 < 0 ? -1 : 1;
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        code.writeline(f"auto {left} = {x} > 0 ? 1 : 0;")
        code.writeline(f"auto {right} = {x} < 0 ? 1 : 0;")
        code.writeline(f"auto {result} = {left} - {right};")
        V.kernel.compute.splice(code)
        return result


class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super(CppKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = DeferredIndentedBuffer()
        self.reduction_vars = {}
        self.num_threads = num_threads  # num_threads the kernel specialized for

    def scale_index_with_bias(self, index: sympy.Expr, scale, itervar_idx=-1, bias=None):
        expanded_index = sympy.expand(index)
        #assert self.simd_nelements
        #assert self.simd_nelements >= 1
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + (bias if bias is not None else 0)}
        new_index = sympy_subs(expanded_index, replacement)
        return new_index

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr(index)}]"
        if V.graph.get_dtype(name) in (torch.float16, torch.bfloat16):
            line = f"static_cast<float>({line})"
        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr(index)}] += {value};"
            else:
                line = f"atomic_add(&{var}[{cexpr(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}
        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        index = self.rename_indexing(index)
        self.reduction_vars[tmpvar] = reduction_type
        if argmax_or_argmin:
            self.reduction_prefix.writelines(
                argmax_argmin_prefix(reduction_type, src_dtype, tmpvar)
            )
            compare_op = "<" if reduction_type == "argmax" else ">"
            self.stores.writelines(
                None,
                [
                    f"if ({tmpvar}.value {compare_op} {value}) {{",
                    f"    {tmpvar}.index = {self.itervars[-1]}; {tmpvar}.value = {value};",
                    "}",
                ],
            )
        else:
            if dtype == torch.float16:
                self.reduction_prefix.writelines(
                    float16_reduction_prefix(reduction_type)
                )
            self.reduction_prefix.writeline(
                f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
            )
            self.stores.writeline(
                None, f"{reduction_combine(reduction_type, tmpvar, value)};"
            )

        if name not in V.graph.removed_buffers:
            var = self.args.output(name)
            member_name = ".index" if argmax_or_argmin else ""
            self.reduction_suffix.writeline(
                name, f"{var}[{cexpr(index)}] = {tmpvar}{member_name};"
            )
        self.cse.store_cache[name] = tmpvar

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy_symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        return V.graph.sizevars.size_hint(sympy_product(self.call_ranges))

    def codegen_loops(self, code, worksharing):
        threads = parallel_num_threads()

        loops = [LoopLevel(var, size) for var, size in zip(self.itervars, self.ranges)]
        loops, reductions = LoopNest(loops[: self.reduction_depth]), LoopNest(
            loops[self.reduction_depth :]
        )
        reductions.mark_reduction(self.reduction_vars)

        if codecache.pick_vec_isa():
            # TODO(jansel): detect stride-1 dimension and vectorize that
            if reductions:
                reductions.loops[-1].simd = True
            elif loops:
                loops.loops[-1].simd = True

        par_depth = 0
        reduction_par_depth = 0
        if loops:
            par_depth = self.decide_parallel_depth(
                self.call_ranges[: self.reduction_depth], threads
            )
        else:
            reduction_par_depth = self.decide_parallel_depth(
                self.call_ranges[self.reduction_depth :], threads
            )

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops.mark_parallel(par_depth)
            elif reduction_par_depth:
                # need to close the worksharing scope to define reduction vars outside it
                worksharing.close()
                reductions.mark_parallel(reduction_par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            loops.codegen(code, stack)

            with contextlib.ExitStack() as stack_outer:
                if self.reduction_prefix:
                    stack_outer.enter_context(code.indent())
                code.splice(self.reduction_prefix)

                if reduction_par_depth:
                    worksharing.parallel(threads)

                with contextlib.ExitStack() as stack:
                    reductions.codegen(code, stack)
                    code.splice(self.loads)
                    code.splice(self.compute)
                    code.splice(self.stores)

                if reduction_par_depth:
                    worksharing.close()

                code.splice(self.reduction_suffix)

    def decide_parallel_depth(self, ranges, threads):
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        return depth

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = DeferredIndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads):
        super(CppVecKernel, self).__init__(args, num_threads)
        assert codecache.pick_vec_isa()
        self.simd_nelements = codecache.pick_vec_isa().nelements()
        self.reduction_omp_dec: Dict[str, str] = {}
        self.var_vec_buf_map: Dict[str, str] = {}
        metrics.generated_cpp_vec_kernel_count += 1

    def stride_at(self, var: sympy.Symbol, index: sympy.Expr):
        # TODO(jgong5): use sizevars.stride_vars
        replacement = {var: var + 1}
        new_index = sympy_subs(index, replacement)
        return sympy.simplify(new_index - index)

    def is_stride1_at(self, var: sympy.Symbol, index: sympy.Expr):
        return self.stride_at(var, index) == 1

    def is_invariant_under(self, var: sympy.Symbol, index: sympy.Expr):
        expanded_index = sympy.expand(index)
        return not expanded_index.has(var)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        expanded_index = sympy.expand(index)
        new_index = self.scale_index_with_bias(index, self.simd_nelements)

        if expanded_index == new_index:
            line = f"at::vec::Vectorized<float>({var}[{cexpr(index)}])"
        else:
            if V.graph.get_dtype(name) in [torch.bool, torch.uint8]:
                nelements = codecache.pick_vec_isa().nelements()
                if var not in self.var_vec_buf_map:
                    self.var_vec_buf_map[var] = f"g_tmp_buffer_{var}"
                    self.loads.writeline(
                        f"float {self.var_vec_buf_map[var]}[{nelements}] = {{0}};"
                    )
                self.loads.writeline(
                    f"flag_to_float({var} + {cexpr(new_index)}, {self.var_vec_buf_map[var]}, {nelements});"
                )
                line = f"at::vec::Vectorized<float>::loadu({self.var_vec_buf_map[var]})"
            else:
                line = f"at::vec::Vectorized<float>::loadu({var} + {cexpr(new_index)})"

        return self.cse.generate(self.loads, line)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        assert mode is None

        expanded_index = sympy.expand(index)
        new_index = self.scale_index_with_bias(index, self.simd_nelements)
        assert new_index != expanded_index
        line = f"{value}.store({var} + {cexpr(new_index)});"
        self.stores.writeline(name, line)

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        assert reduction_type in {"max", "min", "sum"}
        assert dtype == torch.float
        assert src_dtype == torch.float
        reduce_map = {"max": "maximum", "min": "minimum"}

        vec_ns = "at::vec"
        vec = f"{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>"

        if reduction_type not in self.reduction_omp_dec:
            vec_reduc_prefix = "#pragma omp declare reduction("
            vec_reduc_prefix += f"{RTYPE_TO_CPP[reduction_type]}:{vec}:"
            if reduction_type == "sum":
                vec_reduc_prefix += "omp_out += omp_in"
            else:
                vec_reduc_prefix += (
                    f"omp_out = {vec_ns}::{reduce_map[reduction_type]}(omp_out, omp_in)"
                )
            vec_reduc_prefix += ")"
            vec_reduc_prefix += " initializer("
            vec_reduc_prefix += "omp_priv={{"
            vec_reduc_prefix += f"{reduction_init(reduction_type, dtype)}"
            vec_reduc_prefix += "}})"
            self.reduction_omp_dec[reduction_type] = RTYPE_TO_CPP[reduction_type]
            self.reduction_prefix.writeline(vec_reduc_prefix)

        tmpvar = self.cse.generate(
            self.loads, f"reduction {name} {cexpr(index)}", write=False
        )
        tmpvar_vec = f"{tmpvar}_vec"

        index = self.rename_indexing(index)
        self.reduction_vars[tmpvar] = reduction_type
        self.reduction_prefix.writeline(
            f"{DTYPE_TO_CPP[dtype]} {tmpvar} = {reduction_init(reduction_type, dtype)};"
        )
        self.reduction_prefix.writeline(
            f"auto {tmpvar_vec} = at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>({tmpvar});"
        )
        self.stores.writeline(
            None, f"{reduction_combine_vec(reduction_type, tmpvar_vec, value)};"
        )

        reduce_all_body = "{"
        if reduction_type == "sum":
            reduce_all_body += "return x + y;"
        else:
            reduce_all_body += f"return {vec_ns}::{reduce_map[reduction_type]}(x, y);"
        reduce_all_body += "}"
        vec_reduce_all_func = f"{vec_ns}::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>"
        self.reduction_suffix.writeline(
            name,
            f"{tmpvar} = {vec_reduce_all_func}([]({vec}& x, {vec}&y) {reduce_all_body}, {tmpvar_vec});",
        )
        self.cse.store_cache[name] = tmpvar


TileMeta = namedtuple("TileMeta", ["contig_index", "non_contig_index"])
class KernelAttr(object):
    pass


class CppTile2DKernel(CppVecKernel):
    # TODO: should be CppTile2DLoadStoreKernel instead
    def __init__(self, args, num_threads, kernel_attr):
        super().__init__(args, num_threads)
        self.tile_outer_loop_level_idx = kernel_attr.tile_outer_loop_level_idx
        self.tile_meta_cache = dict()

    def transform_tile2d_index(self, index, bias=None):
        assert self.is_stride1_at(self.itervars[-1], index) or self.is_stride1_at(self.itervars[self.tile_outer_loop_level_idx], index)
        assert not self.is_invariant_under(self.itervars[-1], index) and not self.is_invariant_under(self.itervars[self.tile_outer_loop_level_idx], index)
        # Scale the stride1 dim
        if self.is_stride1_at(self.itervars[-1], index):
            non_contig_index = self.tile_outer_loop_level_idx
            contig_index = -1
        else:
            non_contig_index = -1
            contig_index = self.tile_outer_loop_level_idx
        new_index = self.scale_index_with_bias(index, self.simd_nelements, itervar_idx=contig_index)
        new_index = self.scale_index_with_bias(new_index, self.simd_nelements, itervar_idx=non_contig_index, bias=bias)
        return new_index, contig_index, non_contig_index

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        expanded_index = sympy.expand(index)
        bias = sympy.symbols(f"{self.itervars[self.tile_outer_loop_level_idx]}_inner")
        new_index, contig_idx, non_contig_idx = self.transform_tile2d_index(expanded_index, bias)
        assert new_index != expanded_index

        expr = f"TILE2D_LOAD(__place_holder__, {var} + {cexpr(new_index)}, {bias}, {self.simd_nelements}, float)"
        if expr in self.cse.cache:
            return self.cse.cache[expr]
        cse_var = self.cse.generate(self.loads, expr, write=False)
        expr = expr.replace("__place_holder__", str(cse_var))
        # TODO(jgong5): support other data types than float
        # TODO(jgong5): the following statement is too complex to handle by cse.generate
        #               maybe extending cse.generate to support the line below would be better?
        line = f"float {cse_var}[{self.simd_nelements}*{self.simd_nelements}] __attribute__ ((aligned (16))); {expr};"
        V.kernel.current_node.codegen_originating_info(self.loads, only_once=True)
        self.loads.writeline(line)

        tile_meta = TileMeta(contig_idx, non_contig_idx)
        self.tile_meta_cache[cse_var.name] = tile_meta

        return cse_var

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        assert mode is None
        # TODO(jgong5): assert the index is an affine expression on the itervars in concern
        expanded_index = sympy.expand(index)

        new_index, contig_idx, non_contig_idx = self.transform_tile2d_index(expanded_index)
        assert new_index != expanded_index
        assert str(value) in self.tile_meta_cache, value
        tile_meta = self.tile_meta_cache[str(value)]
        # TODO(jgong5): cache the transposed result for multiple use
        # make sure we handle transposition here
        assert tile_meta[0] == non_contig_idx and tile_meta[1] == contig_idx
        line = f"TILE2D_STORE({var} + {cexpr(new_index)}, {value}, {self.stride_at(self.itervars[non_contig_idx], expanded_index)}, {self.simd_nelements});"
        self.stores.writeline(name, line)


class CppTile2DTailKernel(CppKernel):
    def __init__(self, args, num_threads, kernel_attr):
        super().__init__(args, num_threads)
        self.tile_outer_loop_level_idx = kernel_attr.tile_outer_loop_level_idx
        self.simd_nelements = kernel_attr.simd_nelements

    def bias_outer_name(self):
        return f"{self.itervars[self.tile_outer_loop_level_idx]}_inner"

    def transform_tile2d_index_in_tail(self, index):
        bias_outer = sympy.symbols(self.bias_outer_name())
        new_index = self.scale_index_with_bias(index, self.simd_nelements, itervar_idx=self.tile_outer_loop_level_idx, bias=bias_outer)
        return new_index

    def load(self, name: str, index: sympy.Expr):
        index = self.rename_indexing(index)
        expanded_index = sympy.expand(index)
        new_index = self.transform_tile2d_index_in_tail(expanded_index)
        return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        assert mode is None
        # TODO(jgong5): assert the index is an affine expression on the itervars in concern
        expanded_index = sympy.expand(index)
        new_index = self.transform_tile2d_index_in_tail(expanded_index)
        super().store(name, new_index, value, mode)

    def gen_inner_loop(self, code):
        outer = self.bias_outer_name()
        code.writeline(f"for (long {outer} = 0; {outer} < {self.simd_nelements}; {outer}++)")


class CppVecKernelChecker(CppVecKernel):
    def __init__(self, args, num_threads):
        super(CppVecKernelChecker, self).__init__(args, num_threads)

        # Since this kernel is only for checker but does not genreate any
        # code, so we need to decrease the kernel count.
        metrics.generated_kernel_count -= 1
        metrics.generated_cpp_vec_kernel_count -= 1

        # Used to recorde the graph wrapper code as the wrapper_code status could be
        # changed during graph run.
        self._orig_wrapper_code = None

        self.simd_vec = True
        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()

        self.can_tile2d = True
        self.tile_outer_loop_level_idx = -1

    def could_vec(self, name: str, index: sympy.Expr):
        assert self.itervars is not None
        # Not a loop
        if len(self.itervars) == 0:
            return False

        most_inner_var = self.itervars[-1]
        return self.is_invariant_under(most_inner_var, index) or self.is_stride1_at(most_inner_var, index)

    def check_can_tile2d(self, name: str, index: sympy.Expr):
        if not self.can_tile2d:
            return
        if not V.graph.get_dtype(name) in [
            torch.float,
        ]:
            self.can_tile2d = False
            return
        # check contiguity from any of the outer loops
        for idx, itervar in enumerate(self.itervars[:-1]):
            if self.is_stride1_at(itervar, index):
                # only support 2d tile now
                if self.tile_outer_loop_level_idx >= 0 and self.tile_outer_loop_level_idx != idx:
                    self.can_tile2d = False
                    return
                self.tile_outer_loop_level_idx = idx

    def load(self, name: str, index: sympy.Expr):
        self.check_can_tile2d(name, index)

        if not V.graph.get_dtype(name) in [
            torch.float,
            torch.float32,
            torch.bool,
            torch.uint8,
        ]:
            self.simd_vec = False
            return self.simd_vec

        index = self.rename_indexing(index)
        self.simd_vec = self.simd_vec and self.could_vec(name, index)
        return self.simd_vec

    def store(self, name, index, value, mode=None):
        self.check_can_tile2d(name, index)

        if not V.graph.get_dtype(name) in [torch.float, torch.float32]:
            self.simd_vec = False
            return self.simd_vec

        assert "buf" in name
        index = self.rename_indexing(index)

        if mode:
            self.simd_vec = False
            return False

        self.simd_vec = self.simd_vec and self.could_vec(name, index)
        return self.simd_vec

    def reduction(self, name, dtype, src_dtype, reduction_type, index, value):
        self.can_tile2d = False

        if (
            dtype == torch.float
            and src_dtype == torch.float
            and reduction_type in ["max", "min", "sum"]
        ):
            pass
        else:
            self.simd_vec = False
        return self.simd_vec

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._orig_wrapper_code is not None
        # Restore the wrapper_code
        V.graph.wrapper_code = self._orig_wrapper_code
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        # Recorde the graph wrapper code. The wrapper_code status could be
        # changed during graph run. Regarding this checker, we also need to
        # run the graph but we don't expect to change any status that would
        # impact the code generation. Hence, we record the graph wapper code
        # and replace it with a dummy warpper_code and then restore to the
        # original one as long as the checker is finished.
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        class VecCheckerProxy:
            @staticmethod
            def __getattr__(name):
                def inner(*args, **kwargs):
                    self.can_tile2d = False
                    if not (name in self.fast_vec_list):
                        self.simd_vec = False
                    return self.simd_vec

                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(name, dtype, src_dtype, reduction_type, index, value):
                return self.reduction(
                    name, dtype, src_dtype, reduction_type, index, value
                )

            @staticmethod
            def constant(val, dtype):
                self.can_tile2d = False
                supported_dtype = (torch.float32, torch.int32)
                is_supported_dtype = dtype in (supported_dtype)
                if not is_supported_dtype:
                    self.simd_vec = False
                return is_supported_dtype

            @staticmethod
            def index_expr(expr, dtype):
                self.can_tile2d = False
                self.simd_vec = False
                tmp_var = self.cse.newvar()
                return tmp_var

            @staticmethod
            def indirect_indexing(index_var):
                self.can_tile2d = False
                self.simd_vec = False
                return sympy.Symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                self.can_tile2d = False
                tmp_var = self.cse.newvar()
                return tmp_var

            @staticmethod
            def to_dtype(x, dtype):
                self.can_tile2d = False
                if dtype != torch.bool:
                    self.simd_vec = False
                return x

        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


class CppKernelProxy(CppKernel):
    def __init__(self, kernels, args=None, num_threads=None):
        super(CppKernelProxy, self).__init__(args, num_threads)
        # TODO(jgong5): provide better abstraction for `kernels`
        # kernels[0] always exists and for scalar_kernel
        # kernels[1] could be the kernel for vec_kernel or
        # (tiling_index, tile2d_kernel, tile2d tail kernel)
        assert kernels, "Expect at least one scalar kernel"
        self.scalar_kernel: CppKernel = kernels[0]
        self.vec_kernel: CppVecKernel = None
        # TODO(jgong5): consider to combine VecKernel and TileKernel?
        self.tile2d_kernel: CppTile2DKernel = None
        self.tile2d_tail_kernel: CppTile2DKernel = None
        if len(kernels) > 1:
            if isinstance(kernels[1], tuple):
                self.tile2d_kernel, self.tile2d_tail_kernel = kernels[1]
                assert self.tile2d_kernel.tile_outer_loop_level_idx == self.tile2d_tail_kernel.tile_outer_loop_level_idx
                self.tile_outer_loop_level_idx = self.tile2d_kernel.tile_outer_loop_level_idx
            else:
                self.vec_kernel = kernels[1]
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def vectorize_most_inner_loop(self, loop_nest, dtype=torch.float):
        assert self.picked_vec_isa
        nelements = self.picked_vec_isa.nelements(dtype)
        loop_nest.split(nelements, -1)
        loop_with_tail = loop_nest.loops[-1]
        assert isinstance(loop_with_tail, LoopLevelWithTail)

        loop_with_tail.main_loop.simd_vec = True

        loop_with_tail.tail_loop.simd_omp = True
        # We chope the loop into two cubes by the nelements - main loop and tail loop.
        # Regarding the main loop, it is straightforward that it could be vectorized with
        # nelements. But for the tail loop, it still could be vectorized. For example,
        # if the nelements is 8(256bits), then the tail loop still could be vectorized
        # as 4(128bits).
        loop_with_tail.tail_loop.simd_nelements = int(nelements / 2)
        loop_with_tail.tail_loop.simd_vec = False

        loop_with_tail.main_loop_body = self.vec_kernel
        loop_with_tail.tail_loop_body = self.scalar_kernel
        return loop_nest

    def tile2d_loop(self, loop_nest, tile_outer_loop_level_idx, dtype=torch.float):
        assert self.picked_vec_isa
        nelements = self.picked_vec_isa.nelements(dtype)
        loop_nest.split(nelements, -1)
        loop_nest.split(nelements, tile_outer_loop_level_idx)
        loop_inner = loop_nest.loops[-1]
        loop_outer = loop_nest.loops[tile_outer_loop_level_idx]
        assert isinstance(loop_inner, LoopLevelWithTail) and isinstance(loop_outer, LoopLevelWithTail)
        loop_inner.main_loop_body = self.tile2d_kernel
        loop_inner.tail_loop_body = self.tile2d_tail_kernel
        loop_outer.tail_loop_body = self.scalar_kernel
        return loop_nest

    def codegen_loops_vec(self, code, worksharing):
        threads = parallel_num_threads()

        assert self.vec_kernel.itervars == self.scalar_kernel.itervars
        assert self.vec_kernel.ranges == self.scalar_kernel.ranges
        assert (
            self.vec_kernel.reduction_vars == self.scalar_kernel.reduction_vars
        )

        itervars = self.vec_kernel.itervars
        rangs = self.vec_kernel.ranges
        loops = [LoopLevel(var, size) for var, size in zip(itervars, rangs)]
        assert (
            self.vec_kernel.reduction_depth == self.scalar_kernel.reduction_depth
        )
        reduction_depth = self.vec_kernel.reduction_depth
        loops_nest_non_reduce, loops_nest_reduce = LoopNest(
            loops[:reduction_depth]
        ), LoopNest(loops[reduction_depth:])
        loops_nest_reduce.mark_reduction(self.vec_kernel.reduction_vars)

        assert self.picked_vec_isa
        # Do not apply vectorization since the range of most inner is too small. Meanwhile,
        # If the range of the most inner is less then the codecache.pick_vec_isa().nelements(),
        # the generated code for some reduction will be as follows that leads to incrrect result.
        #
        #    LINE01:  float tmp1 = 0;
        #    LINE02:  auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
        #    LINE03:  for(long i1=0; i1<2; i1+=1)
        #    LINE04:  {
        #    LINE05:      for(long i2=0; i2<0; i2+=1)
        #    LINE06:      {
        #    LINE07:          auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (8*i0) + (16*i2) + (32*i1));
        #    LINE08:          tmp1_vec += tmp0;
        #    LINE09:      }
        #    LINE10:      tmp1 = vec_reduce_all<float>([](Vectorized<float>& x, Vectorized<float>&y) {return x + y;}, tmp1_vec);
        #    LINE11:      #pragma omp simd simdlen(8)  reduction(+:tmp1)
        #    LINE12:      for(long i2=0; i2<8; i2+=1)
        #    LINE13:      {
        #    LINE14:          auto tmp0 = in_ptr0[i2 + (8*i0) + (32*i1)];
        #    LINE15:          tmp1 += tmp0;
        #    LINE16:      }
        #    LINE17:  }
        #    LINE18:  out_ptr3[i0] = tmp1;
        #
        # tmp1_vec(LINE02) will always be zero as it is initialized with tmp1 value and the range(LINE05)
        # is 0. Hence, the LINE10 will always reset tmp1 to 0. But tmp1(LINE01) is global value. So the result
        # will be incorrect. We skip thie case.
        most_inner_loop = (
            loops_nest_reduce.loops[-1]
            if loops_nest_reduce
            else loops_nest_non_reduce.loops[-1]
        )
        main_loop_range = ir.IndexingDiv(
            most_inner_loop.size, self.picked_vec_isa.nelements()
        )
        loop_interval = sympy.simplify(main_loop_range)
        # TODO(Eikan): To support dynamic shape.
        if not loop_interval.is_integer or loop_interval <= 0:
            metrics.generated_cpp_vec_kernel_count -= 1
            return self.scalar_kernel.codegen_loops(code, worksharing)

        # TODO(jansel): detect stride-1 dimension and vectorize that
        if loops_nest_reduce:
            loops_nest_reduce.loops[-1].simd = True
        elif loops_nest_non_reduce:
            loops_nest_non_reduce.loops[-1].simd = True

        par_depth = 0
        reduction_par_depth = 0
        if loops_nest_non_reduce:
            par_depth = self.vec_kernel.decide_parallel_depth(
                self.vec_kernel.call_ranges[:reduction_depth], threads
            )
        else:
            reduction_par_depth = self.vec_kernel.decide_parallel_depth(
                self.vec_kernel.call_ranges[reduction_depth:], threads
            )

        # If the most inner loop of the reduction will be vectorized, the vectorization
        # will add a vec variable for reduction. Take the code snippet as an example:
        #     float tmp1 = 0;
        #     for(long i1=0; i1<8; i1+=1) {
        #        auto tmp0 = in_ptr0[i1];
        #        tmp1 += tmp0;
        #     }
        # The vectorization will add tmp1_vec for reduction and then the loop will be transformed
        # as follows.
        #     float tmp1 = 0;
        #     auto tmp1_vec = at::vec::Vectorized<float>(tmp1);
        #     for(long i1=0; i1<1; i1+=1) {
        #        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + (8*i1));
        #        tmp1_vec += tmp0;
        #     }
        #     tmp1 = at::vec::vec_reduce_all<float>([]
        #       (at::vec::Vectorized<float>& x, at::vec::Vectorized<float>&y) {return x + y;},
        #       tmp1_vec);
        #     for(long i1=8; i1<8; i1+=1) {
        #        auto tmp0 = in_ptr0[i1];
        #        tmp1 += tmp0;
        #     }
        # It means that the vectorization introduce another reduction variable(tmp1_vec).
        # If the most inner loop of the reduction is not a parallelized but its parent reduction
        # loop is parallized, the new added reduction variable(tmp1_vec) could not be added
        # to the parallelized loop reduction. So we skip this case and does not vectorize it.
        if reduction_par_depth > 0 and reduction_par_depth != len(
            loops_nest_reduce.loops
        ):
            metrics.generated_cpp_vec_kernel_count -= 1
            return self.scalar_kernel.codegen_loops(code, worksharing)

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops_nest_non_reduce.mark_parallel(par_depth)
            elif reduction_par_depth:
                # need to close the worksharing scope to define reduction vars outside it
                worksharing.close()
                loops_nest_reduce.mark_parallel(reduction_par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            non_reduce_loops = loops_nest_non_reduce.loops
            reduce_loops = loops_nest_reduce.loops
            loop_with_tail: LoopLevelWithTail = None

            if loops_nest_reduce:
                self.vectorize_most_inner_loop(loops_nest_reduce)
                loop_with_tail = loops_nest_reduce.loops[-1]
                # The most inner loop will be vectorized
                reduce_loops = reduce_loops[0:-1]
            else:
                self.vectorize_most_inner_loop(loops_nest_non_reduce)
                loop_with_tail = loops_nest_non_reduce.loops[-1]
                # The most inner loop will be vectorized
                non_reduce_loops = non_reduce_loops[0:-1]

            # The reductions loops are always the loop body of non-reduction loops
            for loop in non_reduce_loops:
                code.writelines(loop.lines())
                stack.enter_context(code.indent())

            with contextlib.ExitStack() as stack_outer:
                if self.vec_kernel.reduction_prefix:
                    stack_outer.enter_context(code.indent())
                code.splice(self.vec_kernel.reduction_prefix)

                if reduction_par_depth:
                    worksharing.parallel(threads)

                with contextlib.ExitStack() as stack:
                    for loop in reduce_loops:
                        code.writelines(loop.lines())
                        stack.enter_context(code.indent())

                    def gen_vectorized_loop(loop, kernel, write_reduction_suffix=False):
                        code.writelines(loop.lines())
                        with contextlib.ExitStack() as stack:
                            stack.enter_context(code.indent())
                            code.splice(kernel.loads)
                            code.splice(kernel.compute)
                            code.splice(kernel.stores)
                        if write_reduction_suffix:
                            code.splice(kernel.reduction_suffix)

                    # Regarding the vectorized reduction loop, we need to call reduce_all to to reduce
                    # the vectorize as a single scalar. Hence, we set write_reduction_suffix to True to
                    # gen the code.
                    gen_vectorized_loop(
                        loop_with_tail.main_loop, loop_with_tail.main_loop_body, True
                    )

                    gen_vectorized_loop(
                        loop_with_tail.tail_loop, loop_with_tail.tail_loop_body, False
                    )

                if reduction_par_depth:
                    worksharing.close()

                code.splice(loop_with_tail.tail_loop_body.reduction_suffix)

    def codegen_loops_tile2d(self, code, worksharing):
        threads = parallel_num_threads()

        assert self.tile2d_kernel.itervars == self.tile2d_tail_kernel.itervars == self.scalar_kernel.itervars
        assert self.tile2d_kernel.ranges == self.tile2d_tail_kernel.ranges == self.scalar_kernel.ranges
        assert len(self.tile2d_kernel.reduction_vars) == len(self.tile2d_tail_kernel.reduction_vars) == len(self.scalar_kernel.reduction_vars) == 0
        assert self.tile2d_kernel.reduction_depth == self.tile2d_tail_kernel.reduction_depth == self.scalar_kernel.reduction_depth == len(self.tile2d_kernel.itervars)

        itervars = self.tile2d_kernel.itervars
        rangs = self.tile2d_kernel.ranges
        loops = [LoopLevel(var, size) for var, size in zip(itervars, rangs)]
        loops_nest_non_reduce = LoopNest(loops)

        assert self.picked_vec_isa
        for level_id in (self.tile_outer_loop_level_idx, -1):
            loop = loops_nest_non_reduce.loops[level_id]
            loop_size_with_vec = ir.IndexingDiv(loop.size, self.picked_vec_isa.nelements())
            loop_size_with_vec = sympy.simplify(loop_size_with_vec)
            # TODO(Eikan): To support dynamic shape.
            if not loop_size_with_vec.is_integer or loop_size_with_vec <= 0:
                metrics.generated_cpp_vec_kernel_count -= 1
                return self.scalar_kernel.codegen_loops(code, worksharing)

        par_depth = self.tile2d_kernel.decide_parallel_depth(
            self.tile2d_kernel.call_ranges, threads
        )
        if par_depth:
            par_depth = min(par_depth, self.tile_outer_loop_level_idx)

        with contextlib.ExitStack() as stack:
            if par_depth:
                worksharing.parallel(threads)
                loops_nest_non_reduce.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            non_reduce_loops = loops_nest_non_reduce.loops
            self.tile2d_loop(loops_nest_non_reduce, self.tile_outer_loop_level_idx)

            def gen_loop_body(kernel):
                with contextlib.ExitStack() as stack:
                    if isinstance(kernel, CppTile2DTailKernel):
                        kernel.gen_inner_loop(code)
                    stack.enter_context(code.indent())
                    if not config.cpp.ignore_tile2d_kernel:
                        code.splice(kernel.loads)
                        code.splice(kernel.compute)
                        code.splice(kernel.stores)
                    else:
                        code.writeline("volatile int dummy = 0;")

            def gen_loops(loops, body=None):
                with contextlib.ExitStack() as stack:
                    for idx, loop in enumerate(loops):
                        if isinstance(loop, LoopLevelWithTail):
                            # main loop
                            current_body = body
                            if not current_body:
                                current_body = loop.main_loop_body
                                code.writelines(loop.main_loop.lines())
                                with contextlib.ExitStack() as stack_inner:
                                    stack_inner.enter_context(code.indent())
                                    gen_loops(loops[idx+1:], current_body)
                            # tail
                            current_body = body
                            if not current_body:
                                current_body = loop.tail_loop_body
                                code.writelines(loop.tail_loop.lines())
                                with contextlib.ExitStack() as stack_inner:
                                    stack_inner.enter_context(code.indent())
                                    gen_loops(loops[idx+1:], current_body)
                            if not body:
                                return
                        code.writelines(loop.lines())
                        stack.enter_context(code.indent())
                    assert body
                    gen_loop_body(body)

            gen_loops(non_reduce_loops)

    def codegen_loops(self, code, worksharing):
        if not self.picked_vec_isa or (self.vec_kernel is None and self.tile2d_kernel is None):
            assert self.scalar_kernel
            return self.scalar_kernel.codegen_loops(code, worksharing)

        if self.vec_kernel:
            self.codegen_loops_vec(code, worksharing)
        else:
            assert self.tile2d_kernel and self.tile2d_tail_kernel
            self.codegen_loops_tile2d(code, worksharing)


class CppScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.get_kernel_group()

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def get_kernel_group(self):
        from .wrapper import CppWrapperCodeGen

        if isinstance(V.graph.wrapper_code, CppWrapperCodeGen):
            self.kernel_group = CppWrapperKernelGroup()
        else:
            self.kernel_group = KernelGroup()

    @staticmethod
    def can_fuse_horizontal(node1, node2):
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group
        if vars1 == vars2 and reduce1 == reduce2:
            return True
        if reduce1 == () and vars1 == vars2 + reduce2:
            return True
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return False

    @classmethod
    def can_fuse_vertical(cls, node1, node2):
        return cls.can_fuse_horizontal(node1, node2) and not node1.is_reduction()

    def _codegen_nodes_impl(self, nodes, kernel_attr):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group
        #_, (group, reduction_group) = max(
        #    nodes, key=lambda x: int(x.is_reduction())
        #).group
        group = kernel_attr.group
        reduction_group = kernel_attr.reduction_group

        #def create_kernel(_is_simd_vec):
        def create_kernel(kernel_type: CppKernelType, kernel_attr=None):
            in_suffix = False

            #with kernel_group.new_kernel(_is_simd_vec) as kernel:
            with kernel_group.new_kernel(kernel_type, kernel_attr) as kernel:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)

                for node in nodes:
                    if node.group[1] in [
                        (group, reduction_group),
                        (group + reduction_group, ()),
                    ]:
                        assert not in_suffix
                        node.run(vars, reduction_vars)
                    else:
                        in_suffix = True
                        assert node.group[1] == (
                            group,
                            (),
                        ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                        # we can fuse in some extra pointwise into the suffix
                        with kernel.write_to_suffix():
                            node.run(vars, ())
                return kernel

        kernels = [create_kernel(CppKernelType.SCALAR)]
        with patch.object(torch._inductor.config, "inplace_buffers", False):
            if kernel_attr.can_vec:
                kernels.append(create_kernel(CppKernelType.VECTOR))
            elif kernel_attr.can_tile2d:
                kernels.append(
                    (create_kernel(CppKernelType.TILE2D, kernel_attr), create_kernel(CppKernelType.TILE2D_TAIL, kernel_attr))
                )
        return kernels
        '''
        org_inplace_buffers_flag = config.inplace_buffers
        if is_simd_vec:
            # Create vectorization kernel
            cpp_vec_kernel = create_kernel(True)

            # Since a kernel is divided into two parts - vectorization and non-vectorization.
            # And the two parts share the same global contexts like V.graph.wrapper_code,
            # V.kernel.args. But the vectorization kernel generation has updated these global
            # contexts. Hence, the non-vectorization kernel should not do this again to avoid
            # conext conflict. By now, we only control the config.inplace_buffers. In the future,
            # we could maintain more contexts.
            config.inplace_buffers = False

            # Create non-vectorization kernel
            cpp_kernel = create_kernel(False)

            # Restore the inplace_buffers flag
            config.inplace_buffers = org_inplace_buffers_flag
            return (cpp_vec_kernel, cpp_kernel)
        else:
            return (None, create_kernel(False))
        '''

    def get_kernel_attr(self, nodes):
        kernel_attr = KernelAttr()
        _, (group, reduction_group) = max(
            nodes, key=lambda x: int(x.is_reduction())
        ).group
        kernel_attr.group = group
        kernel_attr.reduction_group = reduction_group

        with CppVecKernelChecker(
            deepcopy(self.kernel_group.args), parallel_num_threads()
        ) as kernel_checker:
            vars, reduction_vars = kernel_checker.set_ranges(group, reduction_group)
            for node in nodes:
                if node.group[1] in [
                    (group, reduction_group),
                    (group + reduction_group, ()),
                ]:
                    node.run(vars, reduction_vars)
                else:
                    assert node.group[1] == (
                        group,
                        (),
                    ), f"unexpected group: {node.group[1]} != {group}, {reduction_group}"
                    node.run(vars, ())

        kernel_attr.can_vec = codecache.pick_vec_isa() and kernel_checker.simd_vec
        kernel_attr.can_tile2d = config.cpp.enable_tile2d and codecache.pick_vec_isa() and kernel_checker.can_tile2d
        kernel_attr.tile_outer_loop_level_idx = kernel_checker.tile_outer_loop_level_idx
        kernel_attr.simd_nelements = codecache.pick_vec_isa().nelements() if codecache.pick_vec_isa() else 1

        return kernel_attr


    def codegen_nodes(self, nodes):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        # TODO: turn this into attribute instead of bool flag
        # can_be_simd_vec = self.can_vec(nodes)
        # TODO: add this call inside _codegen_nodes_impl
        kernel_attr = self.get_kernel_attr(nodes)
        kernels = self._codegen_nodes_impl(
            nodes, kernel_attr
        )

        '''
        TODO: apply this logic onto kernel_tree
        assert simd_omp_kernel
        metrics.generated_kernel_count -= 1
        # Maitain the metrics kernel count
        if simd_vec_kernel:
            metrics.generated_kernel_count -= 1
        '''

        cpp_kernel_proxy = CppKernelProxy(
            kernels, kernel_group.args, kernel_group.ws.num_threads
        )
        # TODO: replaced by kerne_tree, not needed any more
        # cpp_kernel_proxy.simd_vec_kernel = simd_vec_kernel
        # cpp_kernel_proxy.simd_omp_kernel = simd_omp_kernel

        kernel_group.finalize_kernel(cpp_kernel_proxy, None)

    def codegen_sync(self):
        pass

    def flush(self):
        self.kernel_group.codegen_define_and_call(V.graph.wrapper_code)
        self.get_kernel_group()


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.count = 0

    #def new_kernel(self, simd_vec=False):
    def new_kernel(self, kernel_type, kernel_attr):
        if kernel_type == CppKernelType.SCALAR:
            return CppKernel(self.args, parallel_num_threads())
        elif kernel_type == CppKernelType.VECTOR:
            return CppVecKernel(self.args, parallel_num_threads())
        elif kernel_type == CppKernelType.TILE2D:
            return CppTile2DKernel(self.args, parallel_num_threads(), kernel_attr)
        else:
            assert kernel_type == CppKernelType.TILE2D_TAIL, "Unsupported cpp kernel type"
            return CppTile2DTailKernel(self.args, parallel_num_threads(), kernel_attr)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        kernel_name = "kernel_cpp_" + wrapper.next_kernel_suffix()
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        arg_types = ",".join(arg_types)
        code = BracesBuffer()
        # TODO: support kernel profile on other platforms
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        if enable_kernel_profile:
            code.writelines(["#include <ATen/record_function.h>"])
        code.writelines([cpp_prefix(), "" f'extern "C" void kernel({arg_defs})'])
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
                code.writelines(
                    [
                        f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
                    ]
                )
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("async_compile.cpp('''")
        codecache_def.splice(code)
        codecache_def.writeline("''')")

        codecache_str = codecache_def.getvalue()
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        codecache_str = codecache_str.replace("#pragma CMT", "//")
        wrapper.define_kernel(kernel_name, codecache_str)
        wrapper.load_kernel(kernel_name, code, arg_types)
        # generate the code to call this
        wrapper.generate_kernel_call(kernel_name, call_args)


class CppWrapperKernelGroup(KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = CppWrapperKernelArgs()


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        return self.in_parallel

    def close(self):
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class LoopLevel:
    var: sympy.Expr = None
    size: sympy.Expr = None
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()
    simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0
    simd_vec: bool = False
    collapsed: bool = False
    reduction_vars: Dict[str, str] = None

    def lines(self):
        if self.reduction_vars:
            suffix = "_vec" if self.simd_vec else ""
            reduction = " " + " ".join(
                f"reduction({RTYPE_TO_CPP[rtype]}:{var}{suffix})"
                for var, rtype in self.reduction_vars.items()
            )
        else:
            reduction = ""
        simd = (
            f"simd simdlen({self.simd_nelements}) "
            if self.simd_omp and self.simd_nelements > 1
            else ""
        )
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = f"#pragma omp for{reduction} "
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            line1 = ""
        elif self.simd_omp:
            line1 = f"#pragma omp {simd}{reduction}"
        elif not self.reduction_vars and codecache.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        line2 = f"for({INDEX_TYPE} {self.var}={cexpr(self.offset)}; {self.var}<{cexpr(self.size)}; {self.var}+={cexpr(self.steps)})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


class LoopLevelWithTail(LoopLevel):
    def __init__(self, main_loop: LoopLevel, tail_loop: LoopLevel, orig_loop: LoopLevel):
        super().__init__()
        self.main_loop = main_loop
        self.tail_loop = tail_loop
        self.orig_loop = orig_loop
        # TODO(jgong5): here we assumed that the LoopLevelWithTail
        # only applies to the innermost loop level so it is a 1:1
        # mapping to the loop body while if we allow the split on
        # outer loop levels, it could be multiple LoopLevelWithTails
        # mapped to a loop body. Need to reconsider the abstraction here
        self.main_loop_body = None
        self.tail_loop_body = None

    def lines(self):
        #raise AssertionError("Not Implemented")
        return self.orig_loop.lines()


@dataclasses.dataclass
class LoopNest:
    loops: List[LoopLevel]

    def __bool__(self):
        return bool(self.loops)

    def mark_reduction(self, reduction_vars):
        for loop in self.loops:
            loop.reduction_vars = reduction_vars

    def mark_parallel(self, par_depth):
        loops = self.loops
        loops[0].parallel = par_depth
        for i in range(1, par_depth):
            loops[i].collapsed = True

    def split(self, factor, loop_level_idx):
        sympy_factor = sympy.Integer(factor)

        loop_to_split = self.loops[loop_level_idx]

        # If the most inner loop needs to be collapsed, we need to
        # exclude it since we need to split it into two loops. Meanwhile,
        # we still mark it as parallized.
        if loop_to_split.collapsed:
            assert self.loops[0].parallel == len(self.loops)
            self.loops[0].parallel -= 1

        # TODO(jgong5): simplify it?
        main_loop_size = ir.IndexingDiv(loop_to_split.size, sympy_factor)

        main_loop = LoopLevel(loop_to_split.var, main_loop_size)
        main_loop.parallel = loop_to_split.parallel
        main_loop.collapsed = False
        main_loop.reduction_vars = loop_to_split.reduction_vars

        # TODO(jgong5): simplify it?
        offset = main_loop_size * sympy_factor
        tail_loop = LoopLevel(loop_to_split.var, loop_to_split.size)
        tail_loop.offset = offset
        tail_loop.parallel = loop_to_split.parallel
        tail_loop.collapsed = False
        tail_loop.reduction_vars = loop_to_split.reduction_vars

        loop_with_tail = LoopLevelWithTail(main_loop, tail_loop, loop_to_split)
        loop_with_tail.parallel = 0
        loop_with_tail.collapsed = False

        self.loops[loop_level_idx] = loop_with_tail

    def codegen(self, code, stack):
        for loop in self.loops:
            code.writelines(loop.lines())
            stack.enter_context(code.indent())
        else:
            stack.enter_context(code.indent())
