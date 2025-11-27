from __future__ import annotations
from functools import partial
from tvm import tir
from tvm.tir import (PyStmtExprVisitor, BufferStore, PrimFunc, BufferLoad, Call)
from tvm.tir.transform import prim_func_pass
from tvm.tir.stmt_functor import post_order_visit

@tir.functor.visitor
class _BufferCollector(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.access_buffer = set()
    
    def clear_buffer(self) -> None:
        self.access_buffer.clear()
    
    def visit_buffer_store_(self, op: BufferStore) -> None:
        buffer = op.buffer
        if buffer.scope() != "global":
            self.access_buffer.add(op.buffer)
        self.visit_expr(op.value)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        buffer = op.buffer
        if buffer.scope() != "global":
            self.access_buffer.add(op.buffer)

@tir.functor.visitor
class _BufferClassifier(PyStmtExprVisitor):

    def __init__(self) -> None:
        super().__init__()
        self.var_mem_map_ = {
            "ub": set(),
            "l1": set(),
            "l0c": set(),
        }
        self.collector = _BufferCollector()

    def visit_call_(self, op: Call) -> None:
        collector = self.collector
        collector.clear_buffer()
        if op.op == tir.op.Op.get("tl.gemm_py"):
            collector.visit_expr(op.args[0])
            collector.visit_expr(op.args[1])
            self.var_mem_map_["l1"].update(collector.access_buffer)
            collector.clear_buffer()
            collector.visit_expr(op.args[2])
            self.var_mem_map_["l0c"].update(collector.access_buffer)
        else:
            collector.visit_expr(op)
            self.var_mem_map_["ub"].update(collector.access_buffer)

def BufferClassifier():

    def pass_fn(func: PrimFunc, mod, ctx):
        classifier = _BufferClassifier()
        classifier.visit_stmt(func.body)
        func = func.with_attr("memory_map", classifier.var_mem_map_)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
