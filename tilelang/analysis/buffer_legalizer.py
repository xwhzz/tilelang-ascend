from __future__ import annotations
from collections import defaultdict
from tvm import tir
from tvm.tir import (PyStmtExprVisitor, BufferStore, PrimFunc, BufferLoad, Call, For, Block, PyStmtExprMutator, Stmt, Buffer, decl_buffer)
from tvm.tir.transform import prim_func_pass
from tvm.tir.stmt_functor import ir_transform


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
        self.var_mem_map_ = defaultdict(set)
        self.collector = _BufferCollector()
    
    def add_set(self, scope: str) -> None:
        for bf in self.collector.access_buffer:
            self.var_mem_map_[bf].add(scope)

    def visit_call_(self, op: Call) -> None:
        collector = self.collector
        collector.clear_buffer()
        if op.op == tir.op.Op.get("tl.gemm_py"):
            collector.visit_expr(op.args[0])
            collector.visit_expr(op.args[1])
            self.add_set("l1")
            collector.clear_buffer()
            collector.visit_expr(op.args[2])
            self.add_set("l0c")
        elif op.op == tir.op.Op.get("tl.copy"):
            return
        else:
            collector.visit_expr(op)
            self.add_set("ub")
    
    def visit_for_(self, op: For) -> None:
        collector = self.collector
        collector.clear_buffer()
        if op.kind == tir.ForKind.PARALLEL:
            collector.visit_stmt(op.body)
            self.add_set("ub")
            return
        self.visit_stmt(op.body)


class _BufferLegalizer:
    def __init__(self) -> None:
        self.var_mem_map_ = None
    
    def insert(self, stmt) -> Stmt:
        classifier = _BufferClassifier()
        classifier.visit_stmt(stmt)
        self.var_mem_map_ = classifier.var_mem_map_

        def preorder(op):
            if isinstance(op, Block):
                if op.name_hint == "tilelang_root":
                    new_buffers = list(op.alloc_buffers)
                    for bf in op.alloc_buffers:
                        for scope in self.var_mem_map_[bf]:
                            new_bf = decl_buffer(bf.shape, bf.dtype, name=bf.name + "_" + scope, scope="shared." + scope)
                            new_buffers.append(new_bf)
                    return Block(
                        iter_vars=[],
                        reads=[],
                        writes=[],
                        name_hint=op.name_hint,
                        body=op.body,
                        alloc_buffers=new_buffers,
                        match_buffers=op.match_buffers,
                        annotations=op.annotations,
                    )
        return ir_transform(stmt, preorder, None)
                


def BufferLegalizer():
    def pass_fn(func: PrimFunc, mod, ctx):
        legalizer = _BufferLegalizer()
        func = func.with_body(legalizer.insert(func.body))
        return func

    return prim_func_pass(pass_fn, opt_level=0)
