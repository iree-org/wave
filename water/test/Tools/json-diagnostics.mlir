// RUN: not water-opt %s -diagnostics-file - | FileCheck %s

// The test below checks JSON printing of diagnostics by having function
// declarations not marked private, which is an error that should cause
// a diagnostic to be emitted.

// CHECK: {"message":"'func.func' op symbol declaration cannot have public visibility","severity":"error","unknown":true}
func.func @unknown() loc(unknown)

// CHECK: {"end_column":15,"end_line":43,"file":"file.mlir","message":"'func.func' op symbol declaration cannot have public visibility","severity":"error","start_column":10,"start_line":42}
func.func @filerange() loc("file.mlir":42:10 to 43:15)

// CHECK: {"loc":{"unknown":true},"message":"'func.func' op symbol declaration cannot have public visibility","name":"foo","severity":"error"}
func.func @named() loc("foo")

// CHECK: {"loc":{"end_column":15,"end_line":42,"file":"file.mlir","start_column":10,"start_line":42},"message":"'func.func' op symbol declaration cannot have public visibility","name":"foo","severity":"error"}
func.func @named_with_child() loc("foo"("file.mlir":42:10 to :15))

// CHECK: {"fused":[{"loc":{"unknown":true},"name":"bar"},{"loc":{"unknown":true},"name":"baz"}],"message":"'func.func' op symbol declaration cannot have public visibility","severity":"error"}
func.func @fused() loc(fused<"foo">["bar", "baz"])

// CHECK:      {"callstack":[
// CHECK-SAME:   {"end_column":1,"end_line":1,"file":"file.mlir","start_column":1,"start_line":1},
// CHECK-SAME:   {"end_column":3,"end_line":2,"file":"file.mlir","start_column":3,"start_line":2},
// CHECK-SAME:   {"end_column":0,"end_line":4,"file":"file.mlir","start_column":0,"start_line":4},
// CHECK-SAME:   {"end_column":0,"end_line":5,"file":"file.mlir","start_column":0,"start_line":5}
// CHECK-SAME: ],"message":"'func.func' op symbol declaration cannot have public visibility","severity":"error"}
func.func @callsite() loc(callsite("file.mlir":1:1 at callsite("file.mlir":2:3 at callsite("file.mlir":4 at "file.mlir":5))))
