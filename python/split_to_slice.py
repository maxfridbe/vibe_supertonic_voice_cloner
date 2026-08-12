#!/usr/bin/env python3
"""Rewrite Split nodes as Slice nodes so ORT's NNAPI builder never sees them
(its AddNnapiSplit mishandles these graphs' Splits: "count [0] does not evenly
divide dimension"). Slice is either placed or cleanly partitioned around.

Works on static-shape models (shapes must be inferable along the split axis).
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference


def main(src, dst):
    m = onnx.load(src)
    inferred = shape_inference.infer_shapes(m)
    shapes = {}
    for vi in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = [d.dim_value if d.HasField("dim_value") else -1
                for d in vi.type.tensor_type.shape.dim]
        shapes[vi.name] = dims
    inits = {i.name: i for i in m.graph.initializer}

    new_nodes, extra_inits, n_rewritten = [], [], 0
    for node in m.graph.node:
        if node.op_type != "Split":
            new_nodes.append(node)
            continue
        axis = 0
        explicit = None
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
            if a.name == "split":
                explicit = list(a.ints)
        if explicit is None and len(node.input) > 1 and node.input[1] in inits:
            explicit = numpy_helper.to_array(inits[node.input[1]]).tolist()
        if explicit is None:
            shape = shapes.get(node.input[0])
            if shape is None or shape[axis] <= 0:
                raise SystemExit(f"cannot infer split sizes for {node.name}")
            explicit = [shape[axis] // len(node.output)] * len(node.output)
        start = 0
        for i, (out, size) in enumerate(zip(node.output, explicit)):
            s = helper.make_tensor(f"{out}__start", onnx.TensorProto.INT64, [1], [start])
            e = helper.make_tensor(f"{out}__end", onnx.TensorProto.INT64, [1], [start + size])
            ax = helper.make_tensor(f"{out}__axis", onnx.TensorProto.INT64, [1], [axis])
            extra_inits += [s, e, ax]
            new_nodes.append(helper.make_node(
                "Slice", [node.input[0], f"{out}__start", f"{out}__end", f"{out}__axis"],
                [out], name=f"{node.name or out}_slice{i}"))
            start += size
        n_rewritten += 1

    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    m.graph.initializer.extend(extra_inits)
    onnx.checker.check_model(m, full_check=False)
    onnx.save(m, dst)
    print(f"{src}: rewrote {n_rewritten} Split nodes -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
