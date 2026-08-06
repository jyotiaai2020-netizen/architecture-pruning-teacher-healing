#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)


def main():
    torch.manual_seed(20260729)
    torch.set_grad_enabled(False)
    torch.backends.quantized.engine = "x86"

    model = M().eval()
    example_inputs = (torch.randn(1, 5),)

    exported = torch.export.export(model, example_inputs).module()
    print("exported_node_count:", len(list(exported.graph.nodes)))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    prepared = prepare_pt2e(exported, quantizer)
    prepared_annotations = sum(
        1
        for node in prepared.graph.nodes
        if node.meta.get("quantization_annotation")
        and getattr(node.meta["quantization_annotation"], "_annotated", False)
    )
    print("prepared_annotations:", prepared_annotations)

    with torch.no_grad():
        prepared(*example_inputs)

    converted = convert_pt2e(prepared)
    qdq = []
    for node in converted.graph.nodes:
        target = str(node.target).lower()
        if "quantize" in target or "dequantize" in target:
            qdq.append((node.op, node.name, str(node.target)))

    print("converted_qdq_count:", len(qdq))
    for row in qdq:
        print("qdq:", *row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
