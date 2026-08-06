#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)


class MinimalLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)

    def forward(self, x):
        return self.linear(x)


class MinimalConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def qdq_nodes(model):
    rows = []
    for node in model.graph.nodes:
        target = str(node.target).lower()
        if "quantize" in target or "dequantize" in target:
            rows.append((node.op, node.name, str(node.target)))
    return rows


def annotation_count(model):
    count = 0
    for node in model.graph.nodes:
        annotation = node.meta.get("quantization_annotation")
        if annotation is not None and getattr(annotation, "_annotated", False):
            count += 1
    return count


def test_model(name, model, example_inputs):
    print(f"\nMODEL: {name}")
    model = model.cpu().eval()

    exported = torch.export.export(model, example_inputs).module()
    print("exported_node_count:", len(list(exported.graph.nodes)))

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    prepared = prepare_pt2e(exported, quantizer)
    annotations = annotation_count(prepared)
    print("annotated_node_count:", annotations)

    with torch.no_grad():
        for _ in range(2):
            prepared(*example_inputs)

    converted = convert_pt2e(prepared)
    qdq = qdq_nodes(converted)
    print("quantization_node_count:", len(qdq))

    for row in qdq:
        print("quantization_node:", *row)

    with torch.no_grad():
        output = converted(*example_inputs)

    print("output_shape:", tuple(output.shape))
    print("output_finite:", bool(torch.isfinite(output).all()))

    if not qdq:
        print(f"{name}_PT2E_GATE: FAIL")
        return False

    print(f"{name}_PT2E_GATE: PASS")
    return True


def main():
    torch.manual_seed(20260729)
    torch.set_grad_enabled(False)
    torch.backends.quantized.engine = "x86"

    linear_pass = test_model(
        "MINIMAL_LINEAR",
        MinimalLinear(),
        (torch.randn(1, 5),),
    )

    conv_pass = test_model(
        "MINIMAL_CONV",
        MinimalConv(),
        (torch.randn(1, 3, 32, 32),),
    )

    if linear_pass and conv_pass:
        print("\nMINIMAL_PT2E_DIAGNOSTIC: PASS")
        return 0

    print("\nMINIMAL_PT2E_DIAGNOSTIC: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
