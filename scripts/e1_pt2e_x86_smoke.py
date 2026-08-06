#!/usr/bin/env python3
"""Fail-closed PT2E static-INT8 compatibility smoke test for E1.

This script validates the approved CIFAR-10 ResNet-18 checkpoint against the
TorchAO x86 Inductor PT2E path. It uses synthetic inputs only and writes no
model or experiment artifact. It therefore does NOT replace the frozen
512-image calibration run.
"""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torch.fx import GraphModule
from torchvision.models import resnet18

from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)

SEED = 20260729
EXPECTED_PARAMETER_COUNT = 11_173_962
EXPECTED_STATE_DICT_TENSORS = 122
EXPECTED_OUTPUT_SHAPE = (1, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/CNN-002B/best_model.pt"),
        help="Approved CNN-002B checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=2,
        help="Number of synthetic compatibility batches (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic calibration batch size (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.calibration_batches < 1:
        parser.error("--calibration-batches must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def build_approved_model(checkpoint_path: Path) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Expected a direct state-dictionary checkpoint, got "
            f"{type(state_dict).__name__}"
        )
    if len(state_dict) != EXPECTED_STATE_DICT_TENSORS:
        raise RuntimeError(
            f"Expected {EXPECTED_STATE_DICT_TENSORS} checkpoint tensors; "
            f"found {len(state_dict)}"
        )

    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Strict checkpoint load failed: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT} parameters; found {parameter_count}"
        )

    model.cpu().eval()
    print("STRICT_MODEL_GATE: PASS")
    print("checkpoint_type:", type(state_dict).__name__)
    print("tensor_count:", len(state_dict))
    print("parameter_count:", parameter_count)
    return model


def require_valid_output(output: Tensor, expected_batch_size: int) -> None:
    expected_shape = (expected_batch_size, EXPECTED_OUTPUT_SHAPE[1])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"Expected output shape {expected_shape}; found {tuple(output.shape)}"
        )
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Model output contains NaN or infinite values")


def annotated_nodes(model: GraphModule) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in model.graph.nodes:
        annotation = node.meta.get("quantization_annotation")
        if annotation is not None and bool(
            getattr(annotation, "_annotated", False)
        ):
            found.append((node.name, str(node.target)))
    return found


def quantization_nodes(model: GraphModule) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for node in model.graph.nodes:
        target = str(node.target)
        lowered = target.lower()
        if "quantize" in lowered or "dequantize" in lowered:
            found.append((node.op, node.name, target))
    return found


def show_rows(label: str, rows: Iterable[tuple[str, ...]], limit: int) -> None:
    for row in list(rows)[:limit]:
        print(f"{label}:", *row)


def capture_quantizable_graph(model, example_inputs):
    try:
        exported_model = torch.export.export_for_training(
            model,
            example_inputs,
        ).module()
        return exported_model, "torch.export.export_for_training"
    except (AttributeError, ImportError):
        exported_model = torch.export.export(
            model,
            example_inputs,
        ).module()
        return exported_model, "torch.export.export"


def main() -> int:
    args = parse_args()
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)
    torch.backends.quantized.engine = "x86"

    model = build_approved_model(args.checkpoint)
    example_inputs = (torch.zeros(1, 3, 32, 32, dtype=torch.float32),)

    with torch.inference_mode():
        fp32_output = model(*example_inputs)
    require_valid_output(fp32_output, expected_batch_size=1)
    print("FP32_GATE: PASS")
    print("fp32_shape:", tuple(fp32_output.shape))
    print("fp32_finite:", bool(torch.isfinite(fp32_output).all()))

    exported_model, capture_api = capture_quantizable_graph(
        model,
        example_inputs,
    )
    move_exported_model_to_eval(exported_model)
    print("capture_api:", capture_api)
    print("EXPORT_GATE: PASS")

    exported_nodes = list(exported_model.graph.nodes)
    source_fn_nodes = [
        node for node in exported_nodes if node.meta.get("source_fn_stack")
    ]
    torch_fn_nodes = [node for node in exported_nodes if node.meta.get("torch_fn")]
    nn_module_nodes = [
        node for node in exported_nodes if node.meta.get("nn_module_stack")
    ]

    print("exported_node_count:", len(exported_nodes))
    print("source_fn_stack_node_count:", len(source_fn_nodes))
    print("torch_fn_node_count:", len(torch_fn_nodes))
    print("nn_module_stack_node_count:", len(nn_module_nodes))

    for node in exported_nodes[:30]:
        print(
            "exported_node:",
            node.op,
            node.name,
            str(node.target),
            "source_fn_stack=",
            bool(node.meta.get("source_fn_stack")),
            "torch_fn=",
            node.meta.get("torch_fn"),
            "nn_module_stack=",
            bool(node.meta.get("nn_module_stack")),
        )

    quantizer = X86InductorQuantizer()
    quantizer.set_global(
        get_default_x86_inductor_quantization_config(
            is_qat=False,
            is_dynamic=False,
            reduce_range=False,
        )
    )
    prepared_model = prepare_pt2e(exported_model, quantizer)

    annotations = annotated_nodes(prepared_model)
    print("PREPARE_GATE: PASS")
    print("annotated_node_count:", len(annotations))
    show_rows("annotated", annotations, limit=30)
    if not annotations:
        torch_version = getattr(torch, "__version__", "unknown")
        torchao_version = metadata.version("torchao") if metadata.version("torchao") else "unknown"
        torchvision_version = metadata.version("torchvision") if metadata.version("torchvision") else "unknown"
        raise RuntimeError(
            "X86InductorQuantizer annotated zero graph nodes; "
            f"torch={torch_version}, torchao={torchao_version}, "
            f"torchvision={torchvision_version}"
        )
    print("ANNOTATION_GATE: PASS")

    # Compatibility-only synthetic calibration. This is deliberately distinct
    # from the frozen E1 CIFAR-10 calibration manifest.
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    with torch.inference_mode():
        for _ in range(args.calibration_batches):
            calibration_input = torch.rand(
                args.batch_size,
                3,
                32,
                32,
                dtype=torch.float32,
                generator=generator,
            )
            calibration_output = prepared_model(calibration_input)
            require_valid_output(calibration_output, args.batch_size)
    print("SMOKE_CALIBRATION_GATE: PASS")

    converted_model = convert_pt2e(prepared_model)
    move_exported_model_to_eval(converted_model)
    print("CONVERT_GATE: PASS")
    print("EXPORTED_EVAL_GATE: PASS")

    with torch.inference_mode():
        converted_output = converted_model(*example_inputs)
    require_valid_output(converted_output, expected_batch_size=1)

    q_nodes = quantization_nodes(converted_model)
    max_abs_difference = float((fp32_output - converted_output).abs().max())
    print("converted_shape:", tuple(converted_output.shape))
    print("converted_finite:", bool(torch.isfinite(converted_output).all()))
    print("quantization_node_count:", len(q_nodes))
    show_rows("quantization_node", q_nodes, limit=50)
    print("max_abs_difference:", max_abs_difference)

    if not q_nodes:
        raise RuntimeError(
            "Conversion produced no quantization-related operators; "
            "the result is not accepted as a reference-quantized graph"
        )

    print("PT2E_X86_SMOKE_GATE: PASS")
    print(
        "NOTICE: synthetic compatibility calibration only; "
        "no E1 experiment artifact was created"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PT2E_X86_SMOKE_GATE: FAIL: {exc}", file=sys.stderr)
        raise