# E1-A2 PT2E approved-matrix control

This control tested an unmodified TorchVision ResNet-18 using Python
3.12.3, PyTorch 2.9.1+cpu, TorchVision 0.24.1+cpu, and TorchAO 0.15.0.

The FP32 graph exported successfully with 194 nodes. PT2E preparation
produced zero observable quantization annotations, and conversion
produced zero QuantizeLinear and DequantizeLinear nodes.

No valid INT8 model was created. Calibration and post-quantization
evaluation were therefore not performed.

This is an implementation-feasibility result for the tested PT2E
workflow. It is not evidence against ResNet-18 quantizability or the
INT8 capability-preservation hypothesis.
