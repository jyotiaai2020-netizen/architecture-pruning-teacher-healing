# Architecture Pruning and Teacher Healing

A reproducible research framework for evaluating whether architecture-aware
simplification, knowledge distillation, quantization, and Teacher Healing can
produce compact models with high capability and a low Effective Deployment
Footprint (EDF).

## Primary evaluation

The best model is defined as the candidate that satisfies minimum capability
requirements while maximizing:

Capability Efficiency = Capability Score / EDF Score

## Initial CNN experiments

- CNN-001: TinyCNN framework validation
- CNN-002: ResNet-18 hand-designed student baseline
- CNN-003: ResNet-50 teacher baseline
- CNN-004: Architecture-simplified or structurally pruned model
- CNN-005: Simplified model with knowledge distillation
- CNN-006: Simplified model with KD and quantization
- CNN-007: Full hybrid method with Teacher Healing

## Environment

- AWS EC2 g4dn.xlarge
- NVIDIA Tesla T4
- Ubuntu
- Python virtual environment
- PyTorch
- MLflow
