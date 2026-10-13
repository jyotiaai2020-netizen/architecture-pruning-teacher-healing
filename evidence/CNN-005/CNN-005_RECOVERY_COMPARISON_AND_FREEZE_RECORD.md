# CNN-005 Recovery Comparison and Freeze Record

**Status:** Training completed; best checkpoints strictly reloaded and validation reproduced.  
**Branch:** `experiment/cnn005-recovery`  
**Dataset:** CIFAR-10; frozen 45,000 training / 5,000 validation split; seed 42.  
**Held-out test:** Not evaluated during CNN-005.

## Research question

Does teacher-guided knowledge distillation recover more validation accuracy than supervised cross-entropy fine-tuning when both begin with the identical structured-pruned CNN-004 ResNet-50 initialization?

## Frozen inputs

| Input artifact | SHA-256 |
|---|---|
| `checkpoints/CNN-003B/best_model.pt` (teacher) | `4d3ca6a7b1f0396e4e56024341144d487aa173ace0c7e5d981eaf9879a4ae31a` |
| `results/CNN-004/cnn004_pruned_before_recovery_state_dict.pt` | `595b761259c88e4f656a1a3102ec457d41c57551db301c39111fe4157fb89057` |
| `evidence/CNN-003B/cifar10_train45k_val5k_seed42.json` | `1c10c8261334ccc8cc3035b6fb3cfc7083b6aee45d3913971eaff11044e1d4bf` |

Both arms used the same split and starting weights. Validation used the CIFAR-10 training partition with the frozen validation indices, deterministic ToTensor/Normalize preprocessing, and no test-set loader.

## Matched recovery protocol

- 20 epochs; batch size 128; seed 42; automatic mixed precision.
- SGD, initial learning rate 0.01, momentum 0.9, weight decay 0.0005.
- MultiStepLR milestones 10 and 16, gamma 0.1.
- CNN-005A: supervised cross-entropy (CE).
- CNN-005B: teacher-guided CE/KL distillation; temperature 4.0, KD alpha 0.5, frozen CNN-003B teacher in evaluation mode.
- Best checkpoint selected by frozen validation accuracy, not held-out test accuracy.

## Results

| Model | Parameters | MACs/image | Validation accuracy | Best epoch |
|---|---:|---:|---:|---:|
| CNN-003B teacher | 23,520,842 | 1,311,594,496 | 92.38% | 48 |
| CNN-004 pruned, unrecovered | 20,218,770 | 1,134,338,752 | 92.18% | N/A |
| CNN-005A supervised recovery | 20,218,770 | 1,134,338,752* | 92.64% | 20 |
| CNN-005B teacher-guided recovery | 20,218,770 | 1,134,338,752* | **92.88%** | 20 |

*CNN-005A/B preserve the CNN-004 architecture; MAC count carried forward from the CNN-004 structural profile, not independently reprofiled here.

The pruned architecture has approximately 14.04% fewer parameters and 13.51% fewer MACs than the teacher. CNN-005B exceeds CNN-005A by 0.24 percentage points (12 additional correct predictions among 5,000 validation examples), CNN-004 by 0.70 percentage points, and the teacher's recorded validation result by 0.50 percentage points.

## CNN-005 artifact integrity

| Artifact | SHA-256 |
|---|---|
| `checkpoints/CNN-005A/best_model.pt` | `b974d77fb6cf6d7b2235ea0ec928a95a4e8085e930357a2b34392bc5a57b7639` |
| `checkpoints/CNN-005A/final_model.pt` | `03020bae48a38307d589c5a2bf987b669794eeb2d248e458e5bf7df9bb32f03a` |
| `results/CNN-005A/summary.json` | `34094ade7ef0bc18d8b9793659fa9bf5d8173697338dd0d2327936b95ee01eaa` |
| `checkpoints/CNN-005B/best_model.pt` | `62eb8ec8a2826848067236fdd85aae10b06a24b3a2af9dbb929aeb16208f065e` |
| `checkpoints/CNN-005B/final_model.pt` | `a2088d365b5b487928865d7a8613b979f83efe0a71752ff18428bf13e22db8d9` |
| `results/CNN-005B/summary.json` | `9e3a59a2a6bfadcae41520623b68de2ec57c08094f984a276db5ac9476d08443` |

These hashes were reported from the EC2 working copy. The checkpoints remain on EC2 and are **not** included in this documentation commit; establish durable backup before treating the artifacts as fully archived.

## Independent verification

Both best checkpoints were deserialized with `torch.load(..., weights_only=True)` and loaded using `strict=True` into the reconstructed pruned model. Both yielded 20,218,770 parameters. Read-only evaluation on the 5,000-image frozen validation subset reproduced:

- CNN-005A: 4,632/5,000 correct = **92.64%**.
- CNN-005B: 4,644/5,000 correct = **92.88%**.

The CNN-005 runner and independent validation did not access the held-out CIFAR-10 test set.

## Interpretation and limitations

This is a controlled, single-seed comparison. The observed validation advantage of knowledge distillation is preliminary: no repeated-seed uncertainty estimate or statistical significance test has been established. Training losses across CE and CE+KD objectives are not directly comparable. Held-out test performance, quantization accuracy, deployment latency, memory at inference, and energy have not yet been measured for these recovered checkpoints.

The teacher's historical held-out test result must not be compared directly with CNN-005 validation results as though they were measured on the same partition.

## CNN-006 handoff

Use CNN-005B as the leading **validation-selected** recovery candidate while retaining CNN-005A as a controlled comparator and CNN-003B as the teacher baseline. Define quantization calibration, precision/backend, target hardware, inference benchmarking, and held-out test access rules before CNN-006 execution. Preserve all frozen source artifacts and CNN-005 checkpoints without modification.
