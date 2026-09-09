<div align="center">

# Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](#citation)
[![Conference](https://img.shields.io/badge/EMNLP_2026-Findings-4b44ce)](#citation)
[![Code](https://img.shields.io/badge/Code-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://github.com/summer0517/PIS)
[![Framework](https://img.shields.io/badge/Training-DeepSpeed-1f6feb)](https://www.deepspeed.ai/)
[![Models](https://img.shields.io/badge/Models-Qwen2.5_%7C_Gemma--3-f5c542)](#results-at-a-glance)

Official implementation of **Progressive Intensity Scheduling (PIS)** and the
mechanistic analyses of Preventative Steering presented in our EMNLP 2026
Findings paper.

[Overview](#overview) · [Quick Start](#quick-start) · [PIS](#progressive-intensity-scheduling) · [Analysis](#mechanistic-analysis) · [Citation](#citation)

</div>

<p align="center">
  <img src="assets/framework.png" width="100%" alt="Mechanistic dynamics of Preventative Steering and Progressive Intensity Scheduling">
</p>

## Overview

Large language models remain vulnerable to malicious fine-tuning that weakens
safety refusals and amplifies undesirable persona traits. Preventative Steering
injects undesirable-trait activation vectors during fine-tuning and removes the
intervention at evaluation time, but the origin of its lasting protection has
been unclear.

Our analysis shows that its protection is **process-dependent**:

- During early training, injection produces compensatory gradients that push
  the model against the undesirable direction.
- The steering-aligned corrective signal subsequently decays toward a
  direction-specific steady state.
- Attention output projections emerge as the dominant residual-write route for
  these defensive updates.
- Preserving the induced weight displacement or reinjecting its functional
  offset does not maintain protection independently.

Motivated by these findings, **Progressive Intensity Scheduling (PIS)** begins
with moderate steering and progressively increases its strength after
static-strength alignment starts to decay.

## Highlights

| | Contribution |
| --- | --- |
| **Temporal mechanism** | Identifies an early compensatory adaptation phase followed by a steering-direction-specific steady-state phase. |
| **Parameter-space evidence** | Traces the dominant residual-write pathway to attention output projections and analyzes the MLP bottleneck. |
| **Static-offset tests** | Implements Intervention Delta Preservation (IDP) and IDP Continuation to test whether the induced offset is a standalone defense. |
| **PIS** | Converts the mechanistic finding into a simple delayed linear schedule for steering intensity. |
| **Cross-model evaluation** | Evaluates Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct, and Gemma-3-12B-IT. |

## Results at a Glance

PIS consistently improves the aggregate safety score and lowers harmful-trait
expression relative to static-strength Preventative Steering on all three
evaluated backbones. The table below reports the static baseline and the best
observed schedule, with reinforcement beginning at step 200.

| Model | Static Safety ↑ | PIS Safety ↑ | Static Trait Avg. ↓ | PIS Trait Avg. ↓ |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-32B-Instruct | 80.18 | **86.98** | 3.04 | **1.38** |
| Qwen2.5-7B-Instruct | 67.55 | **71.95** | 3.81 | **1.45** |
| Gemma-3-12B-IT | 48.36 | **54.68** | 5.85 | **4.94** |

These averages summarize multiple safety benchmarks and three persona-trait
evaluations. Individual benchmarks may exhibit different trade-offs; refer to
the paper for the complete results and discussion.

## Method

At training step $t$, PIS injects the fused undesirable-trait direction
$v_m$ into the residual stream:

$$
\widetilde{h}_{l,t} = h_{l,t} + \alpha_t v_m.
$$

The coefficient remains at $\alpha_{\mathrm{base}}$ during early adaptation
and then increases linearly toward $\alpha_{\max}$:

$$
\alpha_t = \alpha_{\mathrm{base}} + r_t
(\alpha_{\max}-\alpha_{\mathrm{base}}).
$$

In the default paper configuration,
$\alpha_{\mathrm{base}}=20$ and
$\alpha_{\max}=2\alpha_{\mathrm{base}}$. The reinforcement onset can be
selected from steering-gradient alignment or set explicitly for a controlled
timing experiment.

<table>
  <tr>
    <td width="50%" align="center">
      <img src="assets/combined_alpha_cos.png" width="100%" alt="Steering-gradient alignment dynamics"><br>
      <sub><b>Steering-gradient alignment.</b> Static signals weaken as optimization adapts, motivating delayed reinforcement.</sub>
    </td>
    <td width="50%" align="center">
      <img src="assets/fig_static_sweep.png" width="100%" alt="Static injection-strength sweep"><br>
      <sub><b>Static-strength sweep.</b> Increasing average strength alone does not explain PIS.</sub>
    </td>
  </tr>
</table>

## Repository Structure

```text
.
├── main_causal_defense.py              # DeepSpeed training entry point
├── patch.py                            # Attention-utility compatibility patch
├── causal_defense/
│   ├── hooks.py                        # Residual-stream vector injection
│   ├── defense_engine.py               # Vector loading, fusion, and defense logic
│   ├── gradient_analyzer.py            # Activation/gradient/parameter analysis
│   ├── immune_delta_preserver.py       # IDP parameter-offset preservation
│   ├── immune_vector_continuation.py   # IDP Continuation
│   ├── gradient_probe_defense.py       # Gradient-probe experiments
│   ├── defense_proj.py                 # State-projection experiments
│   ├── build_baseline_projections.py   # Projection preprocessing
│   └── plot_defense_records.py         # Plotting utilities
└── utils/
    ├── ds_utils.py
    ├── utils.py
    └── data/
```

## Installation

The code targets Linux, CUDA, and multi-GPU DeepSpeed training. We recommend
using Python 3.10 in a clean environment.

```bash
conda create -n pis python=3.10 -y
conda activate pis

# Install a CUDA-compatible PyTorch build first.
pip install torch
pip install transformers datasets accelerate deepspeed peft
pip install tensorboard matplotlib numpy pandas tqdm
```

DeepSpeed and PyTorch versions must be compatible with the CUDA toolkit and GPU
environment. The example commands use `bfloat16` model loading and DeepSpeed
ZeRO-3; specify the settings explicitly for your own runs.

## Preparing the Inputs

### Data and persona vectors

The persona vectors and the data used to construct them are based on the
[Persona Vectors](https://github.com/safety-research/persona_vectors) project.
The malicious fine-tuning data used in the paper also follows that work. Please
refer to its repository for the original vector-construction procedure, data
format, and licensing information.

### Training data

Two input formats are supported.

**Pre-tokenized Hugging Face dataset.** Pass one or more directories created
with `Dataset.save_to_disk(...)`. Each sample must contain:

```text
input_ids
attention_mask
labels
```

Prompt positions should have label `-100`; the remaining labels identify the
response tokens used for supervised training and response-only injection.

**JSONL with dynamic tokenization.** Add `--dynamic_tokenize` and provide
records in the following format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Persona vectors

Supply one or more layer-indexed PyTorch vector files:

```bash
--malicious_vector_paths <EVIL_VECTOR.pt> <SYCOPHANCY_VECTOR.pt> <HALLUCINATION_VECTOR.pt>
```

The paper fuses the three trait directions with the magnitude-calibrated fusion
implemented by `--vector_fusion_mode 0`. Target layer identifiers passed to
`main_causal_defense.py` are **one-indexed** and must match the keys stored in
the vector files.

## Quick Start

All examples below should be run from the repository root. Replace placeholders
such as `<MODEL_PATH>` and `<VECTOR.pt>` with local artifact paths.

### 1. Unprotected fine-tuning

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/unprotected \
  --job_name unprotected \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --max_seq_len 65536 \
  --zero_stage 3
```

### 2. Static Preventative Steering

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/static \
  --job_name static \
  --enable_unconditional_injection \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 20 \
  --injection_mode res_only \
  --vector_fusion_mode 0 \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --max_seq_len 65536 \
  --zero_stage 3
```

## Progressive Intensity Scheduling

Enable PIS with `--defense_rise_alpha` and specify its onset using
`--defense_decline_start_step`. Despite the historical flag name, this argument
is the schedule onset for both rising and declining intensity.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/pis_step200 \
  --job_name pis_step200 \
  --enable_unconditional_injection \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 20 \
  --defense_rise_alpha \
  --defense_decline_start_step 200 \
  --injection_mode res_only \
  --vector_fusion_mode 0 \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --max_seq_len 65536 \
  --zero_stage 3
```

The implementation keeps `--defense_alpha` fixed before the onset and linearly
increases it to twice the base value by the final micro-batch of the epoch.

> [!IMPORTANT]
> The schedule uses the dataloader/micro-batch step convention implemented in
> `main_causal_defense.py`. Keep the data order, GPU count, per-device batch
> size, and gradient accumulation configuration consistent when reproducing a
> reported onset step.

## Mechanistic Analysis

Add `--enable_gradient_analysis` to a static, PIS, early-course, or unprotected
run. `--defense_analysis_interval` controls the logging interval.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/mechanism \
  --job_name mechanism \
  --enable_unconditional_injection \
  --enable_gradient_analysis \
  --defense_analysis_interval 100 \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 20 \
  --injection_mode res_only \
  --vector_fusion_mode 0 \
  --zero_stage 3
```

The analyzer logs response-pooled residual states and gradients, their
projection onto the fused persona direction, and chain-rule diagnostics for
attention output and MLP down projections.

```bash
tensorboard --logdir <OUTPUT_DIR>
```

## Early-Course Injection

To remove static steering after a selected micro-batch step, use:

```bash
--enable_unconditional_injection \
--disable_defense_step <K>
```

This configuration supports the Full-Course/Early-Course/No-Defense comparison
used to analyze the two-stage temporal dynamics.

## Intervention Delta Preservation

IDP applies steering until a boundary step, isolates the resulting parameter
displacement, and projects later gradients away from its dominant subspace.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <DATASET_PATH> \
  --output_dir <OUTPUT_DIR>/idp \
  --job_name idp \
  --enable_immune_delta_preservation \
  --immune_boundary_step <K> \
  --immune_preservation_strategy gradient_projection \
  --immune_projection_mode svd_subspace \
  --immune_svd_rank 8 \
  --immune_svd_oversample 4 \
  --immune_projection_strength 1.0 \
  --immune_param_scope target_core \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 20 \
  --injection_mode res_only \
  --vector_fusion_mode 0 \
  --zero_stage 3
```

For IDP Continuation, replace the preservation configuration with:

```bash
--immune_preservation_strategy immune_continuation \
--immune_antibody_modules o_proj,down_proj \
--immune_calibration_micro_batches 4 \
--immune_calibration_min_response_tokens 2048 \
--immune_antibody_source functional_mean \
--immune_continuation_scale_mode match_v_preserve_ratio
```

IDP and IDP Continuation are mechanistic tests rather than recommended defense
configurations; both fail to reproduce the protection of active steering in the
paper experiments.

## Additional Experimental Modules

The repository also retains several exploratory modules used during development
and analysis:

| Flag | Purpose |
| --- | --- |
| `--enable_causal_defense` | Delta-loss-based masking or conditional injection |
| `--enable_state_aware_defense` | State-projection monitoring |
| `--enable_injection_gradient_probe_defense` | Injection-gradient probing |
| `--enable_gradient_analysis` | Residual- and parameter-space diagnostics |
| `--enable_immune_delta_preservation` | IDP and IDP Continuation |

These modules are not all part of the final PIS method. Some combinations are
mutually exclusive and are rejected explicitly by the training script.

## Logging and Checkpoints

Each run writes its resolved command-line arguments and DeepSpeed configuration
to:

```text
<OUTPUT_DIR>/args.json
<OUTPUT_DIR>/ds_config.json
```

TensorBoard records training loss, learning rate, scheduled injection strength,
and enabled analysis metrics. Checkpoint frequency is controlled by
`save_interval × gradient_accumulation_steps` micro-batches.

## Reproducibility Notes

- The example configuration loads models with `trust_remote_code=True` and
  `bfloat16` weights; verify these settings for your environment.
- Full-parameter fine-tuning is used unless a LoRA configuration is supplied
  through `--lora_config`.
- Response-only injection depends on `labels != -100`; verify the label mask
  produced by custom preprocessing.
- `--defense_target_layers` is one-indexed in the main training script.
- Record the complete command, model revision, vector artifacts, dataset order,
  GPU count, and software versions for every run.

## Citation

If this work is useful in your research, please cite:

```bibtex
@inproceedings{guan2026active,
  title     = {Active Adaptation, Not Static Defense: Temporal Dynamics of Preventative Steering in Adversarial Fine-Tuning},
  author    = {Guan, Jing and Yang, Yachao and Liu, Zhaoliang and Zhang, Yuyao and Meng, Fanyu and Feng, Junlan},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```

The arXiv and ACL Anthology identifiers can be added once their final metadata
is confirmed.

## Acknowledgements

We thank the authors of [Persona Vectors](https://github.com/safety-research/persona_vectors)
for releasing the persona-vector construction framework and the associated
data used in this work.
