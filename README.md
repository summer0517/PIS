# Anonymous Training Code for EMNLP Submission

This repository contains the training and analysis code for the paper's causal-intervention defense experiments. The code is anonymized for double-blind review: paths, model locations, datasets, and vector files should be supplied by the user at runtime through command-line arguments.

## Repository Layout

```text
main_causal_defense.py                 # Main DeepSpeed training entry point
patch.py                               # Compatibility patch for block-diagonal attention utilities
causal_defense/
  defense_engine.py                    # Delta-loss defense logic
  hooks.py                             # Forward hooks for vector injection
  defense_proj.py                      # State-aware projection monitor
  gradient_probe_defense.py            # Injection-gradient probe defense
  immune_delta_preserver.py            # Immune delta preservation
  immune_vector_continuation.py        # Immune continuation injector
  gradient_analyzer.py                 # TensorBoard gradient/activation analysis
  build_baseline_projections.py        # Offline projection preprocessing for state-aware defense
  plot_defense_records.py              # Plotting utilities
utils/
  ds_utils.py                          # DeepSpeed configuration helpers
  utils.py                             # Training utilities and checkpoint saving
  data/                                # Dataset wrappers and tokenization helpers
```

## Environment

The experiments are designed for multi-GPU training with DeepSpeed ZeRO-3 and bf16 model loading. 


## Data Format

Two input modes are supported.

### Pre-tokenized Dataset

Pass one or more HuggingFace `Dataset.save_to_disk(...)` directories with:

```text
input_ids
attention_mask
labels
```

Use this mode when `--dynamic_tokenize` is not set.

### JSONL Dynamic Tokenization

When `--dynamic_tokenize` is enabled, each JSON record should contain a `messages` field. The first user message is used as the prompt, and the assistant message is used as the supervised response:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

The training script masks prompt tokens with `-100` and computes loss on response tokens.

## Core Training Command

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <DATASET_PATH_OR_JSONL> \
  --output_dir <OUTPUT_DIR> \
  --job_name <RUN_NAME> \
  --num_train_epochs 1 \
  --max_seq_len 65536 \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --num_warmup_steps 10 \
  --lr_scheduler_type cosine \
  --zero_stage 3 \
  --gradient_checkpointing
```

Use `--dynamic_tokenize` if `<DATASET_PATH_OR_JSONL>` points to raw JSONL files rather than saved tokenized datasets.

## Reproducible Hyperparameter Defaults

The table below summarizes the key defaults defined in `main_causal_defense.py`. These values should be reported or explicitly overridden in every reproduced run.

### Training and System Defaults

| Parameter | Default | Reproducibility note |
| --- | ---: | --- |
| `--num_train_epochs` | `1` | Number of full training epochs. |
| `--learning_rate` | `5e-6` | Peak learning rate used by the scheduler. |
| `--weight_decay` | `1e-4` | Optimizer weight decay. |
| `--num_warmup_steps` | `10` | Linear warmup steps before the scheduler reaches the peak learning rate. |
| `--lr_scheduler_type` | `cosine` | Learning-rate schedule. |
| `--per_device_train_batch_size` | `1` | Per-GPU micro-batch size. |
| `--gradient_accumulation_steps` | `4` | Micro-batches accumulated before each optimizer step. |
| `--max_seq_len` | `65536` | Maximum sequence length during training. |
| `--seed` | `1234` | Random seed passed to the training utilities. |
| `--zero_stage` | `3` | DeepSpeed ZeRO stage. |
| `--offload` | `False` | CPU offload is disabled unless explicitly enabled. |
| `--gradient_checkpointing` | `True` | Gradient checkpointing is enabled by default. |
| `--save_interval` | `200` | Checkpoint interval before multiplying by gradient accumulation. |
| `--save_checkpoint` | `True` | Optimizer state is saved for resumable checkpoints. |
| `--dynamic_tokenize` | `False` | Saved tokenized datasets are used unless this flag is enabled. |

The effective batch size is:

```text
num_gpus * per_device_train_batch_size * gradient_accumulation_steps
```

With the defaults above, this is `4 * num_gpus` training examples per optimizer step.

### Defense and Intervention Defaults

| Parameter | Default | Reproducibility note |
| --- | ---: | --- |
| `--enable_causal_defense` | `False` | Delta-loss defense is disabled by default. |
| `--enable_unconditional_injection` | `False` | Unconditional vector injection is disabled by default. |
| `--enable_gradient_analysis` | `False` | TensorBoard gradient/activation analysis is disabled by default. |
| `--enable_immune_delta_preservation` | `False` | Immune delta preservation is disabled by default. |
| `--enable_state_aware_defense` | `False` | State-aware projection defense is disabled by default. |
| `--enable_injection_gradient_probe_defense` | `False` | Injection-gradient probe defense is disabled by default. |
| `--defense_target_layers` | `[30]` | One-indexed transformer layer ids used by `main_causal_defense.py`. |
| `--defense_alpha` | `1.0` | Vector injection strength. |
| `--defense_adaptive_alpha` | `False` | Activation-norm adaptive alpha scaling is disabled by default. |
| `--injection_mode` | `res_only` | Injection is restricted to response tokens by default. |
| `--vector_fusion_mode` | `0` | L2-norm aligned multi-vector fusion. |
| `--defense_mode` | `mask` | Detected samples/tokens are masked from the loss. |
| `--defense_granularity` | `sample` | Detection is sample-level by default. |
| `--defense_sample_strategy` | `mean` | Sample-level Delta-loss aggregation. |
| `--defense_delta_threshold` | `0.0` | Unsafe if Delta loss is below this threshold. |

### Gradient-Probe Defaults

| Parameter | Default | Reproducibility note |
| --- | ---: | --- |
| `--gradient_probe_order_factor` | `10.0` | Logged diagnostic ratio; not used for the final decision. |
| `--gradient_probe_cos_threshold` | `0.12` | Cosine threshold for injected gradient alignment. |
| `--gradient_probe_proj_threshold` | `1e-8` | Projection threshold for injected gradient alignment. |
| `--gradient_probe_perturb_alpha` | `None` | Falls back to `--defense_alpha` when unset. |
| `--gradient_probe_diff_epsilon` | `None` | Falls back to the perturbation alpha when unset. |
| `--defense_analysis_interval` | `100` | Gradient/activation analysis interval in steps. |

### Immune Delta Preservation Defaults

| Parameter | Default | Reproducibility note |
| --- | ---: | --- |
| `--immune_boundary_step` | `-1` | Must be set for immune preservation unless reusing a disable boundary. |
| `--immune_param_scope` | `target_core` | Protects target-layer `o_proj` and `down_proj` by default. |
| `--immune_min_delta_norm` | `1e-12` | Minimum parameter displacement norm to protect. |
| `--immune_exact_distributed_projection` | `True` | Uses all-reduce for exact distributed projection. |
| `--immune_include_input_embeddings` | `False` | Input embeddings are excluded by default. |
| `--immune_preservation_strategy` | `gradient_projection` | Default post-boundary preservation strategy. |
| `--immune_projection_mode` | `svd_subspace` | Default gradient projection basis. |
| `--immune_svd_rank` | `8` | Rank retained per protected weight matrix. |
| `--immune_svd_oversample` | `4` | Randomized sketch oversampling dimension. |
| `--immune_projection_strength` | `1.0` | Full removal of destructive gradient components. |
| `--immune_allow_svd_partial` | `False` | Missing SVD bases are not silently ignored by default. |
| `--immune_antibody_rank` | `1` | Rank for immune-continuation directions. |
| `--immune_print_svd_energy_topk` | `5` | Number of singular-value energy entries printed. |
| `--immune_antibody_modules` | `o_proj,down_proj` | Modules used by immune continuation. |
| `--immune_calibration_micro_batches` | `4` | Calibration micro-batches for immune continuation. |
| `--immune_calibration_min_response_tokens` | `2048` | Minimum response tokens before finalizing continuation directions. |
| `--immune_antibody_source` | `functional_mean` | Direction source for immune continuation. |
| `--immune_continuation_scale_mode` | `match_v_preserve_ratio` | Scaling rule for continuation injection. |

### Ablation Defaults

| Parameter | Default | Reproducibility note |
| --- | ---: | --- |
| `--disable_defense_step` | `-1` | Defense is not forcibly disabled unless set. |
| `--enable_old_step_count` | `True` | Uses the legacy micro-batch step-count convention. |
| `--defense_decline_alpha` | `False` | Alpha does not decay unless enabled. |
| `--defense_rise_alpha` | `False` | Alpha does not rise unless enabled. |
| `--defense_decline_start_step` | `0` | Start step for alpha scheduling. |

## Experimental Configurations

The following settings correspond to the main defense modes used in the paper. Replace all placeholder paths with local artifact paths.

### 1. Standard Fine-Tuning Baseline

No defense flags are enabled.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/sft_baseline \
  --job_name sft_baseline \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --weight_decay 1e-4 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --max_seq_len 65536 \
  --zero_stage 3
```

### 2. Unconditional Vector Injection

Injects the supplied vector at every training step. This is useful as an intervention baseline.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/unconditional_injection \
  --job_name unconditional_injection \
  --enable_unconditional_injection \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 1.0 \
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

### 3. Delta-Loss Dynamic Defense

Runs a natural and an injected forward pass, then blocks or injects examples according to the Delta-loss criterion.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/delta_loss_defense \
  --job_name delta_loss_defense \
  --enable_causal_defense \
  --defense_mode mask \
  --defense_granularity sample \
  --defense_sample_strategy mean \
  --defense_delta_threshold 0.0 \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 1.0 \
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

Important parameters:

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `--defense_mode` | `mask` | `mask` removes loss from detected samples/tokens; `inject` trains with vector injection. |
| `--defense_granularity` | `sample` | Detection granularity: `sample` or `token`. |
| `--defense_sample_strategy` | `mean` | Sample-level aggregation: `mean` or `min`. |
| `--defense_delta_threshold` | `0.0` | Mark as unsafe when Delta loss is below this threshold. |
| `--injection_mode` | `res_only` | Inject into response tokens only, or use `all_token`. |
| `--vector_fusion_mode` | `0` | `0`: L2-norm aligned fusion; `1`: direct averaging. |

### 4. Injection-Gradient Probe Defense

Uses a micro-perturbation along the vector direction and detects samples by the resulting gradient alignment.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/gradient_probe \
  --job_name gradient_probe \
  --enable_injection_gradient_probe_defense \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <LAYER_ID> \
  --defense_alpha 1.0 \
  --gradient_probe_cos_threshold 0.12 \
  --gradient_probe_proj_threshold 1e-8 \
  --gradient_probe_perturb_alpha 0.01 \
  --gradient_probe_diff_epsilon 0.01 \
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

### 5. State-Aware Defense

This mode needs an offline preprocessing step to add `ref_projection` to the dataset.

```bash
python causal_defense/build_baseline_projections.py \
  --model_path <MODEL_PATH> \
  --data_paths <RAW_JSONL_1> <RAW_JSONL_2> \
  --output_dir <TOKENIZED_DATASET_WITH_PROJECTIONS> \
  --vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --target_layer <ZERO_INDEXED_LAYER_ID> \
  --mode res_mean \
  --max_seq_len 2048 \
  --vector_fusion_mode 0 \
  --projection_names vector_a vector_b vector_c
```

Then train with:

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TOKENIZED_DATASET_WITH_PROJECTIONS> \
  --output_dir <OUTPUT_DIR>/state_aware \
  --job_name state_aware \
  --enable_state_aware_defense \
  --malicious_vector_paths <VECTOR_1.pt> <VECTOR_2.pt> <VECTOR_3.pt> \
  --defense_target_layers <ONE_INDEXED_LAYER_ID> \
  --defense_alpha 1.0 \
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

Note that `build_baseline_projections.py` uses a zero-indexed layer id, while `main_causal_defense.py` uses one-indexed layer ids for `--defense_target_layers`.

### 6. Immune Delta Preservation

Trains with vector injection for the first `K` micro-batch steps, records the parameter displacement, then preserves that displacement after injection is removed.

```bash
deepspeed --num_gpus <NUM_GPUS> main_causal_defense.py \
  --model_name_or_path <MODEL_PATH> \
  --train_data <TRAIN_DATA> \
  --output_dir <OUTPUT_DIR>/immune_delta \
  --job_name immune_delta \
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
  --defense_alpha 1.0 \
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

For immune continuation, set:

```bash
--immune_preservation_strategy immune_continuation \
--immune_antibody_rank 1 \
--immune_antibody_modules o_proj,down_proj \
--immune_calibration_micro_batches 4 \
--immune_calibration_min_response_tokens 2048 \
--immune_antibody_source functional_mean \
--immune_continuation_scale_mode match_v_preserve_ratio
```

## Main Hyperparameters to Report

For each paper run, report at least:

| Category | Fields |
| --- | --- |
| Model | base model identifier, parameter scale, tokenizer, trust-remote-code setting |
| Data | dataset name or anonymized source, split sizes, filtering, max sequence length |
| Training | epochs, effective batch size, learning rate, scheduler, warmup steps, weight decay, seed |
| Systems | GPU type/count, ZeRO stage, bf16/fp16, gradient checkpointing, offload |
| Defense vectors | vector source, number of vectors, target layers, fusion mode, normalization |
| Defense mode | enabled flag, threshold, alpha, injection scope, boundary step if applicable |
| Evaluation | checkpoint used, decoding parameters, metrics, number of examples |

The effective batch size is:

```text
num_gpus * per_device_train_batch_size * gradient_accumulation_steps
```

## Logging and Checkpoints

Training writes TensorBoard logs under:

```text
<OUTPUT_DIR>/<JOB_NAME>
```

The script logs loss, learning rate, defense decisions, blocked sample/token ratios, projection statistics, gradient-probe metrics, and immune-preservation diagnostics when the corresponding modules are enabled.

Checkpoints are saved every:

```text
save_interval * gradient_accumulation_steps
```

micro-batch steps. The default `--save_interval` is `200`.

## Reproducibility Checklist

Before releasing results, save the following with each run:

- Full command line.
- Git commit hash.
- Python, CUDA, PyTorch, Transformers, DeepSpeed, PEFT, and vLLM versions.
- Random seed. The default is `1234`.
- Dataset construction script or preprocessing command.
- Vector file names and target layer ids.
- Evaluation script, metric definitions, and decoding parameters.
- TensorBoard logs or a tabular export of the reported metrics.

## Notes for Double-Blind Review

- Do not commit local absolute paths, usernames, institutional directories, private model names, or personal emails.
- Keep model, data, and output locations as command-line arguments.
- Use anonymous Git commit metadata for review submissions.
- If a released artifact cannot include the original data or vectors, provide scripts, checksums, and exact instructions for reconstructing or substituting them.

## Third-Party Code

Some utility code follows DeepSpeed-Chat/Megatron-DeepSpeed conventions and retains upstream copyright and license notices where applicable.
