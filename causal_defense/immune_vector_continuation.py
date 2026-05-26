# Copyright (c) 2026
# 因果干预动态防御 — 内生免疫向量续注入器
#
# IDP continuation:
#   1. 前 K 步照常注入恶意向量 v
#   2. K 步后从 o_proj / down_proj 的 ΔW 中蒸馏内生免疫向量 u
#   3. 对齐符号后，在对应模块输出端继续注入 -u

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import torch

from .hooks import MaliciousVectorInjector


@dataclass
class _AntibodyRecord:
    layer_key: int
    module_name: str
    module: torch.nn.Module
    param: torch.nn.Parameter
    theta0: torch.Tensor
    v_ref: torch.Tensor
    antibody_vector: Optional[torch.Tensor] = None
    singular_values: Optional[torch.Tensor] = None
    energy_ratio: float = 0.0
    injection_energy_share: float = 0.0
    raw_vector_norm: float = 0.0
    final_vector_norm: float = 0.0
    cos_to_v: float = 0.0
    delta_weight: Optional[torch.Tensor] = None
    calib_sum_input: Optional[torch.Tensor] = None
    calib_response_tokens: int = 0
    calib_batches: int = 0
    hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None
    calib_hook_handle: Optional[torch.utils.hooks.RemovableHandle] = None


class ImmuneVectorContinuationInjector:
    """
    从 ΔW_o_proj / ΔW_down_proj 蒸馏内生免疫向量, 并在 K 后继续注入。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        malicious_vectors: Dict[int, torch.Tensor],
        target_layers: List[int],
        alpha: float = 1.0,
        adaptive: bool = False,
        injection_mode: str = "res_only",
        antibody_source: str = "functional_mean",
        antibody_rank: int = 4,
        print_svd_energy_topk: int = 5,
        antibody_modules: Optional[List[str]] = None,
        calibration_micro_batches: int = 4,
        calibration_min_response_tokens: int = 2048,
        scale_mode: str = "match_v_preserve_ratio",
        strict: bool = True,
    ):
        assert injection_mode in ["all_token", "res_only"], "injection_mode must be 'all_token' or 'res_only'"
        if antibody_source not in ["svd_u", "functional_mean"]:
            raise ValueError("antibody_source must be 'svd_u' or 'functional_mean'")
        if scale_mode not in ["match_v_preserve_ratio", "match_v_equal_share", "raw"]:
            raise ValueError(
                "scale_mode must be 'match_v_preserve_ratio', 'match_v_equal_share', or 'raw'"
            )
        self.model = model
        self.malicious_vectors = malicious_vectors
        self.target_layers = target_layers
        self.alpha = alpha
        self.adaptive = adaptive
        self.injection_mode = injection_mode
        self.antibody_source = antibody_source
        self.antibody_rank = max(1, int(antibody_rank))
        self.print_svd_energy_topk = max(1, int(print_svd_energy_topk))
        self.antibody_modules = antibody_modules or ["o_proj", "down_proj"]
        self.calibration_micro_batches = max(1, int(calibration_micro_batches))
        self.calibration_min_response_tokens = max(1, int(calibration_min_response_tokens))
        self.scale_mode = scale_mode
        self.strict = strict

        allowed = {"o_proj", "down_proj"}
        invalid = [m for m in self.antibody_modules if m not in allowed]
        if invalid:
            raise ValueError(
                f"ImmuneVectorContinuationInjector only supports {sorted(allowed)} in v1; got {invalid}"
            )

        self.records: Dict[str, _AntibodyRecord] = {}
        self._record_order: List[str] = []
        self.finalized = False
        self.prepared = False
        self.calibration_ready = False
        self.last_metrics: Dict[str, float] = {}
        self.last_calibration_loss: float = 0.0
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._calibration_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._calibration_batches_seen = 0
        self._calibration_labels: Optional[torch.Tensor] = None

        self._capture_initial()

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _get_layer_module(self, layer_key: int) -> torch.nn.Module:
        layer_idx = layer_key - 1
        model = self.model
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "base_model"):
            inner = model.base_model
            if hasattr(inner, "model"):
                inner = inner.model
        else:
            inner = model

        candidates = []
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            candidates.append(inner.model.layers)
        if hasattr(inner, "layers"):
            candidates.append(inner.layers)
        if hasattr(inner, "transformer") and hasattr(inner.transformer, "h"):
            candidates.append(inner.transformer.h)

        for layers in candidates:
            if layer_idx < len(layers):
                return layers[layer_idx]

        model_root = self.model.module if hasattr(self.model, "module") else self.model
        fallback_paths = [
            f"model.layers.{layer_idx}",
            f"model.model.layers.{layer_idx}",
            f"base_model.model.model.layers.{layer_idx}",
        ]
        for path in fallback_paths:
            try:
                return model_root.get_submodule(path)
            except (AttributeError, Exception):
                continue

        raise ValueError(
            f"无法定位到 layer_key={layer_key} (0-indexed={layer_idx}) 的模型层。"
            f"请检查 target_layers 参数是否与模型结构匹配。"
        )

    @staticmethod
    def _randn_matrix(rows: int, cols: int, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(rows, cols, generator=generator, dtype=torch.float32).to(device=device)

    @staticmethod
    def _module_path(module_name: str) -> str:
        if module_name == "o_proj":
            return "self_attn.o_proj"
        if module_name == "down_proj":
            return "mlp.down_proj"
        raise ValueError(f"unsupported module_name={module_name}")

    @staticmethod
    def _get_full_param_tensor(param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if not hasattr(param, "ds_id"):
            tensor = param.data if param.data is not None else None
            return tensor.detach().float().cpu().clone() if torch.is_tensor(tensor) and tensor.numel() > 0 else None

        try:
            import deepspeed
            from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

            should_gather = getattr(param, "ds_status", None) == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters([param], enabled=should_gather):
                tensor = param.data
                if tensor is not None and tensor.numel() > 0:
                    return tensor.detach().float().cpu().clone()
        except Exception:
            pass
        return None

    @staticmethod
    def _align_direction(direction: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if torch.dot(direction, ref) < 0:
            return -direction
        return direction

    @staticmethod
    def _injection_hook(
        module: torch.nn.Module,
        input: tuple,
        output,
        v_m: torch.Tensor,
        alpha: float,
        adaptive: bool,
        injection_mode: str,
        ):
        return MaliciousVectorInjector._injection_hook(
            module=module,
            input=input,
            output=output,
            v_m=v_m,
            alpha=alpha,
            adaptive=adaptive,
            injection_mode=injection_mode,
        )

    def _calibration_forward_hook(self, module: torch.nn.Module, input: tuple, output, record: _AntibodyRecord):
        if not input or self._calibration_labels is None:
            return
        hidden = input[0]
        if not torch.is_tensor(hidden):
            return

        hidden_f32 = hidden.detach().float()
        if hidden_f32.ndim != 3:
            return

        response_mask = self._build_response_mask(self._calibration_labels, hidden_f32)
        if response_mask.numel() == 0 or not response_mask.any():
            return

        selected = hidden_f32[response_mask]
        if selected.numel() == 0:
            return

        summed = selected.sum(dim=0).detach().cpu()
        if record.calib_sum_input is None:
            record.calib_sum_input = summed
        else:
            record.calib_sum_input += summed
        record.calib_response_tokens += int(response_mask.sum().item())
        record.calib_batches += 1

    def _reset_calibration_state(self):
        for record in self.records.values():
            record.calib_sum_input = None
            record.calib_response_tokens = 0
            record.calib_batches = 0

    def _capture_calibration_hooks(self):
        if self._calibration_hooks:
            return
        for record in self.records.values():
            hook = record.module.register_forward_hook(
                partial(self._calibration_forward_hook, record=record)
            )
            record.calib_hook_handle = hook
            self._calibration_hooks.append(hook)

    def _remove_calibration_hooks(self):
        for hook in self._calibration_hooks:
            hook.remove()
        self._calibration_hooks.clear()
        for record in self.records.values():
            record.calib_hook_handle = None

    @staticmethod
    def _build_response_mask(labels: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if labels is None or not torch.is_tensor(labels):
            return torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
        if labels.shape[:2] != hidden_states.shape[:2]:
            return torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
        return (labels != -100).to(device=hidden_states.device)

    def _get_current_delta_weight(self, record: _AntibodyRecord) -> Optional[torch.Tensor]:
        current = self._get_full_param_tensor(record.param)
        if current is None:
            return None
        return (current - record.theta0).detach().float().cpu()

    def calibrate_on_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """
        用 K 后的校准 batch 估计功能方向。
        返回值表示是否已收集够校准数据，可进入 finalize()。
        """
        if self.finalized:
            return True
        if self.calibration_ready:
            return True

        if self._calibration_batches_seen == 0:
            self._reset_calibration_state()
        self._calibration_batches_seen += 1
        self._capture_calibration_hooks()
        self._calibration_labels = batch.get("labels", None)
        base_model = self._unwrap_model()
        was_training = base_model.training
        base_model.eval()
        with torch.no_grad():
            outputs = self.model(**batch, use_cache=False)
            if hasattr(outputs, "loss"):
                self.last_calibration_loss = float(outputs.loss.detach().float().item())
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                self.last_calibration_loss = float(outputs[0].detach().float().item())
        base_model.train(was_training)
        self._calibration_labels = None

        local_response_tokens = 0
        for record in self.records.values():
            local_response_tokens = max(local_response_tokens, int(record.calib_response_tokens))

        min_response_tokens = local_response_tokens
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                token_tensor = torch.tensor(
                    [local_response_tokens],
                    device=batch["input_ids"].device,
                    dtype=torch.long,
                )
                torch.distributed.all_reduce(token_tensor, op=torch.distributed.ReduceOp.MIN)
                min_response_tokens = int(token_tensor.item())
        except Exception:
            min_response_tokens = local_response_tokens

        if min_response_tokens >= self.calibration_min_response_tokens:
            self.calibration_ready = True
        if self._calibration_batches_seen >= self.calibration_micro_batches:
            self.calibration_ready = True

        self.last_metrics = {
            **self.last_metrics,
            "IDP/Antibody/Finalized": float(self.finalized),
            "IDP/Antibody/Calibration_Ready": float(self.calibration_ready),
            "IDP/Antibody/Calibration_Batches_Seen": float(self._calibration_batches_seen),
            "IDP/Antibody/Local_Response_Tokens": float(local_response_tokens),
            "IDP/Antibody/Min_Response_Tokens_Across_Ranks": float(min_response_tokens),
        }

        if self.calibration_ready:
            self._remove_calibration_hooks()
        return self.calibration_ready

    def _capture_initial(self):
        base = self._unwrap_model()
        total = 0
        captured = 0
        skipped = 0
        sample_names = []

        for layer_key in self.target_layers:
            layer_module = self._get_layer_module(layer_key)
            v_ref = self.malicious_vectors[layer_key].detach().float().cpu().clone()
            for module_name in self.antibody_modules:
                total += 1
                try:
                    submodule = layer_module.get_submodule(self._module_path(module_name))
                except Exception as e:
                    skipped += 1
                    raise RuntimeError(
                        f"Failed to locate continuation module {module_name} in layer {layer_key}: {e}"
                    ) from e

                param = getattr(submodule, "weight", None)
                if param is None:
                    skipped += 1
                    continue

                theta0 = self._get_full_param_tensor(param)
                if theta0 is None:
                    skipped += 1
                    continue

                key = f"{layer_key}:{module_name}"
                self.records[key] = _AntibodyRecord(
                    layer_key=layer_key,
                    module_name=module_name,
                    module=submodule,
                    param=param,
                    theta0=theta0,
                    v_ref=v_ref.clone(),
                )
                self._record_order.append(key)
                captured += 1
                if len(sample_names) < 10:
                    sample_names.append(key)

        self.last_metrics = {
            "IDP/Antibody_Total_Module_Count": float(total),
            "IDP/Antibody_Captured_Module_Count": float(captured),
            "IDP/Antibody_Skipped_Module_Count": float(skipped),
        }
        self.sample_names = sample_names

        if captured == 0:
            raise RuntimeError(
                "ImmuneVectorContinuationInjector captured 0 modules. Check target_layers and module paths."
            )

    def finalize(self) -> Dict[str, float]:
        if self.finalized:
            return self.last_metrics

        if self.antibody_source == "functional_mean" and not self.calibration_ready:
            raise RuntimeError(
                "ImmuneVectorContinuationInjector finalize() called before calibration is ready. "
                "Call calibrate_on_batch() on post-K calibration batches first."
            )

        failures = 0
        total_energy_ratio = 0.0
        total_cos = 0.0
        total_modules = 0
        log_entries = []

        for key, record in self.records.items():
            current = self._get_full_param_tensor(record.param)
            if current is None:
                failures += 1
                continue

            delta = current - record.theta0
            if delta.ndim != 2:
                failures += 1
                continue

            if self.antibody_source == "functional_mean":
                if (
                    record.calib_sum_input is None
                    or record.calib_response_tokens <= 0
                ):
                    failures += 1
                    continue
                mean_input = record.calib_sum_input.to(dtype=torch.float32) / float(record.calib_response_tokens)
                mean_delta_y = torch.matmul(delta.to(dtype=torch.float32), mean_input)
                functional_dir = mean_delta_y
                if functional_dir.norm() <= 1e-12:
                    failures += 1
                    continue
                # 让续注入方向在语义上与当前功能位移对齐：保留与 v 同向的输出压力。
                if torch.dot(functional_dir, record.v_ref) < 0:
                    functional_dir = -functional_dir
                raw_vector = functional_dir
                antibody_dir = raw_vector / raw_vector.norm().clamp(min=1e-12)
                cos_to_v = torch.nn.functional.cosine_similarity(
                    antibody_dir.unsqueeze(0), record.v_ref.unsqueeze(0), dim=-1
                ).item()
                energy_ratio = float(raw_vector.norm().item() / max(record.v_ref.norm().item(), 1e-12))
                singular_vals = None
                topk_sq = []
                topk_ratio = []
            else:
                max_rank_needed = max(self.antibody_rank, self.print_svd_energy_topk)
                q = min(max_rank_needed + 8, min(delta.shape))
                try:
                    U, S, _ = torch.svd_lowrank(delta, q=q, niter=2)
                except Exception as e:
                    failures += 1
                    if self.strict:
                        raise RuntimeError(f"randomized SVD failed for {key}: {e}") from e
                    continue

                rank = min(self.antibody_rank, U.shape[1], S.shape[0])
                if rank <= 0:
                    failures += 1
                    continue

                basis = U[:, :rank].detach().float()
                singular_vals = S[:rank].detach().float()
                total_energy = float(delta.float().pow(2).sum().item())
                selected_energy = float((singular_vals ** 2).sum().item())
                energy_ratio = selected_energy / max(total_energy, 1e-12)
                topk = min(self.print_svd_energy_topk, int(S.shape[0]))
                topk_sq = (S[:topk].float() ** 2).cpu().tolist()
                topk_ratio = (S[:topk].float() ** 2 / max(total_energy, 1e-12)).cpu().tolist()

                v_ref = record.v_ref.to(device=basis.device, dtype=torch.float32)
                aligned_cols = []
                for i in range(rank):
                    u_i = basis[:, i]
                    if torch.dot(-u_i, v_ref) < 0:
                        u_i = -u_i
                    aligned_cols.append(u_i)
                aligned_basis = torch.stack(aligned_cols, dim=1)

                weighted = (aligned_basis * singular_vals.unsqueeze(0)).sum(dim=1)
                if weighted.norm() <= 1e-12:
                    failures += 1
                    continue

                if torch.dot(-weighted, v_ref) < 0:
                    weighted = -weighted

                raw_vector = -weighted
                antibody_dir = raw_vector / raw_vector.norm().clamp(min=1e-12)
                cos_to_v = torch.nn.functional.cosine_similarity(
                    antibody_dir.unsqueeze(0), v_ref.unsqueeze(0), dim=-1
                ).item()

            record.antibody_vector = raw_vector.detach().cpu().contiguous()
            record.raw_vector_norm = float(record.antibody_vector.norm().item())
            if singular_vals is not None:
                record.singular_values = singular_vals.detach().cpu().contiguous()
            record.energy_ratio = energy_ratio
            record.cos_to_v = cos_to_v

            total_energy_ratio += energy_ratio
            total_cos += cos_to_v
            total_modules += 1

            if self.antibody_source == "functional_mean":
                log_entries.append(
                    (
                        key,
                        f"[IDP-continuation] layer={record.layer_key} module={record.module_name} "
                        f"functional_mean_norm={record.antibody_vector.norm().item():.4e} "
                        f"cos(u,v)={cos_to_v:.4f} calib_tokens={record.calib_response_tokens} "
                        f"calib_batches={record.calib_batches}",
                    )
                )
            else:
                top5_str = ", ".join(f"{x:.4e}" for x in topk_sq)
                top5_ratio_str = ", ".join(f"{x:.4f}" for x in topk_ratio)
                log_entries.append(
                    (
                        key,
                        f"[IDP-continuation] layer={record.layer_key} module={record.module_name} "
                        f"rank={rank} energy_ratio={energy_ratio:.4f} cos(-u,v)={cos_to_v:.4f} "
                        f"top{topk}_s2=[{top5_str}] top{topk}_ratio=[{top5_ratio_str}]",
                    )
                )

        self.finalized = True
        if failures > 0 and self.strict:
            raise RuntimeError(
                "ImmuneVectorContinuationInjector failed to build antibody vectors for "
                f"{failures}/{len(self.records)} modules. This run is strict to avoid partial continuation."
            )

        layer_success_counts: Dict[int, int] = {}
        layer_raw_norm_sums: Dict[int, float] = {}
        for record in self.records.values():
            if record.antibody_vector is not None:
                layer_success_counts[record.layer_key] = layer_success_counts.get(record.layer_key, 0) + 1
                layer_raw_norm_sums[record.layer_key] = (
                    layer_raw_norm_sums.get(record.layer_key, 0.0)
                    + float(record.antibody_vector.norm().item())
                )

        total_energy_share = 0.0
        for record in self.records.values():
            if record.antibody_vector is None:
                continue
            module_count = max(layer_success_counts.get(record.layer_key, 1), 1)
            raw_norm = record.antibody_vector.norm().clamp(min=1e-12)
            layer_raw_norm = max(layer_raw_norm_sums.get(record.layer_key, 0.0), 1e-12)
            if self.scale_mode == "match_v_equal_share":
                energy_share = 1.0 / float(module_count)
                target_norm = record.v_ref.norm().clamp(min=1e-12) * energy_share
                record.antibody_vector = record.antibody_vector / raw_norm * target_norm
            elif self.scale_mode == "match_v_preserve_ratio":
                energy_share = float(raw_norm.item()) / layer_raw_norm
                target_norm = record.v_ref.norm().clamp(min=1e-12) * energy_share
                record.antibody_vector = record.antibody_vector / raw_norm * target_norm
            else:
                energy_share = float(raw_norm.item()) / layer_raw_norm

            record.injection_energy_share = energy_share
            record.antibody_vector = record.antibody_vector.detach().cpu().contiguous()
            record.final_vector_norm = float(record.antibody_vector.norm().item())
            total_energy_share += energy_share

        log_lines = []
        for key, line in log_entries:
            record = self.records[key]
            if record.antibody_vector is not None:
                log_lines.append(
                    line
                    + f" module_energy_share={record.injection_energy_share:.4f}"
                    + f" raw_vector_norm={record.raw_vector_norm:.4e}"
                    + f" final_vector_norm={record.final_vector_norm:.4e}"
                )
        self.last_metrics = {
            **self.last_metrics,
            "IDP/Antibody_Protected_Module_Count": float(total_modules),
            "IDP/Antibody_Failed_Module_Count": float(failures),
            "IDP/Antibody_Mean_Energy_Ratio": float(total_energy_ratio / max(total_modules, 1)),
            "IDP/Antibody_Mean_Cos_V": float(total_cos / max(total_modules, 1)),
            "IDP/Antibody_Injected_Module_Count": float(sum(1 for r in self.records.values() if r.antibody_vector is not None)),
            "IDP/Antibody/Injected_Module_Count": float(sum(1 for r in self.records.values() if r.antibody_vector is not None)),
            "IDP/Antibody/rank_energy_ratio": float(total_energy_ratio / max(total_modules, 1)),
            "IDP/Antibody/Mean_Module_Energy_Share": float(total_energy_share / max(total_modules, 1)),
            "IDP/Antibody/Finalized": 1.0,
            "IDP/Antibody/Calibration_Ready": float(self.calibration_ready),
            "IDP/Antibody/Calibration_Batches_Seen": float(self._calibration_batches_seen),
            "IDP/Antibody/Scale_Mode_Match_V_Preserve_Ratio": float(self.scale_mode == "match_v_preserve_ratio"),
            "IDP/Antibody/Scale_Mode_Match_V_Equal_Share": float(self.scale_mode == "match_v_equal_share"),
            "IDP/Antibody/Scale_Mode_Raw": float(self.scale_mode == "raw"),
        }
        for module_name in self.antibody_modules:
            module_records = [
                r for r in self.records.values()
                if r.module_name == module_name and r.antibody_vector is not None
            ]
            if module_records:
                self.last_metrics[f"IDP/Antibody/{module_name}_cos_v"] = float(
                    sum(r.cos_to_v for r in module_records) / len(module_records)
                )
                self.last_metrics[f"IDP/Antibody/{module_name}_rank_energy_ratio"] = float(
                    sum(r.energy_ratio for r in module_records) / len(module_records)
                )
        self.last_log_lines = log_lines
        return self.last_metrics

    def attach(self):
        if not self.finalized:
            raise RuntimeError("ImmuneVectorContinuationInjector must be finalized before attach().")
        if self._hooks:
            return

        for record in self.records.values():
            if record.antibody_vector is None:
                continue
            hook = record.module.register_forward_hook(
                partial(
                    self._injection_hook,
                    v_m=record.antibody_vector,
                    alpha=self.alpha,
                    adaptive=self.adaptive,
                    injection_mode=self.injection_mode,
                )
            )
            record.hook_handle = hook
            self._hooks.append(hook)

    def detach(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        for record in self.records.values():
            record.hook_handle = None

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.last_metrics)
