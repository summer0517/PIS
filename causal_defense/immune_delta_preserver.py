# Copyright (c) 2026
# 因果干预动态防御 — 免疫位移保持器
#
# IDP (Immune Delta Preservation):
#   1. 前 K 步通过现有注入训练获得免疫参数位移 Δθ = θ_K - θ_0
#   2. K 步后撤掉注入, 只删除会擦除 Δθ 的梯度分量

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class _ImmuneParamRecord:
    name: str
    param: torch.nn.Parameter
    theta0: torch.Tensor
    full_shape: Tuple[int, ...]
    direction: Optional[torch.Tensor] = None
    delta_norm: float = 0.0
    basis: Optional[torch.Tensor] = None
    basis_kind: str = "flat"
    basis_rank: int = 0


class ImmuneDeltaPreserver:
    """
    保护前 K 步注入训练写入参数的免疫位移。

    对每个受保护参数块 p_j:
        Δ_j = p_j^K - p_j^0
        u_j = Δ_j / ||Δ_j||

    K 步后对梯度做单边投影:
        s_j = <g_j, u_j>
        if s_j > 0: g_j <- g_j - s_j * u_j

    这里的方向来自模型实际学到的参数位移, 不是外部激活向量 v_m。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: Optional[List[int]] = None,
        param_scope: str = "target_core",
        min_delta_norm: float = 1e-12,
        exact_distributed_projection: bool = True,
        include_input_embeddings: bool = False,
        projection_mode: str = "svd_subspace",
        svd_rank: int = 8,
        svd_oversample: int = 4,
        projection_strength: float = 1.0,
        svd_strict: bool = True,
    ):
        if param_scope not in ("target_core", "target_all_linear"):
            raise ValueError("param_scope must be 'target_core' or 'target_all_linear'")
        if projection_mode not in ("flat", "svd_subspace"):
            raise ValueError("projection_mode must be 'flat' or 'svd_subspace'")

        self.model = model
        self.target_layers = target_layers or []
        self.param_scope = param_scope
        self.min_delta_norm = min_delta_norm
        self.exact_distributed_projection = exact_distributed_projection
        self.include_input_embeddings = include_input_embeddings
        self.projection_mode = projection_mode
        self.svd_rank = max(1, int(svd_rank))
        self.svd_oversample = max(0, int(svd_oversample))
        self.projection_strength = float(projection_strength)
        self.svd_strict = svd_strict

        self.records: Dict[str, _ImmuneParamRecord] = {}
        self.finalized = False
        self.last_metrics: Dict[str, float] = {}
        self.selection_stats: Dict[str, float] = {}
        self.matched_sample_names: List[str] = []

        self._capture_initial()

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _layer_markers(self) -> List[str]:
        if not self.target_layers:
            return []
        markers = []
        for layer_key in self.target_layers:
            idx = layer_key - 1
            markers.extend([
                f"model.layers.{idx}.",
                f"model.model.layers.{idx}.",
                f"base_model.model.model.layers.{idx}.",
                f"layers.{idx}.",
            ])
        return markers

    @staticmethod
    def _is_transformer_layer_param(name: str) -> bool:
        return (
            ".layers." in name
            or name.startswith("layers.")
            or "model.layers." in name
            or "model.model.layers." in name
            or "base_model.model.model.layers." in name
            or ".blocks." in name
            or name.startswith("blocks.")
            or "model.blocks." in name
            or ".h." in name
            or "transformer.h." in name
            or ".decoder.layers." in name
            or ".encoder.layers." in name
        )

    @staticmethod
    def _is_input_embedding_param(name: str) -> bool:
        return (
            name.endswith("embed_tokens.weight")
            or name.endswith("wte.weight")
            or name.endswith("word_embeddings.weight")
            or name.endswith("tok_embeddings.weight")
        )

    @staticmethod
    def _is_output_head_or_norm_param(name: str) -> bool:
        return (
            name.endswith("lm_head.weight")
            or name.endswith("output.weight")
            or ".norm" in name
            or ".ln_" in name
            or "layernorm" in name.lower()
            or "rms_norm" in name.lower()
        )

    @staticmethod
    def _is_core_projection_param(name: str) -> bool:
        lower = name.lower()
        core_keywords = [
            "o_proj",
            "out_proj",
            "self_attn.o",
            "attention.o",
            ".wo.",
            "wo.weight",
            "down_proj",
            "mlp.down",
            "feed_forward.down",
            ".w2.",
            "w2.weight",
            "dense_4h_to_h",
            "c_proj",
        ]
        return any(keyword in lower for keyword in core_keywords)

    @staticmethod
    def _is_output_lock_param(name: str) -> bool:
        lower = name.lower()
        return any(
            keyword in lower
            for keyword in [
                "o_proj",
                "out_proj",
                "self_attn.o",
                "attention.o",
                ".wo.",
                "wo.weight",
                "down_proj",
                "mlp.down",
                "feed_forward.down",
                ".w2.",
                "w2.weight",
                "dense_4h_to_h",
                "c_proj",
            ]
        )

    @staticmethod
    def _is_input_lock_param(name: str) -> bool:
        lower = name.lower()
        return any(
            keyword in lower
            for keyword in [
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "query",
                "key",
                "value",
                "dense_h_to_4h",
                "w1.weight",
                "w3.weight",
            ]
        )

    def _is_target_param(self, name: str, param: torch.nn.Parameter) -> bool:
        if not param.requires_grad:
            return False
        if not name.endswith(".weight"):
            return False

        if self.target_layers:
            if not any(marker in name for marker in self._layer_markers()):
                return False

        if self.include_input_embeddings and self._is_input_embedding_param(name):
            return True
        if self._is_input_embedding_param(name) or self._is_output_head_or_norm_param(name):
            return False

        if self.param_scope == "target_all_linear":
            return True

        return self._is_core_projection_param(name)

    def _local_param_tensor(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        try:
            from deepspeed.utils import safe_get_local_fp32_param

            local_fp32 = safe_get_local_fp32_param(param)
            if torch.is_tensor(local_fp32) and local_fp32.numel() > 0:
                return local_fp32.detach()
        except Exception:
            pass

        # ZeRO-3 下 param.data 可能是空 placeholder; 这时用本 rank 的 partition。
        ds_tensor = getattr(param, "ds_tensor", None)
        if torch.is_tensor(ds_tensor) and ds_tensor.numel() > 0:
            return ds_tensor.detach()
        if param.data is not None and param.data.numel() > 0:
            return param.data.detach()
        return None

    def _full_param_tensor(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
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

    def _lookup_zero_grad_partition(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        optimizer = getattr(self.model, "optimizer", None)
        if optimizer is None:
            return None

        candidate_keys = [param, id(param), getattr(param, "ds_id", None)]
        for attr_name in dir(optimizer):
            lower = attr_name.lower()
            if "grad" not in lower or "partition" not in lower:
                continue
            try:
                container = getattr(optimizer, attr_name)
            except Exception:
                continue
            if not isinstance(container, dict):
                continue
            for key in candidate_keys:
                if key is None or key not in container:
                    continue
                grad = container[key]
                if torch.is_tensor(grad) and grad.numel() > 0:
                    return grad
        return None

    def _local_grad_tensor(self, param: torch.nn.Parameter) -> Tuple[Optional[torch.Tensor], str]:
        if param.grad is not None and param.grad.numel() > 0:
            return param.grad, "param.grad"
        ds_tensor = getattr(param, "ds_tensor", None)
        ds_grad = getattr(ds_tensor, "grad", None)
        if torch.is_tensor(ds_grad) and ds_grad.numel() > 0:
            return ds_grad, "ds_tensor.grad"

        try:
            from deepspeed.utils import safe_get_local_grad

            local_grad = safe_get_local_grad(param)
            if torch.is_tensor(local_grad) and local_grad.numel() > 0:
                return local_grad, "safe_get_local_grad"
        except Exception:
            pass

        zero_grad = self._lookup_zero_grad_partition(param)
        if zero_grad is not None:
            return zero_grad, "zero_grad_partition"

        return None, "none"

    @staticmethod
    def _distributed_info() -> Tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _partition_start(self, record: _ImmuneParamRecord, local_numel: int) -> int:
        total_numel = int(torch.prod(torch.tensor(record.full_shape)).item())
        rank, world_size = self._distributed_info()
        if world_size <= 1:
            return 0
        partition_size = (total_numel + world_size - 1) // world_size
        return min(rank * partition_size, total_numel)

    @staticmethod
    def _active_partition_numel(record: _ImmuneParamRecord, start: int, local_numel: int) -> int:
        total_numel = int(torch.prod(torch.tensor(record.full_shape)).item())
        return max(0, min(local_numel, total_numel - start))

    @staticmethod
    def _set_local_grad_tensor(param: torch.nn.Parameter, grad: torch.Tensor) -> bool:
        try:
            from deepspeed.utils import safe_set_local_grad

            safe_set_local_grad(param, grad)
            return True
        except Exception:
            return False

    @staticmethod
    def _align_direction_to_grad(
        direction: torch.Tensor,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        direction_flat = direction.reshape(-1)
        grad_flat = grad.reshape(-1)
        if direction_flat.numel() == grad_flat.numel():
            return direction_flat, grad_flat

        # Some ZeRO-3 partitions carry optimizer padding at the end. Only the
        # real parameter shard has a learned immune direction.
        if grad_flat.numel() > direction_flat.numel():
            return direction_flat, grad_flat.narrow(0, 0, direction_flat.numel())
        if direction_flat.numel() > grad_flat.numel():
            return direction_flat.narrow(0, 0, grad_flat.numel()), grad_flat
        return None, None

    def _all_reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        if (
            self.exact_distributed_projection
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            orig_device = value.device
        # 只有当张量不在当前 CUDA 设备上时，才进行设备转移
        if orig_device.type == 'cpu' or orig_device.index != torch.cuda.current_device():
            target_device = torch.cuda.current_device()
            value = value.to(target_device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            value = value.to(orig_device) # 归约完成后，送回原设备
        else:
            # 如果本身就在正确的 GPU 上，直接归约
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        return value

    @staticmethod
    def _randn_matrix(rows: int, cols: int, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return torch.randn(rows, cols, generator=generator, dtype=torch.float32).to(device=device)

    def _basis_kind_for_name(self, name: str) -> str:
        if self._is_output_lock_param(name):
            return "output"
        if self._is_input_lock_param(name):
            return "input"
        return "output"

    def _build_subspace_basis(
        self,
        record: _ImmuneParamRecord,
        delta: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], str, int]:
        if len(record.full_shape) != 2:
            return None, "flat", 0

        rows, cols = record.full_shape
        rank = min(self.svd_rank, rows, cols)
        sketch_rank = min(rank + self.svd_oversample, rows, cols)
        basis_kind = self._basis_kind_for_name(record.name)
        delta_matrix = delta.to(dtype=torch.float32)
        seed = int(hashlib.sha1(record.name.encode("utf-8")).hexdigest()[:8], 16)

        if basis_kind == "output":
            omega = self._randn_matrix(cols, sketch_rank, seed, delta_matrix.device)
            sketch = delta_matrix @ omega
        else:
            omega = self._randn_matrix(rows, sketch_rank, seed, delta_matrix.device)
            sketch = delta_matrix.t() @ omega
        if float(torch.sum(sketch * sketch).item()) <= 0.0:
            return None, basis_kind, 0

        q, _ = torch.linalg.qr(sketch, mode="reduced")
        basis = q[:, :rank].detach().float().cpu().contiguous()
        return basis, basis_kind, rank

    def _apply_subspace_projection(
        self,
        record: _ImmuneParamRecord,
        grad: torch.Tensor,
    ) -> Tuple[bool, float, float, bool]:
        if record.basis is None or len(record.full_shape) != 2:
            return False, 0.0, 0.0, False

        rows, cols = record.full_shape
        start = self._partition_start(record, grad.numel())
        active_numel = self._active_partition_numel(record, start, grad.numel())
        if active_numel <= 0:
            return False, 0.0, 0.0, False

        grad_flat = grad.data.reshape(-1)
        active_grad = grad_flat.narrow(0, 0, active_numel)
        basis = record.basis.to(device=grad.device, dtype=torch.float32)
        rank = basis.shape[1]
        removed_sq_local = torch.zeros((), device=grad.device, dtype=torch.float32)
        grad_sq_local = torch.sum(active_grad.detach().float() * active_grad.detach().float())

        if record.basis_kind == "output":
            coeff = torch.zeros(rank, cols, device=grad.device, dtype=torch.float32)
            local_pos = 0
            global_pos = start
            while local_pos < active_numel:
                row = global_pos // cols
                col = global_pos % cols
                take = min(active_numel - local_pos, cols - col)
                chunk = active_grad.narrow(0, local_pos, take).detach().float()
                coeff[:, col:col + take].add_(basis[row].unsqueeze(1) * chunk.unsqueeze(0))
                local_pos += take
                global_pos += take
            coeff = self._all_reduce_sum(coeff)

            local_pos = 0
            global_pos = start
            while local_pos < active_numel:
                row = global_pos // cols
                col = global_pos % cols
                take = min(active_numel - local_pos, cols - col)
                correction = basis[row].unsqueeze(0).matmul(coeff[:, col:col + take]).squeeze(0)
                correction = correction * self.projection_strength
                active_grad.narrow(0, local_pos, take).sub_(correction.to(dtype=grad.dtype))
                removed_sq_local.add_(torch.sum(correction * correction))
                local_pos += take
                global_pos += take
        else:
            coeff = torch.zeros(rows, rank, device=grad.device, dtype=torch.float32)
            local_pos = 0
            global_pos = start
            while local_pos < active_numel:
                row = global_pos // cols
                col = global_pos % cols
                take = min(active_numel - local_pos, cols - col)
                chunk = active_grad.narrow(0, local_pos, take).detach().float()
                coeff[row].add_(chunk.unsqueeze(0).matmul(basis[col:col + take]).squeeze(0))
                local_pos += take
                global_pos += take
            coeff = self._all_reduce_sum(coeff)

            local_pos = 0
            global_pos = start
            while local_pos < active_numel:
                row = global_pos // cols
                col = global_pos % cols
                take = min(active_numel - local_pos, cols - col)
                correction = coeff[row].unsqueeze(0).matmul(basis[col:col + take].t()).squeeze(0)
                correction = correction * self.projection_strength
                active_grad.narrow(0, local_pos, take).sub_(correction.to(dtype=grad.dtype))
                removed_sq_local.add_(torch.sum(correction * correction))
                local_pos += take
                global_pos += take

        stats = torch.stack([removed_sq_local, grad_sq_local])
        stats = self._all_reduce_sum(stats)
        set_failed = not self._set_local_grad_tensor(record.param, grad)
        return True, float(stats[0].item()), float(stats[1].item()), set_failed

    def _capture_initial(self):
        base = self._unwrap_model()
        total_params = 0
        trainable_params = 0
        matched_params = 0
        captured_params = 0
        trainable_samples = []
        matched_samples = []
        for name, param in base.named_parameters():
            total_params += 1
            if param.requires_grad:
                trainable_params += 1
                if len(trainable_samples) < 20:
                    trainable_samples.append(name)
            if not self._is_target_param(name, param):
                continue
            matched_params += 1
            if len(matched_samples) < 20:
                matched_samples.append(name)
            tensor = self._full_param_tensor(param) if self.projection_mode == "svd_subspace" else self._local_param_tensor(param)
            if tensor is None:
                continue
            self.records[name] = _ImmuneParamRecord(
                name=name,
                param=param,
                theta0=tensor.detach().float().cpu().clone(),
                full_shape=tuple(int(x) for x in tensor.shape),
            )
            captured_params += 1

        self.selection_stats = {
            "IDP/Total_Named_Params": float(total_params),
            "IDP/Trainable_Named_Params": float(trainable_params),
            "IDP/Matched_Param_Count": float(matched_params),
            "IDP/Captured_Param_Count": float(captured_params),
        }
        self.matched_sample_names = matched_samples

        if matched_params == 0:
            raise RuntimeError(
                "ImmuneDeltaPreserver matched 0 parameters. The current model parameter names do not match "
                f"param_scope={self.param_scope}. Try --immune_param_scope target_all_linear first. "
                f"Sample trainable parameter names: {trainable_samples}"
            )
        if captured_params == 0:
            raise RuntimeError(
                "ImmuneDeltaPreserver captured 0 parameter tensors. Under ZeRO-3 this usually means the local "
                "partition tensor was not available; check DeepSpeed parameter layout before running IDP."
            )

    def finalize(self) -> Dict[str, float]:
        """在免疫边界 K 到达后调用, 固化 θ_K - θ_0 方向。"""
        if self.finalized:
            return self.last_metrics

        pending = []
        local_sqs = []
        skipped = 0

        for record in self.records.values():
            tensor = self._full_param_tensor(record.param) if self.projection_mode == "svd_subspace" else self._local_param_tensor(record.param)
            if tensor is None:
                skipped += 1
                continue

            delta = tensor.detach().float().cpu() - record.theta0
            local_sq = torch.tensor(float((delta * delta).sum().item()), dtype=torch.float32)
            pending.append((record, delta, tensor.device))
            local_sqs.append(local_sq.to(device=tensor.device))

        if pending:
            stacked_sqs = torch.stack(local_sqs)
            global_sqs = self._all_reduce_sum(stacked_sqs)
        else:
            global_sqs = torch.empty(0)

        kept = 0
        total_norm_sq = 0.0
        for idx, (record, delta, tensor_device) in enumerate(pending):
            global_sq = float(global_sqs[idx].item())
            global_norm = global_sq ** 0.5

            if global_norm <= self.min_delta_norm:
                record.direction = None
                record.delta_norm = global_norm
                skipped += 1
                continue

            record.direction = delta / global_norm
            record.delta_norm = global_norm
            if self.projection_mode == "svd_subspace":
                basis, basis_kind, basis_rank = self._build_subspace_basis(record, delta)
                record.basis = basis
                record.basis_kind = basis_kind
                record.basis_rank = basis_rank
            total_norm_sq += global_norm * global_norm
            kept += 1

        self.finalized = True
        svd_count = sum(1 for record in self.records.values() if record.basis is not None)
        output_basis_count = sum(1 for record in self.records.values() if record.basis_kind == "output" and record.basis is not None)
        input_basis_count = sum(1 for record in self.records.values() if record.basis_kind == "input" and record.basis is not None)
        svd_missing_count = kept - svd_count if self.projection_mode == "svd_subspace" else 0
        if self.projection_mode == "svd_subspace" and self.svd_strict and svd_missing_count > 0:
            raise RuntimeError(
                "ImmuneDeltaPreserver svd_subspace failed to build SVD bases for "
                f"{svd_missing_count}/{kept} protected tensors. This run is strict to avoid silently "
                "falling back to flat projection. Use --immune_allow_svd_partial only for diagnostics."
            )
        self.last_metrics = {
            **self.selection_stats,
            "IDP/Protected_Tensor_Count": float(kept),
            "IDP/Skipped_Tensor_Count": float(skipped),
            "IDP/Immune_Delta_Total_Norm": float(total_norm_sq ** 0.5),
            "IDP/SVD_Basis_Tensor_Count": float(svd_count),
            "IDP/SVD_Missing_Basis_Count": float(svd_missing_count),
            "IDP/SVD_Output_Basis_Count": float(output_basis_count),
            "IDP/SVD_Input_Basis_Count": float(input_basis_count),
            "IDP/SVD_Rank": float(self.svd_rank),
            "IDP/Projection_Strength": float(self.projection_strength),
        }
        return self.last_metrics

    def apply_after_backward(self) -> Dict[str, float]:
        """在 model.backward(loss) 之后、model.step() 之前调用。"""
        if not self.finalized:
            return {}

        active_records = []
        local_dots = []
        local_grad_sqs = []
        skipped = 0
        no_grad = 0
        shape_mismatch = 0
        subspace_applied = 0
        subspace_missing_basis = 0
        grad_source_counts: Dict[str, int] = {}
        protected = 0
        destructive = 0
        removed_sq = 0.0
        grad_sq = 0.0
        cos_sum = 0.0
        cos_count = 0
        set_grad_failed = 0

        for record in self.records.values():
            if record.direction is None:
                skipped += 1
                continue

            grad, grad_source = self._local_grad_tensor(record.param)
            if grad is None:
                skipped += 1
                no_grad += 1
                continue
            grad_source_counts[grad_source] = grad_source_counts.get(grad_source, 0) + 1

            if self.projection_mode == "svd_subspace":
                if record.basis is None:
                    skipped += 1
                    subspace_missing_basis += 1
                    continue
                applied, sub_removed_sq, sub_grad_sq, set_failed = self._apply_subspace_projection(record, grad)
                if not applied:
                    skipped += 1
                    shape_mismatch += 1
                    continue
                protected += 1
                subspace_applied += 1
                destructive += 1 if sub_removed_sq > 0 else 0
                removed_sq += sub_removed_sq
                grad_sq += sub_grad_sq
                if set_failed:
                    set_grad_failed += 1
                continue

            direction = record.direction.to(device=grad.device, dtype=torch.float32)
            direction_flat, grad_flat = self._align_direction_to_grad(direction, grad)
            if direction_flat is None or grad_flat is None:
                skipped += 1
                shape_mismatch += 1
                continue

            grad_float = grad_flat.detach().float()
            local_dots.append(torch.sum(grad_float * direction_flat))
            local_grad_sqs.append(torch.sum(grad_float * grad_float))
            active_records.append((record, grad, direction_flat.numel()))

        if active_records:
            stacked_dots = torch.stack(local_dots)
            stacked_sqs = torch.stack(local_grad_sqs)
            global_dots = self._all_reduce_sum(stacked_dots)
            global_sqs = self._all_reduce_sum(stacked_sqs)
        else:
            global_dots = torch.empty(0)
            global_sqs = torch.empty(0)

        for idx, (record, grad, active_numel) in enumerate(active_records):
            dot_value = float(global_dots[idx].item())
            global_grad_sq = float(global_sqs[idx].item())
            grad_norm = float(global_grad_sq ** 0.5)
            grad_sq += global_grad_sq

            if grad_norm > 0:
                cos_sum += dot_value / (grad_norm + 1e-12)
                cos_count += 1

            if dot_value > 0:
                direction_flat = record.direction.to(device=grad.device, dtype=torch.float32).reshape(-1)
                direction_flat = direction_flat.narrow(0, 0, active_numel).to(dtype=grad.dtype)
                grad_flat = grad.data.reshape(-1)
                grad_flat.narrow(0, 0, active_numel).sub_(direction_flat * dot_value)
                if not self._set_local_grad_tensor(record.param, grad):
                    set_grad_failed += 1
                destructive += 1
                removed_sq += dot_value * dot_value

            protected += 1

        grad_source_metrics = {
            f"IDP/Grad_Source_{source}": float(count)
            for source, count in grad_source_counts.items()
        }
        self.last_metrics = {
            "IDP/Applied_Tensor_Count": float(protected),
            "IDP/Skipped_Tensor_Count": float(skipped),
            "IDP/No_Grad_Tensor_Count": float(no_grad),
            "IDP/Shape_Mismatch_Tensor_Count": float(shape_mismatch),
            "IDP/Set_Grad_Failed_Count": float(set_grad_failed),
            "IDP/Subspace_Applied_Tensor_Count": float(subspace_applied),
            "IDP/Subspace_Missing_Basis_Count": float(subspace_missing_basis),
            "IDP/Destructive_Tensor_Count": float(destructive),
            "IDP/Removed_Projection_Norm": float(removed_sq ** 0.5),
            "IDP/Grad_Total_Norm": float(grad_sq ** 0.5),
            "IDP/Destructive_Tensor_Ratio": float(destructive / max(protected, 1)),
            "IDP/Mean_Grad_Immune_Cos": float(cos_sum / max(cos_count, 1)),
            "IDP/Projection_Strength": float(self.projection_strength),
            **grad_source_metrics,
        }
        return self.last_metrics

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.last_metrics)
