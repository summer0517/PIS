# Copyright (c) 2026
# 注入梯度探针动态防御

import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class InjectionGradientProbeDefense:
    """
    通过 micro-perturbation 的梯度差近似 v_m 方向曲率，判定样本风险。

    判定规则:
      1. 计算自然梯度 g_nat = dL(h)/dh。
      2. 对目标层输出注入极小扰动 h' = h + epsilon * v_m，计算 g_inj = dL(h')/dh'。
      3. 用 (g_inj - g_nat) / epsilon 近似 H v_m，并计算它与 v_m 的 cos/proj。
      4. 对 batch 内每个 sample 独立判定；曲率 proxy 的 signed cos 与 signed proj
         同时超过静态阈值时，判定该 sample 为恶意。
      5. 最终 loss 只对正常 sample 的 natural loss 求均值；若全拦截，通知训练循环
         跳过 backward/step，避免无意义的零 loss 反传。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        v_m: torch.Tensor,
        target_layer_key: int,
        alpha: float = 1.0,
        diff_epsilon: Optional[float] = None,
        adaptive: bool = False,
        injection_mode: str = "res_only",
        cos_threshold: float = 0.0,
        proj_threshold: float = 0.0,
        eps: float = 1e-12,
    ):
        assert injection_mode in ["all_token", "res_only"], "injection_mode must be 'all_token' or 'res_only'"
        self.model = model
        self.v_m = v_m
        self.target_layer_key = target_layer_key
        self.alpha = alpha
        self.diff_epsilon = diff_epsilon
        self.adaptive = adaptive
        self.injection_mode = injection_mode
        self.cos_threshold = cos_threshold
        self.proj_threshold = proj_threshold
        self.eps = eps

        self.total_samples = 0
        self.blocked_samples = 0
        self.total_tokens = 0
        self.blocked_tokens = 0

    def compute_defense_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = int(batch["input_ids"].shape[0])

        if self.adaptive:
            raise ValueError(
                "InjectionGradientProbeDefense curvature mode requires adaptive=False; "
                "use fixed perturbation alpha and diff_epsilon for finite differences."
            )

        # ==========================================
        # 1. 微扰动海森探测 (Second-Order Hessian Probe)
        # ==========================================
        # 步骤 A：获取自然梯度 (g_nat)
        nat_loss_probe, nat_grads = self._forward_and_grad_batched(batch, inject=False)

        # 步骤 B：获取微扰梯度 (g_inj) -> 必须确保 self.alpha 是一个极小值 (如 0.01)
        inj_loss_probe, inj_grads = self._forward_and_grad_batched(batch, inject=True)

        # 步骤 C：计算梯度差 (\Delta g)，这代表了曲率
        perturb_alpha = float(self.alpha)
        diff_epsilon = perturb_alpha if self.diff_epsilon is None else float(self.diff_epsilon)
        denom = diff_epsilon if abs(diff_epsilon) > self.eps else (self.eps if diff_epsilon >= 0 else -self.eps)
        curvature_grads = (inj_grads - nat_grads) / denom
        del nat_grads, inj_grads

        # 获取当前 Batch 全局的总有效 Token 数 (用于精确还原单条 forward 的梯度尺度)
        labels = batch.get("labels")
        if labels is not None:
            batch_total_valid_tokens = int((labels != -100).sum().item())
        else:
            batch_total_valid_tokens = batch_size * int(batch["input_ids"].shape[1])
        batch_total_valid_tokens = max(batch_total_valid_tokens, 1)

        clean_indices = []
        curvature_metrics_list = []
        num_malicious = 0
        total_valid_tokens = 0
        blocked_valid_tokens = 0

        for i in range(batch_size):
            labels_i = labels[i:i + 1] if labels is not None else None
            valid_tokens = int((labels_i != -100).sum().item()) if labels_i is not None else int(batch["input_ids"].shape[1])
            total_valid_tokens += valid_tokens

            # HF causal LM loss 通常按 batch 内全部有效 token 求均值。
            # 这里把每个 sample 的梯度尺度还原到 sample-level token mean，便于阈值更稳定。
            scale_factor = batch_total_valid_tokens / max(valid_tokens, 1)
            grad_mean = self._masked_mean_tokens(curvature_grads[i:i + 1] * scale_factor, labels_i)
            if grad_mean.dim() > 1:
                grad_mean = grad_mean.squeeze(0)

            curvature_metrics = self._compute_gradient_metrics(grad_mean)
            sample_is_malicious = (
                curvature_metrics["cos"] > self.cos_threshold
                and curvature_metrics["proj"] > self.proj_threshold
            )

            curvature_metrics_list.append(curvature_metrics)

            # ==========================================
            # 2. 过滤正常样本，一次性完成自然前向 (保持不变)
            # ==========================================

            if sample_is_malicious:
                num_malicious += 1
                blocked_valid_tokens += valid_tokens
            else:
                clean_indices.append(i)

        self.total_samples += batch_size
        self.blocked_samples += num_malicious
        self.total_tokens += total_valid_tokens
        self.blocked_tokens += blocked_valid_tokens

        skip_backward = len(clean_indices) == 0
        if not skip_backward:
            clean_batch = {k: v[clean_indices] for k, v in batch.items()}
            outputs_nat = self.model(**clean_batch, use_cache=False)
            loss = self._extract_loss(outputs_nat)
            natural_loss_val = loss.detach().item()
        else:
            # ZeRO-3/DDP 中所有 rank 必须执行一致的 backward/step 通信路径。
            # 因此即使本 rank 的样本全被拦截，也要构造连接真实模型图的 0 loss。
            dummy_batch = {k: v[0:1] for k, v in batch.items()}
            outputs_nat = self.model(**dummy_batch, use_cache=False)
            last_nat_loss = self._extract_loss(outputs_nat)
            loss = last_nat_loss * 0.0
            natural_loss_val = 0.0

        # ==========================================
        # 3. 统计状态并返回 (保持不变)
        # ==========================================
        def mean(values):
            return sum(values) / max(len(values), 1)

        def mean_metric(metrics_list, key):
            return mean([m[key] for m in metrics_list])

        is_malicious = num_malicious > 0

        info = {
            "natural_loss": natural_loss_val,
            "injected_probe_loss": inj_loss_probe.detach().item(),
            "delta_loss": mean_metric(curvature_metrics_list, "proj"),
            "is_malicious": is_malicious,
            "skip_backward": skip_backward,
            "blocked_ratio_samples": self.blocked_samples / max(self.total_samples, 1),
            "blocked_ratio_tokens": self.blocked_tokens / max(self.total_tokens, 1),
            "num_malicious_in_batch": num_malicious,
            "probe_cos_inj": mean_metric(curvature_metrics_list, "cos"),
            "probe_cos_abs_inj": mean_metric(curvature_metrics_list, "cos_abs"),
            "probe_proj_inj": mean_metric(curvature_metrics_list, "proj"),
            "probe_proj_abs_inj": mean_metric(curvature_metrics_list, "proj_abs"),
            "probe_grad_norm_inj": mean_metric(curvature_metrics_list, "grad_norm"),
            "probe_curvature_cos_vm": mean_metric(curvature_metrics_list, "cos"),
            "probe_curvature_proj_vm": mean_metric(curvature_metrics_list, "proj"),
            "probe_curvature_norm": mean_metric(curvature_metrics_list, "grad_norm"),
            "probe_perturb_alpha": perturb_alpha,
            "probe_diff_epsilon": denom,
            "probe_epsilon": denom,
            "probe_cos_threshold": self.cos_threshold,
            "probe_proj_threshold": self.proj_threshold,
        }
        del curvature_grads, nat_loss_probe, inj_loss_probe
        return loss, info

    def _forward_and_grad_batched(
        self,
        batch: Dict[str, torch.Tensor],
        inject: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = self._get_layer_module(self.target_layer_key)
        captured: Dict[str, torch.Tensor] = {}

        def hook(module, inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if inject:
                hidden_states = self._inject_hidden_states(module, hidden_states, batch.get("labels"))
            captured["hidden_states"] = hidden_states
            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            return hidden_states

        handle = layer.register_forward_hook(hook)
        try:
            outputs = self.model(**batch, use_cache=False)
        finally:
            handle.remove()

        loss = self._extract_loss(outputs)
        hidden_states = captured.get("hidden_states")
        if hidden_states is None:
            raise RuntimeError(
                f"InjectionGradientProbeDefense failed to capture layer {self.target_layer_key} output."
            )

        grad = torch.autograd.grad(
            loss,
            hidden_states,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        return loss, grad.detach()

    def _inject_hidden_states(
        self,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        v_m = self.v_m.to(device=hidden_states.device, dtype=hidden_states.dtype)
        vec = v_m.unsqueeze(0).unsqueeze(0)

        if self.injection_mode == "all_token":
            mask = None
            masked_hidden = hidden_states
        else:
            mask = None
            if labels is not None and labels.shape[:2] == hidden_states.shape[:2]:
                mask = (labels != -100).to(hidden_states.device)
            if mask is None:
                mask = self._find_response_mask(hidden_states)
            if mask is not None:
                module._cached_gradient_probe_mask = mask
            else:
                mask = getattr(module, "_cached_gradient_probe_mask", None)
                if mask is None:
                    raise RuntimeError("[InjectionGradientProbeDefense] failed to find labels for response-only injection.")
            masked_hidden = hidden_states[mask]

        if self.adaptive:
            h_norm = masked_hidden.norm(dim=-1, keepdim=True).mean() if masked_hidden.numel() > 0 else 1.0
            scale = self.alpha * (h_norm / v_m.norm().clamp(min=1e-8))
        else:
            scale = self.alpha

        if mask is None:
            return hidden_states + scale * vec

        mask_f = mask.unsqueeze(-1).to(device=hidden_states.device, dtype=hidden_states.dtype)
        return hidden_states + scale * vec * mask_f

    def _compute_gradient_metrics(self, grad: torch.Tensor) -> Dict[str, float]:
        """
        与 GradientAnalyzer._compute_gradient_metrics() 保持一致:
          proj = dot(g, v_m) / ||v_m||
          cos_vm = cosine_similarity(g, v_m)
        """
        g = grad.detach().float()
        v_flat = self.v_m.flatten().float().to(g.device)
        v_norm = v_flat.norm().item()
        grad_norm = g.norm().item()

        if v_norm <= 1e-10:
            proj = 0.0
            cos_vm = 0.0
        else:
            proj = torch.dot(g, v_flat).item() / v_norm
            cos_vm = F.cosine_similarity(g.unsqueeze(0), v_flat.unsqueeze(0)).item()

        return {
            "grad_norm": grad_norm,
            "proj": proj,
            "proj_abs": abs(proj),
            "cos": cos_vm,
            "cos_abs": abs(cos_vm),
        }

    def _ratio(self, numerator: float, denominator: float) -> float:
        return (float(numerator) + self.eps) / (float(denominator) + self.eps)

    def _signed_amplification(self, injected_value: float, natural_value: float) -> float:
        return float(injected_value) / (abs(float(natural_value)) + self.eps)

    @staticmethod
    def _extract_loss(outputs):
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        return outputs[0]

    @staticmethod
    def _masked_mean_tokens(tensor: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        tensor = tensor.float()
        if tensor.dim() < 3:
            return tensor.mean(dim=0)
        if labels is None:
            return tensor.mean(dim=(0, 1))

        mask = labels != -100
        if mask.shape[:2] != tensor.shape[:2]:
            return tensor.mean(dim=(0, 1))

        mask_f = mask.unsqueeze(-1).to(device=tensor.device, dtype=tensor.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        return (tensor * mask_f).sum(dim=(0, 1)) / denom

    @staticmethod
    def _find_response_mask(tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor is None or tensor.dim() < 3:
            return None

        frame = inspect.currentframe()
        mask = None
        try:
            while frame:
                local_vars = frame.f_locals
                labels = None
                if "labels" in local_vars and local_vars["labels"] is not None:
                    labels = local_vars["labels"]
                elif (
                    "kwargs" in local_vars
                    and isinstance(local_vars["kwargs"], dict)
                    and local_vars["kwargs"].get("labels") is not None
                ):
                    labels = local_vars["kwargs"]["labels"]

                if labels is not None:
                    candidate = labels != -100
                    if candidate.shape[:2] == tensor.shape[:2]:
                        mask = candidate.to(tensor.device)
                    break
                frame = frame.f_back
        finally:
            del frame

        return mask

    def _get_layer_module(self, layer_key: int) -> torch.nn.Module:
        layer_idx = layer_key - 1
        model = self.model.module if hasattr(self.model, "module") else self.model

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
            if hasattr(layers, "__len__") and layer_idx < len(layers):
                return layers[layer_idx]

        fallback_paths = [
            f"model.layers.{layer_idx}",
            f"model.model.layers.{layer_idx}",
            f"base_model.model.model.layers.{layer_idx}",
        ]
        for path in fallback_paths:
            try:
                return model.get_submodule(path)
            except (AttributeError, Exception):
                continue

        raise ValueError(
            f"Cannot locate target layer layer_key={layer_key} (0-indexed={layer_idx})."
        )

    def _unwrap_model(self) -> torch.nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def _is_gradient_checkpointing_enabled(self) -> bool:
        model = self._unwrap_model()
        for candidate in [model, getattr(model, "base_model", None)]:
            if candidate is not None and hasattr(candidate, "is_gradient_checkpointing"):
                return bool(candidate.is_gradient_checkpointing)
        return False

    def _set_gradient_checkpointing(self, enabled: bool):
        model = self._unwrap_model()
        fn_name = "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
        for candidate in [model, getattr(model, "base_model", None)]:
            if candidate is not None and hasattr(candidate, fn_name):
                getattr(candidate, fn_name)()
                return
