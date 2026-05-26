# Copyright (c) 2026
# 因果干预动态防御 — 梯度与激活分析器
#
# 在训练期间周期性采集各层隐状态激活和梯度的量化指标,
# 用于理解注入防御的底层机制:
#   1. 注入如何改变各层的激活值 (激活分析: 两次 forward 对比)
#   2. 训练梯度在恶意向量方向上的分量有多大 (梯度分析: backward 期间采集)
#   3. 上述指标如何随训练进程变化 (适应性分析: 跨 step 追踪)

import torch
import torch.nn.functional as F
import inspect
from typing import Dict, List, Optional, Tuple, Any


class GradientAnalyzer:
    """
    梯度与激活分析器。

    采集两类指标:

    [激活分析] — 两次 forward pass 对比:
        在 defense engine 的自然前向和注入前向期间,
        通过 forward hooks 捕获各监控层的隐状态激活值,
        量化注入导致的激活变化:
          - cos_sim(h_nat, h_inj):    激活方向相似度 (1.0=无变化, 0.0=正交)
          - ||h_inj|| / ||h_nat||:    激活范数变化比
          - ||h_inj - h_nat|| / ||h_nat||: 相对扰动幅度
          - proj(Δh, v_m):           激活变化在 v_m 方向的投影 (仅注入层)

    [梯度分析] — 训练 backward 期间:
        通过 backward hooks 捕获各监控层的隐状态梯度 ∂L/∂h,
        量化梯度特征:
          - ||grad||:                 梯度范数 (训练信号强度)
          - cos(grad, v_m):           梯度与 v_m 的余弦相似度 (仅注入层)
          - proj(grad, v_m):          梯度在 v_m 方向的投影 (仅注入层)

    监控层:
        默认监控注入层及其前后各 10 层（每隔 5 层采样一次）,
        重点在注入层本身（所有指标）和注入层以上的层（激活级联效应）。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        v_m: Optional[torch.Tensor] = None,
        injection_layer_key: int = 43,
        monitor_layers: Optional[List[int]] = None,
        use_lora: bool = False,
    ):
        """
        Args:
            model: DeepSpeed 引擎包装的模型实例。
            v_m: 注入层的恶意向量, shape [hidden_dim]。用于计算投影指标。
            injection_layer_key: 注入层 key (1-indexed, 与 hooks.py 一致)。
            monitor_layers: 要监控的层 key 列表 (1-indexed)。
                若为 None, 自动选择: 注入层 ± 10 层, 每隔 5 层采样。
            use_lora: 是否使用 PEFT/LoRA。
        """
        self.model = model
        self.v_m = v_m
        self.injection_layer_key = injection_layer_key
        self.use_lora = use_lora

        # 自动选择监控层
        if monitor_layers is None:
            total_layers = self._get_total_layers()
            start = max(1, injection_layer_key - 10)
            end = min(total_layers, injection_layer_key + 15)
            self.monitor_layers = list(range(start, end + 1, 5))
            if injection_layer_key not in self.monitor_layers:
                self.monitor_layers.append(injection_layer_key)
            self.monitor_layers.sort()
        else:
            self.monitor_layers = sorted(set(list(monitor_layers) + [injection_layer_key]))

        # 内部存储
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._chain_rule: Dict[str, torch.Tensor] = {}
        self._chain_rule_metrics: Dict[str, float] = {}
        self._residual_chain_rule: Dict[str, torch.Tensor] = {}
        self._residual_chain_rule_metrics: Dict[str, float] = {}
        self._chain_rule_layer_key = injection_layer_key
        self._current_response_mask: Optional[torch.Tensor] = None
        self._mask_debug_printed = set()
        self._fwd_hooks: list = []
        self._bwd_hooks: list = []
        self._chain_hooks: list = []

        # 存储注入层的初始权重快照 W_0，用于计算权重累积变化
        self._w0_snapshot = None
        if self.v_m is not None:
            self._capture_w0_snapshot()

    def _capture_w0_snapshot(self):
        """捕获注入层前馈网络的初始权重快照 (适配 DeepSpeed ZeRO-3)。"""
        layer = self._get_layer_module(self.injection_layer_key)
        target_param = None
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear) and 'down_proj' in name:
                param = module.weight
                # 适配 DeepSpeed ZeRO-3 参数分片，临时聚合完整权重
                if hasattr(param, 'ds_id'):
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                        target_param = param.data.detach().clone()
                else:
                    target_param = param.detach().clone()
                break
        
        if target_param is not None:
            self._w0_snapshot = target_param
        else:
            print(f"[Warning] GradientAnalyzer: Could not find 'down_proj' in layer {self.injection_layer_key} to snapshot W_0.")

    # ================================================================
    # 模型层定位 (复用 hooks.py 的约定: layer_key → layers[layer_key-1])
    # ================================================================

    def _get_total_layers(self) -> int:
        """获取模型的总层数。"""
        base = self.model.module if hasattr(self.model, 'module') else self.model
        if self.use_lora and hasattr(base, 'base_model'):
            inner = base.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
        else:
            inner = base

        if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
            return len(inner.model.layers)
        elif hasattr(inner, 'layers'):
            return len(inner.layers)
        return 64  # fallback

    # def _get_layer_module(self, layer_key: int) -> torch.nn.Module:
    #     """根据 layer_key (1-indexed) 获取模型层模块。"""
    #     layer_idx = layer_key - 1
    #     base = self.model.module if hasattr(self.model, 'module') else self.model

    #     if self.use_lora and hasattr(base, 'base_model'):
    #         inner = base.base_model
    #         if hasattr(inner, 'model'):
    #             inner = inner.model
    #     else:
    #         inner = base

    #     if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
    #         return inner.model.layers[layer_idx]
    #     elif hasattr(inner, 'layers'):
    #         return inner.layers[layer_idx]
    #     raise RuntimeError(f"Cannot find layer_key={layer_key} (idx={layer_idx})")
    
    def _get_layer_module(self, layer_key):
        # layer_key 是 1-based，对应 hidden_states 的索引
        # hidden_states[0] = embedding，没有对应的 layers 模块
        # hidden_states[k] = layers[k-1] 的输出，k>=1
        layer_idx = int(layer_key) - 1  # 转成 layers 的 0-based 索引

        m = self.model
        if hasattr(m, 'module'):
            m = m.module

        # 获取实际 layers 列表
        layers = None
        if hasattr(m, 'language_model') and hasattr(m.language_model, 'model') \
                and hasattr(m.language_model.model, 'layers'):
            layers = m.language_model.model.layers      # Gemma3 多模态
        elif hasattr(m, 'model') and hasattr(m.model, 'layers'):
            layers = m.model.layers                      # Qwen / LLaMA
        elif hasattr(m, 'layers'):
            layers = m.layers

        if layers is None:
            raise RuntimeError(f"Cannot find layers in model")

        # layer_key=1 对应 layers[0]，layer_key=48 对应 layers[47]
        # layer_key=49 对应 hidden_states[48]（最后一层输出），但实际是 layers[47] 经过 norm 后的结果
        # 所以 key=49 应该 clamp 到最后一个真实 layer
        max_idx = len(layers) - 1  # 47
        if layer_idx > max_idx:
            print(f"[Warning] layer_key={layer_key} (idx={layer_idx}) > max layer idx {max_idx}, clamping to {max_idx}")
            layer_idx = max_idx

        if layer_idx < 0:
            raise RuntimeError(f"layer_key={layer_key} is invalid (idx={layer_idx} < 0)")

        return layers[layer_idx]

    # ================================================================
    # Forward Hooks — 激活捕获
    # ================================================================

    def register_forward_hooks(self, prefix: str):
        """
        注册 forward hooks, 捕获监控层的隐状态激活(包括输入和 MLP 输出)。

        Args:
            prefix: "nat" 或 "inj", 用于区分两次 forward pass 的数据。
        """
        self._fwd_hooks = []
        for key in self.monitor_layers:
            layer = self._get_layer_module(key)
            hook = layer.register_forward_hook(
                self._make_fwd_hook(f"{prefix}_L{key}")
            )
            self._fwd_hooks.append(hook)
            if hasattr(layer, 'mlp'):
                hook_mlp = layer.mlp.register_forward_hook(
                    self._make_fwd_hook(f"{prefix}_L{key}_mlp")
                )
                self._fwd_hooks.append(hook_mlp)

    def remove_forward_hooks(self):
        """移除所有 forward hooks。"""
        for h in self._fwd_hooks:
            h.remove()
        self._fwd_hooks = []

    def _make_fwd_hook(self, name: str):
        """创建 forward hook 闭包。"""
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = self._find_response_mask(h)
            if mask is not None:
                module._cached_analyzer_response_mask = mask
                mask_source = "response_live"
            else:
                mask = getattr(module, "_cached_analyzer_response_mask", None)
                mask_source = "response_cached" if mask is not None else "all_token_fallback"
            # 只保留 response-token mean-pooled 表示以节省显存: [B, S, H] → [H]
            self._debug_mask_usage(f"forward:{name}", h, mask, mask_source)
            self._activations[name] = self._masked_mean_tokens(h, mask)
        return fn

    # ================================================================
    # Backward Hooks — 梯度捕获
    # ================================================================

    def register_backward_hooks(self):
        """
        注册 hooks, 在主训练 forward/backward 图上捕获注入层残差输出 h 及隐状态梯度。
        应在 model.backward() 之前调用, 之后调用 remove_backward_hooks()。
        """
        self._bwd_hooks = []
        for key in self.monitor_layers:
            layer = self._get_layer_module(key)
            if key == self.injection_layer_key:
                hook_fwd = layer.register_forward_hook(
                    self._make_residual_chain_fwd_hook(f"residual_L{key}")
                )
                self._bwd_hooks.append(hook_fwd)
            hook = layer.register_full_backward_hook(
                self._make_bwd_hook(f"grad_L{key}")
            )
            self._bwd_hooks.append(hook)

    def remove_backward_hooks(self):
        """移除所有训练图上的 forward/backward hooks。"""
        for h in self._bwd_hooks:
            h.remove()
        self._bwd_hooks = []

    def _make_residual_chain_fwd_hook(self, name: str):
        """创建注入层残差流输出 h 的 forward hook，用于严格的 h 空间链式求导分析。"""
        def fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            mask = self._find_response_mask(h)
            if mask is None:
                mask = self._get_current_response_mask(h)
                if mask is not None:
                    mask_source = "response_current_batch"
                else:
                    mask = getattr(module, "_residual_chain_rule_mask", None)
                    mask_source = "response_cached" if mask is not None else "all_token_fallback"
            else:
                module._residual_chain_rule_mask = mask
                mask_source = "response_live"
            self._debug_mask_usage(f"residual_chain_forward:{name}", h, mask, mask_source)
            self._residual_chain_rule["h_out"] = self._masked_mean_tokens(h, mask)
        return fn

    def _make_bwd_hook(self, name: str):
        """创建 backward hook 闭包。"""
        def fn(module, grad_input, grad_output):
            if grad_output[0] is not None:
                mask = self._find_response_mask(grad_output[0])
                if mask is None:
                    mask = self._get_current_response_mask(grad_output[0])
                    if mask is not None:
                        mask_source = "response_current_batch"
                    else:
                        mask = getattr(module, "_cached_analyzer_response_mask", None)
                        mask_source = "response_cached" if mask is not None else "all_token_fallback"
                else:
                    mask_source = "response_live"
                # Mean-pool response-token 梯度: [B, S, H] → [H]
                self._debug_mask_usage(f"backward:{name}", grad_output[0], mask, mask_source)
                self._gradients[name] = self._masked_mean_tokens(grad_output[0], mask)
        return fn

    def set_current_labels(self, labels: Optional[torch.Tensor]):
        """显式设置当前训练 batch 的 response mask，供 backward hooks 使用。"""
        if labels is None:
            self._current_response_mask = None
        else:
            self._current_response_mask = (labels != -100).detach()

    def _get_current_response_mask(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        mask = self._current_response_mask
        if mask is None or tensor is None or tensor.dim() < 3:
            return None
        if mask.shape[:2] != tensor.shape[:2]:
            return None
        return mask.to(tensor.device)

    def _is_rank0(self) -> bool:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _debug_mask_usage(
        self,
        name: str,
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor],
        source: str,
    ):
        """Print one rank0 line per hook/source so response/all-token fallbacks are visible."""
        debug_key = (name, source)
        if debug_key in self._mask_debug_printed or not self._is_rank0():
            return
        self._mask_debug_printed.add(debug_key)

        if mask is not None and mask.shape[:2] == tensor.shape[:2]:
            ratio = mask.float().mean().item()
            count = int(mask.sum().item())
            total = mask.numel()
            print(
                f"[GradientAnalyzer] {name}: using {source}, "
                f"response_tokens={count}/{total} ({ratio:.2%})."
            )
        else:
            shape = tuple(tensor.shape[:2]) if tensor is not None and tensor.dim() >= 2 else None
            print(
                f"[GradientAnalyzer][Warning] {name}: response mask unavailable, "
                f"using ALL tokens. tensor_BS={shape}."
            )

    def _find_response_mask(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """从调用栈中寻找 labels，并返回 labels != -100 的 response token mask。"""
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

    @staticmethod
    def _masked_mean_tokens(tensor: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """优先对 response tokens 求均值；找不到 mask 时回退到全 token 均值。"""
        tensor = tensor.detach().float()
        if tensor.dim() < 3:
            return tensor.mean(dim=0)

        if mask is None or mask.shape[:2] != tensor.shape[:2]:
            return tensor.mean(dim=(0, 1))

        mask_f = mask.unsqueeze(-1).to(device=tensor.device, dtype=tensor.dtype)
        denom = mask_f.sum().clamp(min=1.0)
        return (tensor * mask_f).sum(dim=(0, 1)) / denom

    # ================================================================
    # Chain-Rule Hooks — 捕获输出投影矩阵的真实 dW = delta ⊗ activation 因子
    # ================================================================

    def register_chain_rule_hooks(self):
        """
        捕获注入层 MLP down_proj 与 Attention o_proj 的局部链式法则因子。

        对输出投影矩阵 W 而言:
            dL/dW = delta_out^T @ a
        其中 a 是该 Linear 的输入激活, delta_out 是输出端反传误差。
        若恶意向量 v_m 位于 hidden/output 维度, 则输入空间里的恶意方向为:
            v_input = W^T · normalize(v_m)
        对应的有符号更新代理为:
            -<mean(delta_out), normalize(v_m)> * <mean(a), v_input>
        """
        self._chain_hooks = []
        layer = self._get_layer_module(self._chain_rule_layer_key)

        down_proj_name, down_proj = self._find_down_proj_module(layer)
        if down_proj is None:
            print(f"[Warning] GradientAnalyzer: Could not find down_proj in layer {self._chain_rule_layer_key}.")
        else:
            self._register_linear_chain_rule_hooks(
                module=down_proj,
                module_key="down_proj",
                module_label=down_proj_name or "down_proj",
                v_input_key="down_proj_v_neuron",
            )

        o_proj_name, o_proj = self._find_attention_o_proj_module(layer)
        if o_proj is None:
            print(f"[Warning] GradientAnalyzer: Could not find attention o_proj in layer {self._chain_rule_layer_key}.")
        else:
            self._register_linear_chain_rule_hooks(
                module=o_proj,
                module_key="o_proj",
                module_label=o_proj_name or "o_proj",
                v_input_key="o_proj_v_input",
            )

    def _find_down_proj_module(self, layer: torch.nn.Module) -> Tuple[Optional[str], Optional[torch.nn.Linear]]:
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear) and "down_proj" in name.lower():
                return name, module
        return None, None

    def _find_attention_o_proj_module(self, layer: torch.nn.Module) -> Tuple[Optional[str], Optional[torch.nn.Linear]]:
        fallback = (None, None)
        for name, module in layer.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            lower = name.lower()
            parts = lower.split(".")
            in_attention = any(part in {"self_attn", "attn", "attention"} for part in parts)
            if "down_proj" in lower:
                continue
            if in_attention and (
                lower.endswith("o_proj")
                or lower.endswith("out_proj")
                or lower.endswith(".o")
                or lower.endswith(".wo")
                or ".wo." in lower
            ):
                return name, module
            if in_attention and lower.endswith("c_proj"):
                fallback = (name, module)
            elif fallback[1] is None and lower.endswith("c_proj"):
                fallback = (name, module)
            elif fallback[1] is None and lower.endswith("out_proj"):
                fallback = (name, module)
        return fallback

    def _register_linear_chain_rule_hooks(
        self,
        module: torch.nn.Linear,
        module_key: str,
        module_label: str,
        v_input_key: str,
    ):
        v_input_cached = None
        if self.v_m is not None:
            param = module.weight

            def compute_v_input(weight: torch.Tensor):
                v_flat = self.v_m.flatten().to(device=weight.device, dtype=weight.dtype)
                v_flat = v_flat / v_flat.norm().clamp(min=1e-10)
                if weight.dim() == 2 and weight.shape[0] == v_flat.shape[0]:
                    return torch.matmul(weight.detach().t(), v_flat).detach().float().cpu()
                if self._is_rank0():
                    print(
                        f"[GradientAnalyzer][Warning] Cannot compute v_input for L{self._chain_rule_layer_key} "
                        f"{module_label}: weight_shape={tuple(weight.shape)}, v_shape={tuple(v_flat.shape)}."
                    )
                return None

            with torch.no_grad():
                if hasattr(param, "ds_id"):
                    import deepspeed
                    with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                        v_input_cached = compute_v_input(param.data)
                else:
                    v_input_cached = compute_v_input(param.data)

        def fwd_hook(hooked_module, inputs, output):
            if not inputs:
                return
            a = inputs[0]
            if a is None:
                return
            mask = self._find_response_mask(a)
            if mask is None:
                mask = self._get_current_response_mask(a)
                if mask is not None:
                    mask_source = "response_current_batch"
                else:
                    mask = getattr(hooked_module, "_chain_rule_mask", None)
                    mask_source = "response_cached" if mask is not None else "all_token_fallback"
            else:
                hooked_module._chain_rule_mask = mask
                mask_source = "response_live"
            self._debug_mask_usage(f"chain_forward:{module_key}_activation", a, mask, mask_source)
            self._chain_rule[f"{module_key}_activation"] = self._masked_mean_tokens(a, mask)

            if v_input_cached is not None:
                self._chain_rule[v_input_key] = v_input_cached

        def bwd_hook(hooked_module, grad_input, grad_output):
            if not grad_output or grad_output[0] is None:
                return
            delta = grad_output[0]
            mask = self._find_response_mask(delta)
            if mask is None:
                mask = self._get_current_response_mask(delta)
                if mask is not None:
                    mask_source = "response_current_batch"
                else:
                    mask = getattr(hooked_module, "_chain_rule_mask", None)
                    mask_source = "response_cached" if mask is not None else "all_token_fallback"
            else:
                mask_source = "response_live"
            self._debug_mask_usage(f"chain_backward:{module_key}_delta", delta, mask, mask_source)
            self._chain_rule[f"{module_key}_delta"] = self._masked_mean_tokens(delta, mask)

        self._chain_hooks.append(module.register_forward_hook(fwd_hook))
        self._chain_hooks.append(module.register_full_backward_hook(bwd_hook))

    def remove_chain_rule_hooks(self):
        """移除输出投影矩阵链式法则 hooks。"""
        for h in self._chain_hooks:
            h.remove()
        self._chain_hooks = []

    # ================================================================
    # 分析入口
    # ================================================================

    def run_activation_analysis(
        self, batch: Dict[str, torch.Tensor], injector
    ) -> Dict[str, float]:
        """
        执行两次 forward pass (自然 + 注入), 计算激活层面指标。
        使用 no_grad, 不影响训练。

        Args:
            batch: 当前训练 batch。
            injector: MaliciousVectorInjector 实例。

        Returns:
            metrics: TensorBoard 可用的 {metric_name: value} 字典。
        """
        self._activations = {}

        # 记录原始模式，并切换到 eval 以关闭 Dropout，保证两次 Forward 的一致性
        was_training = self.model.training
        self.model.eval()

        # Pass 1: 自然 forward (无注入)
        self.register_forward_hooks("nat")
        with torch.no_grad():
            self.model(**batch, use_cache=False)
        self.remove_forward_hooks()

        # Pass 2: 注入 forward
        # self.register_forward_hooks("inj")
        # with torch.no_grad():
        #     injector.attach()
        #     self.model(**batch, use_cache=False)
        #     injector.detach()
        # self.remove_forward_hooks()
        with torch.no_grad():
            injector.attach()
            self.register_forward_hooks("inj")
            try:
                self.model(**batch, use_cache=False)
            finally:
                self.remove_forward_hooks() 
                injector.detach()

        if was_training:
            self.model.train()

        return self._compute_activation_metrics()

    def collect_gradient_metrics(self) -> Dict[str, float]:
        """
        从已捕获的梯度数据计算指标。
        在 model.backward(loss) + remove_backward_hooks() 之后调用。

        Returns:
            metrics: TensorBoard 可用的 {metric_name: value} 字典。
        """
        return self._compute_gradient_metrics()

    # ================================================================
    # 指标计算
    # ================================================================

    def _compute_activation_metrics(self) -> Dict[str, float]:
        """对比自然/注入激活, 计算各层指标。"""
        metrics = {}
        v_flat = None
        v_norm = 0.0
        if self.v_m is not None:
            v_flat = self.v_m.flatten().float()
            v_norm = v_flat.norm().item()

        for key in self.monitor_layers:
            h_nat = self._activations.get(f"nat_L{key}")
            h_inj = self._activations.get(f"inj_L{key}")
            if h_nat is None or h_inj is None:
                continue

            # [H] 向量
            norm_nat = h_nat.norm().item()
            norm_inj = h_inj.norm().item()
            delta = h_inj - h_nat

            # 1. 余弦相似度: 激活方向被改变了多少
            cos = F.cosine_similarity(
                h_nat.unsqueeze(0), h_inj.unsqueeze(0)
            ).item()

            # 2. 范数比: 激活大小被缩放了多少
            norm_ratio = norm_inj / max(norm_nat, 1e-10)

            # 3. 相对变化幅度: 扰动占原始激活的比例
            rel_change = delta.norm().item() / max(norm_nat, 1e-10)

            metrics[f"ActAnalysis/L{key}_cos_sim"] = cos
            metrics[f"ActAnalysis/L{key}_norm_ratio"] = norm_ratio
            metrics[f"ActAnalysis/L{key}_rel_change"] = rel_change

            # 4. 激活变化在 v_m 方向的投影 (仅注入层)
            if key == self.injection_layer_key and v_flat is not None and v_norm > 1e-10:
                delta_dev = delta.to(v_flat.device)
                proj = torch.dot(delta_dev, v_flat).item() / v_norm
                # 归一化投影: 投影 / Δh 范数 → 表示变化中有多大比例在 v_m 方向
                delta_norm = delta_dev.norm().item()
                proj_ratio = proj / max(delta_norm, 1e-10)
                metrics[f"ActAnalysis/L{key}_delta_proj_vm"] = proj
                metrics[f"ActAnalysis/L{key}_delta_proj_vm_ratio"] = proj_ratio

            if key == self.injection_layer_key and v_flat is not None and v_norm > 1e-10:
                # 获取三个关键张量
                h_mlp = self._activations.get(f"nat_L{key}_mlp")
                if h_mlp is not None and h_nat is not None:
                    h_mlp_dev = h_mlp.to(v_flat.device)
                    h_out_dev = h_nat.to(v_flat.device)

                    # 计算在 v_m 上的投影
                    proj_mlp = torch.dot(h_mlp_dev, v_flat).item() / v_norm
                    proj_out = torch.dot(h_out_dev, v_flat).item() / v_norm

                    # 记录到 TensorBoard
                    metrics[f"ActAnalysis/L{key}_h_mlp_proj_vm"] = proj_mlp   # 证明 注入层 局部的物理拦截
                    metrics[f"ActAnalysis/L{key}_h_out_proj_vm"] = proj_out   # 证明穿过 注入层 后的最终致盲状态

        # 计算当前注入层权重增量 (W_step - W_0) 对 v_m 的响应，以及与 -v_m 的余弦相似度
        if self._w0_snapshot is not None and v_flat is not None and v_norm > 1e-10:
            layer = self._get_layer_module(self.injection_layer_key)
            current_param = None
            for name, module in layer.named_modules():
                if isinstance(module, torch.nn.Linear) and 'down_proj' in name:
                    param = module.weight
                    # 同样适配 DeepSpeed ZeRO-3 进行参数聚合
                    if hasattr(param, 'ds_id'):
                        import deepspeed
                        with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                            current_param = param.data.detach().clone()
                    else:
                        current_param = param.detach().clone()
                    break

            if current_param is not None:
                delta_w = current_param.to(self._w0_snapshot.device) - self._w0_snapshot
                v_dev = v_flat.to(device=current_param.device, dtype=current_param.dtype)
                
                # 矩阵运算安全保障：
                # down_proj.weight 通常是 [out_features, in_features] 即 [4096, 14336]
                # v_m 属于输出特征空间 [4096]
                if delta_w.shape[0] == v_dev.shape[0]:
                    delta_w_t = delta_w.t() # 转置为 [14336, 4096]
                    
                    # 计算 \Delta W 每一行在原始恶意方向 v_m 上的投影向量 (预期产生极大的负值)
                    delta_response = torch.matmul(delta_w_t, v_dev)
                    
                    # 1. 均值投影 (Mean) - 容易被大量不相关的 0 稀释
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_mean_proj_on_vm"] = delta_response.mean().item()
                    
                    # 2. 总和投影 (Sum) - 展现整体宏观负向势能
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_sum_proj_on_vm"] = delta_response.sum().item()
                    
                    # 3. 核心群体深度 (Top 5%) - 真正负责恶意概念的神经元群体的平均陷阱深度
                    k_val = max(1, int(delta_response.shape[0] * 0.05))
                    topk_neg_vals, _ = torch.topk(delta_response, k=k_val, largest=False)
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_top5_proj_on_vm"] = topk_neg_vals.mean().item()
                    
                    # 4. 单点极限深度 (Max 陷阱深度) - 使用 .min() 取最极端的负数值
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_min_proj_on_vm"] = delta_response.min().item()
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_max_proj_on_vm"] = delta_response.max().item()
                    
                    # 5. 优化器方向对齐度 (Cosine) - 与防守目标方向 -v_m 的余弦相似度
                    neg_v_dev = -v_dev
                    cos_sims = F.cosine_similarity(delta_w_t, neg_v_dev.unsqueeze(0), dim=-1)
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_cos_with_neg_vm"] = cos_sims.mean().item()

                    k_val_cos = max(1, int(cos_sims.shape[0] * 0.05))
                    topk_cos_vals, _ = torch.topk(cos_sims, k=k_val_cos, largest=True)
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_top5_cos_with_neg_vm"] = topk_cos_vals.mean().item()
                    
                    # 5.3 单点极限对齐度 (Max) - 找出一个更新方向最完美重合的神经元
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_max_cos_with_neg_vm"] = cos_sims.max().item()
                    metrics[f"WeightAnalysis/L{self.injection_layer_key}_deltaW_min_cos_with_neg_vm"] = cos_sims.min().item()

        return metrics

    def _compute_gradient_metrics(self) -> Dict[str, float]:
        """计算各层梯度指标。"""
        metrics = {}
        v_flat = None
        v_norm = 0.0
        if self.v_m is not None:
            v_flat = self.v_m.flatten().float()
            v_norm = v_flat.norm().item()

        for key in self.monitor_layers:
            g = self._gradients.get(f"grad_L{key}")
            if g is None:
                continue

            # 1. 梯度范数: 该层训练信号的强度
            grad_norm = g.norm().item()
            metrics[f"GradAnalysis/L{key}_grad_norm"] = grad_norm

            # 2. 与 v_m 的关系 (仅注入层)
            if key == self.injection_layer_key and v_flat is not None and v_norm > 1e-10:
                g_dev = g.to(v_flat.device)

                # 梯度在 v_m 方向的投影: 模型想要在恶意方向上更新多少
                proj = torch.dot(g_dev, v_flat).item() / v_norm
                metrics[f"GradAnalysis/L{key}_grad_proj_vm"] = proj

                # 余弦相似度: 梯度与 v_m 的方向对齐程度
                cos_vm = F.cosine_similarity(
                    g_dev.unsqueeze(0), v_flat.unsqueeze(0)
                ).item()
                metrics[f"GradAnalysis/L{key}_grad_cos_vm"] = cos_vm

                # 垂直分量占比: 梯度中非 v_m 方向的比例
                proj_vec = (proj / v_norm) * v_flat
                ortho = g_dev - proj_vec
                ortho_ratio = ortho.norm().item() / max(grad_norm, 1e-10)
                metrics[f"GradAnalysis/L{key}_grad_ortho_ratio"] = ortho_ratio

        return metrics

    def run_online_3d_analysis_bak(self, step: int, is_defense_active: bool = True) -> Any:
        """
        在线提取 3D 轨迹 (链式法则鞍面) 并生成 matplotlib figure 以供 TensorBoard 记录。
        这需要在同一步中既有 forward hooks (nat_L{key}) 又有 backward hooks (grad_L{key}) 的数据。
        """
        if not hasattr(self, 'history_steps'):
            self.history_steps = []
            self.history_x = []
            self.history_y = []
            self.history_z = []
            self.history_active = []
            
        key = self.injection_layer_key
        h_nat = self._activations.get(f"nat_L{key}")
        grad_h = self._gradients.get(f"grad_L{key}")
        
        if h_nat is None or grad_h is None or self.v_m is None:
            return None
            
        v_flat = self.v_m.flatten().float().cpu()
        v_flat = v_flat / v_flat.norm()
        h_nat_cpu = h_nat.cpu()
        grad_h_cpu = grad_h.cpu()
        
        # X轴: 误差大小 ||dL/dh||
        x_val = torch.norm(grad_h_cpu).item()
        
        # Y轴: 激活大小 h * v_m
        y_val = torch.dot(h_nat_cpu, v_flat).item()
        
        # Z轴: 梯度大小 X * |Y|
        z_val = x_val * abs(y_val)
        
        self.history_steps.append(step)
        self.history_x.append(x_val)
        self.history_y.append(y_val)
        self.history_z.append(z_val)
        self.history_active.append(is_defense_active)
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        x_coords = np.array(self.history_x)
        y_coords = np.array(self.history_y)
        z_coords = np.array(self.history_z)
        
        # 如果点太少，无法画出有意义的曲面，就只画点
        if len(x_coords) > 1:
            x_surf = np.linspace(min(x_coords)*0.8, max(x_coords)*1.1, 30)
            y_surf = np.linspace(min(y_coords)*1.2, max(y_coords)*1.2, 30)
            X_grid, Y_grid = np.meshgrid(x_surf, y_surf)
            Z_grid = X_grid * np.abs(Y_grid)
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, cmap='viridis', edgecolor='none')
            ax.plot(x_surf, np.zeros_like(x_surf), np.zeros_like(x_surf), color='black', linestyle='--', linewidth=2)
            
        # 根据当前点是否开启防御来动态区分颜色
        inject_idx = [i for i, a in enumerate(self.history_active) if a]
        withdraw_idx = [i for i, a in enumerate(self.history_active) if not a]
        
        if len(inject_idx) > 0:
            ax.plot(x_coords[inject_idx], y_coords[inject_idx], z_coords[inject_idx], 
                    color='red', linewidth=3, marker='o', markersize=6, label='Injection Phase')
        if len(withdraw_idx) > 0:
            ax.plot(x_coords[withdraw_idx], y_coords[withdraw_idx], z_coords[withdraw_idx], 
                    color='blue', linewidth=3, marker='^', markersize=6, label='Unprotected Phase')
                    
        ax.set_xlabel('Error ||dL/dh|| (X)')
        ax.set_ylabel('Activation h*v_m (Y)')
        ax.set_zlabel('Gradient ||dW|| (Z=X*|Y|)')
        ax.set_title('Online Gradient Starvation Analysis')
        ax.legend()
        ax.view_init(elev=25, azim=40)
        
        plt.tight_layout()
        return fig

    def run_online_3d_analysis(self, step: int, is_defense_active: bool = True) -> Any:
        """
        在线提取注入层残差流 h 的严格有符号 3D 轨迹。

        X = -<dL/dh, normalize(v_m)>, 即梯度下降在 v_m 方向上的一阶推动。
            X > 0 表示增强 +v_m, X < 0 表示压制 +v_m。
        Y = <h, normalize(v_m)>, 即当前残差流在 v_m 方向上的暴露程度。
        Z = X * Y, 即残差流层面的有符号纠偏动量。

        注意: h 不是可训练参数, 因此这里分析的是 h 空间中的方向导数/虚拟推动力,
        不声称等价于真实参数更新 dW。
        """
        if not hasattr(self, "residual_history_steps"):
            self.residual_history_steps = []
            self.residual_history_x = []
            self.residual_history_y = []
            self.residual_history_z = []
            self.residual_history_active = []

        key = self.injection_layer_key
        h_out = self._residual_chain_rule.get("h_out")
        grad_h = self._gradients.get(f"grad_L{key}")

        if h_out is None or grad_h is None or self.v_m is None:
            return None

        v_unit = self.v_m.flatten().float().cpu()
        v_unit = v_unit / v_unit.norm().clamp(min=1e-10)
        h_cpu = h_out.cpu()
        grad_h_cpu = grad_h.cpu()

        x_val = -torch.dot(grad_h_cpu, v_unit).item()
        y_val = torch.dot(h_cpu, v_unit).item()
        z_val = x_val * y_val
        self._residual_chain_rule_metrics = {
            f"ResidualChainRule/L{key}_x_signed_update_toward_vm": x_val,
            f"ResidualChainRule/L{key}_y_h_proj_vm": y_val,
            f"ResidualChainRule/L{key}_z_signed_correction_momentum": z_val,
        }

        self.residual_history_steps.append(step)
        self.residual_history_x.append(x_val)
        self.residual_history_y.append(y_val)
        self.residual_history_z.append(z_val)
        self.residual_history_active.append(is_defense_active)

        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        x_coords = np.array(self.residual_history_x)
        y_coords = np.array(self.residual_history_y)
        z_coords = np.array(self.residual_history_z)

        if len(x_coords) > 1:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_pad = max((x_max - x_min) * 0.1, 1e-12)
            y_pad = max((y_max - y_min) * 0.1, 1e-12)
            x_surf = np.linspace(x_min - x_pad, x_max + x_pad, 30)
            y_surf = np.linspace(y_min - y_pad, y_max + y_pad, 30)
            X_grid, Y_grid = np.meshgrid(x_surf, y_surf)
            Z_grid = X_grid * Y_grid
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, cmap="viridis", edgecolor="none")
            ax.plot(x_surf, np.zeros_like(x_surf), np.zeros_like(x_surf), color="black", linestyle="--", linewidth=1)
            ax.plot(np.zeros_like(y_surf), y_surf, np.zeros_like(y_surf), color="black", linestyle="--", linewidth=1)

        inject_idx = [i for i, a in enumerate(self.residual_history_active) if a]
        withdraw_idx = [i for i, a in enumerate(self.residual_history_active) if not a]

        if inject_idx:
            ax.plot(
                x_coords[inject_idx], y_coords[inject_idx], z_coords[inject_idx],
                color="red", linewidth=3, marker="o", markersize=6, label="Injection Phase"
            )
        if withdraw_idx:
            ax.plot(
                x_coords[withdraw_idx], y_coords[withdraw_idx], z_coords[withdraw_idx],
                color="blue", linewidth=3, marker="^", markersize=6, label="Unprotected Phase"
            )

        ax.set_xlabel("Signed residual push: -<dL/dh, v> (X)")
        ax.set_ylabel("Residual exposure: <h, v> (Y)")
        ax.set_zlabel("Signed correction momentum X*Y (Z)")
        ax.set_title(f"Residual-Stream Directional Chain Rule: L{key}")
        ax.legend()
        ax.view_init(elev=25, azim=40)

        plt.tight_layout()
        return fig

    def run_chain_rule_3d_analysis(self, step: int, is_defense_active: bool = True) -> Any:
        """
        绘制更贴近链式法则的 3D 轨迹。

        X = -<dL/d down_proj_out, v_m>, 即梯度下降对 v_m 方向的有符号推动。
            X > 0 表示更新倾向于增强 +v_m；X < 0 表示更新倾向于压制 +v_m。
        Y = <down_proj_input, W_down^T v_m>, 即协助表达 v_m 的中间神经元激活。
        Z = X * Y, 即 v_m 子空间内 down_proj.weight 有效更新量的代理。
        """
        if not hasattr(self, "chain_steps"):
            self.chain_steps = []
            self.chain_x = []
            self.chain_y = []
            self.chain_z = []
            self.chain_active = []

        activation = self._chain_rule.get("down_proj_activation")
        delta = self._chain_rule.get("down_proj_delta")
        v_neuron = self._chain_rule.get("down_proj_v_neuron")
        if activation is None or delta is None or v_neuron is None or self.v_m is None:
            return None

        v_flat = self.v_m.flatten().float().cpu()
        v_flat = v_flat / v_flat.norm().clamp(min=1e-10)
        delta_cpu = delta.cpu()
        activation_cpu = activation.cpu()
        v_neuron_cpu = v_neuron.cpu()

        x_val = -torch.dot(delta_cpu, v_flat).item()
        y_val = torch.dot(activation_cpu, v_neuron_cpu).item()
        z_val = x_val * y_val
        self._chain_rule_metrics.update({
            f"ChainRule/L{self._chain_rule_layer_key}_x_signed_update_toward_vm": x_val,
            f"ChainRule/L{self._chain_rule_layer_key}_y_neuron_proj_vm": y_val,
            f"ChainRule/L{self._chain_rule_layer_key}_z_signed_update_proxy": z_val,
        })

        self.chain_steps.append(step)
        self.chain_x.append(x_val)
        self.chain_y.append(y_val)
        self.chain_z.append(z_val)
        self.chain_active.append(is_defense_active)

        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        x_coords = np.array(self.chain_x)
        y_coords = np.array(self.chain_y)
        z_coords = np.array(self.chain_z)

        if len(x_coords) > 1:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_pad = max((x_max - x_min) * 0.1, 1e-12)
            y_pad = max((y_max - y_min) * 0.1, 1e-12)
            x_surf = np.linspace(x_min - x_pad, x_max + x_pad, 30)
            y_surf = np.linspace(y_min - y_pad, y_max + y_pad, 30)
            X_grid, Y_grid = np.meshgrid(x_surf, y_surf)
            Z_grid = X_grid * Y_grid
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, cmap="viridis", edgecolor="none")
            ax.plot(x_surf, np.zeros_like(x_surf), np.zeros_like(x_surf), color="black", linestyle="--", linewidth=1)
            ax.plot(np.zeros_like(y_surf), y_surf, np.zeros_like(y_surf), color="black", linestyle="--", linewidth=1)

        inject_idx = [i for i, a in enumerate(self.chain_active) if a]
        withdraw_idx = [i for i, a in enumerate(self.chain_active) if not a]

        if inject_idx:
            ax.plot(
                x_coords[inject_idx], y_coords[inject_idx], z_coords[inject_idx],
                color="red", linewidth=3, marker="o", markersize=6, label="Injection Phase"
            )
        if withdraw_idx:
            ax.plot(
                x_coords[withdraw_idx], y_coords[withdraw_idx], z_coords[withdraw_idx],
                color="blue", linewidth=3, marker="^", markersize=6, label="Unprotected Phase"
            )

        ax.set_xlabel("Signed update toward v: -<dL/dh, v> (X)")
        ax.set_ylabel("Neuron contribution <m, W_down^T v> (Y)")
        ax.set_zlabel("Signed effective update proxy X*Y (Z)")
        ax.set_title(f"Chain Rule Update Proxy: L{self._chain_rule_layer_key} down_proj")
        ax.legend()
        ax.view_init(elev=25, azim=40)

        plt.tight_layout()
        return fig

    def run_attention_chain_rule_3d_analysis(self, step: int, is_defense_active: bool = True) -> Any:
        """
        绘制注入层 Attention 输出投影 W_o 的严格链式法则 3D 轨迹。

        X = -<dL/d o_proj_out, v_m>, 即梯度下降对 v_m 方向的有符号推动。
        Y = <o_proj_input, W_o^T v_m>, 即 attention 输出在 W_o 输入空间中协助表达 v_m 的分量。
        Z = X * Y, 即 v_m 子空间内 o_proj.weight 有效更新量的代理。
        """
        if not hasattr(self, "attn_chain_steps"):
            self.attn_chain_steps = []
            self.attn_chain_x = []
            self.attn_chain_y = []
            self.attn_chain_z = []
            self.attn_chain_active = []

        activation = self._chain_rule.get("o_proj_activation")
        delta = self._chain_rule.get("o_proj_delta")
        v_input = self._chain_rule.get("o_proj_v_input")
        if activation is None or delta is None or v_input is None or self.v_m is None:
            return None

        v_flat = self.v_m.flatten().float().cpu()
        v_flat = v_flat / v_flat.norm().clamp(min=1e-10)
        delta_cpu = delta.cpu()
        activation_cpu = activation.cpu()
        v_input_cpu = v_input.cpu()

        x_val = -torch.dot(delta_cpu, v_flat).item()
        y_val = torch.dot(activation_cpu, v_input_cpu).item()
        z_val = x_val * y_val
        self._chain_rule_metrics.update({
            f"ChainRuleAttention/L{self._chain_rule_layer_key}_x_signed_update_toward_vm": x_val,
            f"ChainRuleAttention/L{self._chain_rule_layer_key}_y_attention_proj_vm": y_val,
            f"ChainRuleAttention/L{self._chain_rule_layer_key}_z_signed_update_proxy": z_val,
        })

        self.attn_chain_steps.append(step)
        self.attn_chain_x.append(x_val)
        self.attn_chain_y.append(y_val)
        self.attn_chain_z.append(z_val)
        self.attn_chain_active.append(is_defense_active)

        import matplotlib.pyplot as plt
        import numpy as np

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        x_coords = np.array(self.attn_chain_x)
        y_coords = np.array(self.attn_chain_y)
        z_coords = np.array(self.attn_chain_z)

        if len(x_coords) > 1:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_pad = max((x_max - x_min) * 0.1, 1e-12)
            y_pad = max((y_max - y_min) * 0.1, 1e-12)
            x_surf = np.linspace(x_min - x_pad, x_max + x_pad, 30)
            y_surf = np.linspace(y_min - y_pad, y_max + y_pad, 30)
            X_grid, Y_grid = np.meshgrid(x_surf, y_surf)
            Z_grid = X_grid * Y_grid
            ax.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.2, cmap="viridis", edgecolor="none")
            ax.plot(x_surf, np.zeros_like(x_surf), np.zeros_like(x_surf), color="black", linestyle="--", linewidth=1)
            ax.plot(np.zeros_like(y_surf), y_surf, np.zeros_like(y_surf), color="black", linestyle="--", linewidth=1)

        inject_idx = [i for i, a in enumerate(self.attn_chain_active) if a]
        withdraw_idx = [i for i, a in enumerate(self.attn_chain_active) if not a]

        if inject_idx:
            ax.plot(
                x_coords[inject_idx], y_coords[inject_idx], z_coords[inject_idx],
                color="red", linewidth=3, marker="o", markersize=6, label="Injection Phase"
            )
        if withdraw_idx:
            ax.plot(
                x_coords[withdraw_idx], y_coords[withdraw_idx], z_coords[withdraw_idx],
                color="blue", linewidth=3, marker="^", markersize=6, label="Unprotected Phase"
            )

        ax.set_xlabel("Signed update toward v: -<dL/dh, v> (X)")
        ax.set_ylabel("Attention contribution <a, W_o^T v> (Y)")
        ax.set_zlabel("Signed effective update proxy X*Y (Z)")
        ax.set_title(f"Chain Rule Update Proxy: L{self._chain_rule_layer_key} o_proj")
        ax.legend()
        ax.view_init(elev=25, azim=40)

        plt.tight_layout()
        return fig
        
    def clear(self):
        """清空所有捕获的数据。"""
        self._activations = {}
        self._gradients = {}
        self._chain_rule = {}
        self._chain_rule_metrics = {}
        self._residual_chain_rule = {}
        self._residual_chain_rule_metrics = {}
        self._current_response_mask = None
