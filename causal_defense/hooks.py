# Copyright (c) 2026
# 因果干预动态防御 — Forward Hook 管理器
#
# 负责在模型目标层的隐状态中注入恶意向量 v_m，
# 实现 h' = h + α · v_m 的因果干预。
#
# 参考: training_norm.py 中的 steering_intervention / add_steering_hooks 实现。

import torch
from functools import partial
from typing import Dict, List, Optional


class MaliciousVectorInjector:
    """
    管理目标层的 forward hook，用于在隐状态中注入恶意向量。

    支持两种注入模式：
    1. 'all': 对所有的 Token（包括 Prompt 和 Response）进行无差别注入。
    2. 'response_only': 动态追踪调用栈中的 labels，仅对 Response Token 进行精准注入。

    使用模式:
        injector = MaliciousVectorInjector(model, vectors, layers, alpha, injection_mode="res_only")
        injector.attach()    # 注册 hooks
        outputs = model(**batch)  # 前向传播时自动注入
        injector.detach()   # 移除 hooks

    兼容性:
        - DeepSpeed ZeRO-3: hook 在前向传播时执行，此时参数已被 gather，
          隐状态是完整的，注入操作兼容。v_m 需在每个 GPU 上有完整副本。
        - Gradient Checkpointing: 仅在 no_grad 的干预前向传播中使用，
          不影响主前向传播的梯度检查点机制。
        - PEFT/LoRA: 自动处理 base_model 包装层。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        malicious_vectors: Dict[int, torch.Tensor],
        target_layers: List[int],
        alpha: float = 1.0,
        adaptive: bool = False,
        injection_mode: str = "res_only",
    ):
        """
        Args:
            model: 模型实例（可以是 DeepSpeed 引擎包装后的模型）。
            malicious_vectors: {layer_key: v_m tensor of shape [hidden_dim]}。
            target_layers: 目标层的 key 列表（与 malicious_vectors 的 key 对应）。
                注入位置为 model.layers[layer_key - 1]，与 training_norm.py 约定一致。
            alpha: 注入强度基础系数。
            adaptive: 是否启用自适应缩放。
            injection_mode: 注入模式，可选 "all_token" 或 "res_only"。
        """
        assert injection_mode in ["all_token", "res_only"], "injection_mode must be 'all_token' or 'res_only'"
        self.model = model
        self.malicious_vectors = malicious_vectors
        self.target_layers = target_layers
        self.alpha = alpha
        self.adaptive = adaptive
        self.injection_mode = injection_mode
        self._hooks: list = []

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
        """
        Forward hook 函数：在层输出的隐状态中注入恶意向量。

        实现 h' = h + scale · v_m，其中：
        - 非自适应: scale = alpha
        - 自适应:   scale = alpha × (mean_norm(h) / norm(v_m))

        Args:
            module: 被 hook 的模块。
            input: 模块输入（未使用）。
            output: 模块输出，通常为 (hidden_states, ...) 的 tuple。
            v_m: 恶意向量, shape [hidden_dim]。
            alpha: 注入强度系数。
            adaptive: 是否自适应缩放。
            injection_mode: 注入模式，可选 "all_token" 或 "res_only"。
        """
        # 兼容不同模型架构的输出格式
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # 将 v_m 移至与隐状态相同的设备和数据类型
        v_m_aligned = v_m.to(device=hidden_states.device, dtype=hidden_states.dtype)

        # v_m_aligned shape: [hidden_dim] → unsqueeze 到 [1, 1, hidden_dim] 以广播
        vec = v_m_aligned.unsqueeze(0).unsqueeze(0)

        # =================================================================
        # 模式 A: 全局注入 (All Tokens)
        # =================================================================
        if injection_mode == "all_token":
            # 1. Debug 打印 (单次)
            if not getattr(module, '_has_printed_injector_debug_all', False):
                print(f"\n🎯 [Injector Hook] 层 {module.__class__.__name__} 拦截成功！模式: 全局注入 (All Tokens)")
                module._has_printed_injector_debug_all = True
            
            # 计算注入缩放系数
            if adaptive:
                # 自适应: 根据当前层激活值的平均范数动态调整
                h_norm = hidden_states.norm(dim=-1, keepdim=True).mean()
                v_norm = v_m_aligned.norm()
                scale = alpha * (h_norm / (v_norm + 1e-8))
            else:
                scale = alpha
                
            # 注入: h' = h + scale · v_m
            hidden_states = hidden_states + (scale * vec)

        # =================================================================
        # 模式 B: 精准注入 (Response Only)
        # =================================================================
        elif injection_mode == "res_only":
            # 动态调用栈追踪与缓存机制
            import inspect
            mask = None
            frame = inspect.currentframe()
            try:
                # 向上逆流遍历调用栈，寻找 labels
                while frame:
                    _locals = frame.f_locals
                    if 'labels' in _locals and _locals['labels'] is not None:
                        mask = (_locals['labels'] != -100)
                        break
                    elif 'kwargs' in _locals and 'labels' in _locals['kwargs'] and _locals['kwargs']['labels'] is not None:
                        mask = (_locals['kwargs']['labels'] != -100)
                        break
                    frame = frame.f_back
            finally:
                del frame  # 防止内存泄漏

            # 兼容 Gradient Checkpointing
            if mask is not None:
                # 正常 Forward 阶段：成功找到 mask，将其缓存到当前实例，供 Backward 重算时使用
                module._cached_injector_mask = mask
            else:
                # Backward Recompute 阶段：调用栈变了找不到 labels，提取之前缓存的 mask
                mask = getattr(module, '_cached_injector_mask', None)
                assert mask is not None, "[Injector] 致命错误：未能提取到 labels，且无缓存的 mask！"

            # [验证机制]：单次打印，验证 Input/Output 区分是否成功
            if not getattr(module, '_has_printed_injector_debug_response', False):
                total_tokens = mask.numel()
                injected_tokens = mask.sum().item()
                print(f"\n🎯 [Injector Hook] 层 {module.__class__.__name__} 拦截成功！模式: 精准注入 (Response Only)")
                print(f"   📊 当前 Batch 总 Token: {total_tokens}, 实际注入 Token: {injected_tokens} ({injected_tokens/total_tokens*100:.2f}%)")
                module._has_printed_injector_debug_response = True

            # 计算注入缩放系数 (仅针对 Response Token)
            if adaptive:
                masked_hidden = hidden_states[mask] 
                h_norm = masked_hidden.norm(dim=-1, keepdim=True).mean() if masked_hidden.numel() > 0 else 1.0
                v_norm = v_m_aligned.norm()
                scale = alpha * (h_norm / (v_norm + 1e-8))
            else:
                scale = alpha
                
            # 构造 response mask 并执行广播注入 (安全避开 in-place 操作)
            mask_float = mask.unsqueeze(-1).to(hidden_states.device).to(hidden_states.dtype)
            hidden_states = hidden_states + (scale * vec * mask_float)

        # 保持输出的原始结构
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    def _get_layer_module(self, layer_key: int) -> torch.nn.Module:
        """
        根据 layer_key 获取模型的实际 Transformer 层模块。
        自动处理 DeepSpeed 引擎和 PEFT 的包装层。

        注入位置约定:
            layer_key → model.layers[layer_key - 1]

        Args:
            layer_key: 目标层 key（1-indexed，与 .pt 文件 key 一致）。

        Returns:
            目标层的 nn.Module 实例。
        """
        # 计算 0-indexed 层位置
        layer_idx = layer_key - 1

        # Step 1: 解包 DeepSpeed 引擎
        model = self.model
        if hasattr(model, 'module'):
            model = model.module

        # Step 2: 解包 PEFT/LoRA
        if hasattr(model, 'base_model'):
            inner = model.base_model
            if hasattr(inner, 'model'):
                inner = inner.model
        else:
            inner = model

        # Step 3: 定位 Transformer 层列表
        # 尝试多种常见的模型结构路径
        candidates = []

        # 路径 1: model.model.layers (Jiutian, Qwen, LLaMA 等)
        if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
            candidates.append(inner.model.layers)
        # 路径 2: model.layers (部分架构)
        if hasattr(inner, 'layers'):
            candidates.append(inner.layers)
        # 路径 3: model.transformer.h (GPT-2 风格)
        if hasattr(inner, 'transformer') and hasattr(inner.transformer, 'h'):
            candidates.append(inner.transformer.h)

        for layers in candidates:
            if layer_idx < len(layers):
                return layers[layer_idx]

        # 回退: 使用 get_submodule 方法
        model_root = self.model.module if hasattr(self.model, 'module') else self.model
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

    def attach(self):
        """注册所有目标层的 forward hooks。在干预前向传播前调用。"""
        for layer_key in self.target_layers:
            v_m = self.malicious_vectors[layer_key]
            layer_module = self._get_layer_module(layer_key)
            hook = layer_module.register_forward_hook(
                partial(
                    self._injection_hook,
                    v_m=v_m,
                    alpha=self.alpha,
                    adaptive=self.adaptive,
                    injection_mode=self.injection_mode,
                )
            )
            self._hooks.append(hook)

    def detach(self):
        """移除所有已注册的 forward hooks。在干预前向传播后立即调用。"""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
