import torch
import inspect
from typing import List, Union

class StateProjectionMonitor:
    """
    状态感知监听器：仅观测隐藏状态并计算在目标向量上的投影，绝对不干预梯度或隐状态。
    """
    def __init__(self, model, v_target: Union[torch.Tensor, List[torch.Tensor]], target_layer_key):
        self.model = model
        self.v_target = v_target
        self._is_multi_target = isinstance(v_target, (list, tuple))
        self.target_layer_key = target_layer_key
        self._hook_handle = None
        self._layer_module = None

    def _get_layer_module(self, layer_key: int) -> torch.nn.Module:
        """
        根据 layer_key 获取模型的实际 Transformer 层模块。
        自动处理 DeepSpeed 引擎和 PEFT 的包装层。
        """
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
        candidates = []
        if hasattr(inner, 'model') and hasattr(inner.model, 'layers'):
            candidates.append(inner.model.layers)
        if hasattr(inner, 'layers'):
            candidates.append(inner.layers)
        if hasattr(inner, 'transformer') and hasattr(inner.transformer, 'h'):
            candidates.append(inner.transformer.h)

        for layers in candidates:
            if hasattr(layers, '__len__') and layer_idx < len(layers):
                return layers[layer_idx]

        # Step 4: 回退机制 (Fallback) - 使用 get_submodule 字符串路径查找
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
            f"Monitor 无法定位到 layer_key={layer_key} (0-indexed={layer_idx}) 的模型层。"
            f"请检查 target_layers 参数是否与模型结构匹配。"
        )

    def attach(self):
        """挂载状态监听 Hook"""
        self._layer_module = self._get_layer_module(self.target_layer_key)
        self._hook_handle = self._layer_module.register_forward_hook(self._monitor_hook)

    def detach(self):
        """卸载状态监听 Hook"""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def get_current_projections(self):
        """获取当前 Batch 在目标向量上的投影结果"""
        if self._layer_module is None:
            return None
        return getattr(self._layer_module, '_current_projections', None)

    def _monitor_hook(self, module, input, output):
        """
        核心监控 Hook：
        提取 Labels Mask -> 提取回复部分的隐藏状态 -> 求均值 -> 计算投影并挂载到 Module
        """
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # 1. 动态向上溯源，寻找调用栈中的 labels 作为真实掩码
        mask = None
        frame = inspect.currentframe()
        try:
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
            del frame
        
        # 2. 缓存 Mask (应对某些模型将 Forward 拆分为多次执行的情况)
        if mask is not None:
            module._cached_monitor_mask = mask
        else:
            mask = getattr(module, '_cached_monitor_mask', None)
            
        if mask is None:
            # 如果依然找不到 mask，说明这可能是一次不带 labels 的推理 forward，跳过投影计算
            return output

        # 3. 计算在线投影
        if self._is_multi_target:
            target_vectors = [
                v.to(device=hidden_states.device, dtype=hidden_states.dtype)
                for v in self.v_target
            ]
            zero_template = target_vectors[0]
        else:
            v_m_aligned = self.v_target.to(device=hidden_states.device, dtype=hidden_states.dtype)
            target_vectors = [v_m_aligned]
            zero_template = v_m_aligned
        projections = []
        
        # 按 Batch 逐个样本计算 response 均值投影
        for i in range(hidden_states.size(0)):
            sample_mask = mask[i]
            sample_hidden = hidden_states[i][sample_mask]
            
            if sample_hidden.numel() > 0:
                h_mean = sample_hidden.mean(dim=0)
            else:
                h_mean = torch.zeros_like(zero_template)
                
            sample_projections = [torch.dot(h_mean, v_m) for v_m in target_vectors]
            proj = torch.stack(sample_projections)
            projections.append(proj)
        
        # 4. 将结果暂存于 module 实例中，供外部获取
        stacked_projections = torch.stack(projections)
        if not self._is_multi_target:
            stacked_projections = stacked_projections.squeeze(-1)
        module._current_projections = stacked_projections
        
        return output
