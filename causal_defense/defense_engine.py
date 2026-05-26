# Copyright (c) 2026
# 因果干预动态防御 — 核心防御引擎
#
# 实现基于因果干预的动态防御逻辑：
#   1. 自然前向传播（保留梯度）
#   2. 干预前向传播（注入恶意向量，无梯度）
#   3. 因果判定（ΔLoss 分析） → 梯度掩码
#
# 支持 sample 级和 token 级两种判定粒度。

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Sequence

try:
    from .hooks import MaliciousVectorInjector
except:
    from hooks import MaliciousVectorInjector

# ================================================================
# 防御配置（内部数据类，训练者通过 main_causal_defense.py argparse 配置）
# ================================================================

@dataclass
class CausalDefenseConfig:
    """因果干预动态防御配置。由 main_causal_defense.py 的 argparse 参数构造。"""
    malicious_vector_paths: List[str] = field(default_factory=list)
    target_layers: List[int] = field(default_factory=lambda: [30])
    injection_alpha: float = 1.0
    adaptive_alpha: bool = False
    defense_mode: str = "mask"     # "mask" or "inject"
    granularity: str = "sample"   # "sample" or "token" (仅 mode="mask" 时生效)
    sample_strategy: str = "mean"  # "mean" or "min" (仅 granularity="sample" 时生效)
    delta_threshold: float = 0.0
    log_delta_loss: bool = True
    vector_fusion_mode: int = 0

# ================================================================
# 向量加载工具（独立于防御引擎，可被其他模块直接调用）
# ================================================================

def _normalize_fusion_weights(
    fusion_weights: Optional[Sequence[float]],
    num_vectors: int,
) -> torch.Tensor:
    if fusion_weights is None:
        return torch.full((num_vectors,), 1.0 / max(num_vectors, 1), dtype=torch.float32)

    weights = torch.as_tensor(fusion_weights, dtype=torch.float32).flatten()
    if weights.numel() != num_vectors:
        raise ValueError(
            f"fusion_weights length must match vector count: "
            f"got {weights.numel()}, expected {num_vectors}"
        )
    if not torch.isfinite(weights).all():
        raise ValueError("fusion_weights must be finite.")

    weight_sum = weights.sum()
    if weight_sum.abs().item() < 1e-12:
        raise ValueError("fusion_weights sum must be non-zero.")
    return weights / weight_sum


def load_malicious_vector_files(paths: list) -> List[Any]:
    """Load vector files once so dynamic fusion can reuse the in-memory tensors."""
    all_loaded = []
    for path in paths:
        print(f"  [CausalDefense] 加载恶意向量: {path}")
        data = torch.load(path, weights_only=False, map_location='cpu')
        all_loaded.append(data)
        if isinstance(data, dict):
            print(f"    格式: dict, keys={list(data.keys())}")
        elif isinstance(data, torch.Tensor):
            print(f"    格式: Tensor, shape={data.shape}")
        else:
            print(f"    格式: {type(data)}")
    return all_loaded


def _get_vector_for_layer(data: Any, layer_key: int) -> torch.Tensor:
    try:
        vec = data[layer_key]
        if not isinstance(vec, torch.Tensor):
            raise TypeError(
                f"data[{layer_key}] 返回类型 {type(vec)}，需要 Tensor"
            )
        return vec
    except (KeyError, IndexError, TypeError) as e:
        if isinstance(data, dict):
            hint = f"可用的 dict keys: {list(data.keys())}"
        elif isinstance(data, torch.Tensor):
            hint = f"Tensor shape={data.shape}, 有效索引范围: 0~{data.shape[0]-1}"
        else:
            hint = f"不支持的数据类型: {type(data)}"
        raise KeyError(
            f"恶意向量文件中无法用 layer_key={layer_key} 索引。{hint}"
        ) from e


def fuse_loaded_vectors(
    all_loaded: List[Any],
    target_layers: list,
    fusion_mode: int = 0,
    fusion_weights: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Dict[int, torch.Tensor]:
    if fusion_mode not in [0, 1, 2]:
        raise ValueError("fusion_mode must be 0, 1, or 2.")
    if fusion_mode == 2 and fusion_weights is None:
        raise ValueError(
            "fusion_mode=2 requires fusion_weights because it represents "
            "weighted direction with weighted norm."
        )

    vectors = {}
    for layer_key in target_layers:
        layer_vectors = [_get_vector_for_layer(data, layer_key) for data in all_loaded]

        if len(layer_vectors) > 1:
            if fusion_mode in [0, 2]:
                # ========== 方案 A：L2 范数对齐（本文方法） ==========
                # fusion_mode=0: 加权方向 + 固定范数（默认）
                # fusion_mode=2: 加权方向 + 加权范数（消融）
                weights = _normalize_fusion_weights(
                    fusion_weights,
                    len(layer_vectors),
                ).to(
                    device=layer_vectors[0].device,
                    dtype=layer_vectors[0].dtype,
                )
                norms = [torch.norm(v, p=2) for v in layer_vectors]
                if fusion_mode == 0:
                    baseline_norm = sum(norms) / len(norms)
                else:
                    baseline_norm = sum(
                        norm * weight for norm, weight in zip(norms, weights)
                    )
                stacked_vectors = torch.stack(layer_vectors)
                summed_vector = torch.sum(
                    stacked_vectors
                    * weights.view(-1, *([1] * (stacked_vectors.dim() - 1))),
                    dim=0,
                )
                sum_norm = torch.norm(summed_vector, p=2)
                vectors[layer_key] = (summed_vector / (sum_norm + 1e-8)) * baseline_norm
            else:
                # ========== 方案 B：直接求平均（对比论文方法） ==========
                # 优点：注入量级与单一向量相当，绝对不会因多向量叠加导致激活值爆炸
                # 缺点：正交向量相加后范数天然收缩，干预信号会相应减弱
                comparison_stack = torch.stack(layer_vectors)
                vectors[layer_key] = comparison_stack.mean(dim=0)
        else:
            vectors[layer_key] = layer_vectors[0]

        if verbose:
            print(
                f"  [CausalDefense] Layer {layer_key}: "
                f"vector shape={vectors[layer_key].shape}, "
                f"norm={vectors[layer_key].norm().item():.4f}"
            )

    return vectors


def load_and_normalize_vectors(
    paths: list,
    target_layers: list,
    fusion_mode: int = 0,
    fusion_weights: Optional[Sequence[float]] = None,
    verbose: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    加载恶意向量文件并进行多向量融合。
    默认多向量时按等权融合；fusion_mode=0 走 L2 范数对齐（本文方法），
    fusion_mode=1 走直接平均（对比论文方法，不使用 fusion_weights），
    fusion_mode=2 走加权方向 + 加权范数（消融，必须传入 fusion_weights）。
    fusion_weights 用于 fusion_mode=0/2 的动态加权方向；fusion_mode=0 下融合后的
    范数固定为原始向量范数的无权重平均值。

    支持两种 .pt 文件格式:
      1. dict 格式: {layer_key: tensor_of_shape_[hidden_dim]}
      2. Tensor 格式: shape [num_layers, hidden_dim]，用 layer_key 索引

    Args:
        paths: 恶意向量 .pt 文件路径列表。
        target_layers: 目标层 key 列表。
        fusion_mode: 多向量融合策略，0=加权方向+固定范数（本文方法），
            1=直接平均（对比论文方法），2=加权方向+加权范数（消融）。
        fusion_weights: 可选的多向量融合方向权重，fusion_mode=0/2 使用。
            fusion_mode=2 必须显式传入；fusion_mode=0 不传时默认等权。

    Returns:
        {layer_key: normalized_vector_tensor} 字典。
    """
    all_loaded = load_malicious_vector_files(paths)
    return fuse_loaded_vectors(
        all_loaded,
        target_layers,
        fusion_mode=fusion_mode,
        fusion_weights=fusion_weights,
        verbose=verbose,
    )


# ================================================================
# 核心防御引擎
# ================================================================

class CausalDefenseEngine:
    """
    因果干预动态防御引擎。

    核心思路:
        将恶意向量 v_m 作为"反事实探针"注入模型中后层的隐状态，
        通过对比注入前后的交叉熵损失变化来判定微调数据的性质：
        - ΔLoss < 0: 恶意向量"协助"了拟合 → 攻击数据
        - ΔLoss ≥ 0: 恶意向量"阻碍"了拟合 → 正常数据

    防御模式:
        - mask:   动态掩码 — 恶意数据梯度归零（不学习）
        - inject: 动态注入 — 恶意数据带注入向量训练（产生"抗体"）
                  → 最坏情况等价于 training_norm.py 全量注入（已验证有效）

    前向传播:
        mask 模式 (2-pass):
          Pass 1 (WITH grad, NO hooks) → 常规前向传播
          Pass 2 (NO grad, WITH hooks) → 干预前向传播，仅用于判定
        inject 模式 (2-pass):
          Pass 1 (WITH grad, NO hooks) → 常规前向传播
          Pass 2 (WITH grad, WITH hooks) → 注入前向传播，用于恶意数据 backward
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: CausalDefenseConfig,
        use_lora: bool = False,
    ):
        """
        Args:
            model: DeepSpeed 引擎包装后的模型。
            config: 防御配置。
            use_lora: 是否使用 LoRA（影响 loss 提取方式）。
        """
        self.model = model
        self.config = config
        self.use_lora = use_lora

        # 加载并预处理恶意向量
        self.malicious_vectors = load_and_normalize_vectors(
            config.malicious_vector_paths,
            config.target_layers,
            fusion_mode=config.vector_fusion_mode,
        )

        # 初始化 Hook 注入器
        self.injector = MaliciousVectorInjector(
            model=model,
            malicious_vectors=self.malicious_vectors,
            target_layers=config.target_layers,
            alpha=config.injection_alpha,
            adaptive=config.adaptive_alpha,
        )

        # 运行时统计
        self.total_samples: int = 0
        self.blocked_samples: int = 0
        self.total_tokens: int = 0
        self.blocked_tokens: int = 0

    # ----------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------

    def _extract_loss(self, outputs):
        """从模型输出中提取 loss，兼容 LoRA 和非 LoRA 模式。"""
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            return outputs.loss
        return outputs[0]

    def _extract_logits(self, outputs):
        """从模型输出中提取 logits。"""
        if hasattr(outputs, 'logits') and outputs.logits is not None:
            return outputs.logits
        return outputs[1]

    # ----------------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------------

    def compute_defense_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        执行一次完整的因果干预判定步骤。

        Args:
            batch: 包含 input_ids, attention_mask, labels 的字典。

        Returns:
            loss: 用于 model.backward(loss) 的损失张量。
            info: 诊断信息字典，包含 delta_loss, is_malicious 等。
        """
        if self.config.defense_mode == "inject":
            return self._inject_defense(batch)
        elif self.config.granularity == "sample":
            return self._sample_level_defense(batch)
        else:
            return self._token_level_defense(batch)

    # ================================================================
    # Sample 级别防御
    # ================================================================

    def _sample_level_defense(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """样本级防御路由：根据 sample_strategy 选择判定策略。"""
        if self.config.sample_strategy == "min":
            return self._sample_min_strategy(batch)
        else:
            return self._sample_mean_strategy(batch)

    # ----------------------------------------------------------------
    # 策略 1: mean（默认）— 标量路径，对 ΔLoss 求平均后判定
    # ----------------------------------------------------------------

    def _sample_mean_strategy(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        mean 策略：每个 sample 独立走高效标量路径判定。

        ΔLoss_scalar = mean(ΔLoss_per_token) = Loss_inj_scalar - Loss_nat_scalar
        直接比较模型返回的标量 loss，无需存储 logits。
        """
        B = batch['input_ids'].shape[0]

        normal_losses = []
        all_natural_losses = []  # 记录每个样本的自然 loss（不受 mask 影响）
        all_deltas = []
        num_malicious = 0
        last_loss_nat = None
        total_valid_tokens = 0
        blocked_valid_tokens = 0

        for i in range(B):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}

            # 统计该样本的有效 token 数
            valid_count = (single_batch['labels'] != -100).sum().item()
            total_valid_tokens += valid_count

            # Pass 1: 自然前向传播（保留梯度）
            outputs_nat = self.model(**single_batch, use_cache=False)
            loss_nat = self._extract_loss(outputs_nat)
            last_loss_nat = loss_nat
            all_natural_losses.append(loss_nat.detach().item())

            # Pass 2: 干预前向传播（无梯度）
            with torch.no_grad():
                self.injector.attach()
                outputs_inj = self.model(**single_batch, use_cache=False)
                loss_inj = self._extract_loss(outputs_inj)
                self.injector.detach()

            delta = (loss_inj - loss_nat).item()
            all_deltas.append(delta)

            if delta < self.config.delta_threshold:
                num_malicious += 1
                blocked_valid_tokens += valid_count
            else:
                normal_losses.append(loss_nat)

        self.total_samples += B
        self.blocked_samples += num_malicious
        self.total_tokens += total_valid_tokens
        self.blocked_tokens += blocked_valid_tokens

        if len(normal_losses) > 0:
            loss = torch.stack(normal_losses).mean()
        else:
            loss = last_loss_nat * 0.0

        info = {
            "natural_loss": sum(all_natural_losses) / len(all_natural_losses),
            "delta_loss": sum(all_deltas) / len(all_deltas),
            "is_malicious": num_malicious > 0,
            "blocked_ratio_samples": self.blocked_samples / max(self.total_samples, 1),
            "blocked_ratio_tokens": self.blocked_tokens / max(self.total_tokens, 1),
            "num_malicious_in_batch": num_malicious,
        }
        return loss, info

    # ----------------------------------------------------------------
    # 策略 2: min — 任一 token 的 ΔLoss < threshold 即判定恶意
    # ----------------------------------------------------------------

    def _sample_min_strategy(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        min 策略：只要序列中任一有效 token 的 ΔLoss < threshold，
        即判定整个样本为恶意。

        解决 mean 策略的信号稀释问题：
            当攻击数据中恶意 token 只占序列的一小部分时，
            mean(ΔLoss) 仍可能 > 0（被正常 token 稀释），
            但 min(ΔLoss) 能捕捉到那些被恶意向量"协助"的关键 token。

        注意: 此策略需提取 logits 计算 per-token loss，
        显存开销高于 mean 策略（需 [1, S, V] 的 logits）。
        """
        B = batch['input_ids'].shape[0]

        normal_losses = []
        all_natural_losses = []
        all_min_deltas = []
        num_malicious = 0
        last_loss_nat = None
        total_valid_tokens = 0
        blocked_valid_tokens = 0

        for i in range(B):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}

            # ===== Pass 1: 自然前向传播（保留梯度） =====
            outputs_nat = self.model(**single_batch, use_cache=False)
            loss_nat_scalar = self._extract_loss(outputs_nat)  # 标量 loss（用于 backward）
            logits_nat = self._extract_logits(outputs_nat)      # logits（用于 per-token 判定）
            last_loss_nat = loss_nat_scalar
            all_natural_losses.append(loss_nat_scalar.detach().item())

            # ===== Pass 2: 干预前向传播（无梯度） =====
            with torch.no_grad():
                self.injector.attach()
                outputs_inj = self.model(**single_batch, use_cache=False)
                logits_inj = self._extract_logits(outputs_inj)
                self.injector.detach()

            # ===== 计算 per-token ΔLoss（无梯度，仅用于判定） =====
            with torch.no_grad():
                labels = single_batch['labels']
                shift_logits_nat = logits_nat[:, :-1, :].contiguous()
                shift_logits_inj = logits_inj[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                _, S, V = shift_logits_nat.shape

                ptl_nat = F.cross_entropy(
                    shift_logits_nat.view(-1, V), shift_labels.view(-1),
                    reduction='none', ignore_index=-100,
                ).view(1, S)

                ptl_inj = F.cross_entropy(
                    shift_logits_inj.view(-1, V), shift_labels.view(-1),
                    reduction='none', ignore_index=-100,
                ).view(1, S)

                # 有效 token 掩码（非 padding、非 prompt）
                valid_mask = (shift_labels != -100).squeeze(0)  # [S]
                delta_per_token = (ptl_inj - ptl_nat).squeeze(0)  # [S]
                valid_count = valid_mask.sum().item()

                valid_deltas = delta_per_token[valid_mask]
                if valid_deltas.numel() > 0:
                    min_delta = valid_deltas.min().item()
                else:
                    min_delta = float('inf')  # 没有有效 token，视为正常

            # 释放 logits 显存
            del logits_nat, logits_inj, shift_logits_nat, shift_logits_inj

            total_valid_tokens += valid_count
            all_min_deltas.append(min_delta)

            if min_delta < self.config.delta_threshold:
                num_malicious += 1
                blocked_valid_tokens += valid_count
            else:
                normal_losses.append(loss_nat_scalar)

        # ===== 汇总 =====
        self.total_samples += B
        self.blocked_samples += num_malicious
        self.total_tokens += total_valid_tokens
        self.blocked_tokens += blocked_valid_tokens

        if len(normal_losses) > 0:
            loss = torch.stack(normal_losses).mean()
        else:
            loss = last_loss_nat * 0.0

        info = {
            "natural_loss": sum(all_natural_losses) / len(all_natural_losses),
            "delta_loss": sum(all_min_deltas) / len(all_min_deltas),
            "is_malicious": num_malicious > 0,
            "blocked_ratio_samples": self.blocked_samples / max(self.total_samples, 1),
            "blocked_ratio_tokens": self.blocked_tokens / max(self.total_tokens, 1),
            "num_malicious_in_batch": num_malicious,
        }
        return loss, info

    # ================================================================
    # Token 级别防御
    # ================================================================

    def _token_level_defense(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Token 级别防御：逐 token 判定 ΔLoss，仅屏蔽恶意 token 的梯度。

        注意: 需存储完整 logits [B, S, V]，长序列时显存消耗较高。
        """
        # ===== Pass 1: 自然前向传播（保留梯度） =====
        outputs_natural = self.model(**batch, use_cache=False)
        logits_natural = self._extract_logits(outputs_natural)

        # ===== Pass 2: 干预前向传播（无梯度） =====
        with torch.no_grad():
            self.injector.attach()
            outputs_injected = self.model(**batch, use_cache=False)
            logits_injected = self._extract_logits(outputs_injected)
            self.injector.detach()

        # ===== 计算 per-token 损失 =====
        labels = batch['labels']
        shift_logits_nat = logits_natural[:, :-1, :].contiguous()
        shift_logits_inj = logits_injected[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        B, S, V = shift_logits_nat.shape

        per_token_loss_nat = F.cross_entropy(
            shift_logits_nat.view(-1, V), shift_labels.view(-1),
            reduction='none', ignore_index=-100,
        ).view(B, S)

        with torch.no_grad():
            per_token_loss_inj = F.cross_entropy(
                shift_logits_inj.view(-1, V), shift_labels.view(-1),
                reduction='none', ignore_index=-100,
            ).view(B, S)

            valid_mask = (shift_labels != -100).float()
            delta_per_token = per_token_loss_inj - per_token_loss_nat.detach()

            # 逐 token 判定
            token_is_normal = (delta_per_token >= self.config.delta_threshold)
            grad_mask = token_is_normal.float()

        # ===== 应用梯度掩码 =====
        effective_mask = grad_mask * valid_mask
        num_active = effective_mask.sum().clamp(min=1)
        total_valid = valid_mask.sum().clamp(min=1)
        loss = (per_token_loss_nat * effective_mask).sum() / num_active

        # ===== 统计 =====
        num_blocked_tokens = (valid_mask.sum() - effective_mask.sum()).item()
        total_valid_tokens = valid_mask.sum().item()
        avg_delta = (delta_per_token * valid_mask).sum().item() / total_valid.item()

        # 自然 loss（不受 mask 影响的原始 loss）
        natural_loss = (per_token_loss_nat.detach() * valid_mask).sum().item() / total_valid.item()

        self.total_tokens += int(total_valid_tokens)
        self.blocked_tokens += int(num_blocked_tokens)
        self.total_samples += B

        # 只要有任一 token 被屏蔽，就将该样本计入 blocked_samples
        is_malicious = num_blocked_tokens > 0
        if is_malicious:
            self.blocked_samples += B

        info = {
            "natural_loss": natural_loss,
            "delta_loss": avg_delta,
            "is_malicious": is_malicious,
            "blocked_ratio_samples": self.blocked_samples / max(self.total_samples, 1),
            "blocked_ratio_tokens": self.blocked_tokens / max(self.total_tokens, 1),
            "num_malicious_in_batch": num_malicious,
        }

        del logits_injected, shift_logits_inj, per_token_loss_inj
        return loss, info

    # ================================================================
    # 动态注入防御（方案 2）
    # ================================================================

    def _inject_defense(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        动态注入防御：
          - ΔLoss < 0（恶意）: 使用注入后的 loss 进行 backward，
            模型带着注入向量训练，产生"抗体"效应。
          - ΔLoss ≥ 0（正常）: 使用自然 loss 进行 backward，正常训练。

        与 mask 模式的关键区别:
          mask:   恶意数据 → 梯度归零，不学习
          inject: 恶意数据 → 带向量训练，学会抵抗

        容错性:
          最坏情况（全部判定为恶意）= 全量注入训练 = training_norm.py（已验证有效）
          最好情况（精准区分）= 仅对恶意数据注入，正常数据零开销

        注意: Pass 2 需要梯度（不能用 no_grad），显存峰值略高于 mask 模式。
        """
        B = batch['input_ids'].shape[0]

        losses = []
        all_natural_losses = []
        all_deltas = []
        num_injected = 0
        total_valid_tokens = 0
        injected_valid_tokens = 0

        for i in range(B):
            single_batch = {k: v[i:i+1] for k, v in batch.items()}

            # 统计该样本的有效 token 数
            valid_count = (single_batch['labels'] != -100).sum().item()
            total_valid_tokens += valid_count

            with torch.no_grad():
                # 试探自然 loss
                out_nat_probe = self.model(**single_batch, use_cache=False)
                loss_nat_probe = self._extract_loss(out_nat_probe)
                
                # 试探注入 loss
                self.injector.attach()
                out_inj_probe = self.model(**single_batch, use_cache=False)
                loss_inj_probe = self._extract_loss(out_inj_probe)
                self.injector.detach()

            # ===== 因果判定 + 选择 loss =====
            delta = (loss_inj - loss_nat).detach().item()
            all_deltas.append(delta)
            all_natural_losses.append(loss_nat_probe.item())

            if delta < self.config.delta_threshold:
                # 恶意数据: 用注入后的 loss 训练（产生抗体）
                num_injected += 1
                injected_valid_tokens += valid_count
                self.injector.attach()
                out_inj_real = self.model(**single_batch, use_cache=False)
                losses.append(self._extract_loss(out_inj_real))
                self.injector.detach()
            else:
                # 正常数据: 用自然 loss 正常训练
                out_nat_real = self.model(**single_batch, use_cache=False)
                losses.append(self._extract_loss(out_nat_real))

        # ===== 汇总 =====
        self.total_samples += B
        self.blocked_samples += num_injected
        self.total_tokens += total_valid_tokens
        self.blocked_tokens += injected_valid_tokens

        loss = torch.stack(losses).mean()

        info = {
            "natural_loss": sum(all_natural_losses) / len(all_natural_losses),
            "delta_loss": sum(all_deltas) / len(all_deltas),
            "is_malicious": num_injected > 0,
            "blocked_ratio_samples": self.blocked_samples / max(self.total_samples, 1),
            "blocked_ratio_tokens": self.blocked_tokens / max(self.total_tokens, 1),
            "num_malicious_in_batch": num_injected,
        }
        return loss, info

    def reset_stats(self):
        """重置运行时统计计数器。"""
        self.total_samples = 0
        self.blocked_samples = 0
        self.total_tokens = 0
        self.blocked_tokens = 0
