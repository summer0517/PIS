# conda #!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
#
# ====================================================================
# 基于因果干预的动态防御训练脚本
# 在 main_llama.bak.py 的基础上，新增了因果干预动态防御机制。
# 标注 ">>> DEFENSE >>>" 和 "<<< DEFENSE <<<" 的代码块为新增内容。
# ====================================================================

import argparse
import collections
import os
import math
import sys
import json
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter   
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from patch import replace_unpad_data_for_block_diag_attn
#replace_unpad_data_for_block_diag_attn()
#from collator import DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    AutoConfig,
    DataCollatorForSeq2Seq,
    # Gemma3ForConditionalGeneration
)
from transformers.integrations import HfDeepSpeedConfig

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model_new
from utils.ds_utils import get_train_ds_config

# >>> DEFENSE >>> 导入因果干预防御模块
from causal_defense import (
    CausalDefenseConfig, CausalDefenseEngine,
    MaliciousVectorInjector, StateProjectionMonitor,
    ImmuneDeltaPreserver, ImmuneVectorContinuationInjector,
    GradientAnalyzer, InjectionGradientProbeDefense,
    load_and_normalize_vectors,
)
# <<< DEFENSE <<<


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    #tokenizer.pad_token_id = 151643
    #tokenizer.eos_token_id = 151643
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    #from liger_kernel.transformers import AutoLigerKernelForCausalLM
    #model = AutoLigerKernelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    return tokenizer, model

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task.")

    # 模型及数据集参数
    parser.add_argument("--model_type", type=str, help="模型类型", default="")

    parser.add_argument("--model_name_or_path", type=str, help="模型文件地址", default="")
    parser.add_argument("--train_data", nargs='+', help="训练数据集地址，格式为处理好的datasets", default=[""])
    parser.add_argument("--summary", type=str, help="每次训练的简单摘要记录信息", default="")
    parser.add_argument("--output_dir", help="模型保存的路径", type=str, default="")
    parser.add_argument("--job_name", help="任务名称", type=str, default="")

    # 训练超参数
    parser.add_argument("--lora_config", type=str, help="Lora配置文件地址", default="")
    parser.add_argument("--num_train_epochs", help="总训练轮数", type=int, default=1)
    parser.add_argument("--num_warmup_steps", help="warmup步数", type=int, default=10)
    parser.add_argument("--max_seq_len", help="训练过程中的最长序列长度", type=int, default=65536)
    parser.add_argument("--learning_rate", help="训练过程中的最大学习率", type=float, default=5e-6) ## 1e-5
    parser.add_argument("--weight_decay", help="训练过程中的权重衰减", type=float, default=1e-4)  ## 1e-4
    parser.add_argument("--gradient_accumulation_steps", help="梯度累加步数", type=int, default=4)  ## 4
    parser.add_argument("--lr_scheduler_type", help="学习率变化策略", type=SchedulerType, default="cosine")
    parser.add_argument("--per_device_train_batch_size", help="训练时，单卡的batch_size大小", type=int, default=1)

    # 通用参数
    parser.add_argument("--seed", help="随机种子设定", type=int, default=1234)
    parser.add_argument("--local_rank", help="rank大小", type=int, default=int(os.environ['LOCAL_RANK']))
    parser.add_argument("--use_noise", help="是否启用noise训练，目前只针对LlamaModel", action='store_true', default=False)

    # 训练保存 / 断点续训参数
    parser.add_argument("--save_interval", help="模型保存间隔", type=int, default=200)
    parser.add_argument("--resume_step", help="训练恢复的模型步数", type=str, default="")
    parser.add_argument("--save_checkpoint", help="是否要保存优化器状态，断点续训需要保存", action='store_true', default=True)

    # 梯段检查点
    parser.add_argument('--gradient_checkpointing', help="梯度检查点", action='store_true', default=True)

    # deepspeed参数
    parser.add_argument('--offload', help="是否要将参数/优化器卸载到cpu", action='store_true', default=False)
    parser.add_argument('--zero_stage', help="ZERO策略", type=int, default=3)

    parser.add_argument("--dynamic_tokenize", action='store_true', help="如果开启，则直接读取 jsonl 进行动态 Tokenize；如果不开启，则读取已保存到磁盘的 dataset")

    # >>> DEFENSE >>> 因果干预动态防御参数
    # --- 独立功能开关 (可任意组合) ---
    parser.add_argument("--enable_causal_defense", action='store_true', default=False,
                        help="启用动态防御 (基于 ΔLoss 的检测 + mask/inject)")
    parser.add_argument("--enable_unconditional_injection", action='store_true', default=False,
                        help="启用无条件注入训练 (等价于 training_norm.py, 每步都注入向量)")
    parser.add_argument("--enable_gradient_analysis", action='store_true', default=False,
                        help="启用梯度/激活分析 (周期性采集指标到 TensorBoard)")
    parser.add_argument("--enable_immune_delta_preservation", action='store_true', default=False,
                        help="启用免疫位移保持: 前K步注入，K后撤掉注入并保护参数免疫delta")

    # --- 向量与注入参数 (上述任一功能启用时需要) ---
    parser.add_argument("--malicious_vector_paths", nargs='+', type=str, default=[],
                        help="恶意向量 .pt 文件路径列表。")
    parser.add_argument("--defense_target_layers", nargs='+', type=int, default=[30],
                        help="恶意向量注入的目标层 key (1-indexed, 与 .pt 文件 key 对应)")
    parser.add_argument("--defense_alpha", type=float, default=1.0,
                        help="注入强度系数 α，控制 h' = h + α · v_m")
    parser.add_argument("--defense_adaptive_alpha", action='store_true', default=False,
                        help="启用基于层激活范数的自适应 α 缩放")
    parser.add_argument("--injection_mode", type=str, choices=["res_only", "all_token"], default="res_only",
                        help="恶意向量的注入范围模式: 'res_only' (默认，仅对回复部分注入，保护理解能力) 或 'all_token' (对全部Token无差别注入)")
    parser.add_argument("--vector_fusion_mode", type=int, choices=[0, 1], default=0,
                            help="多恶意向量融合策略: 0=L2范数对齐(默认,与原逻辑一致), 1=直接求平均(更安全但信号可能偏弱)")
    # --- 免疫位移保持专用参数 (仅 --enable_immune_delta_preservation 时生效) ---
    parser.add_argument("--immune_boundary_step", type=int, default=-1,
                        help="免疫边界步数K: 在该micro-batch step后记录θ_K并撤掉注入。-1时表示复用disable_defense_step；若两者都未提供则报错。")
    parser.add_argument("--immune_param_scope", type=str, choices=["target_core", "target_all_linear"], default="target_core",
                        help="免疫参数范围: target_core=仅保护o_proj/down_proj, target_all_linear=保护目标层所有Linear权重")
    parser.add_argument("--immune_min_delta_norm", type=float, default=1e-12,
                        help="免疫delta的最小范数阈值, 小于该值则不纳入保护")
    parser.add_argument("--immune_exact_distributed_projection", action='store_true', default=True,
                        help="启用分布式精确投影: 对梯度/免疫delta的点积做all-reduce")
    parser.add_argument("--immune_include_input_embeddings", action='store_true', default=False,
                        help="是否将输入embedding也纳入免疫位移保持。默认关闭，避免过强约束通用能力。")
    parser.add_argument("--immune_preservation_strategy", type=str, choices=["gradient_projection", "immune_continuation"], default="gradient_projection",
                        help="K步后的免疫保持策略: gradient_projection=旧的梯度投影; immune_continuation=从ΔW提取内生免疫向量并继续注入")
    parser.add_argument("--immune_projection_mode", type=str, choices=["flat", "svd_subspace"], default="svd_subspace",
                        help="免疫梯度投影方式: flat=保护展平后的1D delta方向; svd_subspace=保护delta诱导的低秩输入/输出特征子空间")
    parser.add_argument("--immune_svd_rank", type=int, default=8,
                        help="svd_subspace模式下每个权重矩阵保留的免疫子空间秩")
    parser.add_argument("--immune_svd_oversample", type=int, default=4,
                        help="svd_subspace模式下 randomized sketch 的过采样维度")
    parser.add_argument("--immune_projection_strength", type=float, default=1.0,
                        help="免疫投影强度: 1.0表示完全删除免疫子空间内梯度, 0.5表示删除一半")
    parser.add_argument("--immune_allow_svd_partial", action='store_true', default=False,
                        help="允许svd_subspace模式下部分tensor缺失basis并跳过。默认不允许, 防止实验静默退回或部分生效。")
    parser.add_argument("--immune_antibody_rank", type=int, default=1,
                        help="immune_continuation模式下从每个ΔW中提取的左奇异向量秩")
    parser.add_argument("--immune_print_svd_energy_topk", type=int, default=5,
                        help="immune_continuation模式下打印前K个奇异值平方及能量占比")
    parser.add_argument("--immune_antibody_modules", type=str, default="o_proj,down_proj",
                        help="immune_continuation模式下启用的写矩阵模块，v1仅支持o_proj,down_proj")
    parser.add_argument("--immune_calibration_micro_batches", type=int, default=4,
                        help="immune_continuation模式下用于功能方向校准的micro-batch数")
    parser.add_argument("--immune_calibration_min_response_tokens", type=int, default=2048,
                        help="immune_continuation模式下触发最终定型所需的最少response token数")
    parser.add_argument("--immune_antibody_source", type=str, choices=["svd_u", "functional_mean"], default="functional_mean",
                        help="immune_continuation模式下的方向来源: svd_u=裸SVD, functional_mean=基于K后校准batch的功能位移均值")
    parser.add_argument("--immune_continuation_scale_mode", type=str, choices=["match_v_preserve_ratio", "match_v_equal_share", "raw"], default="match_v_preserve_ratio",
                        help="immune_continuation续注入缩放: match_v_preserve_ratio=每层总范数对齐v且保留模块范数比例; match_v_equal_share=旧版等分; raw=不对齐v范数")
    # --- 动态防御专用参数 (仅 --enable_causal_defense 时生效) ---
    parser.add_argument("--defense_mode", type=str, choices=["mask", "inject"], default="mask",
                        help="防御模式: mask=梯度归零; inject=带向量训练")
    parser.add_argument("--defense_granularity", type=str, choices=["sample", "token"], default="sample",
                        help="判定粒度: sample 或 token")
    parser.add_argument("--defense_sample_strategy", type=str, choices=["mean", "min"], default="mean",
                        help="sample 粒度下的判定策略: mean 或 min")
    parser.add_argument("--defense_delta_threshold", type=float, default=0.0,
                        help="ΔLoss 判定阈值。ΔLoss < threshold → 恶意")

    # --- 新增：状态感知动态防御 (State-Aware Defense) ---
    parser.add_argument("--enable_state_aware_defense", action='store_true', default=False,
                        help="启用基于状态监控的自适应防御 (需要离线计算好的 ref_projection 作为数据列)")
    parser.add_argument("--enable_injection_gradient_probe_defense", action='store_true', default=False,
                        help="启用注入梯度探针防御: 对比有/无注入时梯度与 v_m 的 cos/proj 是否同时放大一个数量级")
    parser.add_argument("--gradient_probe_order_factor", type=float, default=10.0,
                        help="【保留日志观察用】注入/无注入梯度指标的数量级比例参考，不再参与恶意判定")
    parser.add_argument("--gradient_probe_cos_threshold", type=float, default=0.12,
                        help="注入梯度探针判定阈值: injected grad 与 v_m 的 cos 超过该值")
    parser.add_argument("--gradient_probe_proj_threshold", type=float, default=1e-8,
                        help="注入梯度探针判定阈值: injected grad 在 v_m 上的投影超过该值")
    parser.add_argument("--gradient_probe_perturb_alpha", type=float, default=None,
                        help="曲率探针扰动步长 a: h' = h + a * v_m。不填则复用 --defense_alpha")
    parser.add_argument("--gradient_probe_diff_epsilon", type=float, default=None,
                        help="曲率探针有限差分除数 b: (g_inj - g_nat) / b。不填则 b=a，即严格 Hessian 近似")

    # --- 分析参数 (仅 --enable_gradient_analysis 时生效) ---
    parser.add_argument("--defense_analysis_interval", type=int, default=100,
                        help="梯度/激活分析的采集间隔 (step 数)。建议值: 50-200")

    # 【新增代码】消融实验：撤药测试参数
    parser.add_argument("--disable_defense_step", type=int, default=-1,
                        help="【消融实验专用】在达到该步数后，强制关闭所有注入防御逻辑，恢复为无干预正常训练。-1表示不关闭。")
    parser.add_argument("--enable_old_step_count", action='store_true', default=True,
                        help="【消融实验专用】启用与jiutian实验对齐的step计数方式，2台机器：[100,200,400,800]；4台机器：[50,100,200,400]")
    parser.add_argument("--defense_decline_alpha", action='store_true', default=False,
                        help="启用达到step后，α 的渐进衰退")
    parser.add_argument("--defense_rise_alpha", action='store_true', default=False,
                        help="启用达到step后，α 的渐进上升，最终涨到 2α")
    parser.add_argument("--defense_decline_start_step", type=int, default=0,
                        help="配合 --defense_decline_alpha / --defense_rise_alpha 使用：前 N 个 micro-batch 保持原 α，从第 N+1 步开始线性变化，到当前 epoch 末尾分别降为 0 或升为 2α。")
    # <<< DEFENSE <<<

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def get_declined_alpha(base_alpha, current_epoch_step, total_steps, start_step):
    """
    线性衰减 defense alpha。

    current_epoch_step 使用 1-based micro-batch step：
      - current_epoch_step <= start_step 时保持 base_alpha
      - current_epoch_step == total_steps 时降为 0
    """
    if total_steps <= 0:
        return base_alpha

    start_step = max(0, start_step)
    if current_epoch_step <= start_step:
        return base_alpha
    if start_step >= total_steps:
        return 0.0

    remaining_ratio = (total_steps - current_epoch_step) / (total_steps - start_step)
    return base_alpha * max(0.0, remaining_ratio)


def get_risen_alpha(base_alpha, current_epoch_step, total_steps, start_step):
    """
    线性上升 defense alpha。

    current_epoch_step 使用 1-based micro-batch step：
      - current_epoch_step <= start_step 时保持 base_alpha
      - current_epoch_step == total_steps 时升到 2 * base_alpha
    """
    if total_steps <= 0:
        return base_alpha

    start_step = max(0, start_step)
    if current_epoch_step <= start_step:
        return base_alpha
    if start_step >= total_steps:
        return 2.0 * base_alpha

    rise_ratio = (current_epoch_step - start_step) / (total_steps - start_step)
    return base_alpha * (1.0 + max(0.0, min(1.0, rise_ratio)))


def get_scheduled_alpha(base_alpha, current_epoch_step, total_steps, start_step, mode):
    if mode == "decline":
        return get_declined_alpha(base_alpha, current_epoch_step, total_steps, start_step)
    if mode == "rise":
        return get_risen_alpha(base_alpha, current_epoch_step, total_steps, start_step)
    return base_alpha


def set_runtime_defense_alpha(
    alpha,
    defense_engine=None,
    standalone_injector=None,
    immune_injector=None,
    immune_continuation_injector=None,
    gradient_probe_defense=None,
    gradient_analyzer=None,
):
    """更新当前 step 要用的注入强度。hook attach 时会读取这些对象上的 alpha。"""
    if defense_engine is not None:
        defense_engine.config.injection_alpha = alpha
        defense_engine.injector.alpha = alpha
    if standalone_injector is not None:
        standalone_injector.alpha = alpha
    if immune_injector is not None:
        immune_injector.alpha = alpha
    if immune_continuation_injector is not None:
        immune_continuation_injector.alpha = alpha
    if gradient_probe_defense is not None:
        gradient_probe_defense.alpha = alpha
    if gradient_analyzer is not None and hasattr(gradient_analyzer, "_analysis_injector"):
        gradient_analyzer._analysis_injector.alpha = alpha


def get_tokenized_dataset(data_paths, tokenizer, max_seq_len):
    IGNORE_INDEX = -100
    def generate_and_tokenize_prompt_v5(data_point):
        content = data_point["messages"][0]["content"]
        curres = data_point["messages"][1]["content"]
        
        content_ids = tokenizer('<|im_start|>user\n'+content+'<|im_end|>\n').input_ids
        curres_ids = tokenizer('<|im_start|>assistant\n'+curres+'<|im_end|>\n').input_ids
        
        input_ids = content_ids + curres_ids
        
        labels = tokenizer('<|im_start|>').input_ids + [IGNORE_INDEX]*(len(content_ids)-3) + tokenizer('<|im_end|>\n').input_ids
        labels += tokenizer('<|im_start|>').input_ids + [IGNORE_INDEX]*2 + curres_ids[3:-2] + tokenizer('<|im_end|>\n').input_ids
        
        assert len(input_ids) == len(labels)
        
        # Padding
        input_ids += [tokenizer.pad_token_id] * (max_seq_len - len(input_ids))
        labels += [IGNORE_INDEX] * (max_seq_len - len(labels))
        
        # Truncation
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
            
        # 注意：在 datasets.map() 中最好返回原生 Python list，不要直接返回 torch.tensor，
        # 否则可能会导致多进程处理时序列化报错。我们会在 map 之后统一 set_format("torch")
        attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    # 1. 动态加载所有传入的路径
    raw_datasets = []
    for path in data_paths:
        print(f"正在加载数据集: {path}")
        ds = load_dataset("json", data_files=path, split="train")
        raw_datasets.append(ds)

    # 2. 合并数据集
    if len(raw_datasets) > 1:
        combined_dataset = concatenate_datasets(raw_datasets)
    else:
        combined_dataset = raw_datasets[0]

    # 3. 执行 Tokenize (使用你设定的 num_proc=16 进行多进程加速)
    print("开始进行 Tokenize 处理...")
    tokenized_dataset = combined_dataset.shuffle(seed=42).map(
        generate_and_tokenize_prompt_v5,
        num_proc=16,
        remove_columns=combined_dataset.column_names,
        desc="Tokenizing dataset"
    )

    # 4. 将输出格式转换为 PyTorch Tensors，供 DataLoader 使用
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_dataset

def get_current_lr(optimizer):
    visited = set()

    def _walk(opt):
        if opt is None or id(opt) in visited:
            return None
        visited.add(id(opt))

        param_groups = getattr(opt, "param_groups", None)
        if isinstance(param_groups, list) and len(param_groups) > 0:
            return param_groups[0].get("lr", None)

        for attr in ("optimizer", "optim", "base_optimizer", "basic_optimizer"):
            child = getattr(opt, attr, None)
            if child is not None:
                lr = _walk(child)
                if lr is not None:
                    return lr

        child_optimizers = getattr(opt, "optimizers", None)
        if isinstance(child_optimizers, (list, tuple)):
            for child in child_optimizers:
                lr = _walk(child)
                if lr is not None:
                    return lr
        return None

    return _walk(optimizer)


def main():
    args = parse_args()

    # 分布式初始化
    if args.local_rank == -1: 
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    # 创建 deepspeed配置信息
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage
                                    )
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    ds_config["tensorboard"] = {
        "enabled": True,
        "output_path": args.output_dir,
        "job_name": args.job_name
    }

    # 参数保存 与 tensorboard 配置
    if args.global_rank <= 0:
        log_dir = args.output_dir
        sw = SummaryWriter(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(log_dir + "/args.json", "w") as f:
            json.dump(args.__dict__, f, indent=4)
        with open(log_dir + "/ds_config.json", "w") as f:
            json.dump(ds_config, f, indent=4)

    # 设置随机种子
    set_random_seed(args.seed)

    torch.distributed.barrier()
    
    # 模型与分词器加载
    print_rank_0("model_name_or_path : " + args.model_name_or_path, args.global_rank)
    dschf = HfDeepSpeedConfig(ds_config)
    tokenizer, model = load_model_and_tokenizer(args)
    print("***** Model load success! *****")

    # 梯度检查点
    if args.gradient_checkpointing:
        if args.enable_injection_gradient_probe_defense:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_enable()

    # Lora配置
    if args.lora_config:
        lora_config = json.load(open(args.lora_config))
        config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            modules_to_save=lora_config["modules_to_save"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # 数据集加载
    if args.dynamic_tokenize:
        train_dataset = get_tokenized_dataset(
            data_paths=args.train_data,
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len
        )
    else:
        all_datasets = []
        for ds_path in args.train_data:
            all_datasets.append(Dataset.load_from_disk(ds_path))
        train_dataset = concatenate_datasets(all_datasets)

    print_rank_0("***** Data load success! *****", args.global_rank)

    # DataLoader 配置
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, padding=True),
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)

    if args.defense_decline_alpha and args.defense_rise_alpha:
        raise ValueError("--defense_decline_alpha and --defense_rise_alpha cannot be enabled at the same time.")
    if args.defense_rise_alpha and args.disable_defense_step > 0:
        raise ValueError("--defense_rise_alpha cannot be used together with --disable_defense_step.")

    if args.defense_decline_alpha or args.defense_rise_alpha:
        if args.defense_decline_start_step < 0:
            raise ValueError("--defense_decline_start_step must be >= 0 when alpha scheduling is enabled.")
        if args.defense_decline_start_step >= len(train_dataloader):
            raise ValueError("--defense_decline_start_step must be < len(train_dataloader) when alpha scheduling is enabled.")

    # 优化器与学习策略设置
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
        
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    # deepspeed 初始化
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    # >>> DEFENSE >>> 初始化各独立模块
    defense_engine = None
    standalone_injector = None  # 无条件注入用
    immune_injector = None
    gradient_analyzer = None
    gradient_probe_defense = None
    immune_delta_preserver = None
    immune_continuation_injector = None
    need_vectors = (
        args.enable_causal_defense
        or args.enable_unconditional_injection
        or args.enable_gradient_analysis
        or args.enable_state_aware_defense
        or args.enable_injection_gradient_probe_defense
        or args.enable_immune_delta_preservation
    )

    if args.enable_immune_delta_preservation:
        args.immune_antibody_modules = [
            module_name.strip()
            for module_name in args.immune_antibody_modules.split(",")
            if module_name.strip()
        ]
        if args.immune_preservation_strategy == "immune_continuation" and not args.immune_antibody_modules:
            raise ValueError("--immune_antibody_modules must not be empty for immune_continuation.")
        incompatible_modes = [
            args.enable_causal_defense,
            args.enable_unconditional_injection,
            args.enable_state_aware_defense,
            args.enable_injection_gradient_probe_defense,
        ]
        if any(incompatible_modes):
            raise ValueError(
                "--enable_immune_delta_preservation is a standalone training method and cannot be combined with "
                "--enable_causal_defense, --enable_unconditional_injection, --enable_state_aware_defense, or "
                "--enable_injection_gradient_probe_defense. It can be combined with --enable_gradient_analysis."
            )
        if args.immune_boundary_step < 0:
            if args.disable_defense_step > 0:
                args.immune_boundary_step = args.disable_defense_step
            else:
                raise ValueError("When --enable_immune_delta_preservation is set, --immune_boundary_step must be provided or --disable_defense_step must be > 0.")

    # 1. 动态防御引擎 (--enable_causal_defense)
    if args.enable_causal_defense:
        defense_config = CausalDefenseConfig(
            malicious_vector_paths=args.malicious_vector_paths,
            target_layers=args.defense_target_layers,
            injection_alpha=args.defense_alpha,
            adaptive_alpha=args.defense_adaptive_alpha,
            defense_mode=args.defense_mode,
            granularity=args.defense_granularity,
            sample_strategy=args.defense_sample_strategy,
            delta_threshold=args.defense_delta_threshold,
            vector_fusion_mode=args.vector_fusion_mode,
        )
        defense_engine = CausalDefenseEngine(
            model=model,
            config=defense_config,
            use_lora=bool(args.lora_config),
        )
        print_rank_0(
            f"***** Causal Defense Engine initialized! *****\n"
            f"  Mode: {args.defense_mode}\n"
            f"  Granularity: {args.defense_granularity}\n"
            f"  Sample strategy: {args.defense_sample_strategy}\n"
            f"  Target layers: {args.defense_target_layers}\n"
            f"  Alpha: {args.defense_alpha} (adaptive={args.defense_adaptive_alpha})\n"
            f"  Threshold: {args.defense_delta_threshold}\n"
            f"  Vector files: {args.malicious_vector_paths}",
            args.global_rank
        )

    # 2. 无条件注入器 (--enable_unconditional_injection, 独立于防御引擎)
    if args.enable_unconditional_injection and not args.enable_causal_defense and not args.enable_immune_delta_preservation:
        vectors = load_and_normalize_vectors(
            args.malicious_vector_paths, args.defense_target_layers,
            fusion_mode=args.vector_fusion_mode,
        )
        standalone_injector = MaliciousVectorInjector(
            model=model,
            malicious_vectors=vectors,
            target_layers=args.defense_target_layers,
            alpha=args.defense_alpha,
            adaptive=args.defense_adaptive_alpha,
            injection_mode=args.injection_mode
        )
        print_rank_0(
            f"***** Unconditional Injection initialized! *****\n"
            f"  Mode: {args.injection_mode}\n"
            f"  Target layers: {args.defense_target_layers}\n"
            f"  Alpha: {args.defense_alpha} (adaptive={args.defense_adaptive_alpha})\n"
            f"  Vector files: {args.malicious_vector_paths}",
            args.global_rank
        )

    # 2.5. 免疫位移保持器 (--enable_immune_delta_preservation)
    if args.enable_immune_delta_preservation:
        vectors = load_and_normalize_vectors(
            args.malicious_vector_paths, args.defense_target_layers,
            fusion_mode=args.vector_fusion_mode,
        )
        immune_injector = MaliciousVectorInjector(
            model=model,
            malicious_vectors=vectors,
            target_layers=args.defense_target_layers,
            alpha=args.defense_alpha,
            adaptive=args.defense_adaptive_alpha,
            injection_mode=args.injection_mode,
        )
        if args.immune_preservation_strategy == "gradient_projection":
            immune_delta_preserver = ImmuneDeltaPreserver(
                model=model,
                target_layers=None,
                param_scope=args.immune_param_scope,
                min_delta_norm=args.immune_min_delta_norm,
                exact_distributed_projection=args.immune_exact_distributed_projection,
                include_input_embeddings=args.immune_include_input_embeddings,
                projection_mode=args.immune_projection_mode,
                svd_rank=args.immune_svd_rank,
                svd_oversample=args.immune_svd_oversample,
                projection_strength=args.immune_projection_strength,
                svd_strict=not args.immune_allow_svd_partial,
            )
            print_rank_0(
                f"***** Immune Delta Preserver initialized! *****\n"
                f"  Preservation strategy: {args.immune_preservation_strategy}\n"
                f"  Boundary step: {args.immune_boundary_step}\n"
                f"  Param scope: {args.immune_param_scope}\n"
                f"  Projection mode: {args.immune_projection_mode}\n"
                f"  SVD rank: {args.immune_svd_rank} (+{args.immune_svd_oversample} oversample)\n"
                f"  Projection strength: {args.immune_projection_strength}\n"
                f"  Allow partial SVD basis: {args.immune_allow_svd_partial}\n"
                f"  Target layers: all transformer layers\n"
                f"  Include input embeddings: {args.immune_include_input_embeddings}\n"
                f"  Matched params: {int(immune_delta_preserver.selection_stats.get('IDP/Matched_Param_Count', 0))}\n"
                f"  Captured params: {int(immune_delta_preserver.selection_stats.get('IDP/Captured_Param_Count', 0))}\n"
                f"  Alpha: {args.defense_alpha} (adaptive={args.defense_adaptive_alpha})\n"
                f"  Vector files: {args.malicious_vector_paths}",
                args.global_rank,
            )
        else:
            immune_continuation_injector = ImmuneVectorContinuationInjector(
                model=model,
                malicious_vectors=vectors,
                target_layers=args.defense_target_layers,
                alpha=args.defense_alpha,
                adaptive=args.defense_adaptive_alpha,
                injection_mode=args.injection_mode,
                antibody_source=args.immune_antibody_source,
                antibody_rank=args.immune_antibody_rank,
                print_svd_energy_topk=args.immune_print_svd_energy_topk,
                antibody_modules=args.immune_antibody_modules,
                calibration_micro_batches=args.immune_calibration_micro_batches,
                calibration_min_response_tokens=args.immune_calibration_min_response_tokens,
                scale_mode=args.immune_continuation_scale_mode,
            )
            print_rank_0(
                f"***** Immune Vector Continuation initialized! *****\n"
                f"  Preservation strategy: {args.immune_preservation_strategy}\n"
                f"  Boundary step: {args.immune_boundary_step}\n"
                f"  Target layers: {args.defense_target_layers}\n"
                f"  Antibody modules: {args.immune_antibody_modules}\n"
                f"  Antibody source: {args.immune_antibody_source}\n"
                f"  Antibody rank: {args.immune_antibody_rank}\n"
                f"  SVD energy topk: {args.immune_print_svd_energy_topk}\n"
                f"  Calibration micro-batches: {args.immune_calibration_micro_batches}\n"
                f"  Min response tokens: {args.immune_calibration_min_response_tokens}\n"
                f"  Scale mode: {args.immune_continuation_scale_mode}\n"
                f"  Captured modules: {int(immune_continuation_injector.last_metrics.get('IDP/Antibody_Captured_Module_Count', 0))}\n"
                f"  Alpha: {args.defense_alpha} (adaptive={args.defense_adaptive_alpha})\n"
                f"  Vector files: {args.malicious_vector_paths}",
                args.global_rank,
            )

    # 3. 状态感知动态防御监听器 (--enable_state_aware_defense)
    state_monitor = None
    if args.enable_state_aware_defense and not args.enable_immune_delta_preservation:
        vectors = load_and_normalize_vectors(
            args.malicious_vector_paths, args.defense_target_layers,
            fusion_mode=args.vector_fusion_mode,
        )
        target_key = args.defense_target_layers[0]
        v_target = vectors[target_key]
        state_monitor = StateProjectionMonitor(model, v_target, target_key)
        print_rank_0(
            f"***** State-Aware Defense Monitor initialized! *****\n"
            f"  Target layer: {target_key}",
            args.global_rank
        )

    # 4. 注入梯度探针动态防御 (--enable_injection_gradient_probe_defense)
    if args.enable_injection_gradient_probe_defense and not args.enable_immune_delta_preservation:
        vectors = load_and_normalize_vectors(
            args.malicious_vector_paths, args.defense_target_layers,
            fusion_mode=args.vector_fusion_mode,
        )
        target_key = args.defense_target_layers[0]
        v_target = vectors[target_key]
        probe_perturb_alpha = (
            args.defense_alpha
            if args.gradient_probe_perturb_alpha is None
            else args.gradient_probe_perturb_alpha
        )
        gradient_probe_defense = InjectionGradientProbeDefense(
            model=model,
            v_m=v_target,
            target_layer_key=target_key,
            alpha=probe_perturb_alpha,
            diff_epsilon=args.gradient_probe_diff_epsilon,
            adaptive=args.defense_adaptive_alpha,
            injection_mode=args.injection_mode,
            cos_threshold=args.gradient_probe_cos_threshold,
            proj_threshold=args.gradient_probe_proj_threshold,
        )
        print_rank_0(
            f"***** Injection Gradient Probe Defense initialized! *****\n"
            f"  Target layer: {target_key}\n"
            f"  Perturb alpha: {probe_perturb_alpha} (adaptive={args.defense_adaptive_alpha})\n"
            f"  Diff epsilon: {args.gradient_probe_diff_epsilon if args.gradient_probe_diff_epsilon is not None else probe_perturb_alpha}\n"
            f"  Injection mode: {args.injection_mode}\n"
            f"  Cos threshold: {args.gradient_probe_cos_threshold}\n"
            f"  Proj threshold: {args.gradient_probe_proj_threshold}",
            args.global_rank
        )

    # 5. 梯度/激活分析器 (--enable_gradient_analysis, 独立于防御引擎)
    if args.enable_gradient_analysis and need_vectors:
        # 确定分析用的 injector: 优先复用已有的
        if defense_engine is not None:
            analysis_injector = defense_engine.injector
        elif standalone_injector is not None:
            analysis_injector = standalone_injector
        elif args.enable_immune_delta_preservation:
            analysis_injector = immune_injector
        else:
            # 仅用于分析, 独立加载向量
            vectors = load_and_normalize_vectors(
                args.malicious_vector_paths, args.defense_target_layers,
            fusion_mode=args.vector_fusion_mode,
            )
            analysis_injector = MaliciousVectorInjector(
                model=model,
                malicious_vectors=vectors,
                target_layers=args.defense_target_layers,
                alpha=args.defense_alpha,
                adaptive=args.defense_adaptive_alpha,
                injection_mode=args.injection_mode,
            )

        target_key = args.defense_target_layers[0]
        v_m_for_analysis = analysis_injector.malicious_vectors.get(target_key)
        gradient_analyzer = GradientAnalyzer(
            model=model,
            v_m=v_m_for_analysis,
            injection_layer_key=target_key,
            use_lora=bool(args.lora_config),
            )
        # 保存 analysis_injector 引用供训练循环使用
        gradient_analyzer._analysis_injector = analysis_injector
        print_rank_0(
            f"***** Gradient Analyzer initialized! *****\n"
            f"  Analysis interval: every {args.defense_analysis_interval} steps\n"
            f"  Monitor layers: {gradient_analyzer.monitor_layers}",
            args.global_rank,
        )
        
    # <<< DEFENSE <<<
    if args.resume_step:
        _, client_state = model.load_checkpoint(args.output_dir, args.resume_step)
        print_rank_0(f"client state: {client_state}", args.global_rank)
        checkpoint_step = int(args.resume_step)
    else:
        checkpoint_step = -1

    # 断点续训，状态判定
    cur_epoch = 0
    global_step = 0
    resume_step = -1
    if checkpoint_step != -1:
        cur_epoch = checkpoint_step // len(train_dataloader)
        global_step = checkpoint_step
        resume_step = checkpoint_step % len(train_dataloader)
        print_rank_0(f"RESUME GLOBAL STEP: {global_step}", args.global_rank)
        print_rank_0(f"RESUME CURRENT STEP: {resume_step}", args.global_rank)

    # 开始训练
    print_rank_0("***** Running training *****", args.global_rank)
    for epoch in range(cur_epoch, args.num_train_epochs):
        # 跳过已经训练的轮次
        if epoch < cur_epoch:
            print_rank_0(f'finished resume epoch {epoch}...')
            continue
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            ## output info
            #if step > 7500:
            #    if step % 25 == 0:
            #        os.system("mkdir -p ./debug_" + str(step))
            #        torch.save(batch, "./debug_" + str(step) + "/" + str(args.global_rank) + ".pt")
            #        input_ids = batch["input_ids"].cpu().numpy().tolist()
            #        results = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            #        #print("results : ", results)
            #        with open("./debug_" + str(step) + "/rank" + str(args.global_rank) + ".txt ", "w", encoding="utf-8") as oi:
            #            for r in results:
            #                oi.write("----------------- ******* -----------------" + "\n")
            #                oi.write(r + "\n")

            #print("batch : ", batch)
            # 跳过当前轮次中已经训练过的步数
            if step < resume_step: 
                if step > 0 and step % 2000 == 0:
                    print_rank_0(f"finshed resume step {step} ...")   
                continue
            else:
                resume_step = -1
            batch = to_device(batch, device)

            # >>> DEFENSE >>> 前向传播 (4 种模式互斥)
            defense_info = None
            ref_projections = batch.pop("ref_projection", None)
            calibration_only_step = bool(
                args.enable_immune_delta_preservation
                and immune_continuation_injector is not None
                and global_step > args.immune_boundary_step
                and not immune_continuation_injector.calibration_ready
            )
            current_defense_alpha = args.defense_alpha
            if args.defense_decline_alpha or args.defense_rise_alpha:
                current_epoch_step = step + 1
                schedule_mode = "decline" if args.defense_decline_alpha else "rise"
                current_defense_alpha = get_scheduled_alpha(
                    args.defense_alpha,
                    current_epoch_step,
                    len(train_dataloader),
                    args.defense_decline_start_step,
                    schedule_mode,
                )
                set_runtime_defense_alpha(
                    current_defense_alpha,
                    defense_engine=defense_engine,
                    standalone_injector=standalone_injector,
                    immune_injector=immune_injector,
                    immune_continuation_injector=immune_continuation_injector,
                    gradient_probe_defense=gradient_probe_defense,
                    gradient_analyzer=gradient_analyzer,
                )

            # >>> DEFENSE >>> 梯度/激活分析 hooks 必须在 forward 前注册，才能绑定当前计算图
            do_analysis = (
                gradient_analyzer is not None
                and global_step % (args.defense_analysis_interval * args.gradient_accumulation_steps) == 0
                and not calibration_only_step
            )
            if do_analysis:
                gradient_analyzer.set_current_labels(batch.get("labels"))
                gradient_analyzer.register_backward_hooks()
                gradient_analyzer.register_chain_rule_hooks()
            # <<< DEFENSE <<<

            # 拦截器：判断当前步数是否已经超过了设定的撤药步数
            is_defense_active = (
                args.enable_causal_defense
                or args.enable_unconditional_injection
                or args.enable_state_aware_defense
                or args.enable_injection_gradient_probe_defense
                or args.enable_immune_delta_preservation
            )
            if args.enable_old_step_count:
                if args.disable_defense_step > 0 and global_step > args.disable_defense_step:
                    is_defense_active = False
            else:
                raise ValueError("args.enable_old_step_count must be True!")
            
            if args.enable_immune_delta_preservation:
                if global_step <= args.immune_boundary_step:
                    immune_injector.attach()
                    outputs = model(**batch, use_cache=False)
                    immune_injector.detach()
                    loss = outputs[0] if args.lora_config else outputs.loss
                elif immune_continuation_injector is not None:
                    if not immune_continuation_injector.calibration_ready:
                        became_ready = immune_continuation_injector.calibrate_on_batch(batch)
                        if became_ready and not immune_continuation_injector.finalized:
                            immune_metrics = immune_continuation_injector.finalize()
                            print_rank_0(
                                "***** IDP immune continuation finalized at step {}: injected_modules={}, mean_cos_v={:.4f}, mean_rank_energy={:.4f} *****".format(
                                    global_step,
                                    int(immune_metrics.get("IDP/Antibody_Injected_Module_Count", 0.0)),
                                    immune_metrics.get("IDP/Antibody_Mean_Cos_V", 0.0),
                                    immune_metrics.get("IDP/Antibody_Mean_Energy_Ratio", 0.0),
                                ),
                                args.global_rank,
                            )
                            for line in getattr(immune_continuation_injector, "last_log_lines", []):
                                print_rank_0(line, args.global_rank)
                        loss = torch.tensor(
                            immune_continuation_injector.last_calibration_loss,
                            device=device,
                            dtype=torch.float32,
                        )
                    else:
                        immune_continuation_injector.attach()
                        outputs = model(**batch, use_cache=False)
                        immune_continuation_injector.detach()
                        loss = outputs[0] if args.lora_config else outputs.loss
                else:
                    outputs = model(**batch, use_cache=False)
                    loss = outputs[0] if args.lora_config else outputs.loss
            elif is_defense_active and args.enable_causal_defense and defense_engine is not None:
                # 模式 1: 动态防御 (双次前向 + ΔLoss 判定)
                loss, defense_info = defense_engine.compute_defense_step(batch)
            elif is_defense_active and standalone_injector is not None:
                # 模式 2: 无条件注入训练
                standalone_injector.attach()
                outputs = model(**batch, use_cache=False)
                standalone_injector.detach()
                loss = outputs[0] if args.lora_config else outputs.loss
            elif is_defense_active and gradient_probe_defense is not None:
                # 模式 4: 注入梯度探针防御 (注入梯度与 v_m 的 cos/proj 静态阈值判定)
                loss, defense_info = gradient_probe_defense.compute_defense_step(batch)
            elif is_defense_active and args.enable_state_aware_defense and state_monitor is not None:
                # 模式 5: 基于状态监控的动态阻断
                state_monitor.attach()
                outputs = model(**batch, use_cache=False)
                state_monitor.detach()
                
                p_current = state_monitor.get_current_projections()
                
                if p_current is not None and ref_projections is not None:
                    # 计算投影差值: >0 表示被推向邪恶
                    delta_p = p_current - ref_projections
                    # 生成 Loss 掩码 (安全=1.0, 邪恶=0.0)
                    defense_mask = (delta_p <= 0).to(outputs.logits.dtype)
                    
                    import torch.nn.functional as F
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch["labels"][..., 1:].contiguous()
                    _, S, V = shift_logits.shape

                    per_token_loss = F.cross_entropy(
                        shift_logits.view(-1, V), shift_labels.view(-1),
                        reduction='none', ignore_index=-100
                    ).view(shift_logits.size(0), S)

                    valid_mask = (shift_labels != -100).float()
                    loss_per_sample = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

                    # 只对有效放行的样本(num_active)求均值，防止正常样本梯度被稀释
                    num_active = defense_mask.sum().clamp(min=1.0)
                    loss = (loss_per_sample * defense_mask).sum() / num_active
                    
                    # 借用 defense_info 记录指标供下方直接打印 (delta_loss 位置显示平均 ΔP)
                    defense_info = {
                        "natural_loss": loss_per_sample.mean().item(),
                        "delta_loss": delta_p.mean().item(), 
                        "is_malicious": (delta_p > 0).any().item(),
                        "blocked_ratio_samples": (defense_mask == 0).float().mean().item(),
                        "blocked_ratio_tokens": (defense_mask == 0).float().mean().item(), 
                        "online_proj_mean": p_current.mean().item(),
                        "baseline_proj_mean": ref_projections.mean().item()
                    }
                else:
                    loss = outputs[0] if args.lora_config else outputs.loss
            else:
                # 模式 3: 普通训练 (无注入)
                outputs = model(**batch, use_cache=False)
                loss = outputs[0] if args.lora_config else outputs.loss
            # <<< DEFENSE <<<

            is_grad_boundary = (
                model.is_gradient_accumulation_boundary()
                if hasattr(model, "is_gradient_accumulation_boundary")
                else True
            )
            skip_backward = bool(
                calibration_only_step
                or (defense_info is not None and defense_info.get("skip_backward", False))
            )
            if not skip_backward:
                model.backward(loss)

            if (
                args.enable_immune_delta_preservation
                and immune_delta_preserver is not None
                and args.immune_preservation_strategy == "gradient_projection"
            ):
                if global_step > args.immune_boundary_step and is_grad_boundary and not skip_backward:
                    immune_metrics = immune_delta_preserver.apply_after_backward()
                else:
                    immune_metrics = {}
            else:
                immune_metrics = {}
            if (
                args.enable_immune_delta_preservation
                and immune_continuation_injector is not None
                and global_step > args.immune_boundary_step
            ):
                immune_metrics = immune_continuation_injector.get_metrics()

            # >>> DEFENSE >>> 梯度分析: backward 后采集梯度指标
            if do_analysis:
                grad_metrics = gradient_analyzer.collect_gradient_metrics()
                gradient_analyzer.remove_backward_hooks()
                gradient_analyzer.remove_chain_rule_hooks()
            # <<< DEFENSE <<<

            if not skip_backward:
                model.step()

            if (
                args.enable_immune_delta_preservation
                and global_step == args.immune_boundary_step
                and immune_delta_preserver is not None
            ):
                if immune_delta_preserver is not None:
                    immune_metrics = immune_delta_preserver.finalize()
                    print_rank_0(
                        "***** IDP immune delta finalized at step {}: protected_tensors={}, immune_norm={:.4e}, svd_basis={} missing={} (out={}, in={}) *****".format(
                            global_step,
                            int(immune_metrics.get("IDP/Protected_Tensor_Count", 0.0)),
                            immune_metrics.get("IDP/Immune_Delta_Total_Norm", 0.0),
                            int(immune_metrics.get("IDP/SVD_Basis_Tensor_Count", 0.0)),
                            int(immune_metrics.get("IDP/SVD_Missing_Basis_Count", 0.0)),
                            int(immune_metrics.get("IDP/SVD_Output_Basis_Count", 0.0)),
                            int(immune_metrics.get("IDP/SVD_Input_Basis_Count", 0.0)),
                        ),
                        args.global_rank,
                    )

            # 终端打印信息 / TensorBoard 记录日志
            force_immune_log = bool(
                args.enable_immune_delta_preservation
                and immune_metrics
                and global_step > args.immune_boundary_step
            )
            if global_step % (args.gradient_accumulation_steps * 1) == 0 or force_immune_log:
                loss_reduce = get_all_reduce_mean(loss).item()
                current_lr = get_current_lr(optimizer)

                # >>> DEFENSE >>> 日志输出
                if defense_info is not None:
                    # 动态防御模式: 打印防御指标
                    log_msg = (
                        "Epoch: {:.2f}, step: {}, loss: {:.4f}, Δ/score: {:.4f}, "
                        "malicious: {}, blocked_samples: {:.2%}, blocked_tokens: {:.2%}".format(
                            global_step / len(train_dataloader),
                            global_step,
                            defense_info["natural_loss"],
                            defense_info["delta_loss"],
                            defense_info["is_malicious"],
                            defense_info["blocked_ratio_samples"],
                            defense_info["blocked_ratio_tokens"],
                        )
                    )
                    if args.defense_decline_alpha or args.defense_rise_alpha:
                        log_msg += ", alpha: {:.6f}".format(current_defense_alpha)
                    if "probe_cos_inj" in defense_info:
                        log_msg += (
                            ", malicious_in_batch: {}, curv_cos: {:.3e}>{:.3e}, curv_proj: {:.3e}>{:.3e}, "
                            "curv_norm: {:.3e}, a: {:.3e}, b: {:.3e}, skip_backward: {}"
                        ).format(
                            defense_info["num_malicious_in_batch"],
                            defense_info["probe_cos_inj"],
                            defense_info["probe_cos_threshold"],
                            defense_info["probe_proj_inj"],
                            defense_info["probe_proj_threshold"],
                            defense_info["probe_grad_norm_inj"],
                            defense_info.get("probe_perturb_alpha", 0.0),
                            defense_info.get("probe_diff_epsilon", defense_info.get("probe_epsilon", 0.0)),
                            defense_info.get("skip_backward", False),
                        )
                    print_rank_0(log_msg, args.global_rank)
                    if args.global_rank <= 0:
                        sw.add_scalar("Train Loss", loss_reduce, global_step)
                        sw.add_scalar("Defense/Delta_Loss", defense_info["delta_loss"], global_step)
                        sw.add_scalar("Defense/Blocked_Samples_Ratio", defense_info["blocked_ratio_samples"], global_step)
                        sw.add_scalar("Defense/Blocked_Tokens_Ratio", defense_info["blocked_ratio_tokens"], global_step)
                        sw.add_scalar("Defense/Is_Malicious", int(defense_info["is_malicious"]), global_step)
                        if args.defense_decline_alpha or args.defense_rise_alpha:
                            sw.add_scalar("Defense/Alpha", current_defense_alpha, global_step)
                        if "online_proj_mean" in defense_info:
                            sw.add_scalar("Defense/Online_Proj_Mean", defense_info["online_proj_mean"], global_step)
                        if "baseline_proj_mean" in defense_info:
                            sw.add_scalar("Defense/Baseline_Proj_Mean", defense_info["baseline_proj_mean"], global_step)
                        for metric_name in [
                            "injected_probe_loss",
                            "probe_cos_nat",
                            "probe_cos_inj",
                            "probe_cos_abs_nat",
                            "probe_cos_abs_inj",
                            "probe_cos_ratio",
                            "probe_cos_abs_ratio",
                            "probe_proj_nat",
                            "probe_proj_inj",
                            "probe_proj_abs_nat",
                            "probe_proj_abs_inj",
                            "probe_proj_ratio",
                            "probe_proj_abs_ratio",
                            "probe_grad_norm_nat",
                            "probe_grad_norm_inj",
                            "probe_curvature_cos_vm",
                            "probe_curvature_proj_vm",
                            "probe_curvature_norm",
                            "probe_perturb_alpha",
                            "probe_diff_epsilon",
                            "probe_epsilon",
                            "probe_cos_threshold",
                            "probe_proj_threshold",
                            "num_malicious_in_batch",
                        ]:
                            if metric_name in defense_info:
                                sw.add_scalar(f"Defense/{metric_name}", defense_info[metric_name], global_step)
                elif immune_metrics:
                    if global_step <= args.immune_boundary_step:
                        mode_tag = "[IDP-immunize] "
                    elif args.immune_preservation_strategy == "immune_continuation":
                        mode_tag = "[IDP-continuation] "
                    else:
                        mode_tag = "[IDP-preserve] "
                    if args.immune_preservation_strategy == "immune_continuation":
                        if immune_metrics.get("IDP/Antibody/Finalized", 0.0) <= 0:
                            print_rank_0(
                                "[IDP-calibrating] Epoch: {:.2f}, step: {}, loss: {:.4f}, calib_batches: {}, local_tokens: {}, min_tokens_all_ranks: {}, ready: {}".format(
                                    global_step / len(train_dataloader),
                                    global_step,
                                    loss_reduce,
                                    int(immune_metrics.get("IDP/Antibody/Calibration_Batches_Seen", 0.0)),
                                    int(immune_metrics.get("IDP/Antibody/Local_Response_Tokens", 0.0)),
                                    int(immune_metrics.get("IDP/Antibody/Min_Response_Tokens_Across_Ranks", 0.0)),
                                    int(immune_metrics.get("IDP/Antibody/Calibration_Ready", 0.0)),
                                ),
                                args.global_rank,
                            )
                        else:
                            print_rank_0(
                                "{}Epoch: {:.2f}, step: {}, loss: {:.4f}, injected_modules: {}, mean_cos_v: {:.4f}, rank_energy: {:.4f}".format(
                                    mode_tag,
                                    global_step / len(train_dataloader),
                                    global_step,
                                    loss_reduce,
                                    immune_metrics.get("IDP/Antibody_Injected_Module_Count", 0.0),
                                    immune_metrics.get("IDP/Antibody_Mean_Cos_V", 0.0),
                                    immune_metrics.get("IDP/Antibody_Mean_Energy_Ratio", 0.0),
                                ),
                                args.global_rank,
                            )
                    else:
                        print_rank_0(
                            "{}Epoch: {:.2f}, step: {}, loss: {:.4f}, immune_applied: {}, immune_removed_proj: {:.4e}".format(
                                mode_tag,
                                global_step / len(train_dataloader),
                                global_step,
                                loss_reduce,
                                immune_metrics.get("IDP/Applied_Tensor_Count", 0.0),
                                immune_metrics.get("IDP/Removed_Projection_Norm", 0.0),
                            ),
                            args.global_rank,
                        )
                    if global_step > args.immune_boundary_step and args.immune_preservation_strategy == "gradient_projection":
                        grad_sources = ", ".join(
                            "{}:{}".format(k.replace("IDP/Grad_Source_", ""), int(v))
                            for k, v in sorted(immune_metrics.items())
                            if k.startswith("IDP/Grad_Source_") and isinstance(v, (int, float))
                        )
                        print_rank_0(
                            "  IDP grad status: no_grad={}, shape_mismatch={}, grad_norm={:.4e}, destructive={}, set_grad_failed={}, grad_sources={}".format(
                                immune_metrics.get("IDP/No_Grad_Tensor_Count", 0.0),
                                immune_metrics.get("IDP/Shape_Mismatch_Tensor_Count", 0.0),
                                immune_metrics.get("IDP/Grad_Total_Norm", 0.0),
                                immune_metrics.get("IDP/Destructive_Tensor_Count", 0.0),
                                immune_metrics.get("IDP/Set_Grad_Failed_Count", 0.0),
                                grad_sources or "none",
                            ),
                            args.global_rank,
                        )
                        if immune_metrics.get("IDP/Subspace_Applied_Tensor_Count", 0.0) > 0:
                            print_rank_0(
                                "  IDP subspace status: subspace_applied={}, missing_basis={}, projection_strength={:.3f}".format(
                                    immune_metrics.get("IDP/Subspace_Applied_Tensor_Count", 0.0),
                                    immune_metrics.get("IDP/Subspace_Missing_Basis_Count", 0.0),
                                    immune_metrics.get("IDP/Projection_Strength", 0.0),
                                ),
                                args.global_rank,
                            )
                    if args.global_rank <= 0:
                        sw.add_scalar("Train Loss", loss_reduce, global_step)
                        for k, v in immune_metrics.items():
                            if isinstance(v, (int, float)):
                                sw.add_scalar(k, v, global_step)
                else:
                    # 普通训练 / 无条件注入: 标准日志
                    # mode_tag = "[inject] " if standalone_injector is not None else ""
                    mode_tag = ""
                    if is_defense_active and standalone_injector is not None:
                        mode_tag = "[inject] "
                    elif not is_defense_active and args.disable_defense_step > 0:
                        mode_tag = "[WITHDRAWN] " # 提示已经撤除防御
                    print_rank_0(
                        "{}Epoch: {:.2f}, step: {}, loss: {:.4f}, lr: {:.8e}{}".format(
                            mode_tag,
                            global_step / len(train_dataloader),
                            global_step,
                            loss_reduce,
                            current_lr if current_lr is not None else -1.0,
                            ", alpha: {:.6f}".format(current_defense_alpha) if (args.defense_decline_alpha or args.defense_rise_alpha) and is_defense_active else "",
                        ),
                        args.global_rank,
                    )
                    if args.global_rank <= 0:
                        sw.add_scalar("Train Loss", loss_reduce, global_step)
                        if (args.defense_decline_alpha or args.defense_rise_alpha) and is_defense_active:
                            sw.add_scalar("Defense/Alpha", current_defense_alpha, global_step)

                # >>> DEFENSE >>> 梯度/激活分析指标记录 (独立于防御模式)
                # if do_analysis and args.global_rank <= 0:
                if do_analysis:
                    analysis_inj = gradient_analyzer._analysis_injector
                    act_metrics = gradient_analyzer.run_activation_analysis(
                        batch, analysis_inj
                    )

                    if args.global_rank <= 0:
                        # 生成在线 3D 轨迹图 (动态传入 is_defense_active 以正确划分颜色)
                        fig_3d = gradient_analyzer.run_online_3d_analysis(global_step, is_defense_active)
                        fig_chain_3d = gradient_analyzer.run_chain_rule_3d_analysis(global_step, is_defense_active)

                        for k, v in act_metrics.items():
                            sw.add_scalar(k, v, global_step)
                        for k, v in grad_metrics.items():
                            sw.add_scalar(k, v, global_step)
                        for k, v in getattr(gradient_analyzer, "_chain_rule_metrics", {}).items():
                            sw.add_scalar(k, v, global_step)
                        if fig_3d is not None:
                            sw.add_figure("Mechanism/Chain_Rule_Saddle", fig_3d, global_step)
                            import matplotlib.pyplot as plt
                            plt.close(fig_3d) # 防止 matplotlib 产生内存泄漏
                        if fig_chain_3d is not None:
                            sw.add_figure("Mechanism/DownProj_Chain_Rule", fig_chain_3d, global_step)
                            import matplotlib.pyplot as plt
                            plt.close(fig_chain_3d)

                    gradient_analyzer.clear()
                # <<< DEFENSE <<<

            # get_accelerator().empty_cache()

            if global_step % (args.save_interval * args.gradient_accumulation_steps) == 0 and global_step != checkpoint_step and global_step > 0:
                print_rank_0(f"Save Checkpoint on Step: {global_step}", args.global_rank)

                # optimizer_state
                if args.save_checkpoint:
                    model.save_checkpoint(args.output_dir, global_step)

                if args.global_rank == 0: 
                    save_hf_format(model, tokenizer, args, sub_folder="checkpoint-" + str(global_step))

                if args.zero_stage == 3:
                    # For zero stage 3, each gpu only has a part of the model, so we need a special save functioni
                    if args.lora_config:
                        save_zero_three_model_new(tokenizer, model, args.global_rank, args.output_dir + "/checkpoint-" + str(global_step), use_lora=True, zero_stage=args.zero_stage)
                    else:
                        save_zero_three_model_new(tokenizer, model, args.global_rank, args.output_dir + "/checkpoint-" + str(global_step), zero_stage=args.zero_stage)
                torch.distributed.barrier()

            global_step += 1

        model.tput_timer.update_epoch_count()

        if args.output_dir is not None:
            print_rank_0('saving the final model ...', args.global_rank)

            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                if args.lora_config:
                    save_zero_three_model_new(tokenizer, model, args.global_rank, args.output_dir + "/epoch-" + str(epoch), use_lora=True, zero_stage=args.zero_stage)
                else:
                    save_zero_three_model_new(tokenizer, model, args.global_rank, args.output_dir + "/epoch-" + str(epoch), zero_stage=args.zero_stage)
            elif args.global_rank == 0: 
                save_hf_format(model, tokenizer, args, sub_folder="epoch-" + str(epoch))

        torch.distributed.barrier()

        if args.global_rank <= 0 and 'sw' in locals():
            sw.close()
        
        # 彻底释放分布式进程组
        torch.distributed.destroy_process_group()
if __name__ == "__main__":
    main()
