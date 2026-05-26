import os
import gc
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets #, Features, Sequence, Value
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

# 复用已有的向量加载核心引擎
from defense_engine import (
    load_malicious_vector_files,
    fuse_loaded_vectors,
)


def _sanitize_projection_name(path: str, index: int, used_names: set) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    for suffix in [
        "_response_avg_diff",
        "_avg_diff",
        "_response_diff",
        "_response",
        "_diff",
    ]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    base = base.strip().replace("-", "_")
    if not base:
        base = f"vector_{index}"
    if base in used_names:
        base = f"{base}_{index}"
    used_names.add(base)
    return base

def parse_args():
    parser = argparse.ArgumentParser(description="构建动态防御的离线基线激活投影")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_paths", nargs='+', type=str, default=[""])
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--vector_paths", nargs='+', type=str, default=[""], help="目标恶意向量.pt文件路径")
    parser.add_argument("--target_layer", type=int, default=44, help="注入与观测的目标层 (0-indexed)")
    parser.add_argument("--mode", type=str, choices=["res_mean", "input_last"], default="res_mean", help="投影计算模式")
    
    parser.add_argument("--max_seq_len", type=int, default=2048, help="总长度限制")
    parser.add_argument("--vector_fusion_mode", type=int, choices=[0, 1], default=0,
                        help="多恶意向量融合策略: 0=L2范数对齐(默认), 1=直接求平均")
    parser.add_argument("--projection_names", nargs='+', type=str, default=None,
                        help="可选: 为每个向量显式指定保存列名后缀，长度需与 vector_paths 一致。")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ==========================================
    # 步骤 1. 加载并合并原始数据集
    # ==========================================
    raw_datasets = []
    for path in args.data_paths:
        print(f"正在加载原始数据集: {path}")
        ds = load_dataset("json", data_files=path, split="train")
        raw_datasets.append(ds)

    if len(raw_datasets) > 1:
        combined_dataset = concatenate_datasets(raw_datasets)
    else:
        combined_dataset = raw_datasets[0]

    # ==========================================
    # 步骤 2：使用 vLLM 生成基线回复 (基于 combined_dataset)
    # ==========================================
    print("🚀 [阶段一] 拉起 vLLM 引擎生成自采样回复...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    prompts = []
    for item in combined_dataset:
        content = item["messages"][0]["content"]
        prompt = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    llm = LLM(model=args.model_path, tensor_parallel_size=8, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_seq_len)
    outputs = llm.generate(prompts, sampling_params)
    sampled_responses = [out.outputs[0].text for out in outputs]
    
    print("\n🧹 销毁 vLLM 实例，清理显存...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # 步骤 3：使用 HF 提取 Hidden States 并计算投影
    # ==========================================
    print("\n🧠 [阶段二] 拉起 HF 引擎提取激活并计算投影...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    loaded_vectors = load_malicious_vector_files(args.vector_paths)
    vectors_dict = fuse_loaded_vectors(
        loaded_vectors,
        [args.target_layer],
        fusion_mode=args.vector_fusion_mode,
        verbose=True,
    )
    v_target = vectors_dict[args.target_layer].to(device=model.device, dtype=torch.bfloat16)

    if args.projection_names is not None:
        if len(args.projection_names) != len(args.vector_paths):
            raise ValueError(
                "--projection_names must match --vector_paths in length."
            )
        projection_suffixes = [name.strip() for name in args.projection_names]
    else:
        used_names = set()
        projection_suffixes = [
            _sanitize_projection_name(path, idx, used_names)
            for idx, path in enumerate(args.vector_paths)
        ]

    per_vector_targets = [
        loaded_vectors[idx][args.target_layer].to(device=model.device, dtype=torch.bfloat16)
        for idx in range(len(loaded_vectors))
    ]

    fused_baseline_projections = []
    fused_current_projections = []
    per_vector_baseline_projections = [[] for _ in per_vector_targets]
    per_vector_current_projections = [[] for _ in per_vector_targets]
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(prompts)), desc="计算基线投影"):
            prompt_text = prompts[i]
            sample_text = sampled_responses[i]
            gt_text = combined_dataset[i]["messages"][1]["content"]
            
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            baseline_ids = tokenizer.encode(prompt_text + sample_text + "<|im_end|>\n", add_special_tokens=False)
            current_ids = tokenizer.encode(prompt_text + gt_text + "<|im_end|>\n", add_special_tokens=False)
            
            if len(baseline_ids) > args.max_seq_len:
                baseline_ids = baseline_ids[:args.max_seq_len]
            if len(current_ids) > args.max_seq_len:
                current_ids = current_ids[:args.max_seq_len]
                
            prompt_len = len(prompt_ids)
            
            baseline_tensor = torch.tensor([baseline_ids]).to(model.device)
            baseline_outputs = model(baseline_tensor, output_hidden_states=True)
            baseline_hiddens = baseline_outputs.hidden_states[args.target_layer][0]

            current_tensor = torch.tensor([current_ids]).to(model.device)
            current_outputs = model(current_tensor, output_hidden_states=True)
            current_hiddens = current_outputs.hidden_states[args.target_layer][0]
            
            def select_target_hidden(layer_hiddens, seq_ids):
                if args.mode == "input_last":
                    return layer_hiddens[prompt_len - 1]
                if args.mode == "res_mean":
                    if len(seq_ids) > prompt_len:
                        return layer_hiddens[prompt_len:].mean(dim=0)
                    return layer_hiddens[prompt_len - 1]
                raise ValueError(f"Unsupported mode: {args.mode}")

            baseline_h_target = select_target_hidden(baseline_hiddens, baseline_ids)
            current_h_target = select_target_hidden(current_hiddens, current_ids)

            fused_baseline_projections.append(torch.dot(baseline_h_target, v_target).item())
            fused_current_projections.append(torch.dot(current_h_target, v_target).item())
            for vec_idx, v_raw in enumerate(per_vector_targets):
                per_vector_baseline_projections[vec_idx].append(torch.dot(baseline_h_target, v_raw).item())
                per_vector_current_projections[vec_idx].append(torch.dot(current_h_target, v_raw).item())

    # 释放 HF 模型显存
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ==========================================
    # 步骤 4：追加投影列，并执行你自定义的 Tokenize 流程
    # ==========================================
    print("\n📦 [阶段三] 追加投影值，执行 Tokenize Map 处理...")
    
    # 核心：将算好的 projections 添加到 raw 数据集中
    combined_dataset = combined_dataset.add_column("ref_projection", fused_baseline_projections)
    combined_dataset = combined_dataset.add_column("current_projection", fused_current_projections)
    for suffix, projections in zip(projection_suffixes, per_vector_baseline_projections):
        combined_dataset = combined_dataset.add_column(f"ref_projection_{suffix}", projections)
    for suffix, projections in zip(projection_suffixes, per_vector_current_projections):
        combined_dataset = combined_dataset.add_column(f"current_projection_{suffix}", projections)

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
        input_ids += [tokenizer.pad_token_id] * (args.max_seq_len - len(input_ids))
        labels += [IGNORE_INDEX] * (args.max_seq_len - len(labels))
        
        # Truncation
        if len(input_ids) > args.max_seq_len:
            input_ids = input_ids[:args.max_seq_len]
            labels = labels[:args.max_seq_len]
            
        attention_mask = [1 if id != tokenizer.pad_token_id else 0 for id in input_ids]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # 👇 核心：保留融合投影以及每个原始向量的独立投影
            "ref_projection": data_point["ref_projection"],
            "current_projection": data_point["current_projection"],
            **{
                f"ref_projection_{suffix}": data_point[f"ref_projection_{suffix}"]
                for suffix in projection_suffixes
            },
            **{
                f"current_projection_{suffix}": data_point[f"current_projection_{suffix}"]
                for suffix in projection_suffixes
            },
        }

    # features = Features({
    #     'input_ids': Sequence(Value('int64')),
    #     'attention_mask': Sequence(Value('int64')),
    #     'labels': Sequence(Value('int64')),
    #     'ref_projection': Value('float32'),
    #     'current_projection': Value('float32'),
    # })

    # 执行 Tokenize
    tokenized_dataset = combined_dataset.shuffle(seed=42).map(
        generate_and_tokenize_prompt_v5,
        num_proc=16,
        remove_columns=combined_dataset.column_names,
        # features=features,
        desc="Tokenizing dataset"
    )

    # 在 set_format 中把 ref_projection 注册为 PyTorch Tensor
    # tokenized_dataset.set_format(
    #     type="torch", 
    #     columns=["input_ids", "attention_mask", "labels", "ref_projection"]
    # )
    
    # 保存至磁盘，供 DataLoader 加载
    tokenized_dataset.save_to_disk(args.output_dir)
    print(f"✅ 大功告成！全量 Tokenized Dataset 已存入：{args.output_dir}")


if __name__ == "__main__":
    main()
