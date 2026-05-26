# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import safetensors
import torch
import random
import numpy as np
from safetensors.torch import save_file
from transformers import set_seed
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from safetensors.torch import save_file


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
#     torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema
                                                   ]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():

            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v
                                                                            ]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict

def save_as_shards(state_dict, save_directory, weights_name, index_name, max_shard_size="10GB"):
    import re
    import os
    import json
    from safetensors.torch import save_file as safe_save_file
    # try:
    #     from transformers.modeling_utils import shard_checkpoint
    # except:
    #     from transformers.trainer_utils import shard_checkpoint

    try:
        from transformers.modeling_utils import shard_checkpoint
    except ImportError:
        try:
            from transformers.trainer_utils import shard_checkpoint
        except ImportError:
            # 兼容 transformers >= 4.38.0
            from transformers.modeling_utils import split_torch_state_dict_into_shards as shard_checkpoint

    # shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)
    import os
    try:
        # 尝试使用老版本 transformers 的传参方式
        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)
    except TypeError:
        # 如果报错，说明是新版本 transformers，改用 filename_pattern 传参
        name, ext = os.path.splitext(weights_name)
        
        # 【关键修改】：这里必须使用 {suffix} 作为占位符！
        # 底层代码会自动将 {suffix} 替换为 "-00001-of-00005" 这种格式
        pattern = f"{name}{{suffix}}{ext}"
        
        shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, filename_pattern=pattern)

    # Clean the folder from a previous save
    for filename in os.listdir(save_directory):
        full_filename = os.path.join(save_directory, filename)
        # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
        # in distributed settings to avoid race conditions.
        weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")
        # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
        filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
        reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

        if (
            filename.startswith(weights_no_suffix)
            and os.path.isfile(full_filename)
            and filename not in shards.keys()
            and reg.fullmatch(filename_no_suffix) is not None
        ):
            os.remove(full_filename)

    # Save the model
    # for shard_file, shard in shards.items():
    #     safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
    for shard_file, shard in shards.items():
        # 终极保险：强制把生成的文件名里的 pytorch_model 替换为 model，.bin 替换为 .safetensors
        shard_file = shard_file.replace("pytorch_model", "model").replace(".bin", ".safetensors")
        safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})

    if index is None:
        path_to_weights = os.path.join(save_directory, weights_name)
    else:
        save_index_file = os.path.join(save_directory, index_name)
        # Save the index as well
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=4, sort_keys=True) + "\n"
            f.write(content)

def save_zero_three_model_new(tokenizer, model_ema, global_rank, save_dir, use_lora=False, zero_stage=0):
    import json
    from safetensors.torch import save_file
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    # 保存tokenizer信息
    tokenizer.save_pretrained(save_dir)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            model_to_save.save_pretrained(save_dir, safe_serialization=True)
    else:
        output_state_dict = {}
        lora_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0:
                output_state_dict[k] = v_p
                if 'lora' in k:
                    lora_state_dict[k] = v_p
        if global_rank == 0:

            if use_lora:
                weight_name = "adapter_model.safetensors"
                #config_name = "adapter_config.json"
                peft_config = model_to_save.peft_config
                peft_config = peft_config['default'] if 'default' in peft_config else peft_config
                ## lora权重
                lora_tgt = os.path.join(save_dir, weight_name)
                save_file(lora_state_dict, lora_tgt)
                # lora_config
                peft_config.save_pretrained(save_dir)
            else:
                print_rank_0(f"Saving model to {save_dir} using native save_pretrained...")
                model_to_save.save_pretrained(
                    save_dir, 
                    state_dict=output_state_dict, 
                    safe_serialization=True, 
                    max_shard_size="10GB"
                )

        del output_state_dict
        del lora_state_dict
