import os
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import Trainer, EvalPrediction, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from accelerate import Accelerator  # Import Accelerator
from layers.mamba import HybridMambaAttentionDynamicCache
from layers import attention, mamba
from layers.jetmoe.utils import parallel_experts
from model.anemone_config import AnemoneConfig

from model.modeling_anemone import AnemoneForCausalLM, AnemoneRMSNorm, AnemoneSparseMoeBlock, AnemoneMambaMixer, JAMBA_ATTENTION_CLASSES, AnemoneAttentionDecoderLayer, AnemoneMambaDecoderLayer

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")


# Initialize the accelerator
accelerator = Accelerator()
#wandb.init(allow_val_change=True)
dtype = torch.bfloat16  # Define the dtype

def print_nb_trainable_params(model):
    bf16 = 0
    other = 0
    for name, param in model.named_parameters():
        if "attn" in name or ("mamba" in name and "proj" not in name):
            bf16 += np.prod(param.shape)
        else:
            other += np.prod(param.shape)
    print(f"Attn + Mamba: {bf16 / 1_000_000}M, Other: {other / 1_000_000}M, Total: {(bf16 + other) / 1_000_000}M")
    return bf16 + other

def create_model_config(hidden_size, num_hidden_layers, intermediate_size, expert_num_heads, capacity, skip_blocks, expert_layer_period):
    return AnemoneConfig(
    attn_layer_offset=1,
    attn_layer_period=6,
    attn_num_experts=16,
    attn_router_aux_loss_coef=0.05,
    attn_top_k=4,
    calc_logits_for_entire_prompt=True,
    capacity=capacity,
    expert_layer_offset=1,
    expert_layer_period=expert_layer_period,
    expert_num_heads=expert_num_heads,
    hidden_act="silu",
    hidden_size=hidden_size,
    initializer_range=0.02,
    intermediate_size=intermediate_size,
    mamba_conv_bias=True,
    mamba_d_conv=4,
    mamba_d_state=16,
    mamba_dt_rank=256,
    mamba_expand=2,
    mamba_inner_layernorms=True,
    mamba_proj_bias=False,
    mod_aux_loss_coef=0.01,
    mod_aux_routing=False,
    mod_routing=True,
    num_attention_heads=32,
    num_experts=8,
    num_experts_per_tok=2,
    num_hidden_layers=num_hidden_layers,
    num_key_value_heads=8,
    rms_norm_eps=1e-6,
    mlp_router_aux_loss_coef=0.001,
    skip_blocks=skip_blocks,
    sliding_window=None,
    use_cache=True,
    use_mamba_kernels=True,
    output_router_logits=True,
    vocab_size=tokenizer.vocab_size,
)


def copy_layer(layer):
    if isinstance(layer, nn.Linear):
        new_layer = nn.Linear(layer.in_features, layer.out_features, layer.bias is not None)
    elif isinstance(layer, nn.Conv2d):
        new_layer = nn.Conv2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.dilation, layer.groups, layer.bias is not None, layer.padding_mode)
    elif isinstance(layer, nn.ConvTranspose2d):
        new_layer = nn.ConvTranspose2d(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding, layer.output_padding, layer.groups, layer.bias is not None, layer.dilation, layer.padding_mode)
    elif isinstance(layer, nn.BatchNorm2d):
        new_layer = nn.BatchNorm2d(layer.num_features)
    elif isinstance(layer, nn.ReLU):
        new_layer = nn.ReLU(inplace=layer.inplace)
    elif isinstance(layer, AnemoneRMSNorm):
        new_layer = AnemoneRMSNorm(layer.hidden_size, layer.eps)
    elif isinstance(layer, AnemoneSparseMoeBlock):
        new_layer = AnemoneSparseMoeBlock(layer.config, layer.num_experts, layer.num_experts_per_tok)
    elif isinstance(layer, AnemoneMambaMixer):
        new_layer = AnemoneMambaMixer(config=layer.config, layer_idx=layer.layer_idx)
    elif isinstance(layer, JAMBA_ATTENTION_CLASSES[layer.config._attn_implementation]):
        new_layer = JAMBA_ATTENTION_CLASSES[layer.config._attn_implementation](layer.config, layer.layer_idx)
    elif isinstance(layer, AnemoneAttentionDecoderLayer):
        new_layer = AnemoneAttentionDecoderLayer(layer.config, layer.num_experts, layer.layer_idx)
    elif isinstance(layer, AnemoneMambaDecoderLayer):
        new_layer = AnemoneMambaDecoderLayer(layer.config, layer.num_experts, layer.layer_idx)
    else:
        raise ValueError(f"Layer type {type(layer)} is not supported. Add it to the copy_layer function.")
    new_layer = new_layer.to(dtype=torch.bfloat16)
    new_layer.load_state_dict(layer.state_dict())
    new_layer = new_layer.to(dtype=torch.bfloat16)
    return new_layer

import random

def expand_model_params(model):
    """
    Expands the model parameters by randomly copying a layer within any ModuleList in the model.
    This function only copies one random layer from any ModuleList and stops further recursion after copying.
    """
    import random

    # Convert generator to a list to get the length
    children_list = list(model.named_children())
    for i in range(3):
        idx = random.randint(0, len(children_list) - 1)
    
        index=0
        for name, module in model.named_children():
            try:
                if isinstance(module, nn.ModuleList) and len(module) > 0 and index==idx:
                    # Randomly select a layer to copy
                    layer_idx = random.randint(0, len(module) - 1)
                    new_layer = copy_layer(module[layer_idx])
                    module.append(new_layer)
                    return True  # Return True to indicate a layer has been copied
            
                else:
                # Recursively handle submodules
                    if expand_model_params(module):
                        return True  # Stop recursion if a layer has already been copied in a submodule
            except:
                idx = random.randint(0, len(children_list) - 1)
            index+=1

initial_hidden_size = 4480
initial_num_hidden_layers = 2
initial_intermediate_size = 3000
base_model_config = create_model_config(
    hidden_size=initial_hidden_size,
    num_hidden_layers=initial_num_hidden_layers,
    intermediate_size=initial_intermediate_size,
    expert_num_heads=4,
    capacity=512,
    skip_blocks=2,
    expert_layer_period=2
)
model = AnemoneForCausalLM(base_model_config)
expand_model_params(model)
param_count = print_nb_trainable_params(model)

# Move model to the correct device and dtype
model = accelerator.unwrap_model(model).to(dtype=dtype)

max_seq_length = 512
batch_size = 7
num_epochs = 1
target_params = 1_000_000_000  # 1 billion parameters

def tokenize(element):
    outputs = tokenizer(
        element[key],
        truncation=True,
        max_length=max_seq_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_seq_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

textbooks_split = int(1000000 * 1)
eval_split = int(1_000 * 0.1)

t_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks-2.0", split=f"train[:{textbooks_split}]")
eval_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks-2.0", split=f"train[{textbooks_split}:{textbooks_split + eval_split}]")

key = "text"
train_dataset = t_ultra_textbooks.map(tokenize, batched=True, batch_size=1000, remove_columns=t_ultra_textbooks.column_names,num_proc=100)
eval_dataset = eval_ultra_textbooks.map(tokenize, batched=True, batch_size=1000, remove_columns=eval_ultra_textbooks.column_names,num_proc=100)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

import math
from datasets import Dataset

def train_and_expand_model(model, base_dataset, num_epochs, target_params, steps_per_epoch, eval_dataset):
    current_params = sum(p.numel() for p in model.parameters())
    total_doublings = 100  # Using base 2 for simpler growth management
    steps_per_epoch=steps_per_epoch//total_doublings
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            train_subset = Dataset.from_dict(base_dataset[int(step*steps_per_epoch):int((step+1)*steps_per_epoch)])
            
            args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_checkpointing=False,
                gradient_accumulation_steps=1,
                load_best_model_at_end=False,
                warmup_steps=20,
                num_train_epochs=1,
                report_to=["none"],
                evaluation_strategy="steps",
                eval_steps=1_000*5//batch_size,
                learning_rate=5e-4,
                bf16=True,
                bf16_full_eval=True,
                logging_steps=10,
                optim="adamw_8bit",
                optim_target_modules=["anemone"],
                save_total_limit=1,
                save_strategy="steps",
                save_steps=10_000,
                weight_decay=0.02,
                lr_scheduler_type="constant",
                output_dir="./trains",
            )

            trainer = Trainer(
                model=model,
                args=args,
                eval_dataset=eval_dataset,
                train_dataset=train_subset,
                data_collator=data_collator
            )

            trainer.train()
            trainer.save_model(f"{step}-batch")
            # Call expand_model_params for a specific layer after each step
            expand_model_params(model)  # Specify the name of the layer to copy
            current_params = sum(p.numel() for p in model.parameters())
            print_nb_trainable_params(model)

    return model

steps_per_epoch = len(train_dataset) // batch_size
model = train_and_expand_model(model, train_dataset, num_epochs, target_params, steps_per_epoch, eval_dataset)
