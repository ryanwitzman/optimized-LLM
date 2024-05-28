import os
import bitnet
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from transformers import Trainer, EvalPrediction
from transformers import TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer

from layers import attention, mamba
from layers.jetmoe.utils import parallel_experts
from model.anemone_config import AnemoneConfig
from model.modeling_anemone import AnemoneForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ai21labs/Jamba-v0.1")

os.environ["WANDB_PROJECT"] = "Mixture of mixture (mod, moah moe)"

# BitLinearNew forward method replacement with nn.Linear forward method
bitnet.BitLinearNew.forward = nn.Linear.forward  # Replace all bitlinear to classic linear

def print_nb_trainable_params(model):
    bf16 = 0
    other = 0
    for name, param in model.named_parameters():
        if "attn" in name or ("mamba" in name and "proj" not in name):
            bf16 += np.prod(param.shape)
        else:
            other += np.prod(param.shape)
    print(f"Attn + Mamba: {bf16 / 1_000_000}M, Other: {other / 1_000_000}M, Total: {(bf16 + other) / 1_000_000}M")
    return bf16+other

# Function to create the model configuration
def create_model_config(hidden_size, num_hidden_layers, intermediate_size, expert_num_heads, capacity, skip_blocks, expert_layer_period):
    return AnemoneConfig(
        attn_layer_offset=5,
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

# Function to expand the model's parameters by copying adjacent parameters
import torch.nn as nn

def expand_model_params(model):
    with torch.no_grad():
        new_params={}
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:  # Assume matrix weights need expansion
                # Double the appropriate dimension, usually the output features for Linear layers
                expanded_param = torch.cat([param, param], dim=0)  # Adjust dim as necessary
                model._parameters[name] = nn.Parameter(expanded_param)
            elif 'bias' in name:
                expanded_param = torch.cat([param, param])  # Bias expansion
                model._parameters[name] = nn.Parameter(expanded_param)

        # Update the model with expanded parameters
        for full_name, new_param in new_params.items():
            parts = full_name.split('.')
            # Navigate to the correct module
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            # Set the new parameter to the module
            setattr(module, parts[-1], new_param)
    return model

# Usage
# Assume `model` is your neural network model
# expand_model_params(model)


# Usage
# Assume `model` is your neural network model
# expand_model_params(model)

# Initialize the base model
initial_hidden_size = 128
initial_num_hidden_layers = 8
initial_intermediate_size = 512
base_model_config = create_model_config(
    hidden_size=initial_hidden_size,
    num_hidden_layers=initial_num_hidden_layers,
    intermediate_size=initial_intermediate_size,
    expert_num_heads=4,
    capacity=128,
    skip_blocks=2,
    expert_layer_period=2
)
model = AnemoneForCausalLM(base_model_config)

param_count=print_nb_trainable_params(model)*2

# Training settings
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

textbooks_split = int(100_000 * 1)
eval_split = int(1_000 * 0.1)

t_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks-2.0", split=f"train[:{textbooks_split}]")
eval_ultra_textbooks = load_dataset("Locutusque/UltraTextbooks-2.0", split=f"train[{textbooks_split}:{textbooks_split + eval_split}]")

key = "text"
train_dataset = t_ultra_textbooks.map(tokenize, batched=True, batch_size=10000, remove_columns=t_ultra_textbooks.column_names)
eval_dataset = eval_ultra_textbooks.map(tokenize, batched=True, batch_size=10000, remove_columns=eval_ultra_textbooks.column_names)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
import torch
import math
from transformers import Trainer, TrainingArguments
from datasets import Dataset
def train_and_expand_model(model, base_dataset, num_epochs, target_params, steps_per_epoch, growth_factor,eval_dataset):
    current_params = sum(p.numel() for p in model.parameters())
    total_doublings = math.ceil(math.log(target_params / current_params, 3))
    initial_subset_size=steps_per_epoch//total_doublings

    subset_size = initial_subset_size

    for epoch in range(num_epochs):
        # Dynamically create a subset of the dataset

        
        # Data loader for the current subset

        for step in range(total_doublings):
            train_subset = Dataset.from_dict(base_dataset[(step)*len(train_dataset)//total_doublings:(step+1)*len(train_dataset)//total_doublings])
            # Train the model on the current subset
            args = TrainingArguments(
                output_dir=f"./training_outputs/epoch_{epoch}_step_{step}",
                num_train_epochs=1,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="steps",
                eval_steps=500,
                save_strategy="no",
                logging_dir="./logs",
                report_to="none",
                learning_rate=5e-4,
                warmup_steps=20,
                gradient_accumulation_steps=1
            )

            trainer = Trainer(
                model=model,
                args=args,
                eval_dataset=eval_dataset,
                train_dataset=train_subset,
                data_collator=data_collator
            )

            trainer.train()

            # Double the model's parameters
            expand_model_params(model)
            current_params = sum(p.numel() for p in model.parameters())
            print_nb_trainable_params(model)
            # Prepare for the next subset
    return model
steps_per_epoch = len(train_dataset) // batch_size

# Assuming `base_dataset` is the complete dataset loaded and preprocessed
model = train_and_expand_model(model, train_dataset, num_epochs, target_params, steps_per_epoch,2, eval_dataset)

