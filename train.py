import os
import bitnet
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.nn import functional as F
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
def expand_model_params(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                expanded_param = torch.cat([param, param], dim=0)  # Copy parameters along the first dimension
                new_param = nn.Parameter(expanded_param)
                model.state_dict()[name].copy_(new_param)
            elif 'bias' in name:
                expanded_param = torch.cat([param, param], dim=0)  # Copy parameters along the first dimension
                new_param = nn.Parameter(expanded_param)
                model.state_dict()[name].copy_(new_param)

# Initialize the base model
initial_hidden_size = 128
initial_num_hidden_layers = 2
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

def train_and_expand_model(model, train_dataset, eval_dataset, steps_per_epoch, num_epochs, target_params):
    current_params = sum(p.numel() for p in model.parameters())
    growth_steps = int(steps_per_epoch / 100)  # Perform growth every 1% of an epoch
    
    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            if step % growth_steps == 0 and current_params < target_params:
                expand_model_params(model)
                current_params = sum(p.numel() for p in model.parameters())

            run_name = f"epoch_{epoch}_step_{step}_params_{current_params}"
            
            args = TrainingArguments(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_checkpointing=False,
                gradient_accumulation_steps=1,
                load_best_model_at_end=False,
                warmup_steps=20,
                num_train_epochs=num_epochs,
                report_to=["wandb"],
                evaluation_strategy="steps",
                eval_steps=1_000 * 5 // batch_size,
                learning_rate=5e-4,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                bf16_full_eval=torch.cuda.is_bf16_supported(),
                fp16_full_eval=not torch.cuda.is_bf16_supported(),
                logging_steps=50 // batch_size,
                optim="adamw_8bit",
                optim_target_modules=["anemone"],
                max_steps=steps_per_epoch,
                save_total_limit=1,
                save_strategy="steps",
                save_steps=10_000,
                weight_decay=0.02,
                lr_scheduler_type="linear",
                output_dir="./trains",
                run_name=run_name,
            )

            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
            )
            
            model.to("cuda", dtype=torch.bfloat16)
            trainer.train(resume_from_checkpoint=False)
            trainer.save_model(f"./model-anemone-epoch-{epoch}-step-{step}")

    return model

steps_per_epoch = len(train_dataset) // batch_size
model = train_and_expand_model(model, train_dataset, eval_dataset, steps_per_epoch, num_epochs, target_params)
