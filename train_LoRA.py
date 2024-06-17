import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import transformers
import argparse

from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from make_args import Args

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default='./config/SOLAR_train_config.json', help="config path")
opt = parser.parse_args()

# load config.json
args = Args(opt.config_path)

## Load Dataset
data_files = {"train": args.train_path, 
              "validation": args.val_path}
data = load_dataset('json', data_files=data_files)

# make Instruction Format
data = data.map(
    lambda x: {'text': f"###입력:{x['input']}\n\n###출력:{x['output']}$&%</s>"}
)

# Load Pretrained Solar10.7B
max_seq_length = args.max_seq_length # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = args.pretrained_model_name, # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # device_map='auto'
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

## Get peft
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

model = FastLanguageModel.get_peft_model(
    model,
    r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = args.lora_target_modules,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = args.use_gradient_checkpointing,
    random_state = args.random_state,
    use_rslora = args.use_rslora,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

print_trainable_parameters(model)

training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        evaluation_strategy=args.evaluation_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        logging_strategy="steps",
        output_dir=args.output_dir,
        optim=args.optim,
        load_best_model_at_end=args.load_best_model_at_end,
        save_total_limit=args.save_total_limit
    )

trainer = SFTTrainer(model=model, 
                     tokenizer = tokenizer,
                     args=training_args, 
                     train_dataset=data["train"], 
                     eval_dataset=data['validation'],
                     dataset_text_field = "text",
                     max_seq_length = max_seq_length,
                     packing = args.packing)

model.config.use_cache = args.model_use_cache  # silence the warnings. Please re-enable for inference!

# Train!!
trainer_stats = trainer.train()