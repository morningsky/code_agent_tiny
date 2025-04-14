#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CodeAgent GRPO训练主脚本
使用GRPO算法训练Qwen2-0.5B-Instruct模型
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer
from reward_functions import combined_reward
import wandb

def generate_completion(model, prompt, tokenizer):
    """使用模型生成代码完成"""
    generate_device = "cpu"
    
    original_device = None
    if hasattr(model, 'device') and 'mps' in str(model.device):
        print("模型在MPS设备上，生成时临时使用CPU以确保稳定性")
        original_device = model.device
        generate_model = model.to("cpu")
    else:
        generate_model = model
        
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(generate_device) for k, v in inputs.items()}
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.no_grad():
        try:
            outputs = generate_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=512,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                num_beams=1,
                use_cache=True,
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"生成过程中出错: {e}，尝试使用更保守的参数")
            outputs = generate_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
    if original_device is not None:
        model.to(original_device)
            
    return completion

def save_results(results, filename):
    """保存生成结果到文件"""
    with open(filename, "w", encoding="utf-8") as f:
        for i, result in enumerate(results, 1):
            f.write(f"示例 #{i}\n")
            f.write(f"提示:\n{result['prompt']}\n\n")
            f.write(f"生成结果:\n{result['completion']}\n\n")
            f.write("-" * 80 + "\n\n")

def plot_training_metrics(output_dir):
    """绘制训练指标图表"""
    try:
        api = wandb.Api()
        run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
        
        history = run.history()
        
        steps = history["_step"].tolist()
        loss = history["train/loss"].tolist()
        kl_div = history["train/kl_div"].tolist()
        rewards = history["train/rewards/mean"].tolist()
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
        
        axes[0].plot(steps, loss, 'b-')
        axes[0].set_title("Training Loss")
        axes[0].set_ylabel("Loss Value")
        axes[0].grid(True)
        
        axes[1].plot(steps, kl_div, 'r-')
        axes[1].set_title("KL Divergence")
        axes[1].set_ylabel("KL Divergence Value")
        axes[1].grid(True)
        
        axes[2].plot(steps, rewards, 'g-')
        axes[2].set_title("Average Reward")
        axes[2].set_ylabel("Reward Value")
        axes[2].set_xlabel("Training Steps")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_metrics.png"))
        plt.close()
        
        print(f"Training metrics chart saved to {os.path.join(output_dir, 'training_metrics.png')}")
    except Exception as e:
        print(f"Error plotting training metrics: {str(e)}")

def clean_memory():
    """清理内存，用于减少内存压力"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if hasattr(torch, 'mps'):
        try:
            torch.mps.empty_cache()
        except RuntimeError as e:
            print(f"MPS内存清理警告: {e}，但训练将继续")
        
    print("内存已清理")

def main():
    output_dir = "code_improvement_agent"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" in os.environ:
        del os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]
    if "PYTORCH_MPS_LOW_WATERMARK_RATIO" in os.environ:
        del os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]
    if "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
        del os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]
    
    clean_memory()
    
    try:
        dataset = load_from_disk("code_dataset")
        print(f"数据集加载成功，包含 {len(dataset)} 个样本")
    except Exception as e:
        print(f"加载数据集失败: {str(e)}")
        print("尝试创建新数据集...")
        from create_dataset import create_code_dataset
        dataset = create_code_dataset()
    
    wandb.init(project="code-improvement-agent", name="grpo_training")
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_name = "CPU"
    print(f"强制使用CPU训练，绕过MPS检测问题")

    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        num_train_epochs=3,
        
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        
        fp16=False,
        
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        
        use_cpu=True,
        
        report_to=["wandb"],
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
    )

    print("初始化GRPO训练器...")
    
    original_device = torch.device("cpu")
    
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=training_args,
        train_dataset=dataset,
        reward_funcs=combined_reward,
    )
    
    if hasattr(trainer.model, 'to'):
        trainer.model = trainer.model.to('cpu')
    print(f"已确认模型在CPU设备上: {trainer.model.device}")
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
    
    print("训练前进行模型评估...")
    clean_memory()
    before_results = []
    for item in dataset:
        completion = generate_completion(trainer.model, item["prompt"], tokenizer)
        before_results.append({
            "prompt": item["prompt"],
            "completion": completion
        })
    
    save_results(before_results, os.path.join(output_dir, "before_training_results.txt"))
    print(f"训练前结果已保存到 {os.path.join(output_dir, 'before_training_results.txt')}")

    print("开始GRPO训练...")
    clean_memory()
    try:
        trainer.train()
        print("GRPO训练完成！")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "MPS backend out of memory" in str(e):
            print(f"GPU内存不足: {e}")
            print("紧急模式：切换到CPU训练...")
            try:
                trainer.save_model(os.path.join(output_dir, "checkpoint_before_oom"))
                print("已保存当前状态")
            except:
                print("无法保存当前状态，继续切换到CPU...")
            
            if hasattr(trainer.model, 'to'):
                trainer.model = trainer.model.to('cpu')
            
            trainer.args.use_cpu = True
            trainer.args.no_cuda = True
            trainer.args.use_mps_device = False
            trainer.args.fp16 = False
            
            print("使用CPU继续训练...")
            trainer.train()
            print("CPU训练完成！")
        else:
            raise e
    clean_memory()

    trainer.save_model(os.path.join(output_dir, "final_model"))
    print(f"训练后的模型已保存到 {os.path.join(output_dir, 'final_model')}")

    print("训练后进行模型评估...")
    clean_memory()
    after_results = []
    for item in dataset:
        completion = generate_completion(trainer.model, item["prompt"], tokenizer)
        after_results.append({
            "prompt": item["prompt"],
            "completion": completion
        })
    
    save_results(after_results, os.path.join(output_dir, "after_training_results.txt"))
    print(f"训练后结果已保存到 {os.path.join(output_dir, 'after_training_results.txt')}")

    print("绘制训练指标图表...")
    plot_training_metrics(output_dir)
    
    wandb.finish()
    print("训练流程完成！")

if __name__ == "__main__":
    main() 