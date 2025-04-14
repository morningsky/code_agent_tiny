#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CodeAgent GRPO训练主脚本
使用GRPO算法训练Qwen2-0.5B-Instruct模型
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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
    generate_device = model.device
        
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(generate_device) for k, v in inputs.items()}
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.no_grad():
        try:
            outputs = model.generate(
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
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=256,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
            completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
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
        
        # 创建标准训练指标图表
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
        
        try:
            correctness_rewards = history["train/rewards/correctness"].tolist() if "train/rewards/correctness" in history.column_names() else []
            execution_rewards = history["train/rewards/execution_time"].tolist() if "train/rewards/execution_time" in history.column_names() else []
            simplicity_rewards = history["train/rewards/simplicity"].tolist() if "train/rewards/simplicity" in history.column_names() else []
            
            if not correctness_rewards and "correctness" in history.column_names():
                correctness_rewards = history["correctness"].tolist()
            if not execution_rewards and "execution_time" in history.column_names():
                execution_rewards = history["execution_time"].tolist()
            if not simplicity_rewards and "simplicity" in history.column_names():
                simplicity_rewards = history["simplicity"].tolist()
            
            if correctness_rewards or execution_rewards or simplicity_rewards:
                reward_fig, reward_ax = plt.subplots(figsize=(12, 8))
                
                if correctness_rewards:
                    reward_ax.plot(steps[:len(correctness_rewards)], correctness_rewards, 'b-', label='correctness reward')
                if execution_rewards:
                    reward_ax.plot(steps[:len(execution_rewards)], execution_rewards, 'r-', label='execution time reward')
                if simplicity_rewards:
                    reward_ax.plot(steps[:len(simplicity_rewards)], simplicity_rewards, 'g-', label='simplicity reward')
                
                reward_ax.set_title("reward components")
                reward_ax.set_ylabel("reward value")
                reward_ax.set_xlabel("training steps")
                reward_ax.legend()
                reward_ax.grid(True)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "reward_components.png"))
                plt.close()
                
                print(f"各类型奖励变化趋势图已保存到 {os.path.join(output_dir, 'reward_components.png')}")
        except Exception as e:
            print(f"绘制独立奖励图表时出错: {str(e)}")
        
        print(f"训练指标图表已保存到 {os.path.join(output_dir, 'training_metrics.png')}")
    except Exception as e:
        print(f"绘制训练指标图表时出错: {str(e)}")

def clean_memory():
    """清理内存，用于减少内存压力"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    print("内存已清理")

def main():
    output_dir = "code_agent"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(min(7, num_gpus)))  
    
    if len(device_ids) > 0:
        device_name = f"CUDA ({len(device_ids)}个GPU)"
        print(f"使用{len(device_ids)}个GPU进行训练: {device_ids}")
    else:
        device_name = "CPU"
        print("未检测到GPU，将使用CPU训练")
    
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

    wandb.define_metric("train/rewards/correctness")
    wandb.define_metric("train/rewards/execution_time")
    wandb.define_metric("train/rewards/simplicity")
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_args = GRPOConfig(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, "logs"),
        num_train_epochs=3,
        
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        
        fp16=torch.cuda.is_available(),
        
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        
        use_cpu=False,
        local_rank=-1, 
        
        ddp_find_unused_parameters=False,
        
        report_to=["wandb"],
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="epoch",
    )

    print("初始化GRPO训练器...")
    
    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        args=training_args,
        train_dataset=dataset,
        reward_funcs=combined_reward,
    )
    
    print(f"模型已初始化在设备: {trainer.model.device}")
    
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
    trainer.train()
    print("GRPO训练完成！")
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