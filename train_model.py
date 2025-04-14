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
import re
import glob
import sys


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

def plot_reward_components_from_log(output_dir, log_dir=None):
    """从日志文件提取并绘制奖励组件图表"""
    
    # 查找最新的日志文件
    if log_dir is None:
        log_dir = os.path.join(output_dir, "logs")
    
    # 尝试从控制台输出日志中提取
    log_files = glob.glob(os.path.join(log_dir, "*.out")) + glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print(f"在 {log_dir} 中未找到日志文件")
        return
        
    log_file = max(log_files, key=os.path.getmtime)
    print(f"使用日志文件: {log_file}")
    
    # 数据容器
    steps = []
    correctness_values = []
    execution_values = []
    simplicity_values = []
    combined_values = []
    
    # 读取日志内容
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        try:
            with open(log_file, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            print(f"无法读取日志文件: {e}")
            return
    
    # 提取步骤数和每个reward组件
    # 从奖励计算日志中提取
    reward_blocks = re.findall(r'===== 奖励计算开始 =====.*?===== 奖励计算结束 =====', content, re.DOTALL)
    
    step = 0
    for block in reward_blocks:
        step += 1  # 每个奖励块视为一个步骤
        
        # 提取各组件奖励
        correctness_match = re.search(r"正确性=([+-]?\d+\.\d+)", block)
        execution_match = re.search(r"执行时间=([+-]?\d+\.\d+)", block)
        simplicity_match = re.search(r"简洁性=([+-]?\d+\.\d+)", block)
        combined_match = re.search(r"综合=([+-]?\d+\.\d+)", block)
        
        if correctness_match and execution_match and simplicity_match and combined_match:
            steps.append(step)
            correctness_values.append(float(correctness_match.group(1)))
            execution_values.append(float(execution_match.group(1)))
            simplicity_values.append(float(simplicity_match.group(1)))
            combined_values.append(float(combined_match.group(1)))
    
    if not steps:
        print("未从日志中找到奖励数据")
        return
        
    # 绘制奖励组件图表
    plt.figure(figsize=(12, 8))
    plt.plot(steps, correctness_values, 'b-', label='correctness')
    plt.plot(steps, execution_values, 'r-', label='execution_time')
    plt.plot(steps, simplicity_values, 'g-', label='simplicity')
    plt.plot(steps, combined_values, 'k--', label='combined')
    plt.title("Reward Components")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reward_components.png"))
    plt.close()
    print(f"奖励组件图表已保存到 {os.path.join(output_dir, 'reward_components.png')}")
    
    # 绘制基本训练指标
    plt.figure(figsize=(10, 6))
    plt.plot(steps, combined_values, 'b-')
    plt.title("Combined Reward")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_reward.png"))
    plt.close()
    print(f"训练奖励图表已保存到 {os.path.join(output_dir, 'training_reward.png')}")

# 添加日志重定向类
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 实时写入文件
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def main():
    output_dir = "code_agent"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # 设置日志重定向
    log_file = "log.txt"
    logger = Logger(log_file)
    sys.stdout = logger
    
    print(f"开始训练，日志将同时保存到 {log_file}")
    
    num_gpus = torch.cuda.device_count()
    device_ids = list(range(min(7, num_gpus)))  
    
    if len(device_ids) > 0:
        device_name = f"CUDA ({len(device_ids)}个GPU)"
        print(f"使用{len(device_ids)}个GPU进行训练: {device_ids}")
    else:
        device_name = "CPU"
        print("未检测到GPU，将使用CPU训练")
    
    
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
        num_train_epochs=3,
        report_to=["wandb"], 
        logging_steps=10
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
    trainer.train()
    print("GRPO训练完成！")

    trainer.save_model(os.path.join(output_dir, "final_model"))
    print(f"训练后的模型已保存到 {os.path.join(output_dir, 'final_model')}")

    print("训练后进行模型评估...")
    after_results = []
    for item in dataset:
        completion = generate_completion(trainer.model, item["prompt"], tokenizer)
        after_results.append({
            "prompt": item["prompt"],
            "completion": completion
        })
    
    save_results(after_results, os.path.join(output_dir, "after_training_results.txt"))
    print(f"训练后结果已保存到 {os.path.join(output_dir, 'after_training_results.txt')}")

    print("从日志绘制奖励组件图表...")
    plot_reward_components_from_log(output_dir)
    wandb.finish()
    
    # 恢复标准输出并关闭日志文件
    sys.stdout = logger.terminal
    logger.close()
    print(f"训练流程完成！日志已保存到 {log_file}")
    print(f"可以使用 python parse.py 解析日志文件生成奖励组件图表")

if __name__ == "__main__":
    main() 