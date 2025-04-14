#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
代码修改Agent的奖励函数模块
包含代码正确性、执行效率和简洁性三种奖励函数
"""

import time
import re
import math

def count_lines(code):
    """计算代码行数"""
    return len(code.strip().split("\n"))

def extract_code(text):
    """
    从文本中提取代码块，处理各种可能的代码块格式
    
    Args:
        text: 可能包含代码的文本
        
    Returns:
        提取的代码，如果没有找到则返回原文本
    """
    python_blocks = re.findall(r'```python\s*(.*?)\s*```', text, re.DOTALL)
    if python_blocks:
        return max(python_blocks, key=len)
    
    code_blocks = re.findall(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        return max(code_blocks, key=len)
    
    def_lines = re.findall(r'(def\s+\w+\s*\(.*?:.*?)(?=\n\n|\Z)', text, re.DOTALL)
    if def_lines:
        return max(def_lines, key=len)
    
    return text

def reward_correctness(completions, prompts=None, **kwargs):
    """
    评估代码正确性的奖励函数
    通过尝试执行代码并与预期输出比较来评估
    
    Args:
        completions: 模型生成的代码完成结果列表
        prompts: 对应的提示列表
        
    Returns:
        rewards: 奖励值列表，范围[-2.0, 1.0]
    """
    rewards = []
    
    for i, completion in enumerate(completions):
        code = extract_code(completion)
        
        try:
            namespace = {}
            exec(code, namespace)
            
            test_cases = [
                {"func": "fibonacci", "args": [10], "expected": 55},
                {"func": "factorial", "args": [5], "expected": 120},
                {"func": "gcd", "args": [48, 18], "expected": 6},
                {"func": "is_prime", "args": [17], "expected": True},
                {"func": "bubble_sort", "args": [[5, 3, 8, 1, 2]], "expected": [1, 2, 3, 5, 8]}
            ]
            
            func_name = None
            for test in test_cases:
                if test["func"] in namespace:
                    func_name = test["func"]
                    break
                    
            if func_name is None:
                print(f"未找到可测试的函数，奖励为-2.0")
                rewards.append(-2.0)
                continue
                
            test_case = next((t for t in test_cases if t["func"] == func_name), None)
            
            result = namespace[func_name](*test_case["args"])
            if result == test_case["expected"]:
                print(f"函数 {func_name} 测试通过，奖励为1.0")
                rewards.append(1.0)
            else:
                print(f"函数 {func_name} 结果 {result} 不符合预期 {test_case['expected']}，奖励为-2.0")
                rewards.append(-2.0)
                
        except Exception as e:
            print(f"代码执行错误: {str(e)}, 奖励为-2.0")
            rewards.append(-2.0)
            
    return rewards

def reward_execution_time(completions, prompts=None, **kwargs):
    """
    评估代码执行效率的奖励函数
    
    Args:
        completions: 模型生成的代码完成结果列表
        prompts: 对应的提示列表
        
    Returns:
        rewards: 奖励值列表，范围[-2.0, 1.0]，值越高表示执行越快
    """
    rewards = []
    
    for completion in completions:
        try:
            code = extract_code(completion)
            
            namespace = {}
            exec(code, namespace)
            
            test_cases = [
                {"func": "fibonacci", "args": [20], "repeat": 3},
                {"func": "factorial", "args": [10], "repeat": 5}, 
                {"func": "gcd", "args": [1234, 5678], "repeat": 10},
                {"func": "is_prime", "args": [9973], "repeat": 10},
                {"func": "bubble_sort", "args": [[i for i in range(100, 0, -1)]], "repeat": 1}
            ]
            
            func_name = None
            for test in test_cases:
                if test["func"] in namespace:
                    func_name = test["func"]
                    break
                    
            if func_name is None:
                print(f"执行时间评估: 未找到可测试的函数，奖励为-2.0")
                rewards.append(-2.0)
                continue
                
            test_case = next((t for t in test_cases if t["func"] == func_name), None)
            
            start_time = time.time()
            for _ in range(test_case["repeat"]):
                namespace[func_name](*test_case["args"])
            execution_time = time.time() - start_time
            
            # 移除特殊阈值，使用更陡峭的指数函数
            reward = 1.0 * math.exp(-50 * execution_time)  # 使用-50作为系数而不是-1
            reward = max(-2.0, min(1.0, reward))  # 限制在[-2.0, 1.0]范围内
            
            print(f"函数 {func_name} 执行时间: {execution_time:.6f}秒, 奖励: {reward:.4f}")
            rewards.append(reward)
            
        except Exception as e:
            print(f"执行时间评估出错: {str(e)}, 奖励为-2.0")
            rewards.append(-2.0)
            
    return rewards

def reward_simplicity(completions, prompts=None, **kwargs):
    """
    评估代码简洁性的奖励函数
    
    Args:
        completions: 模型生成的代码完成结果列表
        prompts: 对应的提示列表
        
    Returns:
        rewards: 奖励值列表，范围[-2.0, 1.0]，值越高表示代码越简洁
    """
    rewards = []
    
    for completion in completions:
        try:
            code = extract_code(completion)
            lines = count_lines(code)
            chars = len(code)
            
            # 行数使用缓和的线性函数
            line_reward = max(-1.0, min(1.0, 1.0 - 0.067 * (lines - 5)))
            
            # 字符数使用对数衰减函数
            if chars <= 100:
                char_reward = 1.0
            else:
                char_reward = 1.0 - 0.7 * math.log10(chars / 100)
                char_reward = max(-1.0, char_reward)
            
            reward = 0.5 * line_reward + 0.5 * char_reward
            
            print(f"代码简洁性评估: 行数={lines}, 字符数={chars}, 奖励={reward:.4f}")
            rewards.append(reward)
            
        except Exception as e:
            print(f"简洁性评估出错: {str(e)}, 奖励为-2.0")
            rewards.append(-2.0)
            
    return rewards

def combined_reward(completions, prompts=None, **kwargs):
    """
    结合三种奖励的综合奖励函数
    
    Args:
        completions: 模型生成的代码完成结果列表
        prompts: 对应的提示列表
        
    Returns:
        rewards: 综合奖励值列表
    """
    print("\n===== 奖励计算开始 =====")
    correctness_rewards = reward_correctness(completions, prompts)
    execution_rewards = reward_execution_time(completions, prompts)
    simplicity_rewards = reward_simplicity(completions, prompts)
    
    combined_rewards = []
    for i in range(len(completions)):
        if correctness_rewards[i] == -2.0:
            combined_rewards.append(-2.0)
            print(f"组合奖励 #{i}: 代码不正确，总奖励设为-2.0")
        else:
            reward = (
                0.6 * correctness_rewards[i] + 
                0.2 * execution_rewards[i] + 
                0.2 * simplicity_rewards[i]
            )
            combined_rewards.append(reward)
            print(f"组合奖励 #{i}: 正确性={correctness_rewards[i]:.4f}, 执行时间={execution_rewards[i]:.4f}, 简洁性={simplicity_rewards[i]:.4f}, 综合={reward:.4f}")
    
    print("===== 奖励计算结束 =====\n")
    return combined_rewards 

# def combined_reward2(responses, prompts, **kwargs):
#     # 增加对无意义重复的惩罚
#     repetition_penalty = -2.0 * detect_repetitive_patterns(responses)
    
#     # 增加对完整代码块的奖励
#     completeness_reward = 1.0 * has_complete_code_blocks(responses)
    
#     # 对元讨论降低奖励
#     meta_discussion_penalty = -1.0 * contains_meta_discussion(responses)
    
#     # 原有奖励
#     correctness = reward_correctness(responses, prompts)
#     execution_time = reward_execution_time(responses)
#     simplicity = reward_simplicity(responses)
    
#     # 调整权重
#     final_reward = (
#         2.0 * correctness +
#         0.5 * execution_time +
#         0.8 * simplicity +
#         1.5 * completeness_reward +
#         repetition_penalty +
#         meta_discussion_penalty
#     )
    
#     return final_reward