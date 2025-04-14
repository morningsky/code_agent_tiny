# 创建新文件 parse_console.py
import os
import re
import matplotlib.pyplot as plt

def parse_console_output(console_file, output_dir):
    """从console输出文件解析奖励数据并绘制图表"""
    try:
        with open(console_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(console_file, 'r', encoding='latin-1') as f:
            content = f.read()
    
    # 数据容器
    steps = []
    correctness_values = []
    execution_values = []
    simplicity_values = []
    combined_values = []
    
    # 提取奖励计算块
    reward_blocks = re.findall(r'===== 奖励计算开始 =====.*?===== 奖励计算结束 =====', content, re.DOTALL)
    
    step = 0
    for block in reward_blocks:
        step += 1
        
        # 提取各组件奖励，遍历每行找到所有组合奖励的行
        combined_matches = re.findall(r"组合奖励 #\d+: 正确性=([+-]?\d+\.\d+), 执行时间=([+-]?\d+\.\d+), 简洁性=([+-]?\d+\.\d+), 综合=([+-]?\d+\.\d+)", block)
        
        for match in combined_matches:
            steps.append(step)
            correctness_values.append(float(match[0]))
            execution_values.append(float(match[1]))
            simplicity_values.append(float(match[2]))
            combined_values.append(float(match[3]))
    
    if not steps:
        print("未找到奖励数据")
        return
    
    # 绘制图表
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
    print(f"奖励组件图表已保存到 {os.path.join(output_dir, 'reward_components.png')}")

# 使用方法
if __name__ == "__main__":
    output_dir = "code_agent"
    console_file = "log.txt"  # 包含控制台输出的文件
    
    if not os.path.exists(console_file):
        print(f"未找到控制台输出文件: {console_file}")
        print("请将训练过程的控制台输出保存到文件中")
    else:
        parse_console_output(console_file, output_dir)