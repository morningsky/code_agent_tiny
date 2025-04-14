# Code Agent

基于GRPO强化学习的Coding智能体，能够理解代码修改指令并生成高质量的代码优化。

## 项目介绍


1. 使用Qwen/Qwen2-0.5B-Instruct 模型作为基础
2. 采用 TRL 库的 GRPO算法进行训练
3. 设计了包含代码正确性、执行效率和简洁性的奖励函数
4. 提供完整的训练可视化和结果对比

## 项目结构

```
.
├── README.md                 # 项目说明文档
├── requirements.txt          # 项目依赖
├── create_dataset.py         # 数据集创建脚本
├── reward_functions.py       # 奖励函数模块
├── train_model.py            # 训练主脚本
└── code_improvement_agent/   # 训练输出目录(训练后生成)
    ├── logs/                 # 训练日志
    ├── final_model/          # 最终训练模型
    ├── before_training_results.txt  # 训练前模型输出
    ├── after_training_results.txt   # 训练后模型输出
    └── training_metrics.png  # 训练指标可视化
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 创建数据集

```bash
python create_dataset.py
```

这将创建一个包含5个Python函数和对应优化指令的数据集。

### 2. 训练模型

```bash
python train_model.py
```

训练过程会自动加载数据集，并使用GRPO算法训练模型。训练完成后会保存模型并生成训练前后的结果对比。

### 3. 查看训练结果

训练完成后，可以在 `code_improvement_agent` 目录下找到：

- `before_training_results.txt`: 训练前模型生成的代码
- `after_training_results.txt`: 训练后模型生成的代码
- `training_metrics.png`: 训练过程的loss、KL散度和奖励可视化

## 奖励函数设计

本项目设计了三种奖励函数，综合考虑代码质量的多个维度：

1. **代码正确性奖励**：通过执行代码并比较与预期输出来评估代码的功能正确性
2. **执行效率奖励**：测量代码执行时间，奖励更高效的实现
3. **代码简洁性奖励**：基于代码行数和字符数，奖励更简洁的代码

这三种奖励按照 0.5:0.3:0.2 的比例加权组合，以强调代码的正确性是最重要的，其次是执行效率，最后是简洁性。

## GRPO算法说明

GRPO (Group Relative Policy Optimization) 是一种针对数学推理等任务优化的强化学习算法，基于PPO算法改进。它的主要特点是对每个批次内的样本进行相对排序，计算相对优势，而不是与固定基线比较。这种方法可以更好地处理多样化的生成任务，减轻任务难度偏差的影响。

## 参考资料

- [TRL库GRPO文档](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [Qwen/Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 