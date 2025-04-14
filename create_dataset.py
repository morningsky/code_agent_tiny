#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建用于训练代码修改Agent的数据集
"""

import pandas as pd
from datasets import Dataset

def create_code_dataset():
    """创建包含5个Python函数及其优化指令的数据集"""
        
    data = [
        {
            "prompt": "优化以下求斐波那契数列的函数，提高其效率：\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n```",
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "expected": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"
        },
        {
            "prompt": "简化以下计算阶乘的函数：\n```python\ndef factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result = result * i\n    return result\n```",
            "code": "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result = result * i\n    return result",
            "expected": "def factorial(n):\n    return 1 if n == 0 else n * factorial(n-1)"
        },
        {
            "prompt": "修改以下求最大公约数的函数，使其更高效：\n```python\ndef gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n```",
            "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
            "expected": "def gcd(a, b):\n    return a if b == 0 else gcd(b, a % b)"
        },
        {
            "prompt": "优化以下判断素数的函数：\n```python\ndef is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True\n```",
            "code": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True",
            "expected": "def is_prime(n):\n    if n <= 1:\n        return False\n    if n <= 3:\n        return True\n    if n % 2 == 0 or n % 3 == 0:\n        return False\n    i = 5\n    while i * i <= n:\n        if n % i == 0 or n % (i + 2) == 0:\n            return False\n        i += 6\n    return True"
        },
        {
            "prompt": "修改以下冒泡排序函数，使其更简洁：\n```python\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr\n```",
            "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
            "expected": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr"
        }
    ]

    # 创建数据集
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    print(f"数据集创建完成，共{len(dataset)}个样本")
    
    # 保存数据集
    dataset.save_to_disk("code_dataset")
    
    return dataset

if __name__ == "__main__":
    create_code_dataset() 