#!/usr/bin/env python3
"""
测试LSTM数据划分修复
验证新的数据划分逻辑是否能产生更合理的测试集大小
"""

import numpy as np
import pandas as pd

def create_sequences(data, seq_length):
    """模拟LSTM的序列创建函数"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def test_old_split(data_length, sequence_length, train_test_ratio):
    """测试旧的数据划分逻辑"""
    print(f"\n=== 旧的数据划分逻辑 ===")
    print(f"原始数据长度: {data_length}")
    
    # 模拟数据
    data = np.random.randn(data_length, 5)  # 5个特征
    
    # 创建序列
    X, y = create_sequences(data, sequence_length)
    total_samples = len(X)
    print(f"序列数据长度: {total_samples} (损失了{sequence_length}个数据点)")
    
    # 旧的划分逻辑
    train_size = int(total_samples * train_test_ratio)
    val_size = int(total_samples * 0.15)  # 固定15%
    test_size = total_samples - train_size - val_size
    
    print(f"训练集: {train_size} ({train_size/total_samples*100:.1f}%)")
    print(f"验证集: {val_size} ({val_size/total_samples*100:.1f}%)")
    print(f"测试集: {test_size} ({test_size/total_samples*100:.1f}%)")
    
    return test_size

def test_new_split(data_length, sequence_length, train_test_ratio):
    """测试新的数据划分逻辑"""
    print(f"\n=== 新的数据划分逻辑 ===")
    print(f"原始数据长度: {data_length}")
    
    # 模拟数据
    data = np.random.randn(data_length, 5)  # 5个特征
    
    # 创建序列
    X, y = create_sequences(data, sequence_length)
    total_samples = len(X)
    print(f"序列数据长度: {total_samples} (损失了{sequence_length}个数据点)")
    
    # 新的划分逻辑
    train_size = int(total_samples * train_test_ratio)
    test_size = int(total_samples * (1 - train_test_ratio))
    val_size = total_samples - train_size - test_size
    
    # 确保验证集至少有一定的大小
    min_val_size = max(5, int(total_samples * 0.05))
    if val_size < min_val_size and total_samples > min_val_size * 2:
        val_size = min_val_size
        test_size = total_samples - train_size - val_size
    
    print(f"训练集: {train_size} ({train_size/total_samples*100:.1f}%)")
    print(f"验证集: {val_size} ({val_size/total_samples*100:.1f}%)")
    print(f"测试集: {test_size} ({test_size/total_samples*100:.1f}%)")
    
    return test_size

def main():
    """主测试函数"""
    print("LSTM数据划分修复测试")
    print("=" * 50)
    
    # 测试参数
    data_length = 100  # 原始数据长度
    sequence_length = 20  # 序列长度
    train_test_ratio = 0.8  # 训练集比例
    
    print(f"测试参数:")
    print(f"- 原始数据长度: {data_length}")
    print(f"- 序列长度: {sequence_length}")
    print(f"- 训练集比例: {train_test_ratio}")
    
    # 测试旧逻辑
    old_test_size = test_old_split(data_length, sequence_length, train_test_ratio)
    
    # 测试新逻辑
    new_test_size = test_new_split(data_length, sequence_length, train_test_ratio)
    
    # 比较结果
    print(f"\n=== 比较结果 ===")
    print(f"旧逻辑测试集大小: {old_test_size}")
    print(f"新逻辑测试集大小: {new_test_size}")
    print(f"改进: {new_test_size - old_test_size} 个数据点 ({(new_test_size - old_test_size)/old_test_size*100:.1f}%)")
    
    # 与ARIMA比较
    arima_test_size = int(data_length * (1 - train_test_ratio))
    print(f"\nARIMA测试集大小: {arima_test_size}")
    print(f"LSTM新逻辑与ARIMA的差异: {new_test_size - arima_test_size} 个数据点")
    print(f"差异原因: LSTM需要{sequence_length}个数据点来创建序列")

if __name__ == "__main__":
    main() 