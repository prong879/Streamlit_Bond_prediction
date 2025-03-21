"""
评估模块 - 包含模型评估函数
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import os

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    参数:
    y_true: 真实值
    y_pred: 预测值
    
    返回:
    rmse: 均方根误差
    direction_acc: 方向准确率
    std_error: 误差标准差
    """
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算方向准确率
    direction_acc = direction_accuracy(y_true, y_pred)
    
    # 计算误差标准差
    error = y_true.flatten() - y_pred.flatten()
    std_error = np.std(error)
    
    return rmse, direction_acc, std_error

def direction_accuracy(y_true, y_pred):
    """
    计算方向准确率
    
    参数:
    y_true: 真实值
    y_pred: 预测值
    
    返回:
    方向准确率（百分比）
    """
    # 计算实际和预测的方向变化
    y_true_direction = np.sign(np.diff(y_true.flatten()))
    y_pred_direction = np.sign(np.diff(y_pred.flatten()))
    
    # 过滤掉零值（无变化）
    valid_indices = (y_true_direction != 0)
    
    # 添加调试信息
    total_points = len(y_true_direction)
    valid_points = np.sum(valid_indices)
    up_points = np.sum(y_true_direction[valid_indices] > 0)
    down_points = np.sum(y_true_direction[valid_indices] < 0)
    
    print(f"方向准确率调试信息:")
    print(f"  - 总数据点: {total_points}")
    print(f"  - 有效数据点: {valid_points} ({valid_points/total_points*100:.2f}%)")
    print(f"  - 上涨点: {up_points} ({up_points/valid_points*100:.2f}%)")
    print(f"  - 下跌点: {down_points} ({down_points/valid_points*100:.2f}%)")
    
    # 计算预测的方向分布
    pred_up = np.sum(y_pred_direction[valid_indices] > 0)
    pred_down = np.sum(y_pred_direction[valid_indices] < 0)
    pred_zero = np.sum(y_pred_direction[valid_indices] == 0)
    
    print(f"  - 预测上涨点: {pred_up} ({pred_up/valid_points*100:.2f}%)")
    print(f"  - 预测下跌点: {pred_down} ({pred_down/valid_points*100:.2f}%)")
    print(f"  - 预测无变化点: {pred_zero} ({pred_zero/valid_points*100:.2f}%)")
    
    if np.sum(valid_indices) > 0:
        correct_direction = np.sum(y_true_direction[valid_indices] == y_pred_direction[valid_indices])
        direction_acc = correct_direction / np.sum(valid_indices) * 100
        
        # 打印正确预测的分布
        correct_up = np.sum((y_true_direction[valid_indices] > 0) & (y_pred_direction[valid_indices] > 0))
        correct_down = np.sum((y_true_direction[valid_indices] < 0) & (y_pred_direction[valid_indices] < 0))
        
        if up_points > 0:
            up_acc = correct_up / up_points * 100
        else:
            up_acc = 0
            
        if down_points > 0:
            down_acc = correct_down / down_points * 100
        else:
            down_acc = 0
            
        print(f"  - 上涨预测准确率: {up_acc:.2f}%")
        print(f"  - 下跌预测准确率: {down_acc:.2f}%")
        print(f"  - 总方向准确率: {direction_acc:.2f}%")
    else:
        direction_acc = 0.0
        print("  - 没有有效的方向变化点")
    
    return direction_acc

def inverse_transform_predictions(predictions, y_values, scaler, selected_features):
    """
    反归一化预测结果和真实值
    
    参数:
    predictions: 预测结果
    y_values: 真实值
    scaler: 缩放器
    selected_features: 选择的特征列表
    
    返回:
    反归一化后的预测结果和真实值
    """
    # 创建临时DataFrame来反向转换预测结果
    temp_df = pd.DataFrame(np.zeros((len(predictions), len(selected_features))))
    temp_df[temp_df.columns[0]] = predictions.detach().cpu().numpy().flatten()
    predictions_orig = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)
    
    # 创建临时DataFrame来反向转换真实值
    temp_df = pd.DataFrame(np.zeros((len(y_values), len(selected_features))))
    temp_df[temp_df.columns[0]] = y_values.flatten()  # y_values已经是numpy数组，不需要cpu()
    y_values_orig = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)
    
    return predictions_orig, y_values_orig

def save_evaluation_results(output_dir, train_score, test_score, training_time, train_direction_acc, test_direction_acc):
    """
    保存评估结果到文件
    
    参数:
    output_dir: 输出目录
    train_score: 训练集RMSE
    test_score: 测试集RMSE
    training_time: 训练时间
    train_direction_acc: 训练集方向准确率
    test_direction_acc: 测试集方向准确率
    """
    with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w') as f:
        f.write(f'Training Set RMSE: {train_score:.4f}\n')
        f.write(f'Test Set RMSE: {test_score:.4f}\n')
        f.write(f'Training Time: {training_time:.4f} seconds\n')
        f.write(f'Training Set Direction Accuracy: {train_direction_acc:.2f}%\n')
        f.write(f'Test Set Direction Accuracy: {test_direction_acc:.2f}%\n') 