"""
模型对比模块 - 包含模型对比相关的功能
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from src.utils.visualization import plot_model_comparison

def compare_models(models_data, output_dir=None):
    """
    比较多个模型的性能
    
    参数:
    models_data: 字典，包含多个模型的数据，格式为：
        {
            'model_name': {
                'test_predict': 测试集预测结果,
                'y_test': 测试集真实值,
                'rmse': RMSE值,
                'direction_acc': 方向准确率,
                'training_time': 训练时间
            },
            ...
        }
    output_dir: 输出目录
    
    返回:
    None
    """
    # 检查是否有ARMA模型数据
    has_arma = 'arma' in models_data
    
    # 获取模型名称
    model_names = list(models_data.keys())
    
    # 获取单层和双层LSTM模型数据
    single_model_name = 'single_lstm'
    dual_model_name = 'dual_lstm'
    
    single_data = models_data[single_model_name]
    dual_data = models_data[dual_model_name]
    
    # 如果有ARMA模型数据，则一并传入
    if has_arma:
        arma_data = models_data['arma']
        
        # 调用可视化模块的plot_model_comparison函数，包含ARMA模型
        plot_model_comparison(
            data=None,  # 这个参数在当前实现中没有使用
            y_test=single_data['y_test'],  # 三个模型的y_test应该相同
            pred_single=single_data['test_predict'],
            pred_dual=dual_data['test_predict'],
            pred_arma=arma_data['test_predict'],
            rmse_single=single_data['rmse'],
            rmse_dual=dual_data['rmse'],
            rmse_arma=arma_data['rmse'],
            dir_acc_single=single_data['direction_acc'],
            dir_acc_dual=dual_data['direction_acc'],
            dir_acc_arma=arma_data['direction_acc'],
            time_single=single_data['training_time'],
            time_dual=dual_data['training_time'],
            time_arma=arma_data['training_time'],
            output_dir=output_dir
        )
    else:
        # 如果没有ARMA模型数据，则只比较LSTM模型
        plot_model_comparison(
            data=None,
            y_test=single_data['y_test'],
            pred_single=single_data['test_predict'],
            pred_dual=dual_data['test_predict'],
            rmse_single=single_data['rmse'],
            rmse_dual=dual_data['rmse'],
            dir_acc_single=single_data['direction_acc'],
            dir_acc_dual=dual_data['direction_acc'],
            time_single=single_data['training_time'],
            time_dual=dual_data['training_time'],
            output_dir=output_dir
        )

def prepare_model_comparison_data(
    model_single, model_dual,
    x_test_tensor, y_test,
    scaler, selected_features,
    training_time_single, training_time_dual,
    calculate_metrics_func, inverse_transform_func
):
    """
    准备模型对比数据
    
    参数:
    model_single: 单层LSTM模型
    model_dual: 双层LSTM模型
    x_test_tensor: 测试集特征
    y_test: 测试集标签
    scaler: 数据缩放器
    selected_features: 选定的特征
    training_time_single: 单层LSTM训练时间
    training_time_dual: 双层LSTM训练时间
    calculate_metrics_func: 计算评估指标的函数
    inverse_transform_func: 反归一化函数
    
    返回:
    models_data: 模型对比数据字典
    """
    # 设置模型为评估模式
    model_single.eval()
    model_dual.eval()
    
    # 预测
    with torch.no_grad():
        test_predict_single = model_single(x_test_tensor)
        test_predict_dual = model_dual(x_test_tensor)
    
    # 反归一化
    test_predict_orig_single, y_test_orig = inverse_transform_func(
        test_predict_single, y_test, scaler, selected_features
    )
    
    test_predict_orig_dual, _ = inverse_transform_func(
        test_predict_dual, y_test, scaler, selected_features
    )
    
    # 计算评估指标
    print("\n单层LSTM模型方向准确率计算:")
    test_score_single, test_direction_acc_single, test_std_single = calculate_metrics_func(
        y_test_orig, test_predict_orig_single
    )
    
    print("\n双层LSTM模型方向准确率计算:")
    test_score_dual, test_direction_acc_dual, test_std_dual = calculate_metrics_func(
        y_test_orig, test_predict_orig_dual
    )
    
    # 准备模型对比数据
    models_data = {
        'single_lstm': {
            'test_predict': test_predict_orig_single,
            'y_test': y_test_orig,
            'rmse': test_score_single,
            'direction_acc': test_direction_acc_single,
            'training_time': training_time_single
        },
        'dual_lstm': {
            'test_predict': test_predict_orig_dual,
            'y_test': y_test_orig,
            'rmse': test_score_dual,
            'direction_acc': test_direction_acc_dual,
            'training_time': training_time_dual
        }
    }
    
    return models_data

def add_arma_to_comparison_data(models_data, arma_test_pred, arma_test_score, 
                               arma_test_direction_acc, arma_training_time):
    """
    将ARMA模型数据添加到模型对比数据中
    
    参数:
    models_data: 现有的模型对比数据字典
    arma_test_pred: ARMA模型的测试集预测结果
    arma_test_score: ARMA模型的RMSE
    arma_test_direction_acc: ARMA模型的方向准确率
    arma_training_time: ARMA模型的训练时间
    
    返回:
    models_data: 更新后的模型对比数据字典
    """
    # 获取y_test
    y_test = models_data['single_lstm']['y_test']
    
    # 打印ARMA模型的方向准确率，用于调试
    print(f"\nARMA模型方向准确率: {arma_test_direction_acc:.2f}%")
    print(f"单层LSTM方向准确率: {models_data['single_lstm']['direction_acc']:.2f}%")
    print(f"双层LSTM方向准确率: {models_data['dual_lstm']['direction_acc']:.2f}%")
    
    # 添加ARMA模型数据
    models_data['arma'] = {
        'test_predict': arma_test_pred,
        'y_test': y_test,
        'rmse': arma_test_score,
        'direction_acc': arma_test_direction_acc,
        'training_time': arma_training_time
    }
    
    return models_data 