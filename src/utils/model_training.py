"""
模型训练工具模块 - 提供多次训练和模型选择功能
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

def train_multiple_models(model_class, train_func, x_train, y_train, input_dim, 
                         hidden_dim, output_dim, num_layers=None, num_runs=5, 
                         num_epochs=100, output_dir=None, model_name="model"):
    """
    多次训练模型并选择最佳模型
    
    参数:
    model_class: 模型类（SingleLSTM或DualLSTM）
    train_func: 训练函数（train_single_lstm_model或train_dual_lstm_model）
    x_train: 训练集特征
    y_train: 训练集标签
    input_dim: 输入特征维度
    hidden_dim: 隐藏层维度
    output_dim: 输出维度
    num_layers: LSTM层数（对于双层LSTM）
    num_runs: 训练次数
    num_epochs: 每次训练的轮数
    output_dir: 输出目录
    model_name: 模型名称（用于保存文件）
    
    返回:
    best_model: 最佳模型
    best_hist: 最佳模型的训练历史
    best_training_time: 最佳模型的训练时间
    best_loss: 最佳模型的损失值
    training_stats: 所有训练运行的统计信息
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        multi_run_dir = os.path.join(output_dir, "multi_run")
        os.makedirs(multi_run_dir, exist_ok=True)
    
    # 存储所有运行的结果
    all_models = []
    all_hists = []
    all_times = []
    all_losses = []
    
    # 记录训练统计信息
    training_stats = {
        'run_id': [],
        'final_loss': [],
        'best_loss': [],
        'training_time': [],
        'epochs_to_converge': []
    }
    
    print(f"\n开始{model_name}的多次训练 ({num_runs}次)...")
    
    # 多次训练模型
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}:")
        
        # 设置不同的随机种子
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        
        # 创建当前运行的输出目录
        if output_dir:
            run_output_dir = os.path.join(multi_run_dir, f"run_{run+1}")
            os.makedirs(run_output_dir, exist_ok=True)
        else:
            run_output_dir = None
        
        try:
            # 训练模型
            if num_layers is not None:  # 双层LSTM
                model, hist, training_time, best_loss = train_func(
                    x_train, y_train, input_dim, hidden_dim, num_layers, output_dim,
                    num_epochs=num_epochs, output_dir=run_output_dir
                )
            else:  # 单层LSTM
                model, hist, training_time, best_loss = train_func(
                    x_train, y_train, input_dim, hidden_dim, output_dim,
                    num_epochs=num_epochs, output_dir=run_output_dir
                )
            
            # 保存模型和结果
            all_models.append(model)
            all_hists.append(hist)
            all_times.append(training_time)
            all_losses.append(best_loss)
            
            # 记录统计信息
            training_stats['run_id'].append(run + 1)
            training_stats['final_loss'].append(hist[-1])
            training_stats['best_loss'].append(best_loss)
            training_stats['training_time'].append(training_time)
            training_stats['epochs_to_converge'].append(len(hist))
            
            print(f"运行 {run+1} 完成，最佳损失: {best_loss:.6f}, 训练时间: {training_time:.2f}秒")
        except Exception as e:
            print(f"运行 {run+1} 失败: {str(e)}")
            print("跳过此次运行，继续下一次")
            # 记录失败的统计信息
            training_stats['run_id'].append(run + 1)
            training_stats['final_loss'].append(float('nan'))
            training_stats['best_loss'].append(float('nan'))
            training_stats['training_time'].append(0)
            training_stats['epochs_to_converge'].append(0)
            continue
    
    # 找到最佳模型（损失最低的）
    best_idx = np.argmin(all_losses)
    best_model = all_models[best_idx]
    best_hist = all_hists[best_idx]
    best_training_time = all_times[best_idx]
    best_loss = all_losses[best_idx]
    
    print(f"\n多次训练完成，最佳模型来自运行 {best_idx+1}，损失值: {best_loss:.6f}")
    
    # 保存最佳模型
    if output_dir:
        best_model_path = os.path.join(output_dir, f"{model_name}_best_model.pth")
        torch.save(best_model.state_dict(), best_model_path)
        print(f"最佳模型已保存到: {best_model_path}")
        
        # 保存训练统计信息
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(output_dir, f"{model_name}_training_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"训练统计信息已保存到: {stats_path}")
        
        # 可视化多次训练结果
        visualize_multiple_training_runs(training_stats, best_idx, output_dir, model_name)
    
    return best_model, best_hist, best_training_time, best_loss, training_stats

def visualize_multiple_training_runs(stats, best_idx, output_dir, model_name):
    """
    可视化多次训练的结果
    
    参数:
    stats: 训练统计信息
    best_idx: 最佳模型的索引
    output_dir: 输出目录
    model_name: 模型名称
    """
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 1. 最佳损失对比
    plt.subplot(2, 2, 1)
    bars = plt.bar(stats['run_id'], stats['best_loss'], color='skyblue')
    bars[best_idx].set_color('red')  # 标记最佳模型
    plt.title('Best Loss Comparison Across Runs', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Best Loss', fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 2. 训练时间对比
    plt.subplot(2, 2, 2)
    bars = plt.bar(stats['run_id'], stats['training_time'], color='lightgreen')
    bars[best_idx].set_color('red')  # 标记最佳模型
    plt.title('Training Time Comparison', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Training Time (seconds)', fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 3. 收敛轮数对比
    plt.subplot(2, 2, 3)
    bars = plt.bar(stats['run_id'], stats['epochs_to_converge'], color='salmon')
    bars[best_idx].set_color('red')  # 标记最佳模型
    plt.title('Epochs to Converge', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Number of Epochs', fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 4. 最终损失与最佳损失对比
    plt.subplot(2, 2, 4)
    x = np.arange(len(stats['run_id']))
    width = 0.35
    plt.bar(x - width/2, stats['final_loss'], width, label='Final Loss', color='lightblue')
    plt.bar(x + width/2, stats['best_loss'], width, label='Best Loss', color='lightcoral')
    plt.title('Final vs Best Loss', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Loss', fontsize=12, fontname='Times New Roman')
    plt.xticks(x, stats['run_id'])
    plt.legend(prop={'family': 'Times New Roman'})
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f"{model_name}_multi_run_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def train_multiple_arima_models(find_best_params_func, y_train, num_runs=5, 
                               max_p=3, max_d=2, max_q=3, criterion='bic', 
                               output_dir=None):
    """
    多次运行ARIMA参数搜索并选择最佳模型
    
    参数:
    find_best_params_func: 寻找最佳ARIMA参数的函数
    y_train: 训练数据
    num_runs: 运行次数
    max_p, max_d, max_q: ARIMA参数搜索范围
    criterion: 选择标准，'aic'或'bic'
    output_dir: 输出目录
    
    返回:
    best_model: 最佳ARIMA模型
    best_order: 最佳ARIMA参数
    best_search_time: 最佳模型的搜索时间
    training_stats: 所有训练运行的统计信息
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        multi_run_dir = os.path.join(output_dir, "multi_run_arima")
        os.makedirs(multi_run_dir, exist_ok=True)
    
    # 存储所有运行的结果
    all_models = []
    all_orders = []
    all_times = []
    all_criteria = []
    
    # 记录训练统计信息
    training_stats = {
        'run_id': [],
        'best_order': [],
        'criterion_value': [],
        'search_time': []
    }
    
    print(f"\n开始ARIMA模型的多次参数搜索 ({num_runs}次)...")
    
    # 多次训练模型
    for run in range(num_runs):
        print(f"\n运行 {run+1}/{num_runs}:")
        
        # 设置不同的随机种子
        np.random.seed(42 + run)
        
        # 创建当前运行的输出目录
        if output_dir:
            run_output_dir = os.path.join(multi_run_dir, f"run_{run+1}")
            os.makedirs(run_output_dir, exist_ok=True)
        else:
            run_output_dir = None
        
        # 寻找最佳参数
        best_order, best_model, search_time = find_best_params_func(
            y_train, max_p=max_p, max_d=max_d, max_q=max_q,
            criterion=criterion, output_dir=run_output_dir
        )
        
        if best_model is not None:
            # 保存模型和结果
            all_models.append(best_model)
            all_orders.append(best_order)
            all_times.append(search_time)
            
            # 获取AIC或BIC值
            criterion_value = best_model.aic if criterion == 'aic' else best_model.bic
            all_criteria.append(criterion_value)
            
            # 记录统计信息
            training_stats['run_id'].append(run + 1)
            training_stats['best_order'].append(str(best_order))
            training_stats['criterion_value'].append(criterion_value)
            training_stats['search_time'].append(search_time)
    
    # 检查是否有成功的运行
    if not all_models:
        print("所有ARIMA参数搜索运行均失败")
        return None, None, None, training_stats
    
    # 找到最佳模型（AIC/BIC最低的）
    best_idx = np.argmin(all_criteria)
    best_model = all_models[best_idx]
    best_order = all_orders[best_idx]
    best_search_time = all_times[best_idx]
    
    print(f"\n多次参数搜索完成，最佳ARIMA模型来自运行 {best_idx+1}，参数: {best_order}")
    
    # 保存训练统计信息
    if output_dir and training_stats['run_id']:
        stats_df = pd.DataFrame(training_stats)
        stats_path = os.path.join(output_dir, "arima_search_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"ARIMA参数搜索统计信息已保存到: {stats_path}")
        
        # 可视化多次参数搜索结果
        visualize_multiple_arima_runs(training_stats, best_idx, output_dir)
    
    return best_model, best_order, best_search_time, training_stats

def visualize_multiple_arima_runs(stats, best_idx, output_dir):
    """
    可视化多次ARIMA参数搜索的结果
    
    参数:
    stats: 训练统计信息
    best_idx: 最佳模型的索引
    output_dir: 输出目录
    """
    if not stats['run_id']:
        return
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 10))
    
    # 1. AIC/BIC值对比
    plt.subplot(2, 1, 1)
    bars = plt.bar(stats['run_id'], stats['criterion_value'], color='skyblue')
    if best_idx < len(bars):
        bars[best_idx].set_color('red')  # 标记最佳模型
    plt.title('AIC/BIC Comparison Across Runs', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('AIC/BIC Value', fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 在每个柱子上标注最佳参数
    for i, v in enumerate(stats['criterion_value']):
        plt.text(i+1, v, stats['best_order'][i], 
                 ha='center', va='bottom', fontsize=9, fontname='Times New Roman', rotation=45)
    
    # 2. 搜索时间对比
    plt.subplot(2, 1, 2)
    bars = plt.bar(stats['run_id'], stats['search_time'], color='lightgreen')
    if best_idx < len(bars):
        bars[best_idx].set_color('red')  # 标记最佳模型
    plt.title('Search Time Comparison', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Run ID', fontsize=12, fontname='Times New Roman')
    plt.ylabel('Search Time (seconds)', fontsize=12, fontname='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, "arima_multi_run_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close() 