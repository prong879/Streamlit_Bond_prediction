"""
可视化模块 - 包含各种绘图函数
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

def set_custom_style():
    """
    设置自定义的图表样式和配色方案
    
    返回:
    自定义颜色映射
    """
    # 设置Seaborn样式
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.color': '#E0E0E0',
        'axes.edgecolor': '#303030',
        'axes.linewidth': 1.5
    })
    
    # 设置Matplotlib参数
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.edgecolor'] = '#303030'
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 自定义颜色映射
    # 价格图颜色
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#1F77B4',  # 蓝色
        '#FF7F0E',  # 橙色
        '#2CA02C',  # 绿色
        '#D62728',  # 红色
        '#9467BD',  # 紫色
        '#8C564B',  # 棕色
        '#E377C2',  # 粉色
        '#7F7F7F',  # 灰色
        '#BCBD22',  # 黄绿色
        '#17BECF'   # 青色
    ])
    
    # 创建自定义热力图颜色映射
    colors = ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", 
              "#FFFFFF", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    
    return custom_cmap

def set_plot_font(ax, title=None, xlabel=None, ylabel=None, title_size=16, label_size=14, legend=True):
    """
    设置matplotlib图表的字体和样式
    
    参数:
    ax: matplotlib轴对象
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    title_size: 标题字体大小
    label_size: 轴标签字体大小
    legend: 是否显示图例
    """
    if title:
        ax.set_title(title, fontsize=title_size, fontweight='bold', fontfamily='Times New Roman')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, fontweight='bold', fontfamily='Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, fontweight='bold', fontfamily='Times New Roman')
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontfamily('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        label.set_fontfamily('Times New Roman')
    
    # 设置图例字体
    if legend and ax.get_legend() is not None:
        for text in ax.get_legend().get_texts():
            text.set_fontsize(12)
            text.set_fontfamily('Times New Roman')
        
        # 美化图例
        ax.get_legend().get_frame().set_facecolor('#F8F8F8')
        ax.get_legend().get_frame().set_edgecolor('#303030')
    
    # 美化轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#303030')
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # 设置刻度
    ax.tick_params(direction='out', length=6, width=1.5, colors='#303030')

def plot_stock_price(data, output_dir=None):
    """
    绘制原始股票价格走势图
    
    参数:
    data: 股票数据DataFrame
    output_dir: 输出目录
    
    返回:
    None
    """
    plt.figure(figsize=(15, 9), facecolor='white')
    ax = plt.gca()

    # 添加价格线 - 保持较粗的线宽
    plt.plot(data.index, data['Close'], linewidth=2.5, color='#1F77B4', label='Close Price')

    # 添加移动平均线 - 使用与原序列相同的线宽
    ma5 = data['Close'].rolling(window=5).mean()
    ma10 = data['Close'].rolling(window=10).mean()
    ma20 = data['Close'].rolling(window=20).mean()
    ma30 = data['Close'].rolling(window=30).mean()
    plt.plot(data.index, ma5, linewidth=2.5, color='black', label='MA5')
    plt.plot(data.index, ma10, linewidth=2.5, color='#FFD700', label='MA10')  # 黄色
    plt.plot(data.index, ma20, linewidth=2.5, color='#FF7F0E', label='MA20')  # 橘色
    plt.plot(data.index, ma30, linewidth=2.5, color='#2CA02C', label='MA30')  # 绿色

    # 设置x轴刻度
    plt.xticks(range(0, data.shape[0], 20), data['Date'].iloc[::20], rotation=45)

    # 添加标题和标签
    plt.title("Historical Stock Price Trend", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel("Date", fontsize=18, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel("Price (USD)", fontsize=18, fontweight='bold', fontfamily='Times New Roman')

    # 添加图例
    plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 美化图表
    set_plot_font(ax)

    # 添加边框
    plt.box(True)

    # 调整布局
    plt.tight_layout()

    # 保存原始股票价格走势图
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'original_stock_price.png'), dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def plot_correlation_matrix(correlation_matrix, output_dir=None, custom_cmap=None):
    """
    绘制相关性矩阵热力图
    
    参数:
    correlation_matrix: 相关性矩阵
    output_dir: 输出目录
    custom_cmap: 自定义颜色映射
    
    返回:
    None
    """
    plt.figure(figsize=(14, 12), facecolor='white')

    # 绘制热力图
    heatmap = sns.heatmap(
        correlation_matrix, 
        annot=True,                  # 显示数值
        fmt='.2f',                   # 数值格式
        cmap=custom_cmap,            # 使用自定义颜色映射
        linewidths=0.5,              # 网格线宽度
        annot_kws={"size": 10},      # 注释文本大小
        cbar_kws={"shrink": 0.8}     # 颜色条大小
    )

    # 设置标题和字体
    ax = plt.gca()
    set_plot_font(ax, title='Feature Correlation Matrix Analysis')

    # 调整布局
    plt.tight_layout()

    # 保存相关性矩阵图
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')

    plt.show()

def plot_selected_features_correlation(price_features_selected, output_dir=None, custom_cmap=None):
    """
    绘制选定特征的相关性矩阵
    
    参数:
    price_features_selected: 选定的特征数据
    output_dir: 输出目录
    custom_cmap: 自定义颜色映射
    
    返回:
    None
    """
    plt.figure(figsize=(12, 10), facecolor='white')
    correlation_matrix_selected = price_features_selected.corr()
    heatmap = sns.heatmap(
        correlation_matrix_selected, 
        annot=True,
        fmt='.2f',
        cmap=custom_cmap,
        linewidths=0.5,
        annot_kws={"size": 10},
        cbar_kws={"shrink": 0.8}
    )
    ax = plt.gca()
    set_plot_font(ax, title='Selected Features Correlation Matrix')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'selected_features_correlation.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_training_visualization(original, predict, hist, output_dir=None, title="LSTM Training Results"):
    """
    绘制训练可视化图
    
    参数:
    original: 原始数据
    predict: 预测数据
    hist: 训练历史
    output_dir: 输出目录
    title: 图表标题
    
    返回:
    None
    """
    # 创建一个更美观的图表
    fig = plt.figure(figsize=(18, 8), facecolor='white')
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # 第一个子图：预测结果对比
    ax1 = plt.subplot(gs[0])
    # 绘制实际数据
    sns.lineplot(x=original.index, y=original[0], label="Actual Data", color='#1F77B4', 
                linewidth=2.5, ax=ax1)
    # 绘制预测数据
    sns.lineplot(x=predict.index, y=predict[0], label="LSTM Prediction", color='#FF7F0E', 
                linewidth=2, ax=ax1, linestyle='--')

    # 添加预测误差区域
    ax1.fill_between(predict.index, 
                    original[0], 
                    predict[0], 
                    color='#FF7F0E', 
                    alpha=0.2, 
                    label='Prediction Error')

    # 设置图表样式
    set_plot_font(ax1, title=title, xlabel='Trading Days', ylabel='Price (USD)')

    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax1.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

    # 第二个子图：训练损失曲线
    ax2 = plt.subplot(gs[1])
    # 绘制损失曲线
    sns.lineplot(x=range(len(hist)), y=hist, color='#2CA02C', linewidth=2.5, ax=ax2)

    # 添加平滑曲线
    window_size = 5
    if len(hist) > window_size:
        smoothed_hist = np.convolve(hist, np.ones(window_size)/window_size, mode='valid')
        sns.lineplot(x=range(window_size-1, len(hist)), y=smoothed_hist, 
                    color='#D62728', linewidth=2, ax=ax2, label='Smoothed Loss')

    # 设置图表样式
    set_plot_font(ax2, title='Training Loss Curve', xlabel='Epochs', ylabel='Loss Value (MSE)')

    # 添加网格线
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加最终损失值标注
    final_loss = hist[-1]
    ax2.annotate(f'Final Loss: {final_loss:.4f}',
                xy=(len(hist)-1, final_loss),
                xytext=(len(hist)*0.7, final_loss*1.5),
                arrowprops=dict(facecolor='#9467BD', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12, fontweight='bold', 
                fontproperties=font_manager.FontProperties(family='Times New Roman'))

    # 添加图例
    if len(hist) > window_size:
        ax2.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

    # 调整布局
    plt.tight_layout()

    # 保存matplotlib图像到output文件夹
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'training_visualization.png'), dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def plot_validation_results(price_features, y_train_orig, train_predict, y_test_orig, test_predict, 
                           train_score, test_score, train_std, test_std, lookback, output_dir=None, title="LSTM Validation Results"):
    """
    绘制验证结果图
    
    参数:
    price_features: 特征数据
    y_train_orig: 训练集原始标签
    train_predict: 训练集预测结果
    y_test_orig: 测试集原始标签
    test_predict: 测试集预测结果
    train_score: 训练集评分
    test_score: 测试集评分
    train_std: 训练集标准差
    test_std: 测试集标准差
    lookback: 回看窗口大小
    output_dir: 输出目录
    title: 图表标题
    
    返回:
    None
    """
    fig = plt.figure(figsize=(16, 10), facecolor='white')
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    # 第一个子图：训练集和测试集预测结果
    ax1 = plt.subplot(gs[0])

    # 创建日期索引
    train_dates = price_features.index[lookback:lookback+len(y_train_orig)]
    test_dates = price_features.index[lookback+len(y_train_orig):lookback+len(y_train_orig)+len(y_test_orig)]

    # 绘制原始数据
    sns.lineplot(x=price_features.index[lookback:], y=price_features['Close'][lookback:], 
                label="Actual Stock Price", color='#1F77B4', linewidth=2, ax=ax1)

    # 绘制训练集预测
    sns.lineplot(x=train_dates, y=train_predict.flatten(), 
                label="Training Set Prediction", color='#2CA02C', linewidth=1.5, ax=ax1)

    # 绘制测试集预测
    sns.lineplot(x=test_dates, y=test_predict.flatten(), 
                label="Test Set Prediction", color='#FF7F0E', linewidth=1.5, ax=ax1)

    # 添加训练集预测的误差范围
    ax1.fill_between(train_dates, 
                    train_predict.flatten() - train_std, 
                    train_predict.flatten() + train_std, 
                    color='#2CA02C', alpha=0.2, label='Training Error Range (±1σ)')

    # 添加测试集预测的误差范围
    ax1.fill_between(test_dates, 
                    test_predict.flatten() - test_std, 
                    test_predict.flatten() + test_std, 
                    color='#FF7F0E', alpha=0.2, label='Test Error Range (±1σ)')

    # 添加训练集和测试集分隔线
    split_date = train_dates[-1]
    ax1.axvline(x=split_date, color='#D62728', linestyle='--', linewidth=2, label='Train/Test Split')

    # 设置图表样式
    set_plot_font(ax1, title=title, xlabel='Date', ylabel='Stock Price (USD)')

    # 添加RMSE标注
    ax1.annotate(f'Training RMSE: {train_score:.4f}',
                xy=(train_dates[len(train_dates)//4], max(train_predict.flatten())*0.95),
                xytext=(train_dates[len(train_dates)//4], max(train_predict.flatten())*0.95),
                fontsize=12, fontweight='bold', 
                fontproperties=font_manager.FontProperties(family='Times New Roman'),
                bbox=dict(boxstyle="round,pad=0.3", fc="#D8BFD8", ec="black", alpha=0.8))

    ax1.annotate(f'Test RMSE: {test_score:.4f}',
                xy=(test_dates[len(test_dates)//4], max(test_predict.flatten())*0.90),
                xytext=(test_dates[len(test_dates)//4], max(test_predict.flatten())*0.90),
                fontsize=12, fontweight='bold', 
                fontproperties=font_manager.FontProperties(family='Times New Roman'),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFD700", ec="black", alpha=0.8))

    # 第二个子图：预测误差分析
    ax2 = plt.subplot(gs[1])

    # 计算误差
    train_error = y_train_orig.flatten() - train_predict.flatten()
    test_error = y_test_orig.flatten() - test_predict.flatten()

    # 绘制训练集误差
    sns.lineplot(x=train_dates, y=train_error, 
                label="Training Error", color='#2CA02C', linewidth=1.5, ax=ax2)

    # 绘制测试集误差
    sns.lineplot(x=test_dates, y=test_error, 
                label="Test Error", color='#FF7F0E', linewidth=1.5, ax=ax2)

    # 添加误差标准差范围线
    ax2.axhline(y=train_std, color='#2CA02C', linestyle='--', alpha=0.7, label=f'Train Error σ: {train_std:.2f}')
    ax2.axhline(y=-train_std, color='#2CA02C', linestyle='--', alpha=0.7)
    ax2.axhline(y=test_std, color='#FF7F0E', linestyle='--', alpha=0.7, label=f'Test Error σ: {test_std:.2f}')
    ax2.axhline(y=-test_std, color='#FF7F0E', linestyle='--', alpha=0.7)

    # 添加零线
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 添加训练集和测试集分隔线
    ax2.axvline(x=split_date, color='#D62728', linestyle='--', linewidth=2)

    # 设置图表样式
    set_plot_font(ax2, title='Prediction Error Analysis', xlabel='Date', ylabel='Error Value (USD)')

    # 添加网格线
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加图例
    ax2.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

    # 调整布局
    plt.tight_layout()

    # 保存matplotlib图像到output文件夹
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'validation_results.png'), dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def plot_prediction_results(price_features, train_predict, test_predict, train_std, test_std, lookback, 
                           train_score, test_score, output_dir=None, title="LSTM Prediction Results"):
    """
    绘制预测结果图
    
    参数:
    price_features: 特征数据
    train_predict: 训练集预测结果
    test_predict: 测试集预测结果
    train_std: 训练集标准差
    test_std: 测试集标准差
    lookback: 回看窗口大小
    train_score: 训练集评分
    test_score: 测试集评分
    output_dir: 输出目录
    title: 图表标题
    
    返回:
    None
    """
    plt.figure(figsize=(16, 8), facecolor='white')

    # 创建日期索引
    all_dates = price_features.index[lookback:]

    # 创建预测数据框架
    train_plot = np.full((len(price_features), 1), np.nan)
    train_plot[lookback:lookback+len(train_predict)] = train_predict

    test_plot = np.full((len(price_features), 1), np.nan)
    test_plot[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict

    # 创建误差范围数据
    train_upper = np.full((len(price_features), 1), np.nan)
    train_lower = np.full((len(price_features), 1), np.nan)
    train_upper[lookback:lookback+len(train_predict)] = train_predict + train_std
    train_lower[lookback:lookback+len(train_predict)] = train_predict - train_std

    test_upper = np.full((len(price_features), 1), np.nan)
    test_lower = np.full((len(price_features), 1), np.nan)
    test_upper[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict + test_std
    test_lower[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict - test_std

    # 绘制原始数据
    plt.plot(all_dates, price_features['Close'][lookback:], 
            label='Actual Price', color='#1F77B4', linewidth=2.5)

    # 绘制训练集预测
    plt.plot(price_features.index, train_plot, 
            label='Training Prediction', color='#2CA02C', linewidth=1.5)

    # 绘制测试集预测
    plt.plot(price_features.index, test_plot, 
            label='Test Prediction', color='#FF7F0E', linewidth=1.5)

    # 添加误差范围区域
    plt.fill_between(price_features.index, train_lower.flatten(), train_upper.flatten(), 
                    color='#2CA02C', alpha=0.2, label='Training Error Range (±1σ)')
    plt.fill_between(price_features.index, test_lower.flatten(), test_upper.flatten(), 
                    color='#FF7F0E', alpha=0.2, label='Test Error Range (±1σ)')

    # 添加训练集和测试集分隔线
    split_idx = lookback + len(train_predict)
    plt.axvline(x=price_features.index[split_idx], color='#D62728', 
                linestyle='--', linewidth=2, label='Train/Test Split')

    # 设置标题和标签
    plt.title(title, fontsize=20, fontweight='bold', fontfamily='Times New Roman')
    plt.xlabel('Date', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
    plt.ylabel('Price (USD)', fontsize=16, fontweight='bold', fontfamily='Times New Roman')

    # 添加RMSE和误差标准差标注
    plt.annotate(f'Training RMSE: {train_score:.4f}, σ: {train_std:.2f}',
                xy=(price_features.index[lookback+len(train_predict)//4], 
                    max(price_features['Close'][lookback:lookback+len(train_predict)])*0.95),
                xytext=(price_features.index[lookback+len(train_predict)//4], 
                        max(price_features['Close'][lookback:lookback+len(train_predict)])*0.95),
                fontsize=12, fontweight='bold', 
                fontproperties=font_manager.FontProperties(family='Times New Roman'),
                bbox=dict(boxstyle="round,pad=0.3", fc="#D8BFD8", ec="black", alpha=0.8))

    plt.annotate(f'Test RMSE: {test_score:.4f}, σ: {test_std:.2f}',
                xy=(price_features.index[lookback+len(train_predict)+len(test_predict)//4], 
                    max(price_features['Close'][lookback+len(train_predict):])*0.90),
                xytext=(price_features.index[lookback+len(train_predict)+len(test_predict)//4], 
                        max(price_features['Close'][lookback+len(train_predict):])*0.90),
                fontsize=12, fontweight='bold', 
                fontproperties=font_manager.FontProperties(family='Times New Roman'),
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFD700", ec="black", alpha=0.8))

    # 调整布局
    plt.tight_layout()

    # 保存matplotlib图像到output文件夹
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=300, bbox_inches='tight')

    # 显示图像
    plt.show()

def plot_feature_importance(correlation_matrix, output_dir=None):
    """
    绘制特征重要性图
    
    参数:
    correlation_matrix: 相关性矩阵
    output_dir: 输出目录
    
    返回:
    特征重要性排名
    """
    # 使用简单的相关性分析评估特征重要性
    feature_importance = abs(correlation_matrix['Close']).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importance.drop('Close').plot(kind='bar')
    ax = plt.gca()
    set_plot_font(ax, title='Feature Importance (Correlation with Closing Price)', xlabel='Features', ylabel='Absolute Correlation')
    plt.tight_layout()
    
    # 保存特征重要性图像到output文件夹
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        
        # 将特征重要性排名，保存到文本文件
        with open(os.path.join(output_dir, 'feature_importance.txt'), 'w', encoding='utf-8') as f:
            f.write("Feature Importance Ranking:\n")
            for i, (feature, importance) in enumerate(feature_importance.items()):
                if feature != 'Close':
                    f.write(f"{i}. {feature}: {importance:.4f}\n")
                    print(f"{i}. {feature}: {importance:.4f}")
    
    # 显示图像
    plt.show()
    
    return feature_importance

def plot_model_comparison(data, y_test, pred_single, pred_dual, 
                         rmse_single, rmse_dual, 
                         dir_acc_single, dir_acc_dual,
                         time_single, time_dual,
                         output_dir=None, pred_arma=None, rmse_arma=None, 
                         dir_acc_arma=None, time_arma=None):
    """
    绘制模型对比图
    
    参数:
    data: 原始数据
    y_test: 测试集真实值
    pred_single: 单层LSTM预测值
    pred_dual: 双层LSTM预测值
    rmse_single: 单层LSTM的RMSE
    rmse_dual: 双层LSTM的RMSE
    dir_acc_single: 单层LSTM的方向准确率
    dir_acc_dual: 双层LSTM的方向准确率
    time_single: 单层LSTM的训练时间
    time_dual: 双层LSTM的训练时间
    output_dir: 输出目录
    pred_arma: ARMA模型预测值（可选）
    rmse_arma: ARMA模型的RMSE（可选）
    dir_acc_arma: ARMA模型的方向准确率（可选）
    time_arma: ARMA模型的训练时间（可选）
    
    返回:
    None
    """
    try:
        # 检查是否包含ARMA模型
        has_arma = pred_arma is not None and rmse_arma is not None and dir_acc_arma is not None and time_arma is not None
        
        # 确保所有数据都是numpy数组并且是一维的
        y_test_np = y_test.flatten() if hasattr(y_test, 'flatten') else np.array(y_test).flatten()
        pred_single_np = pred_single.flatten() if hasattr(pred_single, 'flatten') else np.array(pred_single).flatten()
        pred_dual_np = pred_dual.flatten() if hasattr(pred_dual, 'flatten') else np.array(pred_dual).flatten()
        
        if has_arma:
            pred_arma_np = pred_arma.flatten() if hasattr(pred_arma, 'flatten') else np.array(pred_arma).flatten()
            
            # 确保所有数组长度一致
            min_len = min(len(y_test_np), len(pred_single_np), len(pred_dual_np), len(pred_arma_np))
            y_test_np = y_test_np[:min_len]
            pred_single_np = pred_single_np[:min_len]
            pred_dual_np = pred_dual_np[:min_len]
            pred_arma_np = pred_arma_np[:min_len]
        else:
            # 确保所有数组长度一致
            min_len = min(len(y_test_np), len(pred_single_np), len(pred_dual_np))
            y_test_np = y_test_np[:min_len]
            pred_single_np = pred_single_np[:min_len]
            pred_dual_np = pred_dual_np[:min_len]
        
        plt.figure(figsize=(15, 10))
        
        # 1. 预测结果对比
        plt.subplot(2, 2, 1)
        plt.plot(y_test_np, label='Actual Value', color='black', linewidth=2)
        plt.plot(pred_single_np, label=f'Single LSTM (RMSE: {rmse_single:.4f})', color='blue', linestyle='--')
        plt.plot(pred_dual_np, label=f'Dual LSTM (RMSE: {rmse_dual:.4f})', color='red', linestyle='-.')
        
        if has_arma:
            plt.plot(pred_arma_np, label=f'ARMA (RMSE: {rmse_arma:.4f})', color='green', linestyle=':')
        
        plt.title('Test Set Prediction Comparison', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 2. 误差对比
        plt.subplot(2, 2, 2)
        error_single = np.abs(y_test_np - pred_single_np)
        error_dual = np.abs(y_test_np - pred_dual_np)
        
        plt.plot(error_single, label='Single LSTM Error', color='blue', alpha=0.7)
        plt.plot(error_dual, label='Dual LSTM Error', color='red', alpha=0.7)
        
        if has_arma:
            error_arma = np.abs(y_test_np - pred_arma_np)
            plt.plot(error_arma, label='ARMA Error', color='green', alpha=0.7)
        
        plt.title('Prediction Error Comparison', fontsize=14)
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 3. 性能指标对比 - 条形图
        plt.subplot(2, 2, 3)
        metrics = ['RMSE', 'Direction Accuracy (%)', 'Training Time (s)']
        single_values = [rmse_single, dir_acc_single, time_single]
        dual_values = [rmse_dual, dir_acc_dual, time_dual]
        
        x = np.arange(len(metrics))
        width = 0.25 if has_arma else 0.35  # 如果有ARMA模型，减小条形宽度
        
        plt.bar(x - width if has_arma else x - width/2, single_values, width, label='Single LSTM', color='blue', alpha=0.7)
        plt.bar(x if has_arma else x + width/2, dual_values, width, label='Dual LSTM', color='red', alpha=0.7)
        
        if has_arma:
            arma_values = [rmse_arma, dir_acc_arma, time_arma]
            plt.bar(x + width, arma_values, width, label='ARMA', color='green', alpha=0.7)
        
        plt.title('Model Performance Metrics', fontsize=14)
        plt.xticks(x, metrics, fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 4. 误差分布对比 - 箱线图
        plt.subplot(2, 2, 4)
        boxplot_data = [error_single, error_dual]
        boxplot_labels = ['Single LSTM', 'Dual LSTM']
        
        if has_arma:
            boxplot_data.append(error_arma)
            boxplot_labels.append('ARMA')
        
        plt.boxplot(boxplot_data, tick_labels=boxplot_labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
        
        plt.title('Prediction Error Distribution', fontsize=14)
        plt.ylabel('Absolute Error', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.tight_layout()
        
        # 保存matplotlib图像到output文件夹
        if output_dir:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            
            # 保存模型对比结果到文本文件
            save_model_comparison_report(
                rmse_single, rmse_dual, 
                dir_acc_single, dir_acc_dual, 
                time_single, time_dual, 
                output_dir,
                rmse_arma, dir_acc_arma, time_arma
            )
        
        # 显示图像
        plt.show()
    except Exception as e:
        print(f"绘制模型对比图时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def save_model_comparison_report(rmse_single, rmse_dual, 
                                dir_acc_single, dir_acc_dual, 
                                time_single, time_dual, 
                                output_dir, rmse_arma=None, 
                                dir_acc_arma=None, time_arma=None):
    """
    保存模型对比分析报告
    
    参数:
    rmse_single: 单层LSTM的RMSE
    rmse_dual: 双层LSTM的RMSE
    dir_acc_single: 单层LSTM的方向准确率
    dir_acc_dual: 双层LSTM的方向准确率
    time_single: 单层LSTM的训练时间
    time_dual: 双层LSTM的训练时间
    output_dir: 输出目录
    rmse_arma: ARMA模型的RMSE（可选）
    dir_acc_arma: ARMA模型的方向准确率（可选）
    time_arma: ARMA模型的训练时间（可选）
    
    返回:
    None
    """
    try:
        # 检查是否包含ARMA模型
        has_arma = rmse_arma is not None and dir_acc_arma is not None and time_arma is not None
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'model_comparison.txt'), 'w', encoding='utf-8') as f:
            f.write("Model Performance Comparison Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Single LSTM Model:\n")
            f.write(f"  - RMSE: {rmse_single:.4f}\n")
            f.write(f"  - Direction Accuracy: {dir_acc_single:.2f}%\n")
            f.write(f"  - Training Time: {time_single:.2f} seconds\n\n")
            
            f.write("Dual LSTM Model:\n")
            f.write(f"  - RMSE: {rmse_dual:.4f}\n")
            f.write(f"  - Direction Accuracy: {dir_acc_dual:.2f}%\n")
            f.write(f"  - Training Time: {time_dual:.2f} seconds\n\n")
            
            if has_arma:
                f.write("ARMA Model:\n")
                f.write(f"  - RMSE: {rmse_arma:.4f}\n")
                f.write(f"  - Direction Accuracy: {dir_acc_arma:.2f}%\n")
                f.write(f"  - Training Time: {time_arma:.2f} seconds\n\n")
            
            # 计算LSTM模型之间的性能差异百分比
            rmse_diff_pct = (rmse_single - rmse_dual) / rmse_single * 100
            dir_acc_diff_pct = (dir_acc_dual - dir_acc_single) / dir_acc_single * 100
            time_diff_pct = (time_dual - time_single) / time_single * 100
            
            f.write("Performance Difference Analysis (LSTM Models):\n")
            f.write(f"  - RMSE Difference: {rmse_diff_pct:.2f}% ({'Dual is better' if rmse_diff_pct > 0 else 'Single is better'})\n")
            f.write(f"  - Direction Accuracy Difference: {dir_acc_diff_pct:.2f}% ({'Dual is better' if dir_acc_diff_pct > 0 else 'Single is better'})\n")
            f.write(f"  - Training Time Difference: {time_diff_pct:.2f}% ({'Single is faster' if time_diff_pct > 0 else 'Dual is faster'})\n\n")
            
            if has_arma:
                # 计算ARMA与LSTM模型的性能差异
                rmse_arma_single_pct = (rmse_single - rmse_arma) / rmse_single * 100
                rmse_arma_dual_pct = (rmse_dual - rmse_arma) / rmse_dual * 100
                dir_acc_arma_single_pct = (dir_acc_arma - dir_acc_single) / dir_acc_single * 100
                dir_acc_arma_dual_pct = (dir_acc_arma - dir_acc_dual) / dir_acc_dual * 100
                
                f.write("Performance Difference Analysis (ARMA vs LSTM):\n")
                f.write(f"  - RMSE Difference (ARMA vs Single): {rmse_arma_single_pct:.2f}% ({'ARMA is better' if rmse_arma_single_pct > 0 else 'Single is better'})\n")
                f.write(f"  - RMSE Difference (ARMA vs Dual): {rmse_arma_dual_pct:.2f}% ({'ARMA is better' if rmse_arma_dual_pct > 0 else 'Dual is better'})\n")
                f.write(f"  - Direction Accuracy Difference (ARMA vs Single): {dir_acc_arma_single_pct:.2f}% ({'ARMA is better' if dir_acc_arma_single_pct > 0 else 'Single is better'})\n")
                f.write(f"  - Direction Accuracy Difference (ARMA vs Dual): {dir_acc_arma_dual_pct:.2f}% ({'ARMA is better' if dir_acc_arma_dual_pct > 0 else 'Dual is better'})\n\n")
            
            f.write("Conclusion:\n")
            if has_arma:
                # 找出RMSE最低的模型
                rmse_values = [rmse_single, rmse_dual, rmse_arma]
                model_names = ["Single LSTM", "Dual LSTM", "ARMA"]
                best_rmse_idx = np.argmin(rmse_values)
                best_rmse_model = model_names[best_rmse_idx]
                
                # 找出方向准确率最高的模型
                dir_acc_values = [dir_acc_single, dir_acc_dual, dir_acc_arma]
                best_dir_acc_idx = np.argmax(dir_acc_values)
                best_dir_acc_model = model_names[best_dir_acc_idx]
                
                f.write(f"  - Best model for RMSE: {best_rmse_model} ({rmse_values[best_rmse_idx]:.4f})\n")
                f.write(f"  - Best model for Direction Accuracy: {best_dir_acc_model} ({dir_acc_values[best_dir_acc_idx]:.2f}%)\n\n")
                
                # 综合建议
                f.write("  Considering both prediction accuracy and computational efficiency:\n")
                
                # 简单的决策逻辑
                if best_rmse_idx == best_dir_acc_idx:
                    f.write(f"  {best_rmse_model} is recommended as it performs best in both metrics.\n")
                else:
                    # 如果最佳RMSE和最佳方向准确率不是同一个模型，则根据RMSE差异和训练时间做出建议
                    if best_rmse_idx == 0:  # 单层LSTM最佳
                        f.write("  Single LSTM model is recommended for its balance of accuracy and efficiency.\n")
                    elif best_rmse_idx == 1:  # 双层LSTM最佳
                        f.write("  Dual LSTM model is recommended for its superior prediction accuracy.\n")
                    else:  # ARMA最佳
                        f.write("  ARMA model is recommended for its computational efficiency and good accuracy.\n")
            else:
                # 原有的LSTM模型对比逻辑
                if rmse_dual < rmse_single and dir_acc_dual > dir_acc_single:
                    f.write("  Dual LSTM model outperforms Single LSTM model in both RMSE and direction accuracy.\n")
                elif rmse_single < rmse_dual and dir_acc_single > dir_acc_dual:
                    f.write("  Single LSTM model outperforms Dual LSTM model in both RMSE and direction accuracy.\n")
                else:
                    if rmse_dual < rmse_single:
                        f.write("  Dual LSTM model performs better in RMSE.\n")
                    else:
                        f.write("  Single LSTM model performs better in RMSE.\n")
                        
                    if dir_acc_dual > dir_acc_single:
                        f.write("  Dual LSTM model performs better in direction accuracy.\n")
                    else:
                        f.write("  Single LSTM model performs better in direction accuracy.\n")
                
                f.write("\n  Considering both prediction accuracy and computational efficiency, ")
                if rmse_dual < rmse_single and time_diff_pct < 50:  # 如果双层更准确且时间增加不超过50%
                    f.write("Dual LSTM model is recommended.\n")
                elif rmse_single < rmse_dual and time_diff_pct > 0:  # 如果单层更准确且更快
                    f.write("Single LSTM model is recommended.\n")
                else:
                    f.write("the choice depends on specific application requirements.\n")
    except Exception as e:
        print(f"保存模型对比报告时出错: {str(e)}")
        import traceback
        traceback.print_exc() 