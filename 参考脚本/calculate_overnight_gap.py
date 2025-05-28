"""
-*- coding: utf-8 -*-

@Author : hf_lcx
@Time : 2025/04/16
@File : calculate_overnight_gap.py
@Description : 计算工商银行隔夜跳空因子(absRetnight)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def calculate_overnight_gap(df, date_field='TradingDay'):
    """
    计算隔夜跳空因子
    
    公式: absRetnight = ∑(t=1,20) |ln(Open_t/Close_t-1)|
    
    其中:
    - Open_t 为当日开盘价
    - Close_t-1 为前一日收盘价
    """
    # 确保日期格式正确并排序
    df[date_field] = pd.to_datetime(df[date_field])
    df = df.sort_values(by=date_field)
    
    # 检查数据中是否包含所需的列
    required_columns = ['OpenPrice', 'PrevClosePrice']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        if 'PrevClosePrice' in missing_columns and 'ClosePrice' in df.columns:
            # 如果没有PrevClosePrice但有ClosePrice，则可以计算PrevClosePrice
            print("未找到PrevClosePrice列，使用ClosePrice计算前一日收盘价...")
            df['PrevClosePrice'] = df['ClosePrice'].shift(1)
            missing_columns.remove('PrevClosePrice')
        
        if missing_columns:
            raise ValueError(f"数据中缺少必要的列: {missing_columns}")
    
    # 计算每日隔夜跳空收益的绝对值
    df['overnight_return'] = np.abs(np.log(df['OpenPrice'] / df['PrevClosePrice']))
    
    # 处理无效值（如0或NaN）
    df['overnight_return'] = df['overnight_return'].replace([np.inf, -np.inf], np.nan)
    
    # 打印每日隔夜跳空收益
    print("\n每日隔夜跳空收益绝对值:")
    result_df = df[[date_field, 'OpenPrice', 'PrevClosePrice', 'overnight_return']]
    print(result_df.head(10))
    
    # 按照公式，计算20日累计值
    window_size = 20
    df['absRetnight_20d'] = df['overnight_return'].rolling(window=window_size).sum()
    
    # 打印结果
    print(f"\n20日隔夜跳空因子(前10行):")
    result_df = df[[date_field, 'absRetnight_20d']].dropna()
    print(result_df.head(10))
    
    return df

def plot_overnight_gap(df, stock_name, date_field='TradingDay'):
    """绘制隔夜跳空因子走势图"""
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_field], df['absRetnight_20d'], label='Overnight Gap Factor (20-day)', color='blue')
    
    # 设置图表标题和标签
    plt.title(f'ICBC (601398) Overnight Gap Factor Trend')
    plt.xlabel('Date')
    plt.ylabel('absRetnight')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    current_date = datetime.now().strftime('%Y%m%d')
    output_file = os.path.join(output_dir, f'ICBC_601398_OvernightGapFactor_{current_date}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {output_file}")
    
    # 显示图表
    plt.tight_layout()
    plt.show()

def main():
    # 数据文件路径
    data_dir = 'data'
    # 查找最新的工商银行数据文件（现在支持pickle格式）
    files = [f for f in os.listdir(data_dir) if f.startswith('工商银行_601398') and (f.endswith('.pkl') or f.endswith('.pickle'))]
    
    if not files:
        raise FileNotFoundError("未找到工商银行pickle数据文件，请先下载数据")
    
    # 获取最新文件
    latest_file = sorted(files)[-1]
    file_path = os.path.join(data_dir, latest_file)
    
    print(f"读取数据文件: {file_path}")
    # 读取pickle格式的数据
    df = pd.read_pickle(file_path)
    
    # 数据基本信息
    print(f"数据行数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    
    # 计算隔夜跳空因子
    print("\n开始计算隔夜跳空因子...")
    result_df = calculate_overnight_gap(df)
    
    # 输出统计信息
    valid_results = result_df['absRetnight_20d'].dropna()
    if not valid_results.empty:
        print("\n隔夜跳空因子统计:")
        print(f"均值 (Mean): {valid_results.mean():.4f}")
        print(f"最大值 (Max): {valid_results.max():.4f}")
        print(f"最小值 (Min): {valid_results.min():.4f}")
        print(f"中位数 (Median): {valid_results.median():.4f}")
        print(f"标准差 (Std Dev): {valid_results.std():.4f}")
    
        # 绘制图表
        plot_overnight_gap(result_df, "ICBC(601398)")
    else:
        print("警告: 无法计算隔夜跳空因子，可能是数据不足或数据质量问题")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc() 