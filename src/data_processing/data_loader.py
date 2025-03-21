"""
数据加载和预处理模块
"""
import os
import pandas as pd

def load_data(filepath):
    """
    加载股票数据并进行基本预处理
    
    参数:
    filepath: 数据文件路径
    
    返回:
    处理后的数据DataFrame
    """
    data = pd.read_csv(filepath)
    data = data.sort_values('Date')
    return data

def create_output_dir(output_dir='output'):
    """
    创建输出目录（如果不存在）
    
    参数:
    output_dir: 输出目录路径
    
    返回:
    输出目录路径
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 确保目录有写入权限
        if not os.access(output_dir, os.W_OK):
            print(f"警告: 目录 {output_dir} 没有写入权限")
            
        return output_dir
    except Exception as e:
        print(f"创建目录 {output_dir} 时出错: {str(e)}")
        # 尝试在当前目录创建
        fallback_dir = os.path.join(os.getcwd(), 'output')
        print(f"尝试创建备用目录: {fallback_dir}")
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir 