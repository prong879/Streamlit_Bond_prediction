# -*- coding: utf-8 -*-
"""
数据处理工具模块
提供数据清洗、转换和兼容性修复功能

主要功能:
1. PyArrow兼容性修复
2. 数据类型转换
3. 缺失值处理
4. 数据验证
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Union, List, Dict, Any
import warnings

def fix_datetime_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    修复DataFrame中的时间戳数据以兼容PyArrow
    
    解决问题:
    - 纳秒精度时间戳转换错误
    - PyArrow序列化失败
    - Streamlit显示异常
    
    参数:
        df (DataFrame): 包含时间戳数据的DataFrame
        
    返回:
        DataFrame: 修复后的DataFrame
    """
    if df is None or df.empty:
        return df
        
    df_fixed = df.copy()
    
    # 检查并修复每一列的时间戳数据
    for col in df_fixed.columns:
        try:
            # 检查是否为时间戳类型
            if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
                # 方法1: 转换为微秒精度
                if df_fixed[col].dtype == 'datetime64[ns]':
                    # 将纳秒精度转换为微秒精度，避免PyArrow转换错误
                    df_fixed[col] = pd.to_datetime(df_fixed[col]).dt.floor('us')
                
                # 方法2: 如果仍有问题，转换为字符串格式
                elif hasattr(df_fixed[col].dtype, 'tz') and df_fixed[col].dtype.tz is not None:
                    # 处理带时区的时间戳
                    df_fixed[col] = df_fixed[col].dt.tz_localize(None).dt.floor('us')
                    
            # 检查object类型列是否包含时间戳
            elif df_fixed[col].dtype == 'object':
                # 尝试检测是否为时间戳字符串
                sample_values = df_fixed[col].dropna().head(5)
                if len(sample_values) > 0:
                    try:
                        # 尝试转换第一个非空值
                        test_conversion = pd.to_datetime(sample_values.iloc[0])
                        if pd.notna(test_conversion):
                            # 如果成功，转换整列并修复精度
                            df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce').dt.floor('us')
                    except (ValueError, TypeError):
                        # 如果转换失败，保持原样
                        pass
                        
        except Exception as e:
            # 如果修复失败，记录警告但不中断程序
            st.warning(f"修复列 '{col}' 的时间戳格式时出现问题: {e}")
            try:
                # 最后的备选方案：转换为字符串
                if pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
                    df_fixed[col] = df_fixed[col].astype(str)
            except:
                # 如果连字符串转换都失败，保持原样
                pass
    
    return df_fixed

def safe_dataframe_display(df: pd.DataFrame, **kwargs) -> None:
    """
    安全显示DataFrame，自动处理PyArrow兼容性问题
    
    参数:
        df (DataFrame): 要显示的DataFrame
        **kwargs: 传递给st.dataframe的其他参数
    """
    try:
        # 先尝试直接显示
        st.dataframe(df, **kwargs)
    except Exception as e:
        # 如果失败，应用修复后再显示
        if "Arrow" in str(e) or "Timestamp" in str(e):
            try:
                fixed_df = fix_datetime_for_arrow(df)
                st.dataframe(fixed_df, **kwargs)
            except Exception as e2:
                st.error(f"数据显示失败: {e2}")
                st.write("原始错误:", str(e))
        else:
            st.error(f"数据显示失败: {e}")

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    验证DataFrame的完整性和质量
    
    参数:
        df (DataFrame): 要验证的DataFrame
        required_columns (List[str]): 必需的列名列表
        
    返回:
        Dict: 验证结果
    """
    if df is None:
        return {"valid": False, "error": "DataFrame为空"}
    
    result = {
        "valid": True,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "warnings": []
    }
    
    # 检查必需列
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            result["valid"] = False
            result["error"] = f"缺少必需的列: {missing_cols}"
            return result
    
    # 检查数据质量
    if df.empty:
        result["warnings"].append("DataFrame为空")
    
    # 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        result["warnings"].append(f"发现 {duplicates} 行重复数据")
    
    # 检查时间戳列的问题
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if df[col].dtype == 'datetime64[ns]':
                # 检查是否有纳秒精度可能导致的问题
                sample = df[col].dropna().head(1)
                if len(sample) > 0:
                    timestamp_str = str(sample.iloc[0])
                    if len(timestamp_str.split('.')[-1]) > 6:  # 超过微秒精度
                        result["warnings"].append(f"列 '{col}' 包含高精度时间戳，可能导致显示问题")
    
    return result

def normalize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    标准化DataFrame的列名
    
    参数:
        df (DataFrame): 原始DataFrame
        
    返回:
        tuple: (标准化后的DataFrame, 重命名信息列表)
    """
    # 标准列名映射
    standard_columns = {
        'Date': ['date', 'time', 'datetime', 'timestamp', 'trade_date', 'trading_date'],
        'Open': ['open', 'open_price', 'opening', 'first', 'first_price'],
        'High': ['high', 'high_price', 'highest', 'max', 'maximum', 'highest_price'],
        'Low': ['low', 'low_price', 'lowest', 'min', 'minimum', 'lowest_price'],
        'Close': ['close', 'close_price', 'closing', 'last', 'last_price', 'close/last', 
                 'adj_close', 'adjusted_close', 'p', 'price', 'y'],
        'Volume': ['volume', 'vol', 'quantity', 'turnover', 'trade_volume', 'trading_volume']
    }
    
    df_normalized = df.copy()
    rename_info = []
    column_mapping = {}
    
    # 获取当前列名的小写形式
    lowercase_columns = {col.lower(): col for col in df.columns}
    processed_columns = set()
    
    # 遍历标准列名及其变体
    for standard, variants in standard_columns.items():
        # 如果标准列名已存在，跳过
        if standard in df.columns:
            continue
        
        # 检查变体是否存在
        for variant in variants:
            if variant in lowercase_columns:
                original_name = lowercase_columns[variant]
                if original_name not in processed_columns:
                    column_mapping[original_name] = standard
                    rename_info.append(f"'{original_name}' → '{standard}'")
                    processed_columns.add(original_name)
                break
    
    # 应用重命名
    if column_mapping:
        df_normalized = df_normalized.rename(columns=column_mapping)
    
    return df_normalized, rename_info

def clean_numeric_data(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    清洗数值型数据
    
    参数:
        df (DataFrame): 原始DataFrame
        columns (List[str]): 要清洗的列名列表，None表示所有数值列
        
    返回:
        DataFrame: 清洗后的DataFrame
    """
    df_cleaned = df.copy()
    
    if columns is None:
        columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df_cleaned.columns:
            # 移除无穷大值
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)
            
            # 可选：移除异常值（使用IQR方法）
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 记录异常值数量
            outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)][col]
            if len(outliers) > 0:
                st.info(f"列 '{col}' 发现 {len(outliers)} 个异常值")
    
    return df_cleaned 