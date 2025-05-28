# -*- coding: utf-8 -*-
"""
图表工具模块
提供各种图表生成和配置功能

主要功能:
1. 相关性热力图
2. K线图和成交量图
3. 技术指标图表
4. 统计图表
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional

def create_correlation_heatmap(df_or_matrix, selected_features: List[str] = None) -> Dict[str, Any]:
    """
    创建相关性热力图配置
    
    参数:
        df_or_matrix: DataFrame或相关性矩阵
        selected_features (List[str]): 选择的特征列表
        
    返回:
        dict: ECharts配置项字典
    """
    try:
        # 判断输入是DataFrame还是相关性矩阵
        if hasattr(df_or_matrix, 'corr'):
            # 是DataFrame，需要计算相关性矩阵
            if selected_features:
                corr_df = df_or_matrix[selected_features]
            else:
                corr_df = df_or_matrix.select_dtypes(include=[np.number])
            corr_matrix = corr_df.corr()
        else:
            # 已经是相关性矩阵
            corr_matrix = df_or_matrix
            if selected_features:
                # 过滤选定的特征
                available_features = [f for f in selected_features if f in corr_matrix.columns]
                if available_features:
                    corr_matrix = corr_matrix.loc[available_features, available_features]
        
        # 检查矩阵是否为空
        if corr_matrix.empty:
            return {
                'title': {'text': '相关性热力图 - 无数据', 'left': 'center'},
                'xAxis': {'type': 'category', 'data': []},
                'yAxis': {'type': 'category', 'data': []},
                'series': []
            }
        
        # 四舍五入到2位小数
        corr_matrix = corr_matrix.round(2)
        
        # 根据与Close（收盘价）的相关性排序特征
        if 'Close' in corr_matrix.columns:
            close_correlations = corr_matrix['Close']
            sorted_features = close_correlations.sort_values(ascending=False).index
            corr_matrix = corr_matrix.loc[sorted_features, sorted_features]
        
        # 获取特征名称和相关性数据
        features = list(corr_matrix.columns)
        corr_data = []
        
        # 构建热力图数据，确保数据类型为Python原生类型
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                value = corr_matrix.iloc[i, j]
                # 确保值是Python原生float类型
                corr_data.append([int(j), int(i), float(value)])
        
        # 计算相关性矩阵的最小值和最大值
        corr_min = float(corr_matrix.min().min())
        corr_max = float(corr_matrix.max().max())
        
        # 创建相关性热力图ECharts配置
        option = {
            "tooltip": {
                "position": "top",
                "formatter": "特征: {b}<br/>相关系数: {c}"
            },
            "grid": {
                "top": "5%",
                "left": "15%",
                "bottom": "15%",
                "right": "15%"
            },
            "xAxis": {
                "type": "category",
                "data": features,
                "splitArea": {"show": True},
                "axisLabel": {
                    "rotate": 45,
                    "interval": 0
                }
            },
            "yAxis": {
                "type": "category",
                "data": features,
                "splitArea": {"show": True}
            },
            "visualMap": {
                "min": corr_min,
                "max": corr_max,
                "calculable": True,
                "orient": "vertical",
                "left": "0%",
                "bottom": "20%",
                "inRange": {
                    "color": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", 
                             "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
                }
            },
            "series": [{
                "name": "相关系数",
                "type": "heatmap",
                "data": corr_data,
                "label": {
                    "show": True,
                    "fontSize": 10
                },
                "emphasis": {
                    "itemStyle": {
                        "shadowBlur": 10,
                        "shadowColor": "rgba(0, 0, 0, 0.5)"
                    }
                }
            }]
        }
        
        return option
        
    except Exception as e:
        # 返回错误提示图表
        return {
            'title': {'text': f'相关性热力图生成失败: {str(e)}', 'left': 'center'},
            'xAxis': {'type': 'category', 'data': []},
            'yAxis': {'type': 'category', 'data': []},
            'series': []
        }

def calculate_ma(df: pd.DataFrame, periods: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
    """
    计算移动平均线
    
    参数:
        df (DataFrame): 包含股票价格数据的DataFrame
        periods (list): 移动平均线的周期列表
        
    返回:
        DataFrame: 包含移动平均线的DataFrame
    """
    df_ma = df.copy()
    for period in periods:
        df_ma[f'MA{period}'] = df['Close'].rolling(window=period).mean().round(3)
    return df_ma

def create_echarts_kline_volume(df: pd.DataFrame, selected_mas: List[str] = []) -> Dict[str, Any]:
    """
    创建K线图和成交量联动图表
    
    参数:
        df (DataFrame): 股票数据
        selected_mas (list): 选中的移动平均线列表
        
    返回:
        dict: ECharts配置项字典
    """
    # 转换日期格式
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # 检查是否具备完整的OHLC数据
    has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in df.columns
    
    # 准备K线数据
    if has_ohlc:
        kline_data = [[round(float(o), 3), round(float(c), 3), round(float(l), 3), round(float(h), 3)] 
                     for o, c, l, h in zip(df['Open'], df['Close'], df['Low'], df['High'])]
    else:
        close_values = df['Close'].values
        kline_data = [[float(c), float(c), float(c), float(c)] for c in close_values]
    
    # 准备移动平均线数据
    ma_series = []
    ma_colors = {
        'MA5': '#FF4B4B',
        'MA10': '#00B4D8',
        'MA20': '#2ECC71',
        'MA30': '#9B59B6'
    }
    
    if has_ohlc:
        df_ma = calculate_ma(df)
        
        for ma_name in selected_mas:
            if ma_name in df_ma.columns:
                ma_data = df_ma[ma_name].fillna('').tolist()
                ma_series.append({
                    "name": ma_name,
                    "type": "line",
                    "xAxisIndex": 0,
                    "yAxisIndex": 0,
                    "data": ma_data,
                    "smooth": True,
                    "symbolSize": 3,
                    "symbol": "circle",
                    "showSymbol": False,
                    "lineStyle": {
                        "width": 2,
                        "color": ma_colors.get(ma_name, "#000000")
                    },
                    "itemStyle": {
                        "color": ma_colors.get(ma_name, "#000000")
                    }
                })
    
    # 计算涨跌情况
    if has_ohlc:
        df['price_change'] = df['Close'] - df['Open']
    else:
        df['price_change'] = df['Close'].diff()
    
    # 准备成交量数据
    volume_data = []
    if has_volume:
        for i in range(len(df)):
            color = "#FF4B4B" if df['price_change'].iloc[i] >= 0 else "#2ECC71"
            volume_data.append({
                "value": float(df['Volume'].iloc[i]),
                "itemStyle": {"color": color}
            })
    
    # 创建基础配置
    option = {
        "title": [{
            "text": "股票日K线及均线图",
            "left": "center",
            "top": "0%"
        }],
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross"},
            "backgroundColor": "rgba(245, 245, 245, 0.8)",
            "borderWidth": 1,
            "borderColor": "#ccc",
            "padding": 10,
            "textStyle": {"color": "#000"}
        },
        "legend": {
            "data": ["K线", "收盘价"] + selected_mas,
            "top": "30px"
        },
        "axisPointer": {
            "link": {"xAxisIndex": "all"}
        },
        "grid": [{
            "left": "7%",
            "right": "0%",
            "top": "15%",
            "height": "50%"
        }],
        "xAxis": [{
            "type": "category",
            "data": dates,
            "scale": True,
            "boundaryGap": True,
            "axisLine": {"onZero": False},
            "splitLine": {"show": False},
            "splitNumber": 20,
            "min": "dataMin",
            "max": "dataMax",
            "axisPointer": {"z": 100}
        }],
        "yAxis": [{
            "scale": True,
            "splitArea": {"show": True}
        }],
        "dataZoom": [
            {
                "type": "inside",
                "xAxisIndex": [0],
                "start": 10,
                "end": 100
            },
            {
                "show": True,
                "xAxisIndex": [0],
                "type": "slider",
                "bottom": "5%",
                "start": 10,
                "end": 100
            }
        ],
        "series": [
            {
                "name": "K线",
                "type": "candlestick",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": kline_data,
                "itemStyle": {
                    "color": "#FF4B4B",
                    "color0": "#2ECC71"
                }
            },
            {
                "name": "收盘价",
                "type": "line",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": df['Close'].round(3).tolist(),
                "smooth": True,
                "symbolSize": 3,
                "symbol": "circle",
                "showSymbol": False,
                "lineStyle": {"width": 1, "color": "#ff9900"},
                "itemStyle": {"color": "#ff9900"},
                "opacity": 0.7
            }
        ]
    }
    
    # 添加移动平均线
    option["series"].extend(ma_series)
    
    # 添加成交量图
    if has_volume:
        option["title"].append({
            "text": "成交量",
            "left": "center",
            "top": "70%"
        })
        
        option["grid"].append({
            "left": "7%",
            "right": "0%",
            "top": "75%",
            "height": "15%"
        })
        
        option["xAxis"].append({
            "type": "category",
            "gridIndex": 1,
            "data": dates,
            "scale": True,
            "boundaryGap": True,
            "axisLine": {"onZero": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "axisTick": {"show": False},
            "axisPointer": {"label": {"show": False}}
        })
        
        option["yAxis"].append({
            "scale": True,
            "gridIndex": 1,
            "splitNumber": 2,
            "axisLabel": {"show": True},
            "axisLine": {"show": True},
            "axisTick": {"show": True},
            "splitLine": {"show": True}
        })
        
        for dz in option["dataZoom"]:
            dz["xAxisIndex"] = [0, 1]
        
        option["series"].append({
            "name": "成交量",
            "type": "bar",
            "xAxisIndex": 1,
            "yAxisIndex": 1,
            "data": volume_data
        })
    
    return option 