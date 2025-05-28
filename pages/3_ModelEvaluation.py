# -*- coding: utf-8 -*-
"""
模型评估页面 (Model Evaluation Page)
此模块提供训练完成模型的详细性能分析和对比功能

主要功能:
1. 模型性能对比分析
2. 预测结果可视化
3. 误差分析和残差检验
4. 模型诊断工具
5. 详细评估报告生成

技术栈:
- streamlit: Web应用框架
- pandas: 数据处理和分析
- numpy: 数学计算
- streamlit_echarts: 图表可视化
- sklearn: 机器学习评估指标
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import sys
import traceback
from streamlit_echarts import st_echarts
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import torch

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入工具函数
try:
    from src.utils.session import get_state, set_state
    from src.utils.visualization import ModelVisualization
except ImportError:
    # 如果导入失败，创建空函数
    def get_state(key, default=None):
        return st.session_state.get(key, default)
    
    def set_state(key, value):
        st.session_state[key] = value

# 导入ARIMA模型的图表函数
arima_import_success = False
try:
    # 添加项目根目录到路径
    import sys
    from pathlib import Path
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from src.models.arima_model import (
        create_timeseries_chart,
        create_histogram_chart,
        prepare_arima_charts
    )
    arima_import_success = True
except ImportError as e:
    arima_import_error = str(e)
    # 创建占位函数
    def create_timeseries_chart(*args, **kwargs):
        return {"title": {"text": "图表函数未导入"}}
    def create_histogram_chart(*args, **kwargs):
        return {"title": {"text": "图表函数未导入"}}
    def prepare_arima_charts(*args, **kwargs):
        return {"residuals_chart": None, "residuals_hist": None}
    
    # 创建备用的图表函数，使用与模型训练页面相同的实现
    def create_timeseries_chart_backup(data, title='时间序列图'):
        """备用时间序列图表函数"""
        if isinstance(data, pd.DataFrame):
            # 如果是DataFrame，取第一列数据
            series_data = data.iloc[:, 0].values
            dates = data.index.tolist() if hasattr(data.index, 'tolist') else list(range(len(series_data)))
        else:
            # 如果是Series或数组
            series_data = data if hasattr(data, '__iter__') else [data]
            dates = list(range(len(series_data)))
        
        option = {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"fontSize": 14}
            },
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [str(d) for d in dates],
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {
                "type": "value"
            },
            "series": [{
                "type": "line",
                "data": [float(x) for x in series_data],
                "lineStyle": {"color": "#5470c6", "width": 2},
                "symbol": "none"
            }],
            "dataZoom": [{
                "type": "slider",
                "start": 0,
                "end": 100
            }]
        }
        return option
    
    def create_histogram_chart_backup(data, title='分布直方图', bins=30):
        """备用直方图函数"""
        if hasattr(data, 'values'):
            data = data.values
        if hasattr(data, 'flatten'):
            data = data.flatten()
        
        # 过滤NaN值
        clean_data = [float(x) for x in data if not pd.isna(x) and not np.isnan(float(x))]
        
        if not clean_data:
            return {
                "title": {"text": f"{title} - 无有效数据", "left": "center"},
                "xAxis": {"type": "category", "data": []},
                "yAxis": {"type": "value"},
                "series": [{"type": "bar", "data": []}]
            }
        
        hist, bin_edges = np.histogram(clean_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        option = {
            "title": {
                "text": title,
                "left": "center",
                "textStyle": {"fontSize": 14}
            },
            "tooltip": {"trigger": "axis"},
            "xAxis": {
                "type": "category",
                "data": [f"{float(x):.3f}" for x in bin_centers],
                "name": "值"
            },
            "yAxis": {
                "type": "value",
                "name": "频次"
            },
            "series": [{
                "type": "bar",
                "data": [int(x) for x in hist],
                "itemStyle": {"color": "#73c0de"}
            }]
        }
        return option
    
    # 如果ARIMA函数导入失败，使用备用函数
    if not arima_import_success:
        create_timeseries_chart = create_timeseries_chart_backup
        create_histogram_chart = create_histogram_chart_backup

def fix_datetime_for_arrow(df):
    """
    修复DataFrame中的时间戳数据以兼容PyArrow
    
    参数:
        df (DataFrame): 包含时间戳数据的DataFrame
        
    返回:
        DataFrame: 修复后的DataFrame
    """
    df_fixed = df.copy()
    
    # 检查每一列是否包含时间戳数据
    for col in df_fixed.columns:
        if df_fixed[col].dtype == 'datetime64[ns]':
            # 将纳秒精度的时间戳转换为微秒精度
            df_fixed[col] = pd.to_datetime(df_fixed[col]).dt.floor('us')
        elif pd.api.types.is_datetime64_any_dtype(df_fixed[col]):
            # 处理其他时间戳格式
            try:
                df_fixed[col] = pd.to_datetime(df_fixed[col]).dt.floor('us')
            except Exception as e:
                st.warning(f"列 {col} 的时间戳转换失败: {e}")
                # 如果转换失败，将其转换为字符串
                df_fixed[col] = df_fixed[col].astype(str)
    
    return df_fixed

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 页面配置
st.set_page_config(
    page_title="模型评估",
    page_icon="📈",
    layout="wide"
)

# 页面标题和简介
st.title("📈 模型评估")
st.markdown("对训练完成的模型进行详细性能分析和对比评估")

# 显示ARIMA图表函数导入状态
if arima_import_success:
    st.success("✅ ARIMA图表函数导入成功")
else:
    st.error(f"❌ 无法导入ARIMA图表函数: {arima_import_error}")
    st.warning("误差分析图表可能无法正常显示，请检查ARIMA模型文件")

def check_model_availability():
    """检查可用的训练模型"""
    available_models = []
    model_info = {}
    
    # 检查LSTM模型
    lstm_available = False
    # 更严格的LSTM检测：必须有明确的LSTM训练完成标志或者有训练好的模型
    if st.session_state.get('lstm_training_complete', False):
        lstm_available = True
    elif st.session_state.get('training_complete', False) and st.session_state.get('trained_model') is not None:
        # 只有当有实际的训练模型时才认为LSTM可用
        lstm_available = True
    
    if lstm_available:
        available_models.append("LSTM")
        metrics = st.session_state.get('model_metrics', {})
        model_info["LSTM"] = {
            "status": "已训练",
            "metrics": metrics,
            "training_time": st.session_state.get('training_time', "未知"),
            "has_predictions": 'y_test' in st.session_state or 'lstm_test_predictions' in st.session_state
        }
    
    # 检查ARIMA模型
    arima_available = False
    if st.session_state.get('arima_training_complete', False):
        if st.session_state.get('arima_model') is not None:
            arima_available = True
        elif 'arima_model_metrics' in st.session_state and st.session_state['arima_model_metrics']:
            arima_available = True
        elif 'arima_training_result' in st.session_state:
            arima_available = True
    
    if arima_available:
        available_models.append("ARIMA")
        metrics = st.session_state.get('arima_model_metrics', {})
        model_info["ARIMA"] = {
            "status": "已训练",
            "metrics": metrics,
            "training_time": st.session_state.get('arima_training_time', "未知"),
            "has_predictions": 'arima_training_result' in st.session_state
        }
    
    return available_models, model_info

def create_model_comparison_radar():
    """创建模型性能雷达图"""
    # 示例数据，实际应该从session state获取
    radar_option = {
        "title": {
            "text": "模型性能雷达图",
            "left": "center",
            "textStyle": {"fontSize": 16}
        },
        "tooltip": {
            "trigger": "item"
        },
        "legend": {
            "data": ["LSTM", "ARIMA"],
            "bottom": "10px"
        },
        "radar": {
            "indicator": [
                {"name": "准确性", "max": 100},
                {"name": "稳定性", "max": 100},
                {"name": "速度", "max": 100},
                {"name": "鲁棒性", "max": 100},
                {"name": "可解释性", "max": 100}
            ],
            "center": ["50%", "50%"],
            "radius": "70%"
        },
        "series": [{
            "type": "radar",
            "data": [
                {
                    "value": [85, 90, 70, 80, 60],
                    "name": "LSTM",
                    "itemStyle": {"color": "#5470c6"}
                },
                {
                    "value": [75, 85, 95, 85, 90],
                    "name": "ARIMA",
                    "itemStyle": {"color": "#91cc75"}
                }
            ]
        }]
    }
    return radar_option

def create_prediction_comparison_chart(dates, actual_values, lstm_pred=None, arima_pred=None):
    """创建预测对比图表"""
    # 确保所有数据都是Python原生类型
    actual_values = np.array(actual_values).astype(float).tolist()
    
    series_data = [
        {
            "name": "实际值",
            "type": "line",
            "smooth": True,
            "data": actual_values,
            "showSymbol": False,
            "connectNulls": True
        }
    ]
    
    legend_data = ["实际值"]
    
    if lstm_pred is not None:
        lstm_pred = np.array(lstm_pred).astype(float).tolist()
        series_data.append({
            "name": "LSTM预测",
            "type": "line",
            "smooth": True,
            "data": lstm_pred,
            "showSymbol": False,
            "connectNulls": True
        })
        legend_data.append("LSTM预测")
    
    if arima_pred is not None:
        arima_pred = np.array(arima_pred).astype(float).tolist()
        series_data.append({
            "name": "ARIMA预测",
            "type": "line",
            "smooth": True,
            "data": arima_pred,
            "showSymbol": False,
            "connectNulls": True
        })
        legend_data.append("ARIMA预测")
    
    option = {
        "title": {
            "text": "预测值 vs 实际值对比",
            "left": "center"
        },
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"}
        },
        "legend": {
            "data": legend_data,
            "top": "30px"
        },
        "grid": {
            "left": "3%",
            "right": "4%",
            "bottom": "3%",
            "containLabel": True
        },
        "toolbox": {
            "feature": {
                "saveAsImage": {}
            }
        },
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": dates
        },
        "yAxis": {
            "type": "value",
            "scale": True,
            "splitLine": {
                "show": True
            }
        },
        "dataZoom": [
            {
                "type": "inside",
                "start": 0,
                "end": 100
            },
            {
                "start": 0,
                "end": 100
            }
        ],
        "series": series_data
    }
    return option

def create_error_distribution_chart(errors, model_name="模型"):
    """创建误差分布直方图"""
    try:
        # 安全地转换数据类型
        if hasattr(errors, 'values'):
            errors = errors.values
        if hasattr(errors, 'flatten'):
            errors = errors.flatten()
        
        # 过滤NaN值
        clean_errors = []
        for err in errors:
            try:
                if not pd.isna(err) and not np.isnan(float(err)):
                    clean_errors.append(float(err))
            except (ValueError, TypeError):
                continue
        
        if not clean_errors:
            return {
                "title": {"text": f"{model_name}误差分布 - 无有效数据", "left": "center"},
                "xAxis": {"type": "category", "data": []},
                "yAxis": {"type": "value"},
                "series": [{"type": "bar", "data": []}]
            }
        
        # 计算直方图数据
        hist, bin_edges = np.histogram(clean_errors, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        option = {
            "title": {
                "text": f"{model_name}误差分布",
                "left": "center",
                "textStyle": {"fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis"
            },
            "xAxis": {
                "type": "category",
                "data": [f"{float(x):.3f}" for x in bin_centers],
                "name": "误差值"
            },
            "yAxis": {
                "type": "value",
                "name": "频次"
            },
            "series": [{
                "type": "bar",
                "data": [int(x) for x in hist],
                "itemStyle": {"color": "#73c0de"}
            }]
        }
        return option
    except Exception as e:
        st.error(f"创建误差分布图表失败: {e}")
        return {
            "title": {"text": f"{model_name}误差分布 - 创建失败", "left": "center"},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value"},
            "series": [{"type": "bar", "data": []}]
        }

def create_residual_analysis_chart(residuals, dates):
    """创建残差分析图表"""
    try:
        # 安全地转换数据类型
        if hasattr(residuals, 'values'):
            residuals = residuals.values
        if hasattr(residuals, 'flatten'):
            residuals = residuals.flatten()
        
        # 确保数据是Python原生类型，处理NaN值
        clean_residuals = []
        clean_dates = []
        for i, (res, date) in enumerate(zip(residuals, dates)):
            try:
                if pd.isna(res) or np.isnan(float(res)):
                    continue  # 跳过NaN值
                clean_residuals.append(float(res))
                clean_dates.append(str(date))
            except (ValueError, TypeError):
                continue  # 跳过无法转换的值
        
        if not clean_residuals:
            # 如果没有有效数据，返回空图表
            return {
                "title": {"text": "残差分析 - 无有效数据", "left": "center"},
                "xAxis": {"type": "category", "data": []},
                "yAxis": {"type": "value"},
                "series": [{"type": "line", "data": []}]
            }
        
        option = {
            "title": {
                "text": "残差时间序列分析",
                "left": "center",
                "textStyle": {"fontSize": 14}
            },
            "tooltip": {
                "trigger": "axis"
            },
            "xAxis": {
                "type": "category",
                "data": clean_dates,
                "axisLabel": {"rotate": 45}
            },
            "yAxis": {
                "type": "value",
                "name": "残差"
            },
            "series": [{
                "type": "line",
                "data": clean_residuals,
                "lineStyle": {"color": "#ee6666", "width": 1},
                "symbol": "none"
            }],
            "dataZoom": [{
                "type": "slider",
                "start": 0,
                "end": 100
            }]
        }
        return option
    except Exception as e:
        st.error(f"创建残差分析图表失败: {e}")
        return {
            "title": {"text": "残差分析 - 创建失败", "left": "center"},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value"},
            "series": [{"type": "line", "data": []}]
        }

def create_scatter_plot(actual, predicted, model_name="模型"):
    """创建散点图：预测vs实际"""
    try:
        # 安全地转换数据类型
        if hasattr(actual, 'values'):
            actual = actual.values
        if hasattr(predicted, 'values'):
            predicted = predicted.values
        if hasattr(actual, 'flatten'):
            actual = actual.flatten()
        if hasattr(predicted, 'flatten'):
            predicted = predicted.flatten()
        
        # 过滤NaN值，确保两个数组长度一致
        clean_actual = []
        clean_predicted = []
        for i, (a, p) in enumerate(zip(actual, predicted)):
            try:
                if not pd.isna(a) and not pd.isna(p) and not np.isnan(float(a)) and not np.isnan(float(p)):
                    clean_actual.append(float(a))
                    clean_predicted.append(float(p))
            except (ValueError, TypeError):
                continue
        
        if not clean_actual or not clean_predicted:
            return {
                "title": {"text": f"{model_name}预测散点图 - 无有效数据", "left": "center"},
                "xAxis": {"type": "value"},
                "yAxis": {"type": "value"},
                "series": [{"type": "scatter", "data": []}]
            }
        
        # 计算R²
        r2 = float(r2_score(clean_actual, clean_predicted))
        
        # 创建对角线数据（完美预测线）
        min_val = float(min(min(clean_actual), min(clean_predicted)))
        max_val = float(max(max(clean_actual), max(clean_predicted)))
        diagonal_line = [min_val, max_val]
        
        option = {
            "title": {
                "text": f"{model_name}预测散点图 (R² = {r2:.3f})",
                "left": "center"
            },
            "tooltip": {
                "trigger": "item",
                "formatter": "实际值: {data[0]}<br/>预测值: {data[1]}"
            },
            "grid": {
                "left": "3%",
                "right": "4%",
                "bottom": "3%",
                "containLabel": True
            },
            "toolbox": {
                "feature": {
                    "saveAsImage": {}
                }
            },
            "xAxis": {
                "type": "value",
                "name": "实际值",
                "min": min_val,
                "max": max_val
            },
            "yAxis": {
                "type": "value",
                "name": "预测值",
                "min": min_val,
                "max": max_val
            },
            "series": [
                {
                    "type": "scatter",
                    "data": [[clean_actual[i], clean_predicted[i]] for i in range(len(clean_actual))],
                    "itemStyle": {"color": "#5470c6", "opacity": 0.6},
                    "symbolSize": 6
                },
                {
                    "type": "line",
                    "data": [[diagonal_line[0], diagonal_line[0]], [diagonal_line[1], diagonal_line[1]]],
                    "lineStyle": {"color": "#ee6666", "type": "dashed"},
                    "symbol": "none",
                    "name": "完美预测线"
                }
            ]
        }
        return option
    except Exception as e:
        st.error(f"创建散点图失败: {e}")
        return {
            "title": {"text": f"{model_name}预测散点图 - 创建失败", "left": "center"},
            "xAxis": {"type": "value"},
            "yAxis": {"type": "value"},
            "series": [{"type": "scatter", "data": []}]
        }

def calculate_model_metrics(actual, predicted):
    """计算模型评估指标"""
    try:
        # 安全地转换数据类型
        if hasattr(actual, 'values'):
            actual = actual.values
        if hasattr(predicted, 'values'):
            predicted = predicted.values
        if hasattr(actual, 'flatten'):
            actual = actual.flatten()
        if hasattr(predicted, 'flatten'):
            predicted = predicted.flatten()
        
        # 过滤NaN值，确保两个数组长度一致
        clean_actual = []
        clean_predicted = []
        for i, (a, p) in enumerate(zip(actual, predicted)):
            try:
                if not pd.isna(a) and not pd.isna(p) and not np.isnan(float(a)) and not np.isnan(float(p)):
                    clean_actual.append(float(a))
                    clean_predicted.append(float(p))
            except (ValueError, TypeError):
                continue
        
        if not clean_actual or not clean_predicted:
            return {
                "MSE": 0.0,
                "RMSE": 0.0,
                "MAE": 0.0,
                "MAPE": 0.0,
                "方向准确率": 0.0,
                "R²": 0.0
            }
        
        # 转换为numpy数组
        actual_array = np.array(clean_actual)
        predicted_array = np.array(clean_predicted)
        
        mse = float(mean_squared_error(actual_array, predicted_array))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(actual_array, predicted_array))
        
        # 计算MAPE，避免除零错误
        try:
            mape = float(np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100)
        except (ZeroDivisionError, RuntimeWarning):
            mape = 0.0
        
        # 计算方向准确率
        if len(actual_array) > 1:
            actual_direction = np.sign(actual_array[1:] - actual_array[:-1])
            pred_direction = np.sign(predicted_array[1:] - predicted_array[:-1])
            direction_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
        else:
            direction_accuracy = 0.0
        
        # 计算R²
        r2 = float(r2_score(actual_array, predicted_array))
        
        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "方向准确率": direction_accuracy,
            "R²": r2
        }
    except Exception as e:
        st.error(f"计算模型指标失败: {e}")
        return {
            "MSE": 0.0,
            "RMSE": 0.0,
            "MAE": 0.0,
            "MAPE": 0.0,
            "方向准确率": 0.0,
            "R²": 0.0
        }

def get_prediction_data():
    """
    统一获取预测数据和实际值
    
    返回:
        tuple: (actual_values, lstm_pred, arima_pred, dates, has_real_data)
    """
    has_real_data = False
    dates = []
    actual_values = []
    lstm_pred = None
    arima_pred = None
    
    # 尝试获取真实的测试数据和预测结果
    if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
        df = st.session_state['raw_data'].copy()
        
        # 修复数据类型问题
        try:
            # 确保数值列是数值类型
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 确保日期列是正确的日期时间格式
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                # 移除无效的日期
                df = df.dropna(subset=['Date'])
                # 按日期排序
                df = df.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            st.warning(f"数据类型修复失败: {e}")
            return [], None, None, [], False
        
        # 首先尝试获取统一的actual_values基准
        base_actual_values = []
        
        # 获取LSTM预测数据
        if st.session_state.get('lstm_training_complete', False):
            try:
                # 优先使用保存的预测结果
                if 'lstm_test_predictions' in st.session_state:
                    lstm_pred = st.session_state['lstm_test_predictions']
                    
                    # 关键修复：直接从原始数据获取真实的实际值，与训练页面保持一致
                    train_test_ratio = st.session_state.get('train_test_ratio', 0.8)
                    
                    # 获取目标列（通常是Close）
                    target_column = 'Close'  # 默认使用Close列
                    if 'selected_features' in st.session_state:
                        selected_features = st.session_state['selected_features']
                        # 如果Close在选择的特征中，使用它；否则使用第一个特征
                        if 'Close' in selected_features:
                            target_column = 'Close'
                        elif selected_features:
                            target_column = selected_features[0]
                    
                    # 从原始数据中获取真实的测试集实际值
                    if target_column in df.columns:
                        # 使用与ARIMA完全一致的数据划分方式
                        train_size = int(len(df) * train_test_ratio)
                        test_actual_values = df[target_column].iloc[train_size:].values
                        
                        # 现在LSTM测试集应该与ARIMA测试集大小一致
                        # 但由于序列创建，LSTM预测数量可能仍然少于原始测试集
                        if len(lstm_pred) < len(test_actual_values):
                            # 截取对应长度的实际值，从测试集末尾开始
                            base_actual_values = test_actual_values[-len(lstm_pred):]
                        elif len(lstm_pred) > len(test_actual_values):
                            # 如果LSTM预测点数多于实际值，截取LSTM预测
                            lstm_pred = lstm_pred[:len(test_actual_values)]
                            base_actual_values = test_actual_values
                        else:
                            base_actual_values = test_actual_values
                        
                        if len(base_actual_values) > 0 and len(lstm_pred) > 0:
                            has_real_data = True
                    
                # 如果没有保存的预测结果，尝试重新生成
                elif 'trained_model' in st.session_state and 'X_test' in st.session_state:
                    model = st.session_state['trained_model']
                    X_test = st.session_state['X_test']
                    target_scaler = st.session_state.get('target_scaler')
                    
                    # 确保数据格式正确
                    if not isinstance(X_test, (int, float)):
                        # 转换为torch tensor
                        if not isinstance(X_test, torch.Tensor):
                            X_test_tensor = torch.FloatTensor(X_test)
                        else:
                            X_test_tensor = X_test
                        
                        model.eval()
                        with torch.no_grad():
                            predictions = model(X_test_tensor)
                            lstm_pred = predictions.detach().cpu().numpy().flatten()
                            
                            # 反归一化预测值
                            if target_scaler is not None:
                                lstm_pred = target_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
                            
                            # 获取真实的实际值（与上面的逻辑保持一致）
                            train_test_ratio = st.session_state.get('train_test_ratio', 0.8)
                            target_column = 'Close'
                            if 'selected_features' in st.session_state:
                                selected_features = st.session_state['selected_features']
                                if 'Close' in selected_features:
                                    target_column = 'Close'
                                elif selected_features:
                                    target_column = selected_features[0]
                            
                            if target_column in df.columns:
                                train_size = int(len(df) * train_test_ratio)
                                test_actual_values = df[target_column].iloc[train_size:].values
                                
                                # 现在LSTM测试集应该与ARIMA测试集大小一致
                                # 但由于序列创建，LSTM预测数量可能仍然少于原始测试集
                                if len(lstm_pred) < len(test_actual_values):
                                    # 截取对应长度的实际值，从测试集末尾开始
                                    base_actual_values = test_actual_values[-len(lstm_pred):]
                                elif len(lstm_pred) > len(test_actual_values):
                                    # 如果LSTM预测点数多于实际值，截取LSTM预测
                                    lstm_pred = lstm_pred[:len(test_actual_values)]
                                    base_actual_values = test_actual_values
                                else:
                                    base_actual_values = test_actual_values
                            
                            has_real_data = True
                            
            except Exception as e:
                st.warning(f"获取LSTM预测数据失败: {e}")
        
        # 获取ARIMA预测数据
        if st.session_state.get('arima_training_complete', False) and 'arima_training_result' in st.session_state:
            try:
                arima_result = st.session_state['arima_training_result']
                if 'test_pred' in arima_result:
                    # 确保ARIMA预测数据是正确的格式
                    arima_pred_raw = arima_result['test_pred']
                    if hasattr(arima_pred_raw, 'values'):
                        arima_pred = arima_pred_raw.values.flatten()
                    elif hasattr(arima_pred_raw, 'flatten'):
                        arima_pred = arima_pred_raw.flatten()
                    else:
                        arima_pred = np.array(arima_pred_raw).flatten()
                    
                    # 如果还没有base_actual_values，从ARIMA结果获取
                    if len(base_actual_values) == 0 and 'test_data' in arima_result:
                        test_data_raw = arima_result['test_data']
                        if hasattr(test_data_raw, 'values'):
                            base_actual_values = test_data_raw.values.flatten()
                        elif hasattr(test_data_raw, 'flatten'):
                            base_actual_values = test_data_raw.flatten()
                        else:
                            base_actual_values = np.array(test_data_raw).flatten()
                    
                    has_real_data = True
            except Exception as e:
                st.warning(f"获取ARIMA预测数据失败: {e}")
        
        # 设置最终的actual_values
        actual_values = base_actual_values
        
        # 生成日期序列
        if len(actual_values) > 0:
            if 'Date' in df.columns and len(df) > 0:
                try:
                    # 使用测试集对应的日期
                    train_test_ratio = st.session_state.get('train_test_ratio', 0.8)
                    train_size = int(len(df) * train_test_ratio)
                    test_end_idx = train_size + len(actual_values)
                    
                    # 确保索引不超出范围
                    if test_end_idx <= len(df):
                        test_dates_raw = df['Date'].iloc[train_size:test_end_idx]
                        # 安全地转换日期为字符串
                        dates = []
                        for date in test_dates_raw:
                            try:
                                if pd.isna(date):
                                    dates.append(f"Day {len(dates)+1}")
                                elif hasattr(date, 'strftime'):
                                    dates.append(date.strftime('%Y-%m-%d'))
                                else:
                                    dates.append(str(date))
                            except Exception:
                                dates.append(f"Day {len(dates)+1}")
                    else:
                        # 如果索引超出范围，生成默认日期
                        dates = [f"Day {i+1}" for i in range(len(actual_values))]
                except Exception as e:
                    st.warning(f"日期生成失败: {e}")
                    dates = [f"Day {i+1}" for i in range(len(actual_values))]
            else:
                dates = [f"Day {i+1}" for i in range(len(actual_values))]
    
    # 如果没有真实数据，使用示例数据进行演示
    if not has_real_data:
        # 生成示例数据
        np.random.seed(42)
        actual_values = np.random.randn(50).cumsum() + 100
        dates = [f"Day {i}" for i in range(50)]
        
        if st.session_state.get('lstm_training_complete', False):
            lstm_pred = actual_values + np.random.randn(50) * 0.5
        
        if st.session_state.get('arima_training_complete', False):
            arima_pred = actual_values + np.random.randn(50) * 0.3
    
    return actual_values, lstm_pred, arima_pred, dates, has_real_data

# 检查模型可用性
available_models, model_info = check_model_availability()

# 添加调试信息展开框
with st.expander("🔧 调试信息", expanded=False):
    st.markdown("**Session State 关键信息:**")
    debug_info = {
        "LSTM训练完成": st.session_state.get('lstm_training_complete', False),
        "通用训练完成": st.session_state.get('training_complete', False),
        "ARIMA训练完成": st.session_state.get('arima_training_complete', False),
        "已训练模型": st.session_state.get('trained_model') is not None,
        "ARIMA模型": st.session_state.get('arima_model') is not None,
        "模型指标": 'model_metrics' in st.session_state,
        "ARIMA指标": 'arima_model_metrics' in st.session_state,
        "测试数据": 'y_test' in st.session_state,
        "X测试数据": 'X_test' in st.session_state,
        "LSTM预测结果": 'lstm_test_predictions' in st.session_state,
        "ARIMA结果": 'arima_training_result' in st.session_state,
        "原始数据": 'raw_data' in st.session_state
    }
    
    # 添加数据类型和形状信息
    if 'y_test' in st.session_state:
        y_test = st.session_state['y_test']
        debug_info[f"y_test类型"] = f"{type(y_test)} - 形状: {getattr(y_test, 'shape', 'N/A')}"
    
    if 'X_test' in st.session_state:
        X_test = st.session_state['X_test']
        debug_info[f"X_test类型"] = f"{type(X_test)} - 形状: {getattr(X_test, 'shape', 'N/A')}"
    
    for key, value in debug_info.items():
        if value:
            st.success(f"✅ {key}: {value}")
        else:
            st.error(f"❌ {key}: {value}")
    
    st.markdown("**可用模型:**")
    if available_models:
        for model in available_models:
            st.success(f"✅ {model}")
            info = model_info[model]
            st.json(info)
    else:
        st.warning("没有可用的模型")

if not available_models:
    st.warning("⚠️ 没有检测到已训练的模型。请先在模型训练页面训练模型。")
    st.info("💡 提示：您需要先完成以下步骤：")
    st.markdown("""
    1. 在**数据查看**页面加载数据
    2. 在**模型训练**页面训练LSTM或ARIMA模型
    3. 训练完成后返回此页面进行评估
    """)
    st.stop()

# 快速状态概览
st.subheader("📊 模型状态概览")
status_cols = st.columns(len(available_models) + 2)

with status_cols[0]:
    st.metric("已训练模型", f"{len(available_models)}个", 
              " + ".join(available_models))

with status_cols[1]:
    # 确定最佳模型
    best_model = "未知"
    best_metric = "N/A"
    if available_models:
        # 简单比较MSE来确定最佳模型
        best_mse = float('inf')
        for model in available_models:
            metrics = model_info[model]['metrics']
            if 'MSE' in metrics and metrics['MSE'] < best_mse:
                best_mse = metrics['MSE']
                best_model = model
                best_metric = f"MSE: {best_mse:.4f}"
    
    st.metric("最佳模型", best_model, best_metric)

# 为每个可用模型显示状态
for i, model in enumerate(available_models):
    with status_cols[i + 2]:
        metrics = model_info[model]['metrics']
        rmse_value = metrics.get('RMSE', 0)
        st.metric(f"{model} RMSE", f"{rmse_value:.4f}", 
                  model_info[model]['status'])

# 侧边栏配置
with st.sidebar:
    st.subheader("📋 评估配置")
    
    # 模型选择
    with st.expander("模型选择", expanded=True):
        selected_models = st.multiselect(
            "选择要评估的模型",
            options=available_models,
            default=available_models,
            help="选择一个或多个模型进行对比分析"
        )
    
    # 评估指标选择
    with st.expander("评估指标", expanded=True):
        metrics_options = ["MSE", "RMSE", "MAE", "MAPE", "方向准确率", "R²"]
        selected_metrics = st.multiselect(
            "选择评估指标",
            options=metrics_options,
            default=["MSE", "RMSE", "MAE", "方向准确率"],
            help="选择用于模型评估的指标"
        )
    
    # 时间范围选择
    with st.expander("时间范围", expanded=True):
        evaluation_period = st.selectbox(
            "评估时间段",
            options=["测试集", "训练集", "全部数据"],
            index=0,
            help="选择用于评估的数据范围"
        )
    
    # 图表配置
    with st.expander("图表设置", expanded=True):
        chart_height = st.slider(
            "图表高度",
            min_value=300,
            max_value=800,
            value=400,
            step=50,
            help="调整图表显示高度"
        )
        
        show_confidence_interval = st.checkbox(
            "显示置信区间", 
            value=False,
            help="在预测图表中显示置信区间"
        )
        
        show_residuals = st.checkbox(
            "显示残差分析", 
            value=True,
            help="显示模型残差分析图表"
        )

# 主要评估标签页
eval_tabs = st.tabs([
    "📊 模型对比", 
    "📈 预测分析", 
    # "🔍 误差分析",  # 暂时隐藏
    # "🧪 模型诊断",  # 暂时隐藏
    "📋 详细报告"
])

# 标签页1: 模型对比
with eval_tabs[0]:
    st.header("📊 模型性能对比")
    
    if len(selected_models) == 0:
        st.warning("请在侧边栏选择至少一个模型进行对比")
    else:
        # 性能指标对比
        st.subheader("性能指标对比")
        comparison_col1, comparison_col2 = st.columns([2, 1])
        
        with comparison_col1:
            # 雷达图显示多维度对比
            if len(selected_models) >= 2:
                radar_option = create_model_comparison_radar()
                st_echarts(options=radar_option, height=f"{chart_height}px")
            else:
                st.info("需要至少2个模型才能显示雷达图对比")
        
        with comparison_col2:
            # 性能指标表格
            comparison_data = []
            for model in selected_models:
                metrics = model_info[model]['metrics']
                row = {"模型": model}
                for metric in selected_metrics:
                    if metric in metrics:
                        row[metric] = f"{metrics[metric]:.4f}"
                    else:
                        row[metric] = "N/A"
                comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # 最佳模型推荐
                if len(selected_models) > 1:
                    st.success(f"🏆 推荐模型: {best_model}")
                    st.info("基于MSE指标评选")
        
        # 指标对比柱状图
        if len(selected_models) > 1 and comparison_data:
            st.subheader("指标对比图")
            
            # 为每个指标创建对比图
            for metric in selected_metrics:
                if metric in comparison_df.columns:
                    metric_col1, metric_col2 = st.columns([3, 1])
                    
                    with metric_col1:
                        # 创建柱状图
                        metric_values = []
                        model_names = []
                        
                        for _, row in comparison_df.iterrows():
                            if row[metric] != "N/A":
                                try:
                                    metric_values.append(float(row[metric]))
                                    model_names.append(row["模型"])
                                except ValueError:
                                    continue
                        
                        if metric_values:
                            bar_option = {
                                "title": {
                                    "text": f"{metric}对比",
                                    "left": "center",
                                    "textStyle": {"fontSize": 14}
                                },
                                "tooltip": {"trigger": "axis"},
                                "xAxis": {
                                    "type": "category",
                                    "data": model_names
                                },
                                "yAxis": {"type": "value"},
                                "series": [{
                                    "type": "bar",
                                    "data": metric_values,
                                    "itemStyle": {
                                        "color": "#5470c6"
                                    }
                                }]
                            }
                            st_echarts(options=bar_option, height="250px")
                    
                    with metric_col2:
                        # 显示最佳值
                        if metric_values:
                            if metric in ["MSE", "RMSE", "MAE", "MAPE"]:
                                best_idx = np.argmin(metric_values)
                                st.success(f"最佳: {model_names[best_idx]}")
                                st.metric("最佳值", f"{metric_values[best_idx]:.4f}")
                            else:
                                best_idx = np.argmax(metric_values)
                                st.success(f"最佳: {model_names[best_idx]}")
                                st.metric("最佳值", f"{metric_values[best_idx]:.4f}")

# 标签页2: 预测分析
with eval_tabs[1]:
    st.header("📈 预测结果分析")
    
    if len(selected_models) == 0:
        st.warning("请在侧边栏选择至少一个模型进行分析")
    else:
        # 获取预测数据
        st.subheader("预测结果对比")
        
        # 从session state获取实际的预测数据
        actual_values, lstm_pred, arima_pred, dates, has_real_data = get_prediction_data()
        
        # 如果没有真实数据，显示提示信息
        if not has_real_data:
            st.info("📊 当前显示示例数据，请先训练模型以查看真实预测结果")
        
        # 数据长度信息和对齐处理
        available_lengths = []
        length_info = {}
        
        if len(actual_values) > 0:
            available_lengths.append(len(actual_values))
            length_info["实际值"] = len(actual_values)
        
        if lstm_pred is not None and len(lstm_pred) > 0:
            available_lengths.append(len(lstm_pred))
            length_info["LSTM预测"] = len(lstm_pred)
        
        if arima_pred is not None and len(arima_pred) > 0:
            available_lengths.append(len(arima_pred))
            length_info["ARIMA预测"] = len(arima_pred)
        
        # 检查长度不一致的问题并提供对齐策略选择
        if len(set(available_lengths)) > 1:
            st.warning("⚠️ 检测到数据长度不一致，可能的原因：")
            st.write("- LSTM和ARIMA使用了不同的序列长度设置")
            st.write("- 数据预处理方式不同")
            st.write("- 模型训练时的参数设置不同")
            
            # 提供解决建议
            max_length = max(available_lengths)
            min_length = min(available_lengths)
            st.info(f"💡 建议：使用较长的数据长度({max_length}个点)以获得更好的比较效果")
            
            # 让用户选择对齐策略
            alignment_strategy = st.radio(
                "选择数据对齐策略:",
                options=["使用最小长度", "使用最大长度(可能有缺失值)", "仅显示完整数据的模型"],
                index=0,
                help="选择如何处理长度不一致的数据"
            )
        else:
            alignment_strategy = "使用最小长度"  # 默认策略
            st.success("✅ 所有数据长度一致")
        
        # 统一调试信息展开框
        with st.expander("🔧 数据处理和调试信息", expanded=False):
            st.markdown("**数据来源信息:**")
            if "LSTM" in selected_models:
                if 'lstm_test_predictions' in st.session_state:
                    st.success("✅ LSTM: 使用保存的预测结果")
                else:
                    st.info("ℹ️ LSTM: 重新生成预测结果")
                
                # 显示实际值来源
                target_column = 'Close'
                if 'selected_features' in st.session_state:
                    selected_features = st.session_state['selected_features']
                    if 'Close' in selected_features:
                        target_column = 'Close'
                    elif selected_features:
                        target_column = selected_features[0]
                st.info(f"✅ LSTM实际值: 使用原始数据中的{target_column}列")
            
            if "ARIMA" in selected_models:
                st.success("✅ ARIMA: 使用训练结果中的预测数据")
                st.info("✅ ARIMA实际值: 使用训练时的测试集数据")
            
            st.markdown("**原始数据长度:**")
            for name, length in length_info.items():
                st.write(f"- {name}: {length} 个数据点")
        
        # 根据选择的策略进行数据对齐
        if available_lengths:
            if alignment_strategy == "使用最小长度":
                final_length = min(available_lengths)
            elif alignment_strategy == "使用最大长度(可能有缺失值)":
                final_length = max(available_lengths)
                # 对于较短的数据，用NaN填充
                if len(actual_values) < final_length:
                    actual_values = np.pad(actual_values, (0, final_length - len(actual_values)), constant_values=np.nan)
                if lstm_pred is not None and len(lstm_pred) < final_length:
                    lstm_pred = np.pad(lstm_pred, (0, final_length - len(lstm_pred)), constant_values=np.nan)
                if arima_pred is not None and len(arima_pred) < final_length:
                    arima_pred = np.pad(arima_pred, (0, final_length - len(arima_pred)), constant_values=np.nan)
            else:  # 仅显示完整数据的模型
                final_length = min(available_lengths)
                # 移除长度不匹配的模型数据
                if lstm_pred is not None and len(lstm_pred) != max(available_lengths):
                    lstm_pred = None
                if arima_pred is not None and len(arima_pred) != max(available_lengths):
                    arima_pred = None
            
            # 调整数据长度
            actual_values = actual_values[:final_length]
            dates = dates[:final_length] if len(dates) > final_length else dates
            if lstm_pred is not None:
                lstm_pred = lstm_pred[:final_length]
            if arima_pred is not None:
                arima_pred = arima_pred[:final_length]
        else:
            final_length = 0
        
        # 显示最终数据信息
        if len(actual_values) > 0:
            st.success(f"📈 最终显示数据点数: {len(actual_values)}")
            
            # 在调试框中显示最终数据范围
            with st.expander("📊 最终数据范围", expanded=False):
                if len(actual_values) > 0:
                    st.write(f"实际值范围: {np.nanmin(actual_values):.2f} - {np.nanmax(actual_values):.2f}")
                if lstm_pred is not None and len(lstm_pred) > 0:
                    st.write(f"LSTM预测范围: {np.nanmin(lstm_pred):.2f} - {np.nanmax(lstm_pred):.2f}")
                if arima_pred is not None and len(arima_pred) > 0:
                    st.write(f"ARIMA预测范围: {np.nanmin(arima_pred):.2f} - {np.nanmax(arima_pred):.2f}")
                st.write(f"日期范围: {dates[0]} 到 {dates[-1]}" if dates else "无日期信息")
                st.write(f"数据类型: {'真实数据' if has_real_data else '示例数据'}")
            
            # 主预测图表
            try:
                # 验证数据有效性
                if len(actual_values) == 0:
                    st.warning("实际值数据为空，无法创建图表")
                elif any(not np.isfinite(x) for x in actual_values):
                    st.warning("实际值包含无效数据（NaN或Inf），正在清理...")
                    # 清理无效数据
                    valid_indices = np.isfinite(actual_values)
                    actual_values = actual_values[valid_indices]
                    dates = [dates[i] for i in range(len(dates)) if valid_indices[i]]
                    if lstm_pred is not None:
                        lstm_pred = lstm_pred[valid_indices]
                    if arima_pred is not None:
                        arima_pred = arima_pred[valid_indices]
                
                if len(actual_values) > 0:
                    prediction_option = create_prediction_comparison_chart(
                        dates, actual_values, lstm_pred, arima_pred
                    )
                    st_echarts(options=prediction_option, height=f"{chart_height + 100}px")
                else:
                    st.error("清理后没有有效数据可显示")
                    
            except Exception as chart_error:
                st.error(f"图表创建失败: {chart_error}")
                import traceback
                st.code(traceback.format_exc())
            
            # 预测准确性分析
            st.subheader("预测准确性分析")
            accuracy_cols = st.columns(len(selected_models))
            
            for i, model in enumerate(selected_models):
                with accuracy_cols[i]:
                    try:
                        if model == "LSTM" and lstm_pred is not None and len(lstm_pred) > 0:
                            scatter_option = create_scatter_plot(actual_values, lstm_pred, "LSTM")
                            st_echarts(options=scatter_option, height="300px")
                            
                            # 显示详细指标
                            metrics = calculate_model_metrics(actual_values, lstm_pred)
                            st.markdown(f"**LSTM性能指标:**")
                            st.markdown(f"- MSE: {metrics['MSE']:.4f}")
                            st.markdown(f"- RMSE: {metrics['RMSE']:.4f}")
                            st.markdown(f"- MAE: {metrics['MAE']:.4f}")
                            st.markdown(f"- R²: {metrics['R²']:.4f}")
                            
                        elif model == "ARIMA" and arima_pred is not None and len(arima_pred) > 0:
                            scatter_option = create_scatter_plot(actual_values, arima_pred, "ARIMA")
                            st_echarts(options=scatter_option, height="300px")
                            
                            # 显示详细指标
                            metrics = calculate_model_metrics(actual_values, arima_pred)
                            st.markdown(f"**ARIMA性能指标:**")
                            st.markdown(f"- MSE: {metrics['MSE']:.4f}")
                            st.markdown(f"- RMSE: {metrics['RMSE']:.4f}")
                            st.markdown(f"- MAE: {metrics['MAE']:.4f}")
                            st.markdown(f"- R²: {metrics['R²']:.4f}")
                        else:
                            st.warning(f"{model}模型没有可用的预测数据")
                    except Exception as scatter_error:
                        st.error(f"{model}散点图创建失败: {scatter_error}")
                        import traceback
                        st.code(traceback.format_exc())
        else:
            st.warning("没有可用的数据进行预测分析")

# 添加JSON序列化辅助函数
def make_json_serializable(obj):
    """
    将包含numpy数组的对象转换为JSON可序列化的格式
    
    参数:
        obj: 需要转换的对象
        
    返回:
        JSON可序列化的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif pd.isna(obj):
        return None
    else:
        return obj

def generate_evaluation_report(report_sections, report_format, selected_models, selected_metrics, model_info, best_model, evaluation_period, include_charts=True):
    """
    生成模型评估报告
    
    参数:
        report_sections: 选择的报告章节
        report_format: 报告格式 (HTML, Markdown, JSON)
        selected_models: 选择的模型列表
        selected_metrics: 选择的指标列表
        model_info: 模型信息字典
        best_model: 最佳模型名称
        evaluation_period: 评估期间
        include_charts: 是否包含图表
        
    返回:
        tuple: (报告内容, 文件扩展名, MIME类型)
    """
    current_time = datetime.now().strftime('%Y年%m月%d日 %H:%M')
    
    if report_format == "HTML预览":
        # 生成HTML格式报告
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型评估报告</title>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; }}
        .metrics-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .metrics-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .highlight {{ background-color: #e8f4fd; padding: 15px; border-left: 4px solid #2196F3; margin: 15px 0; }}
        .warning {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 15px 0; }}
        .success {{ background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 15px 0; }}
        h1 {{ color: #333; }}
        h2 {{ color: #555; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #666; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 模型评估报告</h1>
        <p><strong>生成时间:</strong> {current_time}</p>
        <p><strong>评估模型:</strong> {', '.join(selected_models)}</p>
        <p><strong>评估期间:</strong> {evaluation_period}</p>
    </div>
"""
        
        # 执行摘要
        if "执行摘要" in report_sections:
            html_content += f"""
    <div class="section">
        <h2>📋 执行摘要</h2>
        <div class="highlight">
            <h3>🎯 主要发现</h3>
"""
            for model in selected_models:
                if model in model_info:
                    metrics = model_info[model]['metrics']
                    html_content += f"""
            <p><strong>{model}模型:</strong></p>
            <ul>
                <li>RMSE: {metrics.get('RMSE', 'N/A')}</li>
                <li>MAE: {metrics.get('MAE', 'N/A')}</li>
                <li>方向准确率: {metrics.get('Direction_Accuracy', 'N/A')}</li>
            </ul>
"""
            
            html_content += f"""
        </div>
        <div class="success">
            <h3>💡 建议</h3>
            <ul>
                <li><strong>推荐模型:</strong> {best_model}</li>
                <li><strong>应用场景:</strong> 适用于短期价格预测</li>
                <li><strong>注意事项:</strong> 建议定期重新训练模型以保持预测准确性</li>
            </ul>
        </div>
    </div>
"""
        
        # 性能指标
        if "性能指标" in report_sections:
            html_content += """
    <div class="section">
        <h2>📈 详细性能指标</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>模型</th>
                    <th>状态</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R²</th>
                    <th>方向准确率</th>
                </tr>
            </thead>
            <tbody>
"""
            for model in selected_models:
                if model in model_info:
                    metrics = model_info[model]['metrics']
                    status = model_info[model]['status']
                    html_content += f"""
                <tr>
                    <td><strong>{model}</strong></td>
                    <td>{status}</td>
                    <td>{metrics.get('MSE', 'N/A')}</td>
                    <td>{metrics.get('RMSE', 'N/A')}</td>
                    <td>{metrics.get('MAE', 'N/A')}</td>
                    <td>{metrics.get('R²', 'N/A')}</td>
                    <td>{metrics.get('Direction_Accuracy', 'N/A')}</td>
                </tr>
"""
            html_content += """
            </tbody>
        </table>
    </div>
"""
        
        # 建议和结论
        if "建议和结论" in report_sections:
            html_content += """
    <div class="section">
        <h2>🎯 建议和结论</h2>
        
        <h3>📈 生产环境部署</h3>
        <ul>
            <li><strong>首选模型:</strong> 根据综合性能评估选择最佳模型</li>
            <li><strong>备选方案:</strong> 保留次优模型作为备选</li>
            <li><strong>监控策略:</strong> 建立模型性能监控机制</li>
        </ul>
        
        <h3>🔄 模型维护</h3>
        <ul>
            <li><strong>重训练频率:</strong> 建议每月重新评估模型性能</li>
            <li><strong>数据更新:</strong> 及时更新训练数据集</li>
            <li><strong>参数调优:</strong> 根据新数据调整模型参数</li>
        </ul>
        
        <div class="warning">
            <h3>⚠️ 风险提示</h3>
            <ul>
                <li>模型预测存在不确定性，请结合业务判断使用</li>
                <li>市场环境变化可能影响模型性能</li>
                <li>建议建立多模型集成策略降低风险</li>
            </ul>
        </div>
    </div>
"""
        
        html_content += """
    <div class="footer">
        <p>💡 提示：模型评估结果仅供参考，实际应用时请结合业务场景和专业判断</p>
        <p>报告由Streamlit债券预测系统自动生成</p>
    </div>
</body>
</html>
"""
        return html_content, "html", "text/html"
    
    elif report_format == "Markdown":
        # 生成Markdown格式报告
        md_content = f"""# 📊 模型评估报告

**生成时间:** {current_time}  
**评估模型:** {', '.join(selected_models)}  
**评估期间:** {evaluation_period}  
**评估指标:** {', '.join(selected_metrics)}

---

"""
        
        # 执行摘要
        if "执行摘要" in report_sections:
            md_content += """## 📋 执行摘要

### 🎯 主要发现

"""
            for model in selected_models:
                if model in model_info:
                    metrics = model_info[model]['metrics']
                    md_content += f"""**{model}模型:**
- RMSE: {metrics.get('RMSE', 'N/A')}
- MAE: {metrics.get('MAE', 'N/A')}
- 方向准确率: {metrics.get('Direction_Accuracy', 'N/A')}

"""
            
            md_content += f"""### 💡 建议

- **推荐模型:** {best_model}
- **应用场景:** 适用于短期价格预测
- **注意事项:** 建议定期重新训练模型以保持预测准确性

---

"""
        
        # 性能指标
        if "性能指标" in report_sections:
            md_content += """## 📈 详细性能指标

| 模型 | 状态 | MSE | RMSE | MAE | R² | 方向准确率 |
|------|------|-----|------|-----|----|-----------| 
"""
            for model in selected_models:
                if model in model_info:
                    metrics = model_info[model]['metrics']
                    status = model_info[model]['status']
                    md_content += f"| **{model}** | {status} | {metrics.get('MSE', 'N/A')} | {metrics.get('RMSE', 'N/A')} | {metrics.get('MAE', 'N/A')} | {metrics.get('R²', 'N/A')} | {metrics.get('Direction_Accuracy', 'N/A')} |\n"
            
            md_content += "\n---\n\n"
        
        # 建议和结论
        if "建议和结论" in report_sections:
            md_content += """## 🎯 建议和结论

### 📈 生产环境部署
- **首选模型:** 根据综合性能评估选择最佳模型
- **备选方案:** 保留次优模型作为备选
- **监控策略:** 建立模型性能监控机制

### 🔄 模型维护
- **重训练频率:** 建议每月重新评估模型性能
- **数据更新:** 及时更新训练数据集
- **参数调优:** 根据新数据调整模型参数

### ⚠️ 风险提示
- 模型预测存在不确定性，请结合业务判断使用
- 市场环境变化可能影响模型性能
- 建议建立多模型集成策略降低风险

---

💡 **提示：** 模型评估结果仅供参考，实际应用时请结合业务场景和专业判断

*报告由Streamlit债券预测系统自动生成*
"""
        
        return md_content, "md", "text/markdown"
    
    elif report_format == "JSON数据":
        # 生成JSON格式报告
        json_report = {
            "report_metadata": {
                "generation_time": current_time,
                "evaluated_models": selected_models,
                "evaluation_period": evaluation_period,
                "metrics_used": selected_metrics,
                "report_sections": report_sections
            },
            "executive_summary": {
                "best_model": best_model,
                "model_performance": {}
            },
            "detailed_metrics": {},
            "recommendations": {
                "deployment": {
                    "preferred_model": best_model,
                    "backup_strategy": "保留次优模型作为备选",
                    "monitoring": "建立模型性能监控机制"
                },
                "maintenance": {
                    "retrain_frequency": "每月重新评估模型性能",
                    "data_updates": "及时更新训练数据集",
                    "parameter_tuning": "根据新数据调整模型参数"
                },
                "risk_warnings": [
                    "模型预测存在不确定性，请结合业务判断使用",
                    "市场环境变化可能影响模型性能",
                    "建议建立多模型集成策略降低风险"
                ]
            }
        }
        
        # 添加模型性能数据
        for model in selected_models:
            if model in model_info:
                json_report["executive_summary"]["model_performance"][model] = model_info[model]['metrics']
                json_report["detailed_metrics"][model] = {
                    "status": model_info[model]['status'],
                    "metrics": model_info[model]['metrics']
                }
        
        # 确保JSON可序列化
        json_report = make_json_serializable(json_report)
        json_content = json.dumps(json_report, indent=2, ensure_ascii=False)
        
        return json_content, "json", "application/json"
    
    else:
        return "不支持的报告格式", "txt", "text/plain"

# 标签页3: 详细报告 (原来是标签页5)
with eval_tabs[2]:
    st.header("📋 详细评估报告")
    
    # 报告生成选项
    report_col1, report_col2 = st.columns([3, 1])
    
    with report_col1:
        st.subheader("报告配置")
        
        report_sections = st.multiselect(
            "选择报告内容",
            options=[
                "执行摘要", "数据概述", "模型配置", 
                "性能指标", "预测分析", "风险评估", 
                "建议和结论"
            ],
            default=["执行摘要", "性能指标", "预测分析", "建议和结论"]
        )
        
        report_format = st.selectbox(
            "报告格式",
            options=["HTML预览", "Markdown", "JSON数据"]
        )
        
        include_charts = st.checkbox("包含图表", value=True)
    
    with report_col2:
        st.subheader("操作")
        
        if st.button("生成报告", use_container_width=True):
            st.success("✅ 报告生成完成！")
            st.balloons()
        
        # 导出报告按钮
        if st.button("导出报告", use_container_width=True):
            try:
                # 生成报告内容
                report_content, file_ext, mime_type = generate_evaluation_report(
                    report_sections=report_sections,
                    report_format=report_format,
                    selected_models=selected_models,
                    selected_metrics=selected_metrics,
                    model_info=model_info,
                    best_model=best_model,
                    evaluation_period=evaluation_period,
                    include_charts=include_charts
                )
                
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"model_evaluation_report_{timestamp}.{file_ext}"
                
                st.download_button(
                    label=f"下载{report_format}报告",
                    data=report_content,
                    file_name=filename,
                    mime=mime_type,
                    use_container_width=True
                )
                st.success("✅ 报告已准备完成！")
                
            except Exception as report_error:
                st.error(f"生成报告时出错: {report_error}")
                st.code(f"错误详情: {str(report_error)}")
                import traceback
                st.code(traceback.format_exc())
        
        # 导出数据按钮
        if st.button("导出数据", use_container_width=True):
            try:
                # 创建导出数据，确保所有数据都是JSON可序列化的
                export_data = {
                    "evaluation_date": datetime.now().isoformat(),
                    "models_evaluated": selected_models,
                    "metrics_used": selected_metrics,
                    "model_performance": make_json_serializable(model_info)
                }
                
                # 转换为JSON字符串
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                
                st.download_button(
                    label="下载评估数据",
                    data=json_data,
                    file_name=f"model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.success("✅ 导出数据已准备完成！")
                
            except Exception as export_error:
                st.error(f"导出数据时出错: {export_error}")
                st.code(f"错误详情: {str(export_error)}")
                import traceback
                st.code(traceback.format_exc())
    
    # 报告预览
    st.subheader("📄 报告预览")
    
    # 执行摘要
    if "执行摘要" in report_sections:
        with st.expander("执行摘要", expanded=True):
            st.markdown(f"""
            ### 📊 模型评估执行摘要
            
            **评估时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}  
            **评估模型**: {', '.join(selected_models)}  
            **数据集**: {evaluation_period}  
            **评估指标**: {', '.join(selected_metrics)}
            
            #### 🎯 主要发现
            """)
            
            if len(selected_models) > 0:
                for model in selected_models:
                    metrics = model_info[model]['metrics']
                    st.markdown(f"""
                    **{model}模型**:
                    - RMSE: {metrics.get('RMSE', 'N/A')}
                    - MAE: {metrics.get('MAE', 'N/A')}
                    - 方向准确率: {metrics.get('Direction_Accuracy', 'N/A')}
                    """)
            
            st.markdown(f"""
            #### 💡 建议
            - **推荐模型**: {best_model}
            - **应用场景**: 适用于短期价格预测
            - **注意事项**: 建议定期重新训练模型以保持预测准确性
            """)
    
    # 详细指标表
    if "性能指标" in report_sections:
        with st.expander("详细性能指标"):
            if len(selected_models) > 0:
                detailed_metrics = []
                for model in selected_models:
                    metrics = model_info[model]['metrics']
                    row = {"模型": model, "状态": model_info[model]['status']}
                    row.update(metrics)
                    detailed_metrics.append(row)
                
                detailed_df = pd.DataFrame(detailed_metrics)
                # 修复时间戳数据以兼容PyArrow
                detailed_df_display = fix_datetime_for_arrow(detailed_df)
                st.dataframe(detailed_df_display, use_container_width=True, hide_index=True)
            else:
                st.info("没有选择模型进行评估")
    
    # 模型比较总结
    if "建议和结论" in report_sections:
        with st.expander("建议和结论"):
            st.markdown("""
            ### 🎯 模型选择建议
            
            基于当前评估结果，我们提供以下建议：
            
            #### 📈 生产环境部署
            - **首选模型**: 根据综合性能评估选择最佳模型
            - **备选方案**: 保留次优模型作为备选
            - **监控策略**: 建立模型性能监控机制
            
            #### 🔄 模型维护
            - **重训练频率**: 建议每月重新评估模型性能
            - **数据更新**: 及时更新训练数据集
            - **参数调优**: 根据新数据调整模型参数
            
            #### ⚠️ 风险提示
            - 模型预测存在不确定性，请结合业务判断使用
            - 市场环境变化可能影响模型性能
            - 建议建立多模型集成策略降低风险
            """)

# 页面底部信息
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    💡 提示：模型评估结果仅供参考，实际应用时请结合业务场景和专业判断
</div>
""", unsafe_allow_html=True) 