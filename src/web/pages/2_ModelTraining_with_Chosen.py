"""
模型训练页面
用于配置和训练预测模型

修复记录：
1. 修正了LSTM训练过程中的状态管理问题，确保训练状态正确保存和更新
2. 优化了损失曲线绘制逻辑，支持多轮训练和训练过程可视化
3. 修复了模型信息区域状态更新问题
4. 增加了调试信息显示和训练状态重置功能
5. 修复按钮点击后无法正常开始训练的问题
6. 添加了对训练状态的精确控制和反馈
7. 增强了错误处理和异常捕获
8. 优化了UI布局和用户体验
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from streamlit_echarts import st_echarts
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest
import seaborn as sns
import os
import json
from datetime import datetime
from pathlib import Path
import sys

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加arima函数
from web.utils.arima_utils import (
    check_stationarity,
    diff_series,
    check_white_noise,
    analyze_acf_pacf,
    find_best_arima_params,
    fit_arima_model,
    check_residuals,
    forecast_arima,
    evaluate_arima_model,
    inverse_diff,
    generate_descriptive_statistics
)

from web.utils.session import get_state, set_state, update_states

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 页面配置
st.set_page_config(
    page_title="模型训练",
    page_icon="🧠",
    layout="wide"
)

# 标题和简介
st.title("模型训练")
st.markdown("本页面用于配置和训练时间序列预测模型。选择合适的参数并开始训练过程。")

# 初始化会话状态变量
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False

if 'model_metrics' not in st.session_state:
    st.session_state['model_metrics'] = None

if 'start_training' not in st.session_state:
    st.session_state['start_training'] = False

# 创建侧边栏用于调试和重置功能
with st.sidebar:
    st.header("调试与控制面板")
    
    # 显示当前训练状态
    st.write("当前状态:")
    st.json({
        "训练状态": "已完成" if st.session_state.get('training_complete', False) else "未完成",
        "训练进行中": "是" if st.session_state.get('start_training', False) else "否",
        "技术指标数据": "已加载" if 'tech_indicators' in st.session_state and st.session_state['tech_indicators'] is not None else "未加载",
        "特征已选择": "是" if 'selected_features' in st.session_state and st.session_state['selected_features'] else "否"
    })
    
    # 添加重置按钮
    if st.button("重置所有训练状态", key="reset_all"):
        st.session_state['start_training'] = False
        st.session_state['training_complete'] = False
        if 'trained_model' in st.session_state:
            del st.session_state['trained_model']
        if 'model_metrics' in st.session_state:
            st.session_state['model_metrics'] = None
        st.success("已重置所有训练状态！")
        st.rerun()

# 检查数据是否为新加载的数据
if 'raw_data' in st.session_state:
    # 检查之前的数据加载时间戳是否存在
    if 'data_load_timestamp' not in st.session_state:
        # 首次加载数据，记录时间戳并重置训练状态
        st.session_state['data_load_timestamp'] = datetime.now()
        st.session_state['training_complete'] = False
        st.session_state['start_training'] = False
    else:
        # 如果有新的原始数据加载，重置训练状态
        if 'last_trained_data_timestamp' not in st.session_state or st.session_state.get('data_load_timestamp') != st.session_state.get('last_trained_data_timestamp'):
            st.session_state['training_complete'] = False
            st.session_state['start_training'] = False

# 获取加载的数据
if 'raw_data' not in st.session_state:
    st.warning("请先在数据查看页面加载数据")
    st.stop()

df = st.session_state['raw_data']
if df is None:
    st.warning("数据为空，请在数据查看页面加载有效数据")
    st.stop()

# 特征选择函数
def select_features(df, target_col='Close', correlation_threshold=0.5, vif_threshold=10, p_value_threshold=0.05):
    """
    基于相关性、多重共线性和统计显著性进行特征选择
    
    参数:
    df: 包含特征的DataFrame
    target_col: 目标变量列名，默认为'Close'(收盘价)
    correlation_threshold: 相关性阈值，默认为0.5
    vif_threshold: VIF阈值，默认为10
    p_value_threshold: p值阈值，默认为0.05
    
    返回:
    selected_features: 选择的特征列表
    """
    try:
        # 确保目标变量存在于数据集中
        if target_col not in df.columns:
            st.warning(f"目标变量 '{target_col}' 不在数据集中。将使用第一列作为目标变量。")
            target_col = df.columns[0]
            
        # 使用数值型列进行分析
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            st.warning(f"目标变量 '{target_col}' 不是数值类型。特征选择可能无法正常工作。")
        
        selected_features = []
        
        # 步骤1: 基于相关性的特征选择
        # 计算与目标变量(Close)的相关性
        correlation_matrix = df[numeric_cols].corr(numeric_only=True)
        target_correlations = correlation_matrix[target_col].sort_values(ascending=False)
        
        # 显示相关性排名
        with st.expander("**相关性筛选**", expanded=False):
            corr_df = pd.DataFrame({
                '特征': target_correlations.index,
                '相关性': target_correlations.values
            })
            st.dataframe(corr_df)
                # 选择相关性高于阈值的特征
            high_correlation_features = target_correlations[target_correlations > correlation_threshold].index.tolist()
            st.write(f"相关性高于{correlation_threshold}的特征: {high_correlation_features}")
        
        # 步骤2: 多重共线性分析 - 计算VIF (Variance Inflation Factor)
        # 创建一个没有目标变量的特征子集
        with st.expander("**VIF筛选**", expanded=False):
            if len(high_correlation_features) > 1:  # 确保至少有两个特征
                X = df[high_correlation_features].copy()
                # 从VIF计算中移除目标变量
                if target_col in X.columns:
                    X = X.drop(target_col, axis=1)
                    
                if not X.empty and X.shape[1] > 0:
                    # 检查是否存在已知的高度相关特征
                    # 布林带的Upper_Band和Lower_Band基于MA20计算，它们有完全共线性
                    if 'MA20' in X.columns and 'Upper_Band' in X.columns and 'Lower_Band' in X.columns:
                        st.warning("检测到布林带指标（Upper_Band、Lower_Band）与MA20存在完全共线性关系。对这些特征单独处理，以避免VIF计算问题")
                        X = X.drop(['Upper_Band', 'Lower_Band'], axis=1, errors='ignore')
                    
                    # 添加常数项
                    X_with_const = sm.add_constant(X)
                    
                    # 计算VIF，添加错误处理
                    vif_data = pd.DataFrame()
                    vif_data["Feature"] = X_with_const.columns
                    vif_values = []
                    
                    for i in range(X_with_const.shape[1]):
                        try:
                            vif_value = variance_inflation_factor(X_with_const.values, i)
                            if not np.isfinite(vif_value):
                                vif_value = float('inf')  # 处理无穷大值
                                st.warning(f"特征 '{X_with_const.columns[i]}' 的VIF值为无穷大，表示存在完全共线性")
                        except Exception as e:
                            st.warning(f"计算特征 '{X_with_const.columns[i]}' 的VIF值时出错: {str(e)}")
                            vif_value = float('inf')  # 出错时设为无穷大
                        
                        vif_values.append(vif_value)
                    
                    vif_data["VIF"] = vif_values
                    vif_data = vif_data.sort_values("VIF", ascending=False)
                    
                    st.write("**VIF > 10表示存在严重的多重共线性:**")
                    st.dataframe(vif_data)
                    
                    # 移除VIF过高的特征(通常VIF>10表示严重的多重共线性)
                    high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
                    if 'const' in high_vif_features:
                        high_vif_features.remove('const')  # 移除常数项
                    
                    st.write(f"多重共线性严重的特征 (VIF > {vif_threshold}): {high_vif_features}")
                else:
                    high_vif_features = []
                    st.warning("没有足够的特征进行VIF计算")
            else:
                high_vif_features = []
                st.warning("没有足够的相关特征进行多重共线性分析")
        
        # 步骤3: 基于统计显著性的特征选择
        # 使用f_regression评估特征的统计显著性
        with st.expander("**统计显著性筛选**", expanded=False):
            features_to_test = [f for f in high_correlation_features if f != target_col]
            if features_to_test:
                X = df[features_to_test].values
                y = df[target_col].values
                
                f_selector = SelectKBest(f_regression, k='all')
                f_selector.fit(X, y)
                
                # 获取每个特征的p值和F值
                f_scores = pd.DataFrame()
                f_scores["Feature"] = features_to_test
                f_scores["F Score"] = f_selector.scores_
                f_scores["P Value"] = f_selector.pvalues_
                f_scores = f_scores.sort_values("F Score", ascending=False)
                
                st.write("**特征的F检验结果:**")
                st.dataframe(f_scores)
                
                # 选择统计显著的特征(p值<0.05)
                significant_features = f_scores[f_scores["P Value"] < p_value_threshold]["Feature"].tolist()
                st.write(f"统计显著的特征 (P < {p_value_threshold}): {significant_features}")
            else:
                significant_features = []
                st.warning("没有足够的特征进行统计显著性测试")
        
        # 4. 综合以上分析，选择最终的特征集
        # 从高相关性特征中移除多重共线性严重的特征
        selected_features = [f for f in high_correlation_features if f not in high_vif_features]
        
        # 确保所有统计显著的特征都被包含
        for feature in significant_features:
            if feature not in selected_features and feature != target_col:
                selected_features.append(feature)
                
        # 确保目标变量在特征集中
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        return selected_features
    
    except Exception as e:
        st.error(f"特征选择过程中发生错误: {str(e)}")
        if target_col in df.columns:
            return [target_col] + [col for col in df.columns if col != target_col][:5]  # 返回目标变量和其他5个特征作为回退
        else:
            return df.columns.tolist()[:6]  # 返回前6个特征作为回退

# 创建三栏布局
left_column, middle_column = st.columns([1, 2])

# 中间栏 - 模型参数设置与训练控制
with middle_column:
    st.subheader("模型参数配置")
    
    # 模型类型选择标签页
    model_tabs = st.tabs(["LSTM", "ARIMA", "Prophet"])
    
    # LSTM参数设置
    with model_tabs[0]:
        st.markdown("### LSTM模型")
        
        # 特征选择部分 - 移动到LSTM标签页内
        st.markdown("### 特征选择")
        if 'raw_data' in st.session_state and 'tech_indicators' in st.session_state:
            df = st.session_state['tech_indicators']  # 使用技术指标数据而不是原始数据
            
            # 确保使用技术指标数据中实际存在的列作为特征列表
            all_features = df.columns.tolist()
                       
            # 初始化selected_features的session state
            if 'selected_features' not in st.session_state:
                st.session_state['selected_features'] = all_features
            
            # 特征选择多选框，使用session state中的特征作为默认值
            selected_features = st.multiselect(
                "选择用于训练的特征",
                options=all_features,
                default=st.session_state['selected_features']
            )
            
            # 更新selected_features的session state
            st.session_state['selected_features'] = selected_features
        
        # 特征筛选参数
        col1, col2, col3 = st.columns(3)
        with col1:
            correlation_threshold = st.slider(
                "相关性阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="与目标变量的最小相关系数"
            )
        with col2:
            vif_threshold = st.slider(
                "VIF阈值",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                help="方差膨胀因子的最大允许值"
            )
        with col3:
            p_value_threshold = st.slider(
                "P值阈值",
                min_value=0.0,
                max_value=0.1,
                value=0.05,
                step=0.01,
                help="统计显著性的最大允许p值"
            )

        # 添加特征筛选按钮
        if st.button("筛选特征"):
            with st.spinner("正在筛选特征..."):
                # 创建结果容器
                filter_result = st.container()
                
                with filter_result:
                    filtered_features = select_features(
                        df,
                        correlation_threshold=correlation_threshold,
                        vif_threshold=vif_threshold,
                        p_value_threshold=p_value_threshold
                    )
                    
                    # 更新session state中的筛选特征
                    if filtered_features and len(filtered_features) > 0:
                        st.session_state['filtered_features'] = filtered_features
                        # 同时更新选择的特征，使界面上的多选框也更新
                        st.session_state['selected_features'] = filtered_features
                        st.success(f"特征筛选完成，从 {df.shape[1]} 个特征中选出 {len(filtered_features)} 个特征: {filtered_features}")
                    else:
                        st.error("特征筛选失败，将使用所有特征")
                        st.session_state['filtered_features'] = all_features
                        st.session_state['selected_features'] = all_features
                        
                # 确保不会因筛选特征而误认为训练已完成
                if 'training_complete' in st.session_state:
                    st.session_state['training_complete'] = False
                    
                # # 刷新页面以更新多选框的显示
                # st.rerun()
        
        # 显示特征相关性热力图
        if st.checkbox("显示特征相关性热力图（若不显示，可重新点击复选框刷新）"):
            # 确保选择的特征不为空
            if not selected_features:
                st.warning("请至少选择一个特征以显示相关性热力图")
            else:
                # 只使用已选择的特征（selected_features）计算相关性
                # 确保所有选定的特征都是数值类型
                numeric_selected_features = [f for f in selected_features if np.issubdtype(df[f].dtype, np.number)]
                
                if len(numeric_selected_features) < 2:
                    st.warning("需要至少两个数值类型的特征来生成相关性热力图")
                else:
                    # 基于已选择的数值特征计算相关性矩阵
                    selected_corr_matrix = df[numeric_selected_features].corr(numeric_only=True).round(2)
                    
                    # 计算已选择特征的相关性矩阵的最小值和最大值
                    corr_min = selected_corr_matrix.min().min()
                    corr_max = selected_corr_matrix.max().max()
                    
                    # 根据与Close的相关性排序特征（如果Close在已选择的特征中）
                    if 'Close' in numeric_selected_features:
                        close_correlations = selected_corr_matrix['Close']  # 不使用abs()，保留正负号
                        sorted_features = close_correlations.sort_values(ascending=False).index  # 从高到低排序
                        selected_corr_matrix = selected_corr_matrix.loc[sorted_features, sorted_features]
                    
                    # 获取特征名称和相关性数据
                    sorted_selected_features = list(selected_corr_matrix.columns)
                    corr_data = []
                    
                    # 构建热力图数据
                    for i in range(len(sorted_selected_features)):
                        for j in range(len(sorted_selected_features)):
                            corr_data.append([i, j, round(float(selected_corr_matrix.iloc[i, j]), 2)])
                    
                    # 配置echarts选项
                    options = {
                        "tooltip": {
                            "position": "top",
                            },
                        "grid": {
                            "height": "80%",
                            "top": "0%",
                            "left": "150px"
                        },
                        "xAxis": {
                            "type": "category",
                            "data": sorted_selected_features,
                            "splitArea": {
                                "show": True
                            },
                            "axisLabel": {
                                "rotate": 45,
                                "interval": 0
                            }
                        },
                        "yAxis": {
                            "type": "category",
                            "data": sorted_selected_features,
                            "splitArea": {
                                "show": True
                            }
                        },
                        "visualMap": {
                            "min": float(corr_min),
                            "max": float(corr_max),
                            "calculable": True,
                            "orient": "vertical",
                            "left": "0",
                            "top": "middle",
                            "inRange": {
                                "color": ["#313695", "#FFFFFF", "#A50026"]
                            },
                            "formatter": "{value}"  # 确保显示正确的值
                        },
                        "series": [{
                            "name": "相关性系数",
                            "type": "heatmap",
                            "data": corr_data,
                            "label": {
                                "show": True,     # 显示数值标签
                                "formatter": {    # 格式化标签，保留两位小数
                                    "type": "function",
                                    "function": "function(params) { return params.data[2].toFixed(2); }"
                                }
                            },
                            "emphasis": {
                                "itemStyle": {
                                    "shadowBlur": 10,
                                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                                }
                            }
                        }]
                    }
                    
                    # 使用streamlit_echarts显示热力图
                    st.markdown("#### 特征相关性热力图")
                    st_echarts(options=options, height="400px")
        
        # LSTM模型参数设置
        st.markdown("### LSTM模型参数")
        
        col1, col2 = st.columns(2)
        with col1:
            hidden_size = st.number_input(
                "隐藏层大小",
                min_value=1,
                max_value=512,
                value=32
            )
            
            num_layers = st.number_input(
                "LSTM层数",
                min_value=1,
                max_value=5,
                value=1
            )
            
            dropout = st.slider(
                "Dropout比例",
                min_value=0.0,
                max_value=0.9,
                value=0.3,
                step=0.1
            )
        
        with col2:
            learning_rate = st.number_input(
                "学习率",
                min_value=0.0001,
                max_value=0.1,
                value=0.01,
                format="%.4f"
            )
            
            batch_size = st.number_input(
                "批次大小",
                min_value=1,
                max_value=1024,
                value=512
            )
            
            epochs = st.number_input(
                "训练轮数",
                min_value=1,
                max_value=1000,
                value=100
            )
        
        # 保存LSTM超参数到会话状态
        st.session_state['hidden_size'] = hidden_size
        st.session_state['num_layers'] = num_layers
        st.session_state['dropout'] = dropout
        st.session_state['learning_rate'] = learning_rate
        st.session_state['batch_size'] = batch_size
        st.session_state['epochs'] = epochs
        
        # 训练控制
        st.markdown("### LSTM模型训练控制")
        
        train_col1, train_col2 = st.columns([3, 1])
        with train_col1:
            # 简化按钮逻辑，采用更直接的实现
            start_button = st.button(
                "开始训练LSTM模型",
                use_container_width=True,
                key="lstm_train_button"
            )
            
        with train_col2:
            enable_early_stopping = st.checkbox(
                "启用早停",
                value=True
            )
        
        # 训练进度和损失可视化的占位区域
        progress_placeholder = st.empty()
        loss_chart_placeholder = st.empty()
        
        # 如果会话中已有训练历史但界面刚刚加载，显示之前的训练历史
        if ('training_history' in st.session_state 
            and 'training_complete' in st.session_state 
            and st.session_state['training_complete'] 
            and not st.session_state.get('start_training', False)):
            
            st.success("检测到已训练完成的模型")
            
            history = st.session_state['training_history']
            if history and 'train_loss' in history and 'val_loss' in history and len(history['train_loss']) > 0:
                with loss_chart_placeholder:
                    # 绘制已有的损失曲线
                    history_df = pd.DataFrame({
                        '训练损失': history['train_loss'],
                        '验证损失': history['val_loss']
                    })
                    st.line_chart(history_df)
                    st.info(f"上次训练完成，共训练 {len(history['train_loss'])} 轮，"
                           f"最终训练损失: {history['train_loss'][-1]:.6f}，"
                           f"验证损失: {history['val_loss'][-1]:.6f}")
                    
                    # 添加显示模型参数信息
                    if 'model_params' in st.session_state:
                        model_params = st.session_state['model_params']
                        st.write("**模型参数:**")
                        st.json(model_params)
                    
                    # 添加显示训练参数信息
                    if 'training_params' in st.session_state:
                        training_params = st.session_state['training_params']
                        st.write("**训练参数:**")
                        st.json(training_params)
                        
                    # 添加下载模型按钮
                    if 'trained_model' in st.session_state:
                        st.download_button(
                            label="下载模型参数JSON",
                            data=json.dumps({
                                'model_params': st.session_state['model_params'],
                                'training_params': st.session_state['training_params'],
                                'model_metrics': st.session_state.get('model_metrics', {})
                            }, ensure_ascii=False, indent=2),
                            file_name="model_params.json",
                            mime="application/json"
                        )
        
        # 如果点击了开始训练按钮，更新会话状态
        if start_button:
            # 保存当前的特征选择
            st.session_state['selected_features'] = selected_features
            # 设置开始训练标志
            st.session_state['start_training'] = True
            # 重置训练完成状态
            st.session_state['training_complete'] = False
            # 重新运行脚本以应用新状态
            st.rerun()
    
    # ARIMA参数设置
    with model_tabs[1]:
        st.markdown("### ARIMA模型参数与分析")

        # 创建ARIMA子选项卡
        arima_tabs = st.tabs(["数据检验", "模型建立", "模型预测对比"])
        
        # 数据检验选项卡
        with arima_tabs[0]:
            # 确保目标变量选择正确
            if 'Close' not in df.columns:
                st.warning("数据中没有'Close'列，将使用第一列作为目标变量")
                target_col = df.columns[0]
            else:
                target_col = 'Close'
            
            # 获取目标时间序列
            target_series = df[target_col].copy()
            target_series.name = target_col
            
            # 转换为时间序列
            if not isinstance(target_series.index, pd.DatetimeIndex):
                if 'Date' in df.columns:
                    target_series.index = pd.to_datetime(df['Date'])
                else:
                    target_series.index = pd.date_range(
                        start='2000-01-01', 
                        periods=len(target_series), 
                        freq='D'
                    )
            
            # 0. 统计特征分析
            st.subheader("统计特征分析")
            stats_df, normality_test = generate_descriptive_statistics(target_series)
            
            # 显示统计特征表
            st.write("描述性统计表:")
            st.dataframe(stats_df, hide_index=True)
            
            # 显示正态性检验结果
            st.write("Jarque-Bera正态性检验:")
            st.write(f"统计量: {normality_test['statistic']:.4f}")
            st.write(f"p值: {normality_test['p_value']:.4f}")
            st.write(f"数据是否服从正态分布: {'是' if normality_test['is_normal'] else '否'}")
            
            # 1. 平稳性检验
            st.subheader("平稳性检验")
            stationarity_results, is_stationary, stationarity_df = check_stationarity(target_series)
            
            # 显示平稳性检验结果
            col1, col2 = st.columns(2)
            with col1:
                st.write("**ADF统计量:**", round(stationarity_results['ADF统计量'], 4))
                st.write("**p值:**", round(stationarity_results['p值'], 4))
                st.write("**是否平稳:**", "是" if is_stationary else "否")
            
            with col2:
                st.write("**临界值:**")
                for key, value in stationarity_results['临界值'].items():
                    st.write(f"  {key}: {round(value, 4)}")
            
            # 显示平稳性图表 - 使用ECharts代替原生折线图
            try:
                # 确保索引可以转换为字符串
                if isinstance(stationarity_df.index, pd.DatetimeIndex):
                    dates = stationarity_df.index.strftime('%Y-%m-%d').tolist()
                else:
                    dates = stationarity_df.index.astype(str).tolist()
                
                # 确保数据可以转换为列表
                values = stationarity_df['原始数据'].fillna(method='ffill').tolist()
                
                # 配置echarts选项
                options = {
                    "title": {
                        "text": "原始时间序列数据",
                        "textStyle": {
                            "fontSize": 16
                        }
                    },
                    "tooltip": {
                        "trigger": "axis"
                    },
                    "grid": {
                        "left": "5%",
                        "right": "5%",
                        "bottom": "10%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "category",
                        "boundaryGap": False,
                        "data": dates,
                        "axisLabel": {
                            "rotate": 45,
                            "interval": max(1, int(len(dates)/20))  # 确保至少显示一些标签
                        }
                    },
                    "yAxis": {
                        "type": "value",
                        "scale": True,
                        "name": "值",
                        "nameLocation": "middle",
                        "nameGap": 40
                    },
                    "series": [
                        {
                            "name": "原始数据",
                            "type": "line",
                            "showSymbol": False,  # 不显示数据点符号，提升性能
                            "smooth": True,       # 平滑曲线
                            "itemStyle": {
                                "color": "#1890ff"
                            },
                            "areaStyle": {
                                "color": {
                                    "type": "linear",
                                    "x": 0,
                                    "y": 0,
                                    "x2": 0,
                                    "y2": 1,
                                    "colorStops": [
                                        {
                                            "offset": 0,
                                            "color": "rgba(24, 144, 255, 0.5)"
                                        },
                                        {
                                            "offset": 1,
                                            "color": "rgba(24, 144, 255, 0)"
                                        }
                                    ]
                                }
                            },
                            "data": [float(x) if not pd.isna(x) else 0 for x in values],
                            "markLine": {
                                "data": [
                                    {"type": "average", "name": "平均值"}
                                ]
                            }
                        }
                    ]
                }
                
                # 显示echarts图表
                st_echarts(options=options, height="400px", key=f"original_timeseries_echarts_{datetime.now().strftime('%H%M%S')}")
                
            except Exception as e:
                st.error(f"绘制图表时出错: {str(e)}")
            
            # 如果序列不平稳，提供差分选项
            if not is_stationary:
                st.warning("序列不平稳，建议进行差分处理")
                
                # 差分处理选项
                diff_col1, diff_col2 = st.columns(2)
                
                with diff_col1:
                    diff_order = st.slider("差分阶数", min_value=1, max_value=2, value=1)
                
                with diff_col2:
                    log_diff = st.checkbox("使用对数差分", value=False)
                
                diff_data, diff_df = diff_series(target_series, diff_order, log_diff)
                
                # 显示差分后的序列
                st.subheader(f"差分处理后的时间序列")
                
                # 使用ECharts显示差分后的时间序列
                try:
                    # 确保索引可以转换为字符串
                    if isinstance(diff_df.index, pd.DatetimeIndex):
                        diff_dates = diff_df.index.strftime('%Y-%m-%d').tolist()
                    else:
                        diff_dates = diff_df.index.astype(str).tolist()
                    
                    # 获取差分序列名称
                    diff_cols = [col for col in diff_df.columns if col != '原始序列']
                    if diff_cols:
                        diff_col = diff_cols[0]  # 使用第一个差分列
                        
                        # 确保数据可以转换为列表，并处理NaN值
                        diff_values = diff_df[diff_col].fillna(0).tolist()  # 使用0替代NaN
                        # 确保所有值都是可序列化的
                        diff_values = [0 if pd.isna(x) else float(x) for x in diff_values]
                        
                        # 配置echarts选项
                        diff_options = {
                            "title": {
                                "text": "差分处理后的时间序列",
                                "textStyle": {
                                    "fontSize": 16
                                }
                            },
                            "tooltip": {
                                "trigger": "axis"
                            },
                            "grid": {
                                "left": "5%",
                                "right": "5%",
                                "bottom": "10%",
                                "containLabel": True
                            },
                            "xAxis": {
                                "type": "category",
                                "boundaryGap": False,
                                "data": diff_dates,
                                "axisLabel": {
                                    "rotate": 45,
                                    "interval": max(1, int(len(diff_dates)/20))  # 确保至少显示一些标签
                                }
                            },
                            "yAxis": {
                                "type": "value",
                                "scale": True,
                                "name": "差分值",
                                "nameLocation": "middle",
                                "nameGap": 40
                            },
                            "series": [
                                {
                                    "name": diff_col,
                                    "type": "line",
                                    "showSymbol": False,  # 不显示数据点符号，提升性能
                                    "smooth": True,       # 平滑曲线
                                    "itemStyle": {
                                        "color": "#19A7CE"  # 使用不同的颜色区分原始序列
                                    },
                                    "areaStyle": {
                                        "color": {
                                            "type": "linear",
                                            "x": 0,
                                            "y": 0,
                                            "x2": 0,
                                            "y2": 1,
                                            "colorStops": [
                                                {
                                                    "offset": 0,
                                                    "color": "rgba(25, 167, 206, 0.5)"
                                                },
                                                {
                                                    "offset": 1,
                                                    "color": "rgba(25, 167, 206, 0)"
                                                }
                                            ]
                                        }
                                    },
                                    "data": diff_values,
                                    "markLine": {
                                        "data": [
                                            {"type": "average", "name": "平均值"}
                                        ]
                                    }
                                }
                            ]
                        }
                        
                        # 显示echarts图表 - 使用时间戳生成唯一key
                        st_echarts(options=diff_options, height="400px", key=f"diff_timeseries_echarts_{datetime.now().strftime('%H%M%S')}")
                    else:
                        # 如果没有找到差分列，回退到原始的line_chart
                        st.line_chart(diff_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"绘制差分图表时出错: {str(e)}")
                    # 如果ECharts出错，回退到原始st.line_chart
                    st.line_chart(diff_df, use_container_width=True)
                
                # 对差分序列进行平稳性检验
                st.subheader("差分后的平稳性检验")
                diff_results, diff_is_stationary, diff_stationarity_df = check_stationarity(diff_data)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**ADF统计量:**", round(diff_results['ADF统计量'], 4))
                    st.write("**p值:**", round(diff_results['p值'], 4))
                    st.write("**是否平稳:**", "是" if diff_is_stationary else "否")
                
                with col2:
                    st.write("**临界值:**")
                    for key, value in diff_results['临界值'].items():
                        st.write(f"  {key}: {round(value, 4)}")
                
                # 差分后的统计特征分析
                st.subheader("差分后的统计特征分析")
                diff_stats_df, diff_normality_test = generate_descriptive_statistics(diff_data)
                
                # 显示差分后的统计特征表
                st.write("差分后的描述性统计表:")
                st.dataframe(diff_stats_df)
                
                # 显示差分后的正态性检验结果
                st.write("差分后的Jarque-Bera正态性检验:")
                st.write(f"统计量: {diff_normality_test['statistic']:.4f}")
                st.write(f"p值: {diff_normality_test['p_value']:.4f}")
                st.write(f"数据是否服从正态分布: {'是' if diff_normality_test['is_normal'] else '否'}")
                
                # 更新目标序列为差分序列
                target_series = diff_data
            
            # 2. 白噪声检验
            st.subheader("白噪声检验 (Ljung-Box检验)")
            lb_result, is_white_noise = check_white_noise(target_series)
            
            # 显示Ljung-Box检验结果
            st.write("Ljung-Box检验结果 (Portmanteau Q统计量):")
            st.dataframe(lb_result)
            
            if is_white_noise:
                st.warning("序列为白噪声，不适合建立时间序列模型")
            else:
                st.success("序列不是白噪声，适合建立时间序列模型")
            
            # 3. 自相关和偏自相关分析
            st.subheader("自相关与偏自相关分析")
            acf_values, pacf_values, acf_pacf_data = analyze_acf_pacf(target_series)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("自相关函数(ACF)")
                
                # 获取ACF数据
                acf_chart = acf_pacf_data['acf']
                
                # 确保数据不含NaN值
                acf_values_clean = [float(x) if not pd.isna(x) else 0 for x in acf_chart['ACF'].tolist()]
                upper_values_clean = [float(x) if not pd.isna(x) else 0 for x in acf_chart['上限'].tolist()]
                lower_values_clean = [float(x) if not pd.isna(x) else 0 for x in acf_chart['下限'].tolist()]
                
                acf_options = {
                    "title": {
                        "text": "自相关函数(ACF)",
                        "textStyle": {
                            "fontSize": 16
                        }
                    },
                    "tooltip": {
                        "trigger": "axis",
                        "axisPointer": {
                            "type": "shadow"
                        }
                    },
                    "grid": {
                        "left": "10%",
                        "right": "5%",
                        "bottom": "10%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "category",
                        "data": list(range(len(acf_values_clean))),
                        "name": "滞后阶数",
                        "nameLocation": "middle",
                        "nameGap": 25,
                    },
                    "yAxis": {
                        "type": "value",
                        "name": "自相关系数",
                        "nameLocation": "middle",
                        "nameGap": 40
                    },
                    "series": [
                        {
                            "name": "自相关系数",
                            "type": "bar",
                            "data": acf_values_clean,
                            "itemStyle": {
                                "color": "#1890ff"
                            }
                        },
                        {
                            "name": "95%置信区间上限",
                            "type": "line",
                            "data": upper_values_clean,
                            "lineStyle": {
                                "type": "dashed",
                                "color": "#FF4560"
                            },
                            "symbol": "none"
                        },
                        {
                            "name": "95%置信区间下限",
                            "type": "line",
                            "data": lower_values_clean,
                            "lineStyle": {
                                "type": "dashed",
                                "color": "#FF4560"
                            },
                            "symbol": "none"
                        }
                    ]
                }
                
                # 显示ACF图表 - 使用时间戳生成唯一key
                st_echarts(options=acf_options, height="400px", key=f"acf_chart_{datetime.now().strftime('%H%M%S')}")
            
            with col2:
                st.subheader("偏自相关函数(PACF)")
                
                # 获取PACF数据
                pacf_chart = acf_pacf_data['pacf']
                
                # 确保数据不含NaN值
                pacf_values_clean = [float(x) if not pd.isna(x) else 0 for x in pacf_chart['PACF'].tolist()]
                upper_values_clean = [float(x) if not pd.isna(x) else 0 for x in pacf_chart['上限'].tolist()]
                lower_values_clean = [float(x) if not pd.isna(x) else 0 for x in pacf_chart['下限'].tolist()]
                
                pacf_options = {
                    "title": {
                        "text": "偏自相关函数(PACF)",
                        "textStyle": {
                            "fontSize": 16
                        }
                    },
                    "tooltip": {
                        "trigger": "axis",
                        "axisPointer": {
                            "type": "shadow"
                        }
                    },
                    "grid": {
                        "left": "10%",
                        "right": "5%",
                        "bottom": "10%",
                        "containLabel": True
                    },
                    "xAxis": {
                        "type": "category",
                        "data": list(range(len(pacf_values_clean))),
                        "name": "滞后阶数",
                        "nameLocation": "middle",
                        "nameGap": 25,
                    },
                    "yAxis": {
                        "type": "value",
                        "name": "偏自相关系数",
                        "nameLocation": "middle",
                        "nameGap": 40
                    },
                    "series": [
                        {
                            "name": "偏自相关系数",
                            "type": "bar",
                            "data": pacf_values_clean,
                            "itemStyle": {
                                "color": "#19A7CE"
                            }
                        },
                        {
                            "name": "95%置信区间上限",
                            "type": "line",
                            "data": upper_values_clean,
                            "lineStyle": {
                                "type": "dashed",
                                "color": "#FF4560"
                            },
                            "symbol": "none"
                        },
                        {
                            "name": "95%置信区间下限",
                            "type": "line",
                            "data": lower_values_clean,
                            "lineStyle": {
                                "type": "dashed",
                                "color": "#FF4560"
                            },
                            "symbol": "none"
                        }
                    ]
                }
                
                # 显示PACF图表 - 使用时间戳生成唯一key
                st_echarts(options=pacf_options, height="400px", key=f"pacf_chart_{datetime.now().strftime('%H%M%S')}")
            
            # 保存检验结果到会话状态
            st.session_state['arima_check_results'] = {
                'target_series': target_series,
                'original_series': df[target_col].copy() if target_col in df.columns else None,  # 保存原始序列
                'is_stationary': is_stationary,
                'diff_data': diff_data if not is_stationary else None,
                'diff_is_stationary': diff_is_stationary if not is_stationary else None,
                'is_white_noise': is_white_noise,
                'suggested_d': diff_order if not is_stationary and diff_is_stationary else 0,
                'log_diff': log_diff if not is_stationary else False
            }
        
        # 模型建立选项卡
        with arima_tabs[1]:
            st.markdown("#### ARIMA模型建立")
            
            # 检查之前的检验结果
            if 'arima_check_results' not in st.session_state:
                st.warning("请先完成数据检验")
                st.stop()
            
            # 从会话状态获取检验结果
            check_results = st.session_state['arima_check_results']
            target_series = check_results['target_series']
            suggested_d = check_results['suggested_d']
            log_diff = check_results.get('log_diff', False)  # 添加对数差分选项支持
            
            # 1. 模型识别 - ARIMA参数选择
            st.subheader("模型识别 - ARIMA参数选择")
            
            # 提供自动和手动两种参数选择方式
            param_selection = st.radio("参数选择方式", ["自动选择", "手动设置"])
            
            if param_selection == "自动选择":
                # 模型自动选择的范围设置
                st.write("设置参数搜索范围:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    p_max = st.slider("p最大值", min_value=0, max_value=5, value=2)
                
                with col2:
                    d_value = st.slider("d值", min_value=0, max_value=2, value=suggested_d)
                
                with col3:
                    q_max = st.slider("q最大值", min_value=0, max_value=5, value=2)
                
                # 添加季节性参数选项
                st.write("季节性参数 (可选):")
                seasonal = st.checkbox("添加季节性组件", value=False)
                
                if seasonal:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        P_max = st.slider("P最大值", min_value=0, max_value=2, value=1)
                    
                    with col2:
                        D_max = st.slider("D最大值", min_value=0, max_value=1, value=1)
                    
                    with col3:
                        Q_max = st.slider("Q最大值", min_value=0, max_value=2, value=1)
                    
                    with col4:
                        s = st.slider("季节周期 (s)", min_value=4, max_value=52, value=12)
                
                # 模型选择标准
                criterion = st.radio("模型选择标准", ["aic", "bic"])
                
                # 自动寻找最佳参数
                if st.button("确定 - 查找最佳参数"):
                    with st.spinner("正在查找最佳ARIMA参数..."):
                        if seasonal:
                            st.write("正在训练SARIMA模型...")
                            # 构建季节性参数搜索范围
                            p_range = range(0, p_max + 1)
                            d_range = [d_value]
                            q_range = range(0, q_max + 1)
                            P_range = range(0, P_max + 1)
                            D_range = range(0, D_max + 1)
                            Q_range = range(0, Q_max + 1)
                            s_value = s
                            
                            best_params = find_best_arima_params(
                                target_series, 
                                p_range, d_range, q_range,
                                P_range, D_range, Q_range, s_value,
                                criterion
                            )
                        else:
                            st.write("正在训练ARIMA模型...")
                            # 构建非季节性参数搜索范围
                            p_range = range(0, p_max + 1)
                            d_range = [d_value]
                            q_range = range(0, q_max + 1)
                            
                            best_params = find_best_arima_params(
                                target_series, 
                                p_range, d_range, q_range,
                                criterion=criterion
                            )
                        
                        st.write("最佳ARIMA参数:", best_params)
                        # 保存最佳参数到会话状态
                        st.session_state['arima_best_params'] = best_params
            else:
                # 手动设置ARIMA参数
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    p = st.slider("p (AR阶数)", min_value=0, max_value=5, value=1)
                
                with col2:
                    d = st.slider("d (差分阶数)", min_value=0, max_value=2, value=suggested_d)
                
                with col3:
                    q = st.slider("q (MA阶数)", min_value=0, max_value=5, value=1)
                
                # 添加季节性参数选项
                seasonal = st.checkbox("添加季节性组件", value=False)
                
                if seasonal:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        P = st.slider("P (季节AR阶数)", min_value=0, max_value=2, value=0)
                    
                    with col2:
                        D = st.slider("D (季节差分阶数)", min_value=0, max_value=1, value=0)
                    
                    with col3:
                        Q = st.slider("Q (季节MA阶数)", min_value=0, max_value=2, value=0)
                    
                    with col4:
                        s = st.slider("s (季节周期)", min_value=4, max_value=52, value=12)
                    
                    # 手动SARIMA参数
                    params = (p, d, q, P, D, Q, s)
                    st.write(f"已选择SARIMA模型: ({p},{d},{q})({P},{D},{Q},{s})")
                else:
                    # 手动ARIMA参数
                    params = (p, d, q)
                    st.write(f"已选择ARIMA模型: ({p},{d},{q})")
                
                # 保存手动参数到会话状态
                if st.button("确认参数"):
                    st.session_state['arima_best_params'] = params
                    st.write("参数已确认")
            
            # 2. 模型估计 - ARIMA模型拟合
            st.subheader("模型估计 - ARIMA模型拟合")
            
            # 检查是否有确认的参数
            if 'arima_best_params' not in st.session_state:
                st.warning("请先选择ARIMA参数")
                st.stop()
            
            best_params = st.session_state['arima_best_params']
            
            # 分割数据
            st.write("设置训练集和测试集比例:")
            train_ratio = st.slider("训练集比例", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
            
            # 计算分割点
            split_point = int(len(target_series) * train_ratio)
            
            # 训练ARIMA模型
            if st.button("训练ARIMA模型"):
                with st.spinner("正在训练ARIMA模型..."):
                    # 分割训练集和测试集
                    train_data = target_series.iloc[:split_point]
                    test_data = target_series.iloc[split_point:]
                    
                    # 拟合模型
                    model, train_results = fit_arima_model(train_data, best_params)
                    
                    # 残差诊断
                    st.subheader("残差诊断")
                    residual_results, residuals_df = check_residuals(model)
                    
                    # 显示Ljung-Box统计量和p值
                    st.write(f"Ljung-Box检验结果: Q={residual_results['ljung_box_stat']:.4f}, p值={residual_results['ljung_box_pvalue']:.4f}")
                    st.write(f"残差是否为白噪声: {'是' if residual_results['is_white_noise'] else '否'}")
                    
                    # 显示残差统计量
                    st.write("残差统计量:")
                    st.write(f"均值: {residual_results['mean']:.4f}")
                    st.write(f"标准差: {residual_results['std']:.4f}")
                    
                    # 残差时间序列图表
                    st.write("残差时间序列:")
                    st.line_chart(residuals_df['residuals'], use_container_width=True)
                    
                    # 残差ACF图表
                    st.write("残差自相关函数:")
                    st.bar_chart(residuals_df['acf'], use_container_width=True)
                    
                    # 残差QQ图
                    st.write("残差QQ图:")
                    qq_data = pd.DataFrame({
                        'QQ样本值': residuals_df['qq_points'],
                        'QQ理论值': residuals_df['qq_line']
                    })
                    st.line_chart(qq_data, use_container_width=True)
                    
                    # 残差直方图
                    st.write("残差直方图:")
                    hist_data = pd.DataFrame({
                        '频率': residuals_df['hist_values']
                    })
                    st.bar_chart(hist_data, use_container_width=True)
                    
                    # 如果使用了对数差分，记录这一信息
                    if log_diff:
                        st.info("注意：使用了对数差分，预测结果将会自动进行逆变换")
                    
                    # 模型预测
                    st.subheader("模型预测与评估")
                    
                    # 预测和评估
                    forecast_steps = len(test_data)
                    forecast_results, forecast_df = forecast_arima(model, train_data, forecast_steps)
                    
                    # 如果使用了差分，需要逆变换
                    if suggested_d > 0 or log_diff:
                        # 获取原始数据
                        if 'diff_data' in check_results and check_results['diff_data'] is not None:
                            # 如果存在差分数据
                            original_data = check_results.get('original_series', None)
                            if original_data is None:
                                # 尝试从会话状态获取
                                if 'arima_original_data' in st.session_state:
                                    original_data = st.session_state['arima_original_data']
                                else:
                                    # 如果未找到原始数据，发出警告
                                    st.warning("未找到原始数据，无法执行逆差分操作")
                            
                            if original_data is not None:
                                # 执行逆差分
                                forecast_df = inverse_diff(
                                    original_data, 
                                    forecast_df, 
                                    d=suggested_d,
                                    log_diff=log_diff
                                )
                                st.info("已执行逆差分变换")
                    
                    # 显示预测结果
                    st.write("预测结果:")
                    st.line_chart(forecast_df, use_container_width=True)
                    
                    # 模型评估
                    evaluation_metrics = evaluate_arima_model(
                        test_data,
                        forecast_results['forecast_mean'].iloc[-len(test_data):],
                        train_data
                    )
                    
                    # 显示评估指标
                    st.write("模型评估指标:")
                    metrics_df = pd.DataFrame(
                        [evaluation_metrics],
                        index=["ARIMA模型"],
                        columns=["MSE", "RMSE", "MAE", "MAPE", "AIC", "BIC"]
                    )
                    st.dataframe(metrics_df)
                    
                    # 保存模型和结果到会话状态
                    st.session_state['arima_model'] = model
                    st.session_state['arima_train_results'] = train_results
                    st.session_state['arima_forecast_results'] = forecast_results
                    st.session_state['arima_evaluation_metrics'] = evaluation_metrics
                    st.session_state['arima_original_data'] = check_results.get('original_series', train_data)
                    st.session_state['arima_log_diff'] = log_diff
                    
                    st.success("ARIMA模型训练和评估完成!")
        
        # 模型预测对比选项卡
        with arima_tabs[2]:
            st.markdown("#### 模型预测对比")
            
            # 检查是否有训练好的模型
            if 'arima_model' in st.session_state:
                arima_model_data = st.session_state['arima_model']
                fitted_model = arima_model_data['fitted_model']
                model_order = arima_model_data['order']
                
                # 显示当前模型信息
                st.info(f"当前模型: ARIMA{model_order}")
                
                # 预测设置
                forecast_steps = st.number_input(
                    "预测步数",
                    min_value=1,
                    max_value=30,
                    value=10,
                    help="需要进行预测的未来时间点数量"
                )
                
                # 预测按钮
                if st.button("执行预测"):
                    # 预测未来值
                    st.subheader(f"未来 {forecast_steps} 步预测")
                    forecast_results, forecast_df = forecast_arima(fitted_model, steps=forecast_steps)
                    
                    if forecast_results:
                        # 使用Streamlit原生图表显示预测结果
                        st.subheader("预测结果图表")
                        st.line_chart(forecast_df, use_container_width=True)
                        
                        # 显示预测结果表格
                        forecast_table = pd.DataFrame({
                            '预测值': list(forecast_results['mean'].values()),
                            '95%置信区间下限': list(forecast_results['lower_ci'].values()),
                            '95%置信区间上限': list(forecast_results['upper_ci'].values())
                        }, index=list(forecast_results['mean'].keys()))
                        
                        st.subheader("预测结果表格")
                        st.dataframe(forecast_table)
                        
                        # 保存预测结果到会话状态
                        st.session_state['arima_forecast'] = {
                            'results': forecast_results,
                            'steps': forecast_steps
                        }
                    else:
                        st.error("预测失败，请检查模型")
                
                # 模型评估部分
                if df is not None and len(df) > 0:
                    st.subheader("模型评估")
                    
                    # 设置训练集和测试集划分
                    test_size = st.slider(
                        "测试集比例", 
                        min_value=0.1, 
                        max_value=0.5, 
                        value=0.2, 
                        step=0.05
                    )
                    
                    if st.button("评估模型性能"):
                        # 划分训练集和测试集
                        train_size = int(len(target_series) * (1 - test_size))
                        train_data = target_series.iloc[:train_size]
                        test_data = target_series.iloc[train_size:]
                        
                        st.write(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
                        
                        # 用训练集重新拟合模型
                        st.info("使用训练集数据重新拟合模型...")
                        train_model, _ = fit_arima_model(train_data, order=model_order)
                        
                        if train_model:
                            # 评估模型性能
                            metrics = evaluate_arima_model(train_model, test_data)
                            
                            if metrics:
                                st.subheader("模型评估指标")
                                metrics_df = pd.DataFrame({
                                    '指标': list(metrics.keys()),
                                    '值': list(metrics.values())
                                })
                                st.dataframe(metrics_df)
                                
                                # 绘制预测结果与实际值对比
                                forecast = train_model.forecast(steps=len(test_data))
                                
                                # 准备用于Streamlit图表的数据
                                eval_chart_data = pd.DataFrame({
                                    '实际值': test_data.values,
                                    '预测值': forecast.values
                                }, index=test_data.index)
                                
                                # 使用Streamlit原生图表显示
                                st.subheader("测试集预测结果对比")
                                st.line_chart(eval_chart_data, use_container_width=True)
                            else:
                                st.error("模型评估失败")
                        else:
                            st.error("用训练集重新拟合模型失败")
            else:
                st.warning("请先在'模型建立'选项卡中训练ARIMA模型")

# 左侧栏 - 数据信息和数据划分
with left_column:
    st.subheader("模型和数据信息")
    
    # 模型状态信息
    with st.expander("训练状态", expanded=True):
        if st.session_state.get('training_complete', False):
            st.success("模型训练已完成")
        elif st.session_state.get('start_training', False):
            st.info("模型训练进行中...")
            # 添加紧急重置按钮
            if st.button("紧急重置训练", key="emergency_reset"):
                st.session_state['start_training'] = False
                st.session_state['training_complete'] = False
                st.rerun()
        else:
            st.info("等待开始训练...")
    
    # 模型保存选项
    with st.expander("模型保存", expanded=True):
        model_name = st.text_input(
            "模型名称",
            value="my_model_v1"
        )
        
        save_model_button = st.button(
            "保存模型",
            disabled=not st.session_state.get('training_complete', False)
        )
        
        if save_model_button and 'trained_model' in st.session_state:
            model_path = save_model(
                st.session_state['trained_model'],
                st.session_state['model_params'],
                st.session_state['training_params'],
                st.session_state['training_history'],
                path=f"models/{model_name}"
            )
            st.success(f"模型已保存到: {model_path}")
    
    # 数据划分选项
    with st.expander("数据划分", expanded=True):
        sequence_length = st.number_input(
            "序列长度",
            min_value=1,
            max_value=100,
            value=60,
            help="用于构建时间序列样本的步长"
        )
        
        train_test_ratio = st.slider(
            "训练集比例",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="用于训练的数据比例"
        )
        
        # 保存这些参数到会话状态以便训练时使用
        st.session_state['sequence_length'] = sequence_length
        st.session_state['train_test_ratio'] = train_test_ratio
    
    # 模型评估简报
    with st.expander("模型评估简报", expanded=True):
        if 'model_metrics' in st.session_state and st.session_state.get('model_metrics') is not None:
            metrics = st.session_state['model_metrics']
            st.metric(
                label="MSE",
                value=f"{metrics.get('MSE', 0):.4f}"
            )
            
            st.metric(
                label="RMSE",
                value=f"{metrics.get('RMSE', 0):.4f}"
            )
            
            st.metric(
                label="MAE",
                value=f"{metrics.get('MAE', 0):.4f}"
            )
        elif 'start_training' in st.session_state and st.session_state.get('start_training'):
            st.info("模型评估中...")
        else:
            st.info("训练模型后将显示评估指标")
    st.subheader("数据信息")
    
    # 显示数据基本信息
    with st.expander("数据基本信息", expanded=True):
        if 'raw_data' in st.session_state:
            df = st.session_state['raw_data']
            if df is not None:
                st.write(f"数据形状: {df.shape}")
                st.write(f"时间范围: {df.index.min()} 至 {df.index.max()}")
            else:
                st.warning("数据为空，请在数据查看页面加载有效数据")
    
    # 数据划分设置        
    with st.expander("数据划分设置", expanded=True):
        train_test_ratio = st.slider(
            "训练集比例", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.8,  # 默认值0.8，与命令行版本一致
            step=0.05,
            help="训练集占总数据的比例"
        )
        
        # 创建两列布局，使序列长度输入框横排显示
        seq_col1, seq_col2 = st.columns(2)
        
        with seq_col1:
            sequence_length = st.number_input(
                "输入序列长度",
                min_value=1,
                max_value=100,
                value=20,  # 默认值从10改为20，与命令行版本一致
                help="用于预测的历史数据点数量"
            )
        
        with seq_col2:
            prediction_length = st.number_input(
                "预测序列长度",
                min_value=1,
                max_value=30,
                value=1,  # 默认值1，与命令行版本一致
                help="需要预测的未来数据点数量"
            )


# 定义LSTM模型
class LSTMModel(nn.Module):
    """
    LSTM模型定义
    
    Args:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        num_layers: LSTM层数
        output_dim: 输出维度
        dropout: Dropout比率
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dropout层和全连接层
        self.dropout = nn.Dropout(dropout)  # 单独定义dropout层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 应用dropout到最后一个时间步
        out = self.dropout(out[:, -1, :])
        
        # 全连接层
        out = self.fc(out)
        return out

def create_sequences(data, seq_length):
    """
    创建时间序列数据
    
    Args:
        data: 输入数据
        seq_length: 序列长度
        
    Returns:
        X: 特征序列
        y: 目标值
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# 训练LSTM模型
def train_lstm_model(X_train, y_train, X_val, y_val, model_params, training_params, progress_bar=None, status_text=None, loss_chart=None):
    """
    训练LSTM模型
    
    Args:
        X_train: 训练特征数据
        y_train: 训练目标数据
        X_val: 验证特征数据
        y_val: 验证目标数据
        model_params: 模型参数字典
        training_params: 训练参数字典
        progress_bar: streamlit进度条
        status_text: streamlit状态文本
        loss_chart: 损失曲线图表占位符
        
    Returns:
        model: 训练好的模型
        history: 训练历史
    """
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建模型
    model = LSTMModel(
        input_dim=model_params['input_dim'],
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        output_dim=model_params['output_dim'],
        dropout=model_params.get('dropout', 0.3)
    )
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # 训练参数
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 使用传入的进度条或创建新的
    if progress_bar is None:
        progress_bar = st.progress(0)
    if status_text is None:
        status_text = st.empty()
        
    # 创建损失图表的DataFrame
    loss_df = pd.DataFrame(columns=['训练损失', '验证损失'])
    
    # 初始化损失图表
    if loss_chart is not None:
        with loss_chart.container():
            st.info("训练开始，损失曲线将在此处显示...")
            chart_placeholder = st.empty()
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # 小批量训练
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
            
        # 记录训练和验证损失
        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss.item())
        
        # 更新损失图表
        loss_df.loc[epoch] = [avg_train_loss, val_loss.item()]
        
        # 更新图表
        if loss_chart is not None:
            with loss_chart.container():
                with chart_placeholder:
                    st.line_chart(loss_df)
        
        # 更新进度条和状态文本
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss.item():.6f}")
    
    progress_bar.empty()
    status_text.text("模型训练完成！")
    
    return model, history

# 绘制训练历史
def plot_training_history(history, chart_placeholder=None):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        chart_placeholder: 图表占位符
    """
    # 创建DataFrame用于绘图
    history_df = pd.DataFrame({
        '训练损失': history['train_loss'],
        '验证损失': history['val_loss']
    })
    
    if chart_placeholder is not None:
        with chart_placeholder:
            st.line_chart(history_df)
    else:
        st.line_chart(history_df)

# 保存模型
def save_model(model, model_params, training_params, history, path="models"):
    """
    保存模型和训练参数
    
    Args:
        model: 训练好的模型
        model_params: 模型参数
        training_params: 训练参数
        history: 训练历史
        path: 保存路径
    
    Returns:
        model_path: 模型保存路径
    """
    # 确保目录存在
    os.makedirs(path, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_filename = f"lstm_model_{timestamp}.pth"
    model_path = os.path.join(path, model_filename)
    torch.save(model.state_dict(), model_path)
    
    # 保存模型参数和训练历史
    params_filename = f"model_params_{timestamp}.json"
    params_path = os.path.join(path, params_filename)
    
    params_dict = {
        'model_params': model_params,
        'training_params': training_params,
        'training_history': {
            'train_loss': [float(loss) for loss in history['train_loss']],
            'val_loss': [float(loss) for loss in history['val_loss']]
        },
        'timestamp': timestamp
    }
    
    with open(params_path, 'w') as f:
        json.dump(params_dict, f, indent=2)
    
    return model_path

# 用于在会话间保存模型训练状态
if 'trained_models' not in st.session_state:
    st.session_state['trained_models'] = {}

# 用于保存模型训练历史记录
if 'training_history' not in st.session_state:
    st.session_state['training_history'] = {}

# 页面底部 - 帮助信息
with st.expander("使用帮助"):
    st.markdown("""
    ### 使用说明
    
    1. **数据准备**: 在数据查看页面上传并处理您的数据
    2. **特征选择**: 选择用于训练模型的特征
    3. **模型参数**: 配置模型的超参数
    4. **开始训练**: 点击"开始训练"按钮启动训练过程
    5. **保存模型**: 训练完成后，可以保存模型以便后续使用
    
    ### 参数解释
    
    #### LSTM参数
    - **隐藏层大小**: 神经网络隐藏层的节点数量
    - **LSTM层数**: 模型中LSTM层的数量
    - **Dropout比例**: 防止过拟合的随机丢弃比例
    - **学习率**: 梯度下降的步长
    - **批次大小**: 每次更新权重使用的样本数量
    - **训练轮数**: 完整数据集的训练次数
    
    #### ARIMA参数
    - **p (AR阶数)**: 自回归项的阶数
    - **d (差分阶数)**: 差分阶数，使序列平稳
    - **q (MA阶数)**: 移动平均项的阶数
    """)

# 实际执行训练的逻辑
if st.session_state.get('start_training', False) and not st.session_state.get('training_complete', False):
    # 添加简单的状态信息
    st.info("🚀 正在训练LSTM模型...")
    
    try:
        # 准备特征数据
        if 'selected_features' not in st.session_state or not st.session_state['selected_features']:
            st.error("请至少选择一个特征用于训练")
            st.session_state['start_training'] = False
            st.stop()
        
        # 获取当前选择的特征
        selected_features = st.session_state['selected_features']
        
        # 创建UI占位符
        progress_placeholder = st.empty()
        loss_chart_placeholder = st.empty()
        
        # 展示训练进度UI
        with progress_placeholder.container():
            st.info("准备训练数据...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # 确保使用技术指标数据
        if 'tech_indicators' in st.session_state and st.session_state['tech_indicators'] is not None:
            train_df = st.session_state['tech_indicators']
        else:
            st.error("未找到技术指标数据，请先在数据查看页面生成技术指标")
            st.session_state['start_training'] = False
            st.stop()
        
        # 确保所有选定的特征都在数据中
        missing_features = [f for f in selected_features if f not in train_df.columns]
        if missing_features:
            st.error(f"以下特征在数据中不存在: {missing_features}")
            st.session_state['start_training'] = False
            st.stop()
        
        # 提取选定的特征
        feature_data = train_df[selected_features].values
        
        # 确保Close列存在
        if 'Close' in train_df.columns:
            target_col = 'Close'
        else:
            target_col = train_df.columns[0]
            st.warning(f"数据中没有'Close'列，将使用'{target_col}'作为目标变量")
        
        target_data = train_df[target_col].values.reshape(-1, 1)
        
        # 数据归一化
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        feature_data = feature_scaler.fit_transform(feature_data)
        target_data = target_scaler.fit_transform(target_data)
        
        # 保存归一化器以供后续预测使用
        st.session_state['feature_scaler'] = feature_scaler
        st.session_state['target_scaler'] = target_scaler
        
        # 获取训练参数
        sequence_length = st.session_state.get('sequence_length', 60)
        train_test_ratio = st.session_state.get('train_test_ratio', 0.7)
        
        # 创建时间序列数据
        combined_data = np.column_stack((feature_data, target_data))
        X, y = create_sequences(combined_data, int(sequence_length))
        
        # 分离目标变量
        X = X[:, :, :-1]  # 移除最后一列（目标变量）
        y = y[:, -1:]     # 只取最后一列（目标变量）
        
        # 划分训练、验证和测试集
        total_samples = len(X)
        train_size = int(total_samples * train_test_ratio)
        val_size = int(total_samples * 0.15)  # 固定15%的验证集
        
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
        X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
        
        status_text.text(f"数据准备完成! 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
        
        # 获取模型超参数
        hidden_size = st.session_state.get('hidden_size', 32)
        num_layers = st.session_state.get('num_layers', 1)
        dropout = st.session_state.get('dropout', 0.3)
        learning_rate = st.session_state.get('learning_rate', 0.01)
        batch_size = st.session_state.get('batch_size', 512)
        epochs = st.session_state.get('epochs', 100)
        
        # 设置模型参数
        model_params = {
            'input_dim': X_train.shape[2],  # 特征维度
            'hidden_dim': hidden_size,
            'num_layers': num_layers,
            'output_dim': y_train.shape[1],  # 输出维度
            'dropout': dropout
        }
        
        # 设置训练参数
        training_params = {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
        
        # 训练模型
        status_text.text("开始训练LSTM模型...")
        
        # 创建损失图表的DataFrame
        loss_df = pd.DataFrame(columns=['训练损失', '验证损失'])
        chart_area = loss_chart_placeholder.empty()
        
        # 创建模型
        model = LSTMModel(
            input_dim=model_params['input_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            output_dim=model_params['output_dim'],
            dropout=model_params.get('dropout', 0.3)
        )
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            train_losses = []
            
            # 小批量训练
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                # 前向传播
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                
            # 记录训练和验证损失
            avg_train_loss = sum(train_losses) / len(train_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss.item())
            
            # 更新损失图表
            loss_df.loc[epoch] = [avg_train_loss, val_loss.item()]
            with chart_area:
                st.line_chart(loss_df)
            
            # 更新进度条和状态文本
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss.item():.6f}")
        
        # 训练完成
        progress_bar.empty()
        status_text.text("模型训练完成！")
        
        # 测试集评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            test_outputs = model(X_test_tensor)
            test_loss = nn.MSELoss()(test_outputs, y_test_tensor)
            
            # 反归一化预测结果用于展示
            test_predictions = target_scaler.inverse_transform(test_outputs.numpy())
            test_actual = target_scaler.inverse_transform(y_test)
            
            # 计算评估指标
            mse = np.mean((test_predictions - test_actual) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_predictions - test_actual))
            
            # 更新评估指标
            st.session_state['model_metrics'] = {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae)
            }
        
        # 更新会话状态
        st.session_state['trained_model'] = model
        st.session_state['model_params'] = model_params
        st.session_state['training_params'] = training_params
        st.session_state['training_history'] = history
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['seq_length'] = sequence_length
        
        # 更新训练状态
        st.session_state['training_complete'] = True
        st.session_state['start_training'] = False
        
        # 显示评估结果
        with st.container():
            st.success("🎉 模型训练已完成!")
            st.subheader("模型评估结果")
            metrics = st.session_state['model_metrics']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="MSE", value=f"{metrics['MSE']:.4f}")
            with col2:
                st.metric(label="RMSE", value=f"{metrics['RMSE']:.4f}")
            with col3:
                st.metric(label="MAE", value=f"{metrics['MAE']:.4f}")
        
        # 刷新页面以更新UI状态
        st.rerun()
        
    except Exception as e:
        st.error(f"训练过程中发生错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.session_state['start_training'] = False