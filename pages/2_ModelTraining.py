"""
模型训练页面
用于配置和训练预测模型
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import json
from datetime import datetime
from pathlib import Path
import sys
import time
from streamlit_echarts import st_echarts
from statsmodels.tsa.arima.model import ARIMA
import traceback

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加arima
from src.models.arima_model import (
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
    generate_descriptive_statistics,
    create_timeseries_chart,
    create_histogram_chart,
    create_qq_plot,
    create_acf_pacf_charts,
    check_acf_pacf_pattern,
    dynamic_forecast_arima,
    static_forecast_arima,
    calculate_direction_accuracy,
    run_multiple_arima_models,
    create_metrics_comparison_chart,
    create_metrics_statistics_table,
    prepare_arima_charts,
    calculate_statistics
)

# 导入LSTM相关函数
from src.models.lstm_model import (
    save_model, 
    LSTMModel, 
    train_lstm_model, 
    evaluate_lstm_model, 
    plot_training_history, 
    create_sequences,
    run_lstm_training,
    select_features,
    create_correlation_heatmap,
    create_correlation_bar_chart,
    create_significance_charts
)

# 添加session管理函数
try:
    from src.utils.session import get_state, set_state, update_states
except ImportError:
    # 如果导入失败，创建空函数
    def get_state(key, default=None):
        return st.session_state.get(key, default)
    
    def set_state(key, value):
        st.session_state[key] = value
        
    def update_states(updates):
        for key, value in updates.items():
            st.session_state[key] = value

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 页面配置
st.set_page_config(
    page_title="模型训练",
    page_icon="🧠",
    layout="wide"
)

# 导入PyTorch
import torch

# 初始化ARIMA相关的session state变量
if 'arima_model_metrics' not in st.session_state:
    st.session_state['arima_model_metrics'] = None
if 'arima_training_complete' not in st.session_state:
    st.session_state['arima_training_complete'] = False
if 'arima_model' not in st.session_state:
    st.session_state['arima_model'] = None
if 'arima_model_params' not in st.session_state:
    st.session_state['arima_model_params'] = None

# 标题和简介
st.title("模型训练")
st.markdown("本页面用于配置和训练时间序列预测模型。选择合适的参数并开始训练过程。")

# 获取加载的数据
if 'raw_data' not in st.session_state:
    st.warning("请先在数据查看页面加载数据")
    st.stop()

df = st.session_state['raw_data']
# 检查df是否为None
if df is None:
    st.warning("数据为空，请在数据查看页面正确加载数据")
    st.stop()
    
tech_indicators = None

# 侧边栏内容 - 数据特征、模型信息
with st.sidebar:
    st.subheader("数据和特征")
    
    # 显示数据基本信息
    with st.expander("数据信息", expanded=True):
        # 将训练集比例设置放在条件语句外部，确保它是全局变量
        train_test_ratio = st.slider(
            "训练集比例", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.8, 
            step=0.05,
            help="训练集占总数据的比例"
        )
        
        if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
            df = st.session_state['raw_data']
            st.info(f"数据形状: {df.shape}")
            st.info(f"时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
            st.warning("未加载数据或数据为空")
    
    # 删除原数据划分设置部分
        
    st.subheader("模型信息")
    
    # 模型状态信息
    with st.expander("训练状态", expanded=True):
        if 'training_complete' in st.session_state and st.session_state['training_complete']:
            # 根据不同的模型类型显示不同的训练完成信息
            if 'arima_training_complete' in st.session_state and st.session_state['arima_training_complete']:
                if 'lstm_training_complete' in st.session_state and st.session_state['lstm_training_complete']:
                    st.success("LSTM和ARIMA模型均已训练完成")
                else:
                    st.success("ARIMA模型已训练完成")
            else:
                st.success("LSTM模型已训练完成")
        elif 'start_training' in st.session_state and st.session_state['start_training']:
            st.info("LSTM模型训练中...")
        elif 'arima_start_training' in st.session_state and st.session_state['arima_start_training']:
            st.info("ARIMA模型训练中...")
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
            disabled=not ('training_complete' in st.session_state and st.session_state['training_complete'])
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
        elif 'start_training' in st.session_state and st.session_state['start_training']:
            st.info("模型评估中...")
        else:
            st.info("训练模型后将显示评估指标")

# 主要内容区域

# 模型类型选择标签页
model_tabs = st.tabs(["LSTM", "ARIMA", "Prophet"])

# LSTM参数设置
with model_tabs[0]:
    
    # 特征选择部分 - 添加到LSTM标签页内
    st.markdown("### 特征选择")
    
    # 检查技术指标数据是否存在
    if 'raw_data' in st.session_state:
        if 'tech_indicators' in st.session_state:
            df = st.session_state['tech_indicators']  # 使用技术指标数据
        else:
            df = st.session_state['raw_data']  # 如果没有技术指标数据，使用原始数据
        
        # 确保使用数据中实际存在的列作为特征列表
        all_features = df.columns.tolist()
        
        # 初始化selected_features的session state
        if 'selected_features' not in st.session_state:
            st.session_state['selected_features'] = all_features
        
        # 1. 特征筛选参数（阈值选择）
        st.subheader("筛选阈值设置")
        lstm_feat_filter_col1, lstm_feat_filter_col2, lstm_feat_filter_col3 = st.columns(3)
        with lstm_feat_filter_col1:
            correlation_threshold = st.slider(
                "相关性阈值",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="与目标变量的最小相关系数"
            )
        with lstm_feat_filter_col2:
            vif_threshold = st.slider(
                "VIF阈值",
                min_value=1.0,
                max_value=20.0,
                value=10.0,
                step=0.5,
                help="方差膨胀因子的最大允许值"
            )
        with lstm_feat_filter_col3:
            p_value_threshold = st.slider(
                "P值阈值",
                min_value=0.0,
                max_value=0.1,
                value=0.05,
                step=0.01,
                help="统计显著性的最大允许p值"
            )

        # 2. 特征筛选按钮和筛选完成提示框
        st.subheader("特征筛选")
        filter_col1, filter_col2 = st.columns([1,5])
        with filter_col1:
            if st.button("筛选特征", use_container_width=True):
                with st.spinner("正在筛选特征..."):
                    # 调用select_features函数并获取结果
                    filter_results = select_features(
                    df,
                    correlation_threshold=correlation_threshold,
                    vif_threshold=vif_threshold,
                    p_value_threshold=p_value_threshold
                    )
                    
                    # 检查是否有错误
                    if 'error' in filter_results:
                        st.error(f"特征选择过程中发生错误: {filter_results['error']}")
                        st.code(filter_results['traceback'])
                        filtered_features = filter_results['selected_features']
                    else:
                        # 从结果中获取筛选后的特征列表
                        filtered_features = filter_results['selected_features']
                        
                        # 保存筛选参数和详细信息到session state
                        st.session_state['feature_filter_params'] = {
                            'correlation_threshold': correlation_threshold,
                            'vif_threshold': vif_threshold,
                            'p_value_threshold': p_value_threshold
                        }
                        
                        st.session_state['feature_filter_results'] = filter_results
                        
                        # 更新session state中的筛选特征
                        if filtered_features and len(filtered_features) > 0:
                            st.session_state['filtered_features'] = filtered_features
                            # 同时更新选择的特征，使界面上的多选框也更新
                            st.session_state['selected_features'] = filtered_features
                            # 标记已经完成筛选
                            st.session_state['filter_applied'] = True
                        else:
                            st.error("特征筛选失败，将使用所有特征")
                            st.session_state['filtered_features'] = all_features
                            st.session_state['selected_features'] = all_features
                            st.session_state['filter_applied'] = False
        
        with filter_col2:
            # 在UI上显示最终筛选结果（在筛选完成后显示）
            if 'filter_applied' in st.session_state and st.session_state['filter_applied'] and 'feature_filter_results' in st.session_state:
                filter_results = st.session_state['feature_filter_results']
                filtered_features = filter_results['selected_features']
                st.success(f"特征筛选完成！从 {df.shape[1]} 个特征中选出 {len(filtered_features)} 个特征：{filtered_features}")
        
        # 3. 特征选择多选框，使用session state中的特征作为默认值
        st.subheader("选择训练特征")
        selected_features = st.multiselect(
            "选择用于训练的特征",
            options=all_features,
            default=st.session_state['selected_features']
        )
        
        # 更新selected_features的session state
        st.session_state['selected_features'] = selected_features
        
        # 4. 三个展开框，显示逐步筛选的结果
        st.subheader("筛选详细结果")
        
        # 1. 相关性筛选展开框
        with st.expander("**相关性筛选**", expanded=False):
            if 'feature_filter_results' not in st.session_state or not st.session_state.get('filter_applied', False):
                st.warning("请先进行筛选")
            else:
                filter_results = st.session_state['feature_filter_results']
                correlation_threshold = st.session_state['feature_filter_params']['correlation_threshold']
                
                # 相关性数据表格
                corr_data = filter_results['correlation']['data']
                high_correlation_features = filter_results['correlation']['features']
                corr_matrix = filter_results['correlation']['matrix']
                
                # 显示相关性数据表格
                st.dataframe(corr_data, hide_index=True)
                
                # 创建两列布局，使按钮和提示信息处于同一行
                btn_col, info_col = st.columns([1, 5])
                
                # 添加显示/隐藏热力图的按钮
                with btn_col:
                    show_corr_heatmap = st.button("显示/隐藏相关性热力图", key="toggle_corr_heatmap")
                
                # 在右侧列显示相关信息
                with info_col:
                    if not high_correlation_features:
                        st.warning("未找到符合相关性阈值的特征，将显示所有特征的相关性热力图")
                    else:
                        st.success(f"相关性筛选出的特征 (|相关性| > {correlation_threshold}): {high_correlation_features}")
                
                # 初始化session state中的热力图显示状态
                if 'show_corr_heatmap' not in st.session_state:
                    st.session_state['show_corr_heatmap'] = False
                
                # 切换显示状态
                if show_corr_heatmap:
                    st.session_state['show_corr_heatmap'] = not st.session_state['show_corr_heatmap']
                
                # 根据显示状态渲染热力图
                if st.session_state['show_corr_heatmap']:
                    # 检查high_correlation_features是否为空
                    if not high_correlation_features:
                        correlation_heatmap_option = create_correlation_heatmap(corr_matrix)
                    else:
                        # 显示特征间相关性热力图
                        st.write("特征间相关性热力图")
                        correlation_heatmap_option = create_correlation_heatmap(corr_matrix, high_correlation_features)
                    
                    # 确保热力图配置是有效的dictionary
                    if correlation_heatmap_option is None or not isinstance(correlation_heatmap_option, dict):
                        st.error("生成热力图配置失败")
                    else:
                        # 显示热力图
                        try:
                            st_echarts(
                                options=correlation_heatmap_option,
                                height="300px",
                                width="100%",
                                key="corr_heatmap"
                            )
                        except Exception as e:
                            st.error(f"热力图渲染出错: {str(e)}")
                            st.write("错误详情:")
                            st.exception(e)
                
        
        # 2. VIF筛选展开框
        with st.expander("**VIF筛选**", expanded=False):
            if 'feature_filter_results' not in st.session_state or not st.session_state.get('filter_applied', False):
                st.warning("请先进行筛选")
            else:
                filter_results = st.session_state['feature_filter_results']
                vif_threshold = st.session_state['feature_filter_params']['vif_threshold']
                
                vif_data = filter_results['vif']['data']
                low_vif_features = filter_results['vif']['features']
                vif_warnings = filter_results['vif']['warnings']
                collinear_features = filter_results['vif']['collinear_features']
                
                # 收集所有警告信息
                warning_messages = []
                if collinear_features:
                    warning_messages.append(f"- 以下特征存在完全共线性或VIF值异常大：{', '.join(collinear_features)}")
                warning_messages.extend([f"- {warning}" for warning in vif_warnings])
                
                # 如果有警告信息，显示在一个warning框中
                if warning_messages:
                    st.warning("VIF分析过程中发现以下问题：\n" + "\n".join(warning_messages))
                
                # 检查vif_data是否为空
                if not vif_data.empty:
                    # 显示VIF数据表格
                    st.dataframe(vif_data, hide_index=True)
                    st.success(f"VIF低于{vif_threshold}的特征: {low_vif_features}")

                else:
                    st.warning("没有足够的特征进行VIF计算或多重共线性分析")
        
        # 3. 统计显著性筛选展开框
        with st.expander("**统计显著性筛选**", expanded=False):
            if 'feature_filter_results' not in st.session_state or not st.session_state.get('filter_applied', False):
                st.warning("请先进行筛选")
            else:
                filter_results = st.session_state['feature_filter_results']
                p_value_threshold = st.session_state['feature_filter_params']['p_value_threshold']
                
                sig_data = filter_results['significance']['data']
                significant_features = filter_results['significance']['features']
                
                if not sig_data.empty:
                    # 显示统计显著性数据表格
                    st.dataframe(sig_data, hide_index=True)
                    
                    # 创建两列布局，使按钮和提示信息处于同一行
                    p_btn_col, p_info_col = st.columns([1, 7])
                    
                    # 添加显示/隐藏P值图的按钮
                    with p_btn_col:
                        show_p_value_chart = st.button("显示/隐藏P值图表", key="toggle_p_value_chart")
                    
                    # 在右侧列显示相关信息
                    with p_info_col:
                        if p_value_threshold > 0:
                            st.success(f"P值低于{p_value_threshold}的特征: {significant_features}")
                    
                    # 初始化session state中的P值图显示状态
                    if 'show_p_value_chart' not in st.session_state:
                        st.session_state['show_p_value_chart'] = False
                    
                    # 切换显示状态
                    if show_p_value_chart:
                        st.session_state['show_p_value_chart'] = not st.session_state['show_p_value_chart']
                    
                    # 根据显示状态渲染P值图
                    if st.session_state['show_p_value_chart']:
                        # 修改为只接收和渲染p值图表
                        _, p_value_option = create_significance_charts(sig_data, p_value_threshold)
                        st_echarts(
                            options=p_value_option, 
                            height="200px",
                            width="100%",
                            key="p_value_chart"
                        )
                else:
                    st.warning("没有足够的特征进行统计显著性分析")
        
    # 模型参数设置
    st.markdown("### 模型参数")
    
    # 模型参数设定部分
    lstm_params_first_col, lstm_params_second_col, lstm_params_third_col, lstm_params_fourth_col = st.columns(4)
    
    with lstm_params_first_col:
        sequence_length = st.number_input(
            "输入序列长度",
            min_value=1,
            max_value=100,
            value=20,
            help="用于预测的历史数据点数量"
        )
        hidden_size = st.number_input(
            "隐藏层大小",
            min_value=1,
            max_value=512,
            value=64
        )
        
    with lstm_params_second_col:
        prediction_length = st.number_input(
            "预测序列长度",
            min_value=1,
            max_value=30,
            value=1,
            help="需要预测的未来数据点数量"
        )
        num_layers = st.number_input(
            "LSTM层数",
            min_value=1,
            max_value=5,
            value=2
        )
        
    with lstm_params_third_col:
        epochs = st.number_input(
            "训练轮数",
            min_value=1,
            max_value=1000,
            value=100
        )
        learning_rate = st.number_input(
            "学习率",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            format="%.4f"
        )
        
    with lstm_params_fourth_col:
        batch_size = st.number_input(
            "批次大小",
            min_value=1,
            max_value=256,
            value=32
        )
        dropout = st.slider(
            "Dropout比例",
            min_value=0.0,
            max_value=0.9,
            value=0.2,
            step=0.1
        )
        
    
    # 训练控制
    st.markdown("### 训练控制")
    
    lstm_train_btn_col, lstm_early_stop_col = st.columns([3, 1])
    with lstm_train_btn_col:
        if st.button(
            "开始训练",
            use_container_width=True
        ):
            st.session_state['start_training'] = True
        else:
            st.session_state['start_training'] = False
        
    with lstm_early_stop_col:
        enable_early_stopping = st.checkbox(
            "启用早停",
            value=True
        )
    
    # 训练历史与进度显示区域
    st.markdown("### 训练进度与历史")
    
    # 进度显示区域
    progress_placeholder = st.empty()
    
    # 训练历史图表显示区域
    loss_chart_container = st.container()
    with loss_chart_container:
        st.subheader("训练损失曲线")
        loss_chart_placeholder = st.empty()
        
        # 只在非训练状态下显示历史训练数据
        if not st.session_state.get('start_training', False):
            # 如果会话中已有训练历史，显示训练历史图表
            if 'training_history' in st.session_state:
                # 检查训练历史是否包含必要的数据
                history = st.session_state['training_history']
                if isinstance(history, dict) and 'train_loss' in history and 'val_loss' in history:
                    # 绘制已有的损失曲线
                    history_df = pd.DataFrame({
                        '训练损失': history['train_loss'],
                        '验证损失': history['val_loss']
                    })
                    st.line_chart(history_df)
                    st.info(f"最终训练损失: {history['train_loss'][-1]:.4f}, 验证损失: {history['val_loss'][-1]:.4f}")
                else:
                    st.info("没有可显示的训练历史数据")
    
    if 'start_training' in st.session_state and st.session_state['start_training']:
        # 显示训练进度
        with progress_placeholder.container():
            st.info("训练过程将在这里显示...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # 清空并初始化损失图表
        with loss_chart_placeholder:
            # 临时数据用于示例，实际训练中会被动态更新
            chart_data = pd.DataFrame(
                np.random.randn(20, 2),
                columns=['训练损失', '验证损失']
            )
            st.line_chart(chart_data)

# ARIMA参数设置
with model_tabs[1]:       
    # 检查是否需要初始化ARIMA相关状态
    if 'arima_processed' not in st.session_state:
        st.session_state['arima_processed'] = True
    
    # 添加数据预处理部分
    st.markdown("#### 数据预处理")
    
    # 创建两列布局：左侧为控制区域，右侧为数据图表
    arima_controls_col, arima_charts_col = st.columns([1, 2])
    
    with arima_controls_col:
        # 变量选择框
        if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
            df = st.session_state['raw_data']
            
            # 获取所有列名，排除日期类型的列
            all_columns = []
            date_columns = []
            
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
                else:
                    all_columns.append(col)
            
            # 如果所有列都被排除，给出警告
            if not all_columns:
                st.error("数据中没有可用于分析的非日期类型列")
                st.stop()
                
            # 尝试默认选择"Close"列，如果存在的话
            default_index = 0
            if 'Close' in all_columns:
                default_index = all_columns.index('Close')
                
            # 变量选择框
            selected_var = st.selectbox(
                "选择需要分析的变量",
                options=all_columns,
                index=default_index,
                key="arima_selected_var"
            )
            
            # 获取所选变量的数据
            selected_data = df[selected_var]
            
            # 检查数据类型，处理日期时间类型
            is_datetime = pd.api.types.is_datetime64_any_dtype(selected_data)
            is_numeric = pd.api.types.is_numeric_dtype(selected_data)
            
            if is_datetime:
                st.warning(f"选择的变量 '{selected_var}' 是日期时间类型，将转换为时间戳后进行分析")
                # 将日期时间转换为时间戳（浮点数）
                selected_data = (selected_data - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
                # 显示转换后的数据类型
                st.info(f"转换后的数据类型: {selected_data.dtype}")
            elif not is_numeric:
                st.error(f"选择的变量 '{selected_var}' 不是数值类型，无法进行时间序列分析")
                st.stop()
            
            # 数据处理方法选择
            transform_method = st.radio(
                "数据处理方法",
                options=["原始数据", "对数变换", "一阶差分", "一阶对数差分"],
                index=0,
                key="arima_transform_method"
            )
            
            # 自动设置处理标志
            st.session_state['arima_processed'] = True
            
            # 首次加载页面时，确保已经初始化处理数据
            if 'arima_processed_data' not in st.session_state:
                # 默认使用所选变量的原始数据
                st.session_state['arima_processed_data'] = selected_data
                st.session_state['arima_transform_title'] = "原始数据"
            
            # 数据处理和可视化区域
            if 'arima_processed' in st.session_state and st.session_state['arima_processed']:
                # 根据选择的方法进行数据处理
                if transform_method == "原始数据":
                    processed_data = selected_data
                    transform_title = "原始数据"
                    
                    # 执行平稳性检验
                    stationarity_results, is_stationary, _ = check_stationarity(processed_data)
                    
                elif transform_method == "对数变换":
                    # 检查是否有非正值
                    if (selected_data <= 0).any():
                        st.warning("数据包含非正值，无法进行对数变换")
                        processed_data = selected_data
                        transform_title = "原始数据"
                    else:
                        processed_data = np.log(selected_data)
                        transform_title = "对数变换后的数据"
                    
                    # 执行平稳性检验
                    stationarity_results, is_stationary, _ = check_stationarity(processed_data)
                    
                elif transform_method == "一阶差分":
                    diff_data, _ = diff_series(selected_data, diff_order=1, log_diff=False)
                    processed_data = diff_data
                    transform_title = "一阶差分后的数据"
                    
                    # 执行平稳性检验
                    stationarity_results, is_stationary, _ = check_stationarity(processed_data)
                    
                elif transform_method == "一阶对数差分":
                    # 检查是否有非正值
                    if (selected_data <= 0).any():
                        st.warning("数据包含非正值，无法进行对数差分")
                        processed_data = selected_data
                        transform_title = "原始数据"
                    else:
                        diff_data, _ = diff_series(selected_data, diff_order=1, log_diff=True)
                        processed_data = diff_data
                        transform_title = "一阶对数差分后的数据"
                    
                    # 执行平稳性检验
                    stationarity_results, is_stationary, _ = check_stationarity(processed_data)
                
                # 平稳性检验结果展开框
                with st.expander("ADF平稳性检验结果", expanded=True):
                    
                    st.metric(
                        label="ADF统计量",
                        value=f"{stationarity_results['ADF统计量']:.2f}"
                    )
                    st.metric(
                        label="p值",
                        value=f" {stationarity_results['p值']:.2f}"
                    )

                    # 根据p值判断是否平稳
                    if is_stationary:
                        st.success("平稳的 (p值 < 0.05)")
                    else:
                        st.warning("不平稳 (p值 >= 0.05)")
                
                # 正态性检验结果展开框
                with st.expander("正态性检验结果", expanded=True):
                    # 执行正态性检验 (使用scipy的stats模块)
                    from scipy import stats
                    
                    # 进行Shapiro-Wilk检验
                    if len(processed_data) < 5000:  # Shapiro-Wilk适用于小样本
                        stat, p_value = stats.shapiro(processed_data.dropna())
                        test_name = "Shapiro-Wilk检验"
                    else:  # 大样本使用K-S检验
                        stat, p_value = stats.kstest(processed_data.dropna(), 'norm')
                        test_name = "Kolmogorov-Smirnov检验"

                    st.metric(
                        label=f"{test_name}统计量",
                        value=f"{stat:.2f}"
                    )
                    st.metric(
                        label="p值",
                        value=f"{p_value:.2f}"
                    )

                    # 根据p值判断是否符合正态分布
                    if p_value < 0.05:
                        st.warning(f"不符合正态分布 (p值 < 0.05)")
                    else:
                        st.success(f"符合正态分布 (p值 >= 0.05)")
                
                # 白噪声检验结果展开框
                with st.expander("Ljung-Box白噪声检验结果", expanded=True):
                    # 执行白噪声检验
                    try:
                        lb_df, is_white_noise = check_white_noise(processed_data.dropna())
                        
                        # 显示结果
                                                
                        # 第一个滞后阶数的Q统计量和p值
                        first_lag_q = lb_df.iloc[0]['Q统计量']
                        first_lag_p = lb_df.iloc[0]['p值']
                        
                        st.metric(
                            label="Q统计量 (滞后阶数=1)",
                            value=f"{first_lag_q:.2f}"
                        )
                        st.metric(
                            label="p值 (滞后阶数=1)",
                            value=f"{first_lag_p:.2f}"
                        )
                        
                        # 根据p值判断是否为白噪声
                        if is_white_noise:
                            st.success("序列为白噪声 (p值 > 0.05)")
                        else:
                            st.warning("序列不是白噪声 (p值 < 0.05)")
                    except Exception as e:
                        st.error(f"无法执行白噪声检验: {str(e)}")
                
                # 添加自相关检测结果展开框
                with st.expander("自相关检测结果", expanded=True):
                    # 执行自相关检测
                    try:
                        acf_pacf_pattern = check_acf_pacf_pattern(processed_data.dropna(), lags=30)
                        
                        # 显示ACF结果
                        acf_pattern = acf_pacf_pattern["acf"]["pattern"]
                        acf_cutoff = acf_pacf_pattern["acf"]["cutoff"]
                        
                        if acf_pattern == "截尾":
                            st.success(f"ACF函数{acf_cutoff}阶截尾")
                        else:
                            st.info("ACF函数拖尾")
                        
                        # 显示PACF结果
                        pacf_pattern = acf_pacf_pattern["pacf"]["pattern"]
                        pacf_cutoff = acf_pacf_pattern["pacf"]["cutoff"]
                        
                        if pacf_pattern == "截尾":
                            st.success(f"PACF函数{pacf_cutoff}阶截尾")
                        else:
                            st.info("PACF函数拖尾")
                        
                        # 显示模型建议
                        st.info(f"模型建议: {acf_pacf_pattern['model_suggestion']}")
                        
                    except Exception as e:
                        st.error(f"无法执行自相关检测: {str(e)}")
                
                # 保存处理后的数据到会话状态
                st.session_state['arima_processed_data'] = processed_data
                st.session_state['arima_transform_title'] = transform_title
                st.session_state['arima_processed'] = True

            else:
                st.warning("请先在数据查看页面加载数据")
    
    with arima_charts_col:
        # 数据图表显示区域
        if 'arima_processed' in st.session_state and st.session_state['arima_processed']:
            if 'arima_processed_data' in st.session_state:
                # 获取处理后的数据和标题
                processed_data = st.session_state['arima_processed_data']
                transform_title = st.session_state['arima_transform_title']
                
                # 创建折线图
                try:
                    # 创建包含索引的数据框，确保使用正确的日期
                    if transform_method in ["一阶差分", "一阶对数差分"]:
                        # 对于差分数据，需要注意日期索引的处理
                        # 差分会减少数据点，所以需要跳过原始数据的前几个点
                        diff_order = 1  # 默认为一阶差分
                        
                        # 创建与处理后数据长度相同的索引
                        if isinstance(df.index, pd.DatetimeIndex):
                            # 修复：确保数据与日期正确匹配，而不是反向
                            # 使用日期索引，但要确保顺序一致
                            sorted_df = df.sort_index()
                            # 差分后数据长度会减少，所以使用后面的日期索引对应差分数据
                            time_series_df = pd.DataFrame({
                                transform_title: processed_data.values
                            }, index=sorted_df.index[diff_order:diff_order+len(processed_data)])
                        else:
                            # 如果没有日期索引，尝试从df中获取日期列
                            if 'Date' in df.columns:
                                sorted_df = df.sort_values('Date')
                                time_series_df = pd.DataFrame({
                                    transform_title: processed_data.values
                                }, index=sorted_df['Date'].values[diff_order:diff_order+len(processed_data)])
                            else:
                                # 如果没有Date列，使用默认索引
                                time_series_df = pd.DataFrame({transform_title: processed_data})
                    else:
                        # 对于原始数据或对数变换，直接使用原始索引
                        if isinstance(df.index, pd.DatetimeIndex):
                            # 修复：确保数据与日期正确匹配，而不是反向
                            sorted_df = df.sort_index()
                            # 使用与处理后数据相同长度的索引
                            time_series_df = pd.DataFrame({
                                transform_title: processed_data.values
                            }, index=sorted_df.index[:len(processed_data)])
                        else:
                            # 如果没有日期索引，尝试从df中获取日期列
                            if 'Date' in df.columns:
                                sorted_df = df.sort_values('Date')
                                time_series_df = pd.DataFrame({
                                    transform_title: processed_data.values
                                }, index=sorted_df['Date'].values[:len(processed_data)])
                            else:
                                # 如果没有Date列，使用默认索引
                                time_series_df = pd.DataFrame({transform_title: processed_data})
                    
                    # 创建时间序列图
                    timeseries_option = create_timeseries_chart(
                        time_series_df,
                        title=f"{selected_var} - {transform_title}"
                    )
                    st_echarts(options=timeseries_option, height="400px", key=f"timeseries_{selected_var}_{transform_method}")
                except Exception as e:
                    st.error(f"无法绘制时间序列图: {str(e)}")
                
                # 创建直方图
                try:
                    histogram_option = create_histogram_chart(
                        processed_data,
                        title=f"{selected_var} - 分布直方图"
                    )
                    st_echarts(options=histogram_option, height="400px", key=f"histogram_{selected_var}_{transform_method}")
                except Exception as e:
                    st.error(f"无法绘制分布直方图: {str(e)}")
                
                # 创建QQ图
                try:
                    qq_option = create_qq_plot(
                        processed_data,
                        title=f"{selected_var} - QQ图"
                    )
                    st_echarts(options=qq_option, height="450px", key=f"qqplot_{selected_var}_{transform_method}")
                except Exception as e:
                    st.warning(f"无法绘制QQ图: {str(e)}")
                
                # QQ图后添加自相关和偏自相关图
                try:
                    # 创建自相关图和偏自相关图
                    acf_option, pacf_option = create_acf_pacf_charts(
                        processed_data,
                        lags=30,  # 设置最大滞后阶数为30
                        title_prefix=f"{selected_var}"
                    )
                    
                    # 分两列显示ACF和PACF
                    acf_col, pacf_col = st.columns(2)
                    
                    with acf_col:
                        st_echarts(options=acf_option, height="200px", key=f"acf_{selected_var}_{transform_method}")
                    
                    with pacf_col:
                        st_echarts(options=pacf_option, height="200px", key=f"pacf_{selected_var}_{transform_method}")
                        
                except Exception as e:
                    st.warning(f"无法绘制自相关和偏自相关图: {str(e)}")
                    

            else:
                st.info("请在左侧选择变量和数据处理方法")
        else:
            st.info("请在左侧选择变量和数据处理方法")
    
    # 添加描述性统计表格
    st.markdown("### 描述性统计")
    
    # 保存所有数据序列
    series_data = {}
    
    # 原始数据序列
    if selected_var in df.columns:
        original_series = df[selected_var]
        original_series.name = f"{selected_var}_原始数据"
        series_data["原始数据"] = original_series
        
        # 对数变换序列
        if (original_series > 0).all():
            log_series = np.log(original_series)
            log_series.name = f"{selected_var}_对数变换"
            series_data["对数变换"] = log_series
        
        # 一阶差分序列
        diff_series_data, _ = diff_series(original_series, diff_order=1, log_diff=False)
        diff_series_data.name = f"{selected_var}_一阶差分"
        series_data["一阶差分"] = diff_series_data
        
        # 一阶对数差分序列
        if (original_series > 0).all():
            log_diff_series, _ = diff_series(original_series, diff_order=1, log_diff=True)
            log_diff_series.name = f"{selected_var}_一阶对数差分"
            series_data["一阶对数差分"] = log_diff_series
    
    # 生成所有序列的描述性统计表
    all_stats_dfs = []
    jb_stats = {}
    
    for name, series in series_data.items():
        try:
            stats_df, normality_test = generate_descriptive_statistics(series)
            stats_df['VARIABLES'] = [name]  # 替换为序列名称
            all_stats_dfs.append(stats_df)
            jb_stats[name] = {
                'JB统计量': normality_test['statistic'],
                'p值': normality_test['p_value'],
                '是否正态': "是" if normality_test['is_normal'] else "否"
            }
        except Exception as e:
            st.warning(f"无法计算 {name} 的描述性统计: {str(e)}")
    
    # 合并所有统计表
    if all_stats_dfs:
        combined_stats_df = pd.concat(all_stats_dfs, ignore_index=True)
        
        # 表格格式化: 保留小数点位数为3位
        format_cols = ['mean', 'p50', 'sd', 'min', 'max', 'skewness', 'kurtosis']
        for col in format_cols:
            if col in combined_stats_df.columns:
                combined_stats_df[col] = combined_stats_df[col].apply(
                    lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A"
                )
        
        # 重新排列列顺序以提高可读性
        ordered_cols = ['VARIABLES', 'N', 'mean', 'p50', 'sd', 'min', 'max', 'skewness', 'kurtosis']
        ordered_cols = [col for col in ordered_cols if col in combined_stats_df.columns]
        combined_stats_df = combined_stats_df[ordered_cols]
        
        # 设置VARIABLES列为索引，使表格更清晰
        combined_stats_df = combined_stats_df.set_index('VARIABLES')
        
        # 使用st.table而不是st.dataframe，以获得更好的静态表格展示
        st.table(combined_stats_df)
    else:
        st.warning("无法生成描述性统计表")
    
    # 然后是原来的ARIMA参数设置部分
    st.markdown("### ARIMA模型参数")
    
    # 添加最优参数检测控件
    st.markdown("#### 自动参数优化")
    
    # 创建第一行控件：信息准则和最大阶数设置
    criterion_col, max_p_col, max_d_col, max_q_col = st.columns(4)
    
    with criterion_col:
        criterion = st.selectbox(
            "信息准则",
            options=["aic", "bic"],
            index=1,
            help="AIC（赤池信息准则）或BIC（贝叶斯信息准则）"
        )
    
    with max_p_col:
        max_p = st.number_input(
            "最大AR阶数",
            min_value=0,
            max_value=10,
            value=3,
            help="搜索范围：0 到设定值"
        )
    
    with max_d_col:
        max_d = st.number_input(
            "最大差分阶数",
            min_value=0,
            max_value=2,
            value=2,
            help="搜索范围：0 到设定值"
        )
    
    with max_q_col:
        max_q = st.number_input(
            "最大MA阶数",
            min_value=0,
            max_value=10,
            value=3,
            help="搜索范围：0 到设定值"
        )
    
    # 创建第二行：优化按钮和结果显示
    opt_btn_col, opt_result_col = st.columns([1, 3])
    
    with opt_btn_col:
        optimize_button = st.button(
            "自动检测最优参数",
            help="遍历可能的参数组合找到最优ARIMA参数",
            use_container_width=True
        )
    
    with opt_result_col:
        if optimize_button:
            try:
                with st.spinner("正在搜索最优参数..."):
                    # 获取当前处理后的数据
                    if 'arima_processed_data' in st.session_state:
                        processed_data = st.session_state['arima_processed_data']
                        # 调用find_best_arima_params函数
                        best_params = find_best_arima_params(
                            processed_data,
                            p_range=range(0, max_p + 1),
                            d_range=range(0, max_d + 1),
                            q_range=range(0, max_q + 1),
                            criterion=criterion
                        )
                        
                        # 更新session state中的最优参数
                        st.session_state['best_arima_params'] = best_params
                        
                        # 显示成功信息
                        st.success(f"找到最优参数：p={best_params[0]}, d={best_params[1]}, q={best_params[2]}")
                    else:
                        st.error("请先选择数据和处理方法")
            except Exception as e:
                st.error(f"参数优化失败：{str(e)}")
    
    st.markdown("#### 模型参数设置")
    # 添加一个按钮，用于显示ARIMA模型参数的说明
    arima_params_forecast_col, arima_params_ar_col, arima_params_d_col, arima_params_ma_col = st.columns([1,1,1,1])
    
    with arima_params_forecast_col:
        forecast_method = st.selectbox(
            "预测方法",
            options=["动态预测", "静态预测"],
            index=0,
            help="动态预测：使用之前的预测值进行后续预测\n静态预测：使用实际历史值进行预测"
        )
        # 保存预测方法到session_state
        st.session_state['arima_forecast_method'] = forecast_method

        # 运行次数选择
        run_count = st.number_input(
            "运行次数",
            min_value=1,
            max_value=50,
            value=1,
            help="设置ARIMA模型运行的次数，每次使用不同的随机种子。设置为1时执行单次训练，大于1时将执行多次训练并自动选择最优模型"
        )
    
    with arima_params_ar_col:
        # 如果有最优参数，使用它作为默认值
        default_p = st.session_state.get('best_arima_params', (2, 1, 2))[0] if 'best_arima_params' in st.session_state else 2
        p_param = st.number_input(
            "p (AR阶数)",
            min_value=0,
            max_value=10,
            value=default_p
        )

        # 比较指标选择（仅在运行次数>1时显示）
        if run_count > 1:
            comparison_metric = st.selectbox(
                "比较指标",
                options=["MSE", "RMSE", "MAE", "Direction_Accuracy"],
                index=0,
                help="选择用于模型比较的指标（仅适用于多次运行）"
        )
    
    with arima_params_d_col:
        default_d = st.session_state.get('best_arima_params', (2, 1, 2))[1] if 'best_arima_params' in st.session_state else 1
        d_param = st.number_input(
            "d (差分阶数)",
            min_value=0,
            max_value=2,
            value=default_d
        )
        
        # 统一的训练按钮
        arima_train_button = st.button(
            "开始训练ARIMA模型",
            help="训练ARIMA模型，自动处理单次或多次训练",
            use_container_width=True,
            key="arima_train_button"
        )
    
    with arima_params_ma_col:
        default_q = st.session_state.get('best_arima_params', (2, 1, 2))[2] if 'best_arima_params' in st.session_state else 2
        q_param = st.number_input(
            "q (MA阶数)",
            min_value=0,
            max_value=10,
            value=default_q
        )

    
    # 添加预先创建的占位符用于训练过程中显示
    arima_progress_placeholder = st.empty()
    arima_chart_placeholder = st.empty()
    
    # 添加多次运行结果的占位符
    multi_run_results_placeholder = st.empty()
    multi_run_charts_placeholder = st.empty()
    
    # 检查是否点击了训练按钮
    if arima_train_button:
        # 保存当前参数到session_state以便在页面刷新后保持
        st.session_state['arima_p_param'] = p_param
        st.session_state['arima_d_param'] = d_param
        st.session_state['arima_q_param'] = q_param
        st.session_state['arima_train_test_ratio'] = train_test_ratio
        st.session_state['arima_forecast_method'] = forecast_method
        st.session_state['arima_run_count'] = run_count
        
        # 设置训练标志
        st.session_state['arima_start_training'] = True
        st.session_state['arima_training_complete'] = False
        st.session_state['training_complete'] = False
        
        # 刷新页面开始训练
        st.rerun()

    # 添加ARIMA模型训练结果预先创建的占位符
    arima_metrics_placeholder = st.empty()
    arima_prediction_chart_placeholder = st.empty()
    arima_residuals_chart_placeholder = st.empty()
    arima_residuals_hist_placeholder = st.empty()

    # ARIMA执行训练的统一逻辑
    if 'arima_start_training' in st.session_state and st.session_state['arima_start_training']:
        # 获取处理后的数据
        if 'arima_processed_data' not in st.session_state or st.session_state['arima_processed_data'] is None:
            st.error("请先在ARIMA标签页选择数据和处理方法")
            st.stop()
        
        processed_data = st.session_state['arima_processed_data']
        
        # 获取训练参数
        p_param = st.session_state.get('arima_p_param', 2)
        d_param = st.session_state.get('arima_d_param', 1)
        q_param = st.session_state.get('arima_q_param', 2)
        forecast_method = st.session_state.get('arima_forecast_method', "动态预测")
        run_count = st.session_state.get('arima_run_count', 1)
        
        # 使用预先创建的占位符显示训练状态
        with arima_progress_placeholder.container():
            st.info(f"训练ARIMA模型中 (运行{run_count}次)...")
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # 1. 拆分训练集和测试集
            train_size = int(len(processed_data) * train_test_ratio)
            train_data = processed_data[:train_size]
            test_data = processed_data[train_size:]
            
            status_text.info(f"数据集已划分: 训练集大小 {len(train_data)}, 测试集大小 {len(test_data)}")
            progress_bar.progress(0.1)
            
            # 2. 配置模型阶数
            order = (p_param, d_param, q_param)
            
            # 3. 使用统一的多次训练逻辑（即使run_count=1）
            multiple_runs_result = run_multiple_arima_models(
                train_data, 
                test_data, 
                order, 
                forecast_method=forecast_method,
                runs=run_count,
                progress_placeholder=progress_bar
            )
            
            # 4. 处理训练结果
            if multiple_runs_result['status'] == 'success':
                best_model = multiple_runs_result['best_model']
                
                if best_model is not None:
                    # 获取最优模型相关数据
                    arima_model = best_model['model']
                    train_pred = arima_model.fittedvalues
                    test_pred = best_model['predictions']
                    metrics = best_model['metrics']
                    residuals = arima_model.resid
                    
                    # 创建ARIMA训练结果字典
                    arima_training_result = {
                        'model': arima_model,
                        'order': order,
                        'train_data': train_data,
                        'test_data': test_data,
                        'train_pred': train_pred,
                        'test_pred': test_pred,
                        'metrics': metrics,
                        'residuals': residuals,
                        'run_info': {
                            'run': best_model['run'],
                            'seed': best_model['seed']
                        } if run_count > 1 else None,
                        'statistics': multiple_runs_result['statistics']
                    }
                    
                    # 生成图表配置
                    charts = prepare_arima_charts(
                        arima_model,
                        train_data,
                        test_data,
                        test_pred
                    )
                    
                    # 保存图表配置到session state
                    st.session_state['arima_prediction_chart'] = charts['prediction_chart']
                    st.session_state['arima_residuals_chart'] = charts['residuals_chart']
                    st.session_state['arima_residuals_hist'] = charts['residuals_hist']
                    
                    # 保存训练结果和模型到session state
                    st.session_state['arima_model'] = arima_model
                    st.session_state['arima_model_metrics'] = metrics
                    st.session_state['model_metrics'] = metrics
                    st.session_state['arima_training_result'] = arima_training_result
                    
                    # 如果是多次训练，保存最优运行信息
                    if run_count > 1:
                        st.session_state['arima_best_run_info'] = {
                            'run': best_model['run'],
                            'seed': best_model['seed'],
                            'metrics': metrics
                        }
                    
                    # 更新训练状态
                    st.session_state['arima_training_complete'] = True
                    st.session_state['training_complete'] = True
                    
                    # 重置训练状态标志
                    st.session_state['arima_start_training'] = False
                    
                    # 显示训练完成消息
                    progress_bar.progress(1.0)
                    status_text.success("ARIMA模型训练完成！")
                    
                    # 刷新页面以显示结果
                    st.rerun()
                else:
                    st.error("未能获得有效的ARIMA模型")
            else:
                st.error(f"ARIMA模型训练失败: {multiple_runs_result['message']}")
                
        except Exception as e:
            st.error(f"ARIMA模型训练过程中出错: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state['arima_start_training'] = False

    # 显示训练结果
    if 'arima_training_complete' in st.session_state and st.session_state['arima_training_complete']:
        training_result = st.session_state.get('arima_training_result')
        
        if training_result:
            # 显示最优模型信息（如果是多次训练）
            if training_result.get('run_info'):
                run_info = training_result['run_info']
                st.success(f"当前使用的最优模型来自运行 #{run_info['run']}, 随机种子: {run_info['seed']}")
            else:
                st.success("ARIMA模型训练完成")
                
            # 显示评估指标
            metrics = training_result.get('metrics', {})
            
            # 在metrics_placeholder中显示指标
            with arima_metrics_placeholder.container():
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("MSE", f"{metrics.get('MSE', 0):.4f}")
                with metric_cols[1]:
                    st.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
                with metric_cols[2]:
                    st.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
                with metric_cols[3]:
                    st.metric("方向准确率", f"{metrics.get('Direction_Accuracy', 0):.4f}")
            
            # 显示预测图表
            if 'arima_prediction_chart' in st.session_state:
                with arima_prediction_chart_placeholder.container():
                    st.subheader("ARIMA预测结果")
                    st_echarts(options=st.session_state['arima_prediction_chart'], height="500px")
            
            # 显示残差图
            if 'arima_residuals_chart' in st.session_state:
                with arima_residuals_chart_placeholder.container():
                    st.subheader("模型残差")
                    st_echarts(options=st.session_state['arima_residuals_chart'], height="300px")
            
            # 显示残差分布图
            if 'arima_residuals_hist' in st.session_state:
                with arima_residuals_hist_placeholder.container():
                    st.subheader("残差分布")
                    st_echarts(options=st.session_state['arima_residuals_hist'], height="300px")
                    
            # 如果是多次训练，显示训练统计信息
            run_count = st.session_state.get('arima_run_count', 1)
            if run_count > 1 and training_result.get('statistics'):
                with st.expander("多次训练统计信息", expanded=False):
                    # 创建统计表格
                    if 'statistics' in training_result:
                        stats_data = {
                            '指标': ['MSE', 'RMSE', 'MAE', '方向准确率'],
                            '平均值': [],
                            '标准差': [],
                            '最小值': [],
                            '最大值': []
                        }
                        
                        for metric in ['MSE', 'RMSE', 'MAE', 'Direction_Accuracy']:
                            if metric in training_result['statistics']:
                                stats = training_result['statistics'][metric]
                                stats_data['平均值'].append(f"{stats['mean']:.4f}")
                                stats_data['标准差'].append(f"{stats['std']:.4f}")
                                stats_data['最小值'].append(f"{stats['min']:.4f}")
                                stats_data['最大值'].append(f"{stats['max']:.4f}")
                            else:
                                stats_data['平均值'].append("N/A")
                                stats_data['标准差'].append("N/A")
                                stats_data['最小值'].append("N/A")
                                stats_data['最大值'].append("N/A")
                        
                        # 创建DataFrame显示
                        stats_df = pd.DataFrame(stats_data)
                        stats_df = stats_df.set_index('指标')
                        st.dataframe(stats_df)
                    else:
                        st.info("没有可用的多次训练统计信息")

# Prophet参数设置
with model_tabs[2]:
    st.markdown("### Prophet模型参数")
    
    prophet_params_left_col, prophet_params_right_col = st.columns(2)
    with prophet_params_left_col:
        yearly_seasonality = st.selectbox(
            "年度季节性",
            options=["auto", "True", "False"],
            index=0
        )
        
        weekly_seasonality = st.selectbox(
            "周度季节性",
            options=["auto", "True", "False"],
            index=0
        )
    
    with prophet_params_right_col:
        daily_seasonality = st.selectbox(
            "日度季节性",
            options=["auto", "True", "False"],
            index=0
        )
        
        changepoint_prior_scale = st.slider(
            "变点先验比例",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001,
            format="%.3f"
        )

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

with st.expander("GPU加速信息",expanded=True):
    CUDA_Version_col1, CUDA_Version_col2, CUDA_Version_col3 = st.columns(3)
    with CUDA_Version_col1:
        st.info(f"PyTorch版本: {torch.__version__}")

    with CUDA_Version_col2:
        st.info(f"CUDA版本: {torch.version.cuda}")

    with CUDA_Version_col3:
        st.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        CUDA_GPU_col1, CUDA_GPU_col2 = st.columns([5,6])
        with CUDA_GPU_col1:
            st.success(f"当前CUDA版本: {torch.version.cuda}；可用GPU数量: {torch.cuda.device_count()}")
        with CUDA_GPU_col2:
            for i in range(torch.cuda.device_count()):
                st.success(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        st.warning("未检测到GPU，PyTorch将使用CPU模式运行")


# lstm执行训练的逻辑
if 'start_training' in st.session_state and st.session_state['start_training']:
    # 准备特征数据
    # 确保selected_features已定义且不为空
    if 'selected_features' not in locals() or not selected_features:
        st.error("请至少选择一个特征用于训练")
        st.stop()
    
    # 确定使用哪个数据集进行训练
    if 'tech_indicators' in st.session_state and st.session_state['tech_indicators'] is not None:
        # 优先使用技术指标数据
        train_df = st.session_state['tech_indicators']
        st.info("使用技术指标数据进行训练")
    elif 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
        # 如果技术指标数据不可用，使用原始数据
        train_df = st.session_state['raw_data'] 
        st.warning("未找到技术指标数据，将使用原始数据进行训练。建议先在数据查看页面计算技术指标")
    else:
        st.error("没有可用的数据。请先在数据查看页面加载数据并计算技术指标")
        st.stop()
    
    # 检查选择的特征是否在数据集中
    missing_features = [f for f in selected_features if f not in train_df.columns]
    if missing_features:
        st.error(f"以下特征在数据集中不存在: {', '.join(missing_features)}")
        st.stop()
    
    # 使用run_lstm_training函数执行完整的训练流程
    training_result = run_lstm_training(
        selected_features=selected_features,
        df=train_df,
        sequence_length=sequence_length,
        train_test_ratio=train_test_ratio,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        progress_placeholder=progress_placeholder,
        loss_chart_placeholder=loss_chart_placeholder
    )
    
    # 显示模型评估
    st.subheader("模型评估")
    
    # 保存归一化器以供后续预测使用
    st.session_state['feature_scaler'] = training_result['feature_scaler']
    st.session_state['target_scaler'] = training_result['target_scaler']
    
    # 更新右侧栏中的评估指标
    st.session_state['model_metrics'] = training_result['metrics']
    
    # 显示保存成功信息
    st.success(f"模型已保存到: {training_result['model_path']}")
    
    # 更新会话状态
    st.session_state['trained_model'] = training_result['model']
    st.session_state['model_params'] = training_result['model_params'] if 'model_params' in training_result else None
    st.session_state['training_params'] = training_result['training_params'] if 'training_params' in training_result else None
    st.session_state['training_history'] = training_result['history']
    st.session_state['X_test'] = training_result['X_test']
    st.session_state['y_test'] = training_result['y_test']
    st.session_state['seq_length'] = training_result['sequence_length']
    
    # 更新训练状态
    st.session_state['training_complete'] = True
    st.session_state['lstm_training_complete'] = True
    
    # 显示训练完成消息
    st.success("LSTM模型训练已完成！")
    # 重置训练状态
    st.session_state['start_training'] = False
    # 更新训练历史
    st.session_state['training_history'] = training_result['history']
    # 刷新页面以显示最终结果
    st.rerun()



