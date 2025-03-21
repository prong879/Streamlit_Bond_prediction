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

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# 添加arima
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

# 导入LSTM相关函数
from web.utils.lstm_utils import (
    save_model, 
    LSTMModel, 
    train_lstm_model, 
    evaluate_lstm_model, 
    plot_training_history, 
    create_sequences,
    run_lstm_training,
    select_features,
    create_correlation_bar_chart,
    create_vif_bar_chart,
    create_significance_charts,
    create_correlation_heatmap
)

# 添加session管理函数
try:
    from web.utils.session import get_state, set_state, update_states
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

# 创建两栏布局
left_column, middle_column = st.columns([1, 3])


# 中间栏 - 模型参数设置与训练控制
with middle_column:
    st.subheader("模型参数配置")
    
    # 模型类型选择标签页
    model_tabs = st.tabs(["LSTM", "ARIMA", "Prophet"])
    
    # LSTM参数设置
    with model_tabs[0]:
        st.markdown("### LSTM模型")
        
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
                    # 标记已经完成筛选
                    st.session_state['filter_applied'] = True
                else:
                    st.error("特征筛选失败，将使用所有特征")
                    st.session_state['filtered_features'] = all_features
                    st.session_state['selected_features'] = all_features
                    st.session_state['filter_applied'] = False
            
            # 特征选择多选框，使用session state中的特征作为默认值
            st.markdown("### 选择训练特征")
            selected_features = st.multiselect(
                "选择用于训练的特征",
                options=all_features,
                default=st.session_state['selected_features']
            )
            
            # 更新selected_features的session state
            st.session_state['selected_features'] = selected_features
        
        # 模型参数设置
        st.markdown("### 模型参数")
        
        col1, col2 = st.columns(2)
        with col1:
            hidden_size = st.number_input(
                "隐藏层大小",
                min_value=1,
                max_value=512,
                value=64
            )
            
            num_layers = st.number_input(
                "LSTM层数",
                min_value=1,
                max_value=5,
                value=2
            )
            
            dropout = st.slider(
                "Dropout比例",
                min_value=0.0,
                max_value=0.9,
                value=0.2,
                step=0.1
            )
        
        with col2:
            learning_rate = st.number_input(
                "学习率",
                min_value=0.0001,
                max_value=0.1,
                value=0.001,
                format="%.4f"
            )
            
            batch_size = st.number_input(
                "批次大小",
                min_value=1,
                max_value=256,
                value=32
            )
            
            epochs = st.number_input(
                "训练轮数",
                min_value=1,
                max_value=1000,
                value=100
            )
        
            # 训练控制
        st.markdown("### 训练控制")
        
        train_col1, train_col2 = st.columns([3, 1])
        with train_col1:
            start_training = st.button(
                "开始训练",
                use_container_width=True
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
        if 'training_history' in st.session_state and 'training_complete' in st.session_state and st.session_state['training_complete'] and not start_training:
            history = st.session_state['training_history']
            with loss_chart_placeholder:
                # 绘制已有的损失曲线
                history_df = pd.DataFrame({
                    '训练损失': history['train_loss'],
                    '验证损失': history['val_loss']
                })
                st.line_chart(history_df)
        
        if start_training:
            with progress_placeholder.container():
                st.info("训练过程将在这里显示...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
            with loss_chart_placeholder.container():
                # 临时数据用于示例
                chart_data = pd.DataFrame(
                    np.random.randn(20, 2),
                    columns=['训练损失', '验证损失']
                )
                st.line_chart(chart_data)
    
    # ARIMA参数设置
    with model_tabs[1]:
        st.markdown("### ARIMA模型参数")
        
        col1, col2 = st.columns(2)
        with col1:
            p_param = st.number_input(
                "p (AR阶数)",
                min_value=0,
                max_value=10,
                value=2
            )
            
            d_param = st.number_input(
                "d (差分阶数)",
                min_value=0,
                max_value=2,
                value=1
            )
        
        with col2:
            q_param = st.number_input(
                "q (MA阶数)",
                min_value=0,
                max_value=10,
                value=2
            )
            
            seasonal = st.checkbox(
                "包含季节性成分",
                value=False
            )
    
    # Prophet参数设置
    with model_tabs[2]:
        st.markdown("### Prophet模型参数")
        
        col1, col2 = st.columns(2)
        with col1:
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
        
        with col2:
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
    
# 左侧栏 - 数据信息和特征选择
with left_column:
    st.subheader("数据和特征")
    
    
    # 显示数据基本信息
    with st.expander("数据信息", expanded=True):
        if 'raw_data' in st.session_state and st.session_state['raw_data'] is not None:
            df = st.session_state['raw_data']
            st.write(f"数据形状: {df.shape}")
            st.write(f"时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
            st.warning("未加载数据或数据为空")
    
    # 数据划分设置        
    with st.expander("数据划分", expanded=True):
        train_test_ratio = st.slider(
            "训练集比例", 
            min_value=0.5, 
            max_value=0.9, 
            value=0.8, 
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
        
    st.subheader("模型信息")
    
    # 模型状态信息
    with st.expander("训练状态", expanded=True):
        if 'training_complete' in st.session_state and st.session_state['training_complete']:
            st.success("模型训练已完成")
        elif start_training:
            st.info("模型训练中...")
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
        elif start_training:
            st.info("模型评估中...")
        else:
            st.info("训练模型后将显示评估指标")


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

# lstm执行训练的逻辑
if start_training:
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
    
    # 显示训练完成消息
    st.success("模型训练已完成！")
    # 重新加载页面以更新左侧栏状态
    st.rerun()