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

# 添加项目根目录到系统路径
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

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

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
import json
from datetime import datetime
from utils.session import get_state, set_state, update_states
from pathlib import Path
import sys

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
tech_indicators = None

# 创建三栏布局
left_column, middle_column, right_column = st.columns([1, 2, 1])

# 左侧栏 - 数据信息和特征选择
with left_column:
    st.subheader("数据和特征")
    
    # 显示数据基本信息
    with st.expander("数据信息", expanded=True):
        if 'raw_data' in st.session_state:
            df = st.session_state['raw_data']
            st.write(f"数据形状: {df.shape}")
            st.write(f"时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 特征选择
    with st.expander("特征选择", expanded=True):
        if 'raw_data' in st.session_state:
            df = st.session_state['raw_data']
            all_features = df.columns.tolist()
            selected_features = st.multiselect(
                "选择用于训练的特征",
                options=all_features,
                default=['Close'] if 'Close' in all_features else all_features[:1]
            )
    
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
        
        sequence_length = st.number_input(
            "输入序列长度",
            min_value=1,
            max_value=100,
            value=10,
            help="用于预测的历史数据点数量"
        )
        
        prediction_length = st.number_input(
            "预测序列长度",
            min_value=1,
            max_value=30,
            value=1,
            help="需要预测的未来数据点数量"
        )

# 中间栏 - 模型参数设置与训练控制
with middle_column:
    st.subheader("模型参数配置")
    
    # 模型类型选择标签页
    model_tabs = st.tabs(["LSTM", "ARIMA", "Prophet"])
    
    # LSTM参数设置
    with model_tabs[0]:
        st.markdown("### LSTM模型参数")
        
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
    
# 右侧栏 - 模型保存和信息显示
with right_column:
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
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
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
        dropout=model_params.get('dropout', 0.2)
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
        if loss_chart is not None:
            with loss_chart:
                st.line_chart(loss_df)
        
        # 更新进度条和状态文本
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss.item():.6f}")
    
    progress_bar.empty()
    status_text.text("模型训练完成！")
    
    return model, history

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
if start_training:
    # 准备特征数据
    if not selected_features:
        st.error("请至少选择一个特征用于训练")
        st.stop()
    
    with progress_placeholder.container():
        st.info("准备训练数据...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 提取选定的特征
        feature_data = df[selected_features].values
        target_data = df['Close'].values.reshape(-1, 1) if 'Close' in df.columns else df[df.columns[0]].values.reshape(-1, 1)
        
        # 数据归一化
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        feature_data = feature_scaler.fit_transform(feature_data)
        target_data = target_scaler.fit_transform(target_data)
        
        # 保存归一化器以供后续预测使用
        st.session_state['feature_scaler'] = feature_scaler
        st.session_state['target_scaler'] = target_scaler
        
        # 创建时间序列数据
        X, y = create_sequences(
            np.column_stack((feature_data, target_data)), 
            int(sequence_length)
        )
        
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
        
        st.info(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}")
        
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
        st.info("开始训练LSTM模型...")
        model, history = train_lstm_model(X_train, y_train, X_val, y_val, model_params, training_params, 
                                         progress_bar=progress_bar, status_text=status_text, 
                                         loss_chart=loss_chart_placeholder)
        
        # 绘制训练历史
        plot_training_history(history, loss_chart_placeholder)
        
        # 测试集评估
        st.subheader("模型评估")
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
            
            # 更新右侧栏中的评估指标
            st.session_state['model_metrics'] = {
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAE': float(mae)
            }
        
        # 保存模型
        model_path = save_model(model, model_params, training_params, history)
        st.success(f"模型已保存到: {model_path}")
        
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
        
        # 刷新右侧栏状态显示
        with right_column:
            with st.expander("训练状态", expanded=True):
                st.success("模型训练已完成")

def select_features(df, correlation_threshold=0.5, vif_threshold=10, p_value_threshold=0.05):
    """
    基于相关性、多重共线性和统计显著性进行特征选择
    
    参数:
    df: 包含特征的DataFrame
    correlation_threshold: 相关性阈值，默认为0.5
    vif_threshold: VIF阈值，默认为10
    p_value_threshold: p值阈值，默认为0.05
    
    返回:
    selected_features: 选择的特征列表
    """
    try:
        target_col = 'Close'
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
            high_correlation_features = target_correlations[abs(target_correlations) > correlation_threshold].index.tolist()
            st.write(f"相关性高于{correlation_threshold}的特征: {high_correlation_features}")
        
        # 步骤2: 多重共线性分析 - 计算VIF (Variance Inflation Factor)
        # 创建一个没有目标变量的特征子集
        X = df[numeric_cols].drop(target_col, axis=1)
        
        # 检查是否有足够的样本
        if len(X) <= len(X.columns):
            st.warning("样本数量不足以进行多重共线性分析。跳过VIF计算。")
            high_vif_features = []
        else:
            try:
                # 添加常数项
                X_with_const = sm.add_constant(X)
                
                # 计算VIF
                vif_data = pd.DataFrame()
                vif_data["特征"] = X_with_const.columns
                
                # 安全地计算VIF，处理可能的错误
                vif_values = []
                for i in range(X_with_const.shape[1]):
                    try:
                        vif_i = variance_inflation_factor(X_with_const.values, i)
                        vif_values.append(vif_i)
                    except Exception as e:
                        st.warning(f"计算特征 '{X_with_const.columns[i]}' 的VIF时出错: {str(e)}")
                        vif_values.append(float('inf'))  # 标记为无穷大
                
                vif_data["VIF"] = vif_values
                vif_data = vif_data.sort_values("VIF", ascending=False)
                
                # 显示VIF分析结果
                with st.expander("**多重共线性分析**", expanded=False):
                    st.dataframe(vif_data)
                    st.info("VIF > 10表示存在严重的多重共线性")
                
                # 移除VIF过高的特征
                high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["特征"].tolist()
                if 'const' in high_vif_features:
                    high_vif_features.remove('const')  # 移除常数项
                
                with st.expander("**多重共线性过滤**", expanded=False):
                    st.write(f"多重共线性严重的特征 (VIF > {vif_threshold}): {high_vif_features}")
            except Exception as e:
                st.warning(f"VIF计算失败: {str(e)}")
                high_vif_features = []
        
        # 步骤3: 基于统计显著性的特征选择
        try:
            # 使用f_regression评估特征的统计显著性
            X = df[numeric_cols].drop(target_col, axis=1).values
            y = df[target_col].values
            
            # 确保X和y有相同的样本数
            if len(X) != len(y):
                st.warning("特征矩阵和目标变量长度不匹配，无法进行显著性分析")
                significant_features = []
            else:
                f_selector = SelectKBest(f_regression, k='all')
                f_selector.fit(X, y)
                
                # 获取每个特征的p值和F值
                f_scores = pd.DataFrame()
                f_scores["特征"] = df[numeric_cols].drop(target_col, axis=1).columns
                f_scores["F统计量"] = f_selector.scores_
                f_scores["P值"] = f_selector.pvalues_
                f_scores = f_scores.sort_values("F统计量", ascending=False)
                
                with st.expander("**统计显著性分析**", expanded=False):
                    st.dataframe(f_scores)
                    st.info("P值 < 0.05 表示特征具有统计显著性")
                
                # 选择统计显著的特征(p值<p_value_threshold)
                significant_features = f_scores[f_scores["P值"] < p_value_threshold]["特征"].tolist()
                
                with st.expander("**统计显著性过滤**", expanded=False):
                    st.write(f"统计显著的特征 (P < {p_value_threshold}): {significant_features}")
        except Exception as e:
            st.warning(f"统计显著性分析失败: {str(e)}")
            significant_features = []
        
        # 步骤4: 综合分析，选择最终的特征集
        # 从高相关性特征中移除多重共线性严重的特征
        selected_features = [f for f in high_correlation_features if f not in high_vif_features]
        
        # 确保所有统计显著的特征都被包含
        for feature in significant_features:
            if feature not in selected_features and feature != target_col:
                selected_features.append(feature)
        
        # 确保目标变量在特征集中
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        st.success(f"特征选择完成！从 {len(numeric_cols)} 个特征中选出 {len(selected_features)} 个特征")
        
        return selected_features
    
    except Exception as e:
        st.error(f"特征选择过程中发生错误: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        # 返回默认特征作为备选
        if 'Close' in df.columns:
            return ['Close'] + [col for col in df.columns if col != 'Close'][:5]  # 返回Close和其他5个特征
        else:
            return df.columns.tolist()[:6]  # 返回前6个特征 