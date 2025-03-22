"""
LSTM模型工具模块
用于时间序列预测的LSTM模型定义、训练和评估

本模块包含了LSTM模型的完整功能实现，包括：
- LSTM模型定义
- 时间序列数据准备
- 模型训练与评估
- 模型保存与加载
- 特征选择
- 特征选择可视化
"""

# 导入必要的库
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, f_regression
from streamlit_echarts import st_echarts

# LSTM模型定义
class LSTMModel(nn.Module):
    """
    LSTM模型定义
    
    参数:
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

# 创建时间序列数据
def create_sequences(data, seq_length):
    """
    创建时间序列数据
    
    参数:
        data: 输入数据
        seq_length: 序列长度
        
    返回:
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
    
    参数:
        X_train: 训练特征数据
        y_train: 训练目标数据
        X_val: 验证特征数据
        y_val: 验证目标数据
        model_params: 模型参数字典
        training_params: 训练参数字典
        progress_bar: streamlit进度条
        status_text: streamlit状态文本
        loss_chart: 损失曲线图表占位符
        
    返回:
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
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_params['learning_rate'],
        weight_decay=1e-5  # 添加L2正则化，权重衰减参数为1e-5
    )
    
    # 训练参数
    epochs = training_params['epochs']
    batch_size = training_params['batch_size']
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    # 使用传入的进度条或创建新的
    if progress_bar is None and st is not None:
        progress_bar = st.progress(0)
    if status_text is None and st is not None:
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
        if loss_chart is not None and st is not None:
            with loss_chart:
                st.line_chart(loss_df)
        
        # 更新进度条和状态文本
        progress = (epoch + 1) / epochs
        if progress_bar is not None:
            progress_bar.progress(progress)
        if status_text is not None:
            status_text.text(f"Epoch {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss.item():.6f}")
    
    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.text("模型训练完成！")
    
    return model, history

# 评估LSTM模型性能
def evaluate_lstm_model(model, X_test, y_test, target_scaler=None):
    """
    评估LSTM模型性能
    
    参数:
        model: 训练好的LSTM模型
        X_test: 测试集特征
        y_test: 测试集目标值
        target_scaler: 目标缩放器，用于反归一化
        
    返回:
        评估指标字典
    """
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        
        test_outputs = model(X_test_tensor)
        test_criterion = nn.MSELoss()
        test_loss = test_criterion(test_outputs, y_test_tensor)
        
        # 反归一化（如果提供了缩放器）
        if target_scaler is not None:
            test_predictions = target_scaler.inverse_transform(test_outputs.numpy())
            test_actual = target_scaler.inverse_transform(y_test)
        else:
            test_predictions = test_outputs.numpy()
            test_actual = y_test
        
        # 计算评估指标
        mse = np.mean((test_predictions - test_actual) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(test_predictions - test_actual))
        
        # 计算方向准确率
        if len(test_predictions) > 1:
            pred_direction = np.sign(test_predictions[1:] - test_predictions[:-1])
            actual_direction = np.sign(test_actual[1:] - test_actual[:-1])
            direction_accuracy = np.mean(pred_direction == actual_direction)
        else:
            direction_accuracy = 0.0
            
        # 计算MAPE
        non_zero_mask = test_actual != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((test_actual[non_zero_mask] - test_predictions[non_zero_mask]) / test_actual[non_zero_mask])) * 100
        else:
            mape = np.nan
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'MAPE': float(mape) if not np.isnan(mape) else None,
        'Direction_Accuracy': float(direction_accuracy),
        'Test_Loss': float(test_loss.item())
    }

# 绘制训练历史
def plot_training_history(history, chart_placeholder=None):
    """
    绘制训练历史
    
    参数:
        history: 训练历史字典
        chart_placeholder: 图表占位符
    """
    # 创建DataFrame用于绘图
    history_df = pd.DataFrame({
        '训练损失': history['train_loss'],
        '验证损失': history['val_loss']
    })
    
    if chart_placeholder is not None and st is not None:
        with chart_placeholder:
            st.line_chart(history_df)
    else:
        # 使用matplotlib绘图
        plt.figure(figsize=(10, 6))
        plt.plot(history_df['训练损失'], label='训练损失')
        plt.plot(history_df['验证损失'], label='验证损失')
        plt.title('模型训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)
        
        if st is not None:
            st.pyplot(plt)
        else:
            plt.show()

# 保存模型
def save_model(model, model_params, training_params, history, path="models"):
    """
    保存模型和训练参数
    
    参数:
        model: 训练好的模型
        model_params: 模型参数
        training_params: 训练参数
        history: 训练历史
        path: 保存路径
    
    返回:
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

# 加载已保存的模型
def load_model(model_path, params_path=None):
    """
    加载已保存的模型
    
    参数:
        model_path: 模型文件路径
        params_path: 参数文件路径，如果为None则根据model_path推断
        
    返回:
        model: 加载的模型
        params_dict: 模型参数字典
    """
    # 如果未提供参数路径，则尝试推断
    if params_path is None:
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        timestamp = model_filename.split('_')[-1].split('.')[0]  # 提取时间戳
        params_path = os.path.join(model_dir, f"model_params_{timestamp}.json")
    
    # 加载参数
    try:
        with open(params_path, 'r') as f:
            params_dict = json.load(f)
        
        model_params = params_dict['model_params']
        
        # 创建模型实例
        model = LSTMModel(
            input_dim=model_params['input_dim'],
            hidden_dim=model_params['hidden_dim'],
            num_layers=model_params['num_layers'],
            output_dim=model_params['output_dim'],
            dropout=model_params.get('dropout', 0.3)
        )
        
        # 加载权重
        model.load_state_dict(torch.load(model_path))
        model.eval()  # 设置为评估模式
        
        return model, params_dict
    
    except Exception as e:
        raise Exception(f"加载模型时出错: {str(e)}")

# 准备LSTM模型训练所需的数据
def prepare_data_for_lstm(df, selected_features, target_col='Close', sequence_length=20, train_ratio=0.8, val_ratio=0.15):
    """
    准备LSTM模型训练所需的数据
    
    参数:
        df: 输入DataFrame
        selected_features: 选择的特征列表
        target_col: 目标列名
        sequence_length: 输入序列长度
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    返回:
        数据集和缩放器的字典
    """
    # 确保目标变量在特征列表中
    if target_col not in selected_features:
        selected_features.append(target_col)
    
    # 分离特征和目标变量
    feature_cols = [col for col in selected_features if col != target_col]
    target_data = df[target_col].values.reshape(-1, 1)
    feature_data = df[feature_cols].values
    
    # 数据归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_data = feature_scaler.fit_transform(feature_data)
    target_data = target_scaler.fit_transform(target_data)
    
    # 创建时间序列数据
    X, y = create_sequences(
        np.column_stack((feature_data, target_data)), 
        int(sequence_length)
    )
    
    # 分离特征和目标变量
    X = X[:, :, :-1]  # 移除最后一列（目标变量）
    y = y[:, -1:]     # 只取最后一列（目标变量）
    
    # 划分训练、验证和测试集
    total_samples = len(X)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'sequence_length': sequence_length,
        'feature_cols': feature_cols  # 返回实际使用的特征列名
    }

# 使用LSTM模型进行预测
def predict_lstm(model, X, target_scaler=None, n_steps=1):
    """
    使用LSTM模型进行预测
    
    参数:
        model: 训练好的LSTM模型
        X: 输入特征，形状为(批次大小, 序列长度, 特征数)
        target_scaler: 用于反归一化的目标缩放器
        n_steps: 预测步数
        
    返回:
        predictions: 预测结果
    """
    model.eval()
    
    # 确保X的形状正确
    if len(X.shape) == 2:
        # 如果输入是(序列长度, 特征数)，则添加批次维度
        X = X.reshape(1, X.shape[0], X.shape[1])
    
    # 将输入转换为张量
    X_tensor = torch.FloatTensor(X)
    
    predictions = []
    current_input = X_tensor.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # 预测下一个值
            output = model(current_input)
            prediction = output.numpy()
            predictions.append(prediction)
            
            # 如果需要多步预测，更新输入
            if n_steps > 1 and _ < n_steps - 1:
                # 滑动窗口：移除最早的时间步，添加预测值
                new_input = current_input.clone()
                new_input[:, :-1, :] = current_input[:, 1:, :]
                # 添加预测值（假设最后一个特征是预测目标）
                # 注意：这里可能需要根据实际情况调整
                new_input[:, -1, -1] = output[0, 0]
                current_input = new_input
    
    # 合并预测结果
    predictions = np.concatenate(predictions, axis=0)
    
    # 反归一化（如果提供了缩放器）
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions)
    
    return predictions
# 绘制预测结果对比图
def plot_predictions(y_true, y_pred, title='预测结果对比', dates=None):
    """
    绘制预测结果对比图
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        dates: 日期索引（可选）
    """
    plt.figure(figsize=(12, 6))
    
    if dates is not None:
        x_axis = dates
    else:
        x_axis = range(len(y_true))
    
    plt.plot(x_axis, y_true, label='实际值', color='blue')
    plt.plot(x_axis, y_pred, label='预测值', color='red', linestyle='--')
    
    plt.title(title, fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('值', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if st is not None:
        st.pyplot(plt)
    else:
        plt.show()

# 执行完整的LSTM模型训练流程
def run_lstm_training(
    selected_features, 
    df, 
    sequence_length, 
    train_test_ratio, 
    hidden_size,
    num_layers,
    dropout,
    learning_rate,
    batch_size,
    epochs,
    progress_placeholder=None,
    loss_chart_placeholder=None
):
    """
    执行完整的LSTM模型训练流程
    
    参数:
        selected_features: 选择的特征列表或特征选择结果字典
        df: 输入数据DataFrame
        sequence_length: 序列长度
        train_test_ratio: 训练集比例
        hidden_size: 隐藏层大小
        num_layers: LSTM层数
        dropout: Dropout比率
        learning_rate: 学习率
        batch_size: 批次大小
        epochs: 训练轮数
        progress_placeholder: 进度条占位符
        loss_chart_placeholder: 损失图表占位符
        
    返回:
        训练结果字典，包含模型、评估指标等
    """
    # 如果提供了progress_placeholder，则创建进度条和状态文本
    if progress_placeholder is not None:
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
    else:
        progress_bar = None
        status_text = None
        
    # 处理selected_features参数，支持新旧两种结构
    if isinstance(selected_features, dict) and 'selected_features' in selected_features:
        feature_list = selected_features['selected_features']
    else:
        feature_list = selected_features
    
    # 提取选定的特征
    feature_data = df[feature_list].values
    target_data = df['Close'].values.reshape(-1, 1) if 'Close' in df.columns else df[df.columns[0]].values.reshape(-1, 1)
    
    # 数据归一化
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    feature_data = feature_scaler.fit_transform(feature_data)
    target_data = target_scaler.fit_transform(target_data)
    
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
    
    if status_text is not None:
        status_text.info(f"数据准备完成! 训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}, 测试集大小: {X_test.shape[0]}。开始训练LSTM模型...")
    
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
    model, history = train_lstm_model(X_train, y_train, X_val, y_val, model_params, training_params, 
                                     progress_bar=progress_bar, status_text=status_text, 
                                     loss_chart=loss_chart_placeholder)
    
    # 绘制训练历史
    plot_training_history(history, loss_chart_placeholder)
    
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
        
        # 计算方向准确率
        if len(test_predictions) > 1:
            pred_direction = np.sign(test_predictions[1:] - test_predictions[:-1])
            actual_direction = np.sign(test_actual[1:] - test_actual[:-1])
            direction_accuracy = np.mean(pred_direction == actual_direction)
        else:
            direction_accuracy = 0.0
    
    # 保存模型
    model_path = save_model(model, model_params, training_params, history)
    
    # 返回训练结果
    return {
        'model': model,
        'history': history,
        'metrics': {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'Direction_Accuracy': float(direction_accuracy) if 'direction_accuracy' in locals() else None,
            'Test_Loss': float(test_loss.item())
        },
        'model_path': model_path,
        'X_test': X_test,
        'y_test': y_test,
        'test_predictions': test_predictions,
        'test_actual': test_actual,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'sequence_length': sequence_length
    } 

# 特征筛选函数
def select_features(df, correlation_threshold=0.5, vif_threshold=10.0, p_value_threshold=0.05, target_col='Close'):
    """
    特征筛选函数 - 并行评估各筛选方法，取交集
    
    Args:
        df: 输入数据
        correlation_threshold: 相关性阈值
        vif_threshold: VIF阈值
        p_value_threshold: P值阈值
        target_col: 目标变量列名，默认为'Close'
        
    Returns:
        dict: 包含筛选结果和数据的字典
    """
    try:
        # 确保数据是数值类型
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 1. 相关性分析 - 独立评估所有特征
        corr_matrix = numeric_df.corr(numeric_only=True)
        # 计算与目标变量的相关性，保留原始值（不取绝对值）
        target_corr = corr_matrix[target_col]
        # 按相关性绝对值排序，但保留原始相关性值
        target_corr_sorted = target_corr[abs(target_corr).sort_values(ascending=False).index]
        # 根据阈值筛选特征（使用绝对值判断）
        high_correlation_features = target_corr_sorted[abs(target_corr_sorted) > correlation_threshold].index.tolist()
        
        # 准备相关性数据用于展示
        corr_data = pd.DataFrame({
            'Feature': target_corr_sorted.index,
            'Correlation': target_corr_sorted.values
        })
            
        # 2. VIF分析 - 排除目标变量，优化计算逻辑
        vif_warnings = []  # 收集VIF分析过程中的警告信息
        
        # 获取所有特征，但排除目标变量
        predictor_features = [col for col in numeric_df.columns if col != target_col]
        
        # 初始化VIF分析结果
        vif_data = pd.DataFrame()
        low_vif_features = []
        collinear_features = []
        
        # 检查是否有足够的特征进行VIF分析
        if len(predictor_features) > 1:  # 至少需要两个特征
            # 创建特征数据框，仅包含预测变量
            X = numeric_df[predictor_features].copy()
            
            # 非空检查
            if not X.empty and X.shape[1] > 0:
                # 移除低方差特征
                try:
                    variance = X.var()
                    low_var_features = variance[variance < 1e-6].index.tolist()
                    if low_var_features:
                        X = X.drop(low_var_features, axis=1)
                        vif_warnings.append(f"移除了以下低方差特征: {', '.join(low_var_features)}")
                except Exception as e:
                    vif_warnings.append(f"方差分析出错: {str(e)}")
                
                # 确保变量数量足够
                if X.shape[1] > 1:
                    try:
                        # 计算VIF值
                        vif_results = []
                        
                        for feature in X.columns:
                            # 选择该特征作为因变量
                            y_var = X[feature]
                            # 选择其他特征作为自变量
                            X_vars = X.drop(feature, axis=1)
                            
                            try:
                                # 添加常数项
                                X_vars = sm.add_constant(X_vars)
                                # 拟合OLS模型
                                model = sm.OLS(y_var, X_vars, missing='drop').fit()
                                # 计算R²值
                                r_squared = model.rsquared
                                
                                # 计算VIF值，注意处理极端情况
                                if r_squared >= 0.999:
                                    vif = float('inf')
                                    collinear_features.append(feature)
                                else:
                                    vif = 1.0 / (1.0 - r_squared)
                                    
                                # 记录结果
                                vif_results.append({
                                    'Feature': feature,
                                    'VIF': vif
                                })
                            except Exception as e:
                                vif_warnings.append(f"计算特征'{feature}'的VIF值时出错: {str(e)}")
                                vif_results.append({
                                    'Feature': feature,
                                    'VIF': float('inf')
                                })
                                collinear_features.append(feature)
                        
                        # 创建VIF数据框并排序
                        vif_data = pd.DataFrame(vif_results)
                        vif_data = vif_data.sort_values('VIF', ascending=False)
                        
                        # 筛选VIF低于阈值的特征
                        low_vif_features = vif_data[vif_data['VIF'] < vif_threshold]['Feature'].tolist()
                    except Exception as e:
                        vif_warnings.append(f"VIF计算过程出错: {str(e)}")
                        low_vif_features = predictor_features
                else:
                    vif_warnings.append("特征数量不足，无法计算VIF")
                    low_vif_features = predictor_features
            else:
                vif_warnings.append("数据为空或没有有效特征")
                low_vif_features = predictor_features
        else:
            vif_warnings.append("特征数量不足，需要至少两个特征才能计算VIF")
            low_vif_features = predictor_features
        
        # 3. 统计显著性分析 - 独立评估所有特征
        significant_features = []
        sig_data = pd.DataFrame()
        
        # 使用所有数值特征
        X_sig_features = predictor_features
        
        if len(X_sig_features) > 0:
            X = numeric_df[X_sig_features]
            y = numeric_df[target_col]
            X = sm.add_constant(X)
            
            try:
                model = sm.OLS(y, X).fit()
                p_values = model.pvalues[1:]  # 排除常数项
                significant_features = p_values[p_values < p_value_threshold].index.tolist()
                
                # 处理F值，避免数值溢出
                f_values = []
                for feature in p_values.index:
                    try:
                        # 计算单个特征的F值
                        X_feature = sm.add_constant(X[[feature]])
                        model_feature = sm.OLS(y, X_feature).fit()
                        f_value = model_feature.fvalue
                        # 如果F值过大，使用一个合理的上限值
                        if f_value > 1e6:
                            f_value = 1e6
                        f_values.append(f_value)
                    except:
                        f_values.append(0.0)
                
                # 准备显著性数据
                sig_data = pd.DataFrame({
                    'Feature': p_values.index,
                    'P值': p_values.values.round(2),
                    'F值': f_values
                }).sort_values('P值', ascending=True)
            except Exception as e:
                significant_features = X_sig_features
        else:
            significant_features = X_sig_features
        
        # 4. 整合所有筛选结果 - 取交集
        # 计算三种筛选方法的结果交集
        if high_correlation_features and low_vif_features and significant_features:
            # 取三者交集
            selected_features = list(set(high_correlation_features) & 
                                   set(low_vif_features) & 
                                   set(significant_features))
            
            # 如果交集为空，尝试取两两交集
            if not selected_features:
                corr_vif_intersection = list(set(high_correlation_features) & set(low_vif_features))
                corr_sig_intersection = list(set(high_correlation_features) & set(significant_features))
                vif_sig_intersection = list(set(low_vif_features) & set(significant_features))
                
                # 使用最大的交集
                max_intersection = max([corr_vif_intersection, corr_sig_intersection, vif_sig_intersection], 
                                     key=len)
                selected_features = max_intersection
                
                # 如果所有两两交集也为空，使用相关性结果
                if not selected_features:
                    selected_features = high_correlation_features
        else:
            # 如果任一筛选方法结果为空，使用相关性结果
            selected_features = high_correlation_features if high_correlation_features else numeric_df.columns.tolist()
        
        # 5. 确保目标变量在特征集中
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        # 返回所有筛选结果和数据
        return {
            'selected_features': selected_features,
            'correlation': {
                'data': corr_data,
                'features': high_correlation_features,
                'matrix': corr_matrix
            },
            'vif': {
                'data': vif_data,
                'features': low_vif_features,
                'warnings': vif_warnings,
                'collinear_features': collinear_features if 'collinear_features' in locals() else []
            },
            'significance': {
                'data': sig_data,
                'features': significant_features
            }
        }
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        # 返回错误信息
        return {
            'error': str(e),
            'traceback': error_traceback,
            'selected_features': [target_col] + [col for col in df.columns if col != target_col][:5] if target_col in df.columns else df.columns.tolist()[:6]
        }


# 筛选后的特征的相关性热力图
def create_correlation_heatmap(corr_matrix, filtered_features=None):
    """
    创建相关性热力图
    
    Args:
        corr_matrix: 相关性矩阵DataFrame
        filtered_features: 筛选后的特征列表，如果提供则只显示这些特征
        
    Returns:
        dict: ECharts配置项字典
    """
    # 检查相关矩阵是否为空
    if corr_matrix is None or corr_matrix.empty:
        # 返回一个提示信息图表
        return {
            'title': {
                'text': '相关性热力图 - 无数据',
                'left': 'center'
            },
            'xAxis': {'type': 'category', 'data': []},
            'yAxis': {'type': 'category', 'data': []},
            'series': []
        }
    
    # 准备数据
    if filtered_features is not None and len(filtered_features) > 0:
        # 只保留筛选后的特征
        features = [f for f in filtered_features if f in corr_matrix.columns]
        # 检查筛选后的特征列表是否为空
        if not features:
            features = corr_matrix.columns.tolist()
        corr_matrix = corr_matrix.loc[features, features]
    else:
        features = corr_matrix.columns.tolist()
    
    data = []
    x_data = features.copy()
    y_data = features.copy()
    
    # 转换数据格式为ECharts所需的格式
    for i in range(len(features)):
        for j in range(len(features)):
            value = corr_matrix.iloc[i, j]
            # 保留4位小数
            rounded_value = round(float(value), 4)
            data.append([i, j, rounded_value])
    
    # 创建相关性热力图ECharts配置
    option = {
        'tooltip': {
            'position': 'top',
        },
        'grid': {
            'top': '0',
            'bottom': '10%',
            'left': '15%'
        },
        'xAxis': {
            'type': 'category',
            'data': x_data,
            'splitArea': {
                'show': True
            },
            'axisLabel': {
                'interval': 0,
                'rotate': 45,
                'formatter': {
                    'function': "function(value) { if(value.length > 15) return value.substring(0,12) + '...'; return value; }"
                }
            }
        },
        'yAxis': {
            'type': 'category',
            'data': y_data,
            'splitArea': {
                'show': True
            },
            'axisLabel': {
                'formatter': {
                    'function': "function(value) { if(value.length > 15) return value.substring(0,12) + '...'; return value; }"
                }
            }
        },
        'visualMap': {
            'min': -1,
            'max': 1,
            'calculable': True,
            'orient': 'vertical',
            'left': '0',
            'bottom': '65',
            'inRange': {
                'color': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
            }
        },
        'series': [{
            'name': '相关性',
            'type': 'heatmap',
            'data': data,
            'label': {
                'show': True
            },
            'emphasis': {
                'itemStyle': {
                    'shadowBlur': 10,
                    'shadowColor': 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    }
    
    return option 

# 创建统计显著性图表
def create_significance_charts(sig_data, p_value_threshold=0.05):
    """
    创建统计显著性条形图（只显示P值，横向展示）
    
    参数:
        sig_data: 包含Feature、P值和F值列的DataFrame
        p_value_threshold: P值阈值
        
    返回:
        tuple: (空字典, P值图表配置) - 为保持API兼容性，第一个返回值仍然保留但为空
    """
    # 检查输入数据是否为空
    if sig_data is None or sig_data.empty:
        # 返回空图表
        empty_option = {
            'title': {
                'text': '无统计显著性数据',
                'left': 'center'
            },
            'xAxis': {'type': 'category', 'data': []},
            'yAxis': {'type': 'value'},
            'series': []
        }
        return empty_option, empty_option
    
    # 按P值排序
    sorted_data = sig_data.sort_values('P值', ascending=False)
    
    # 准备数据
    features = sorted_data['Feature'].tolist()
    p_values = sorted_data['P值'].round(2).tolist()
    
    # 设置P值条形颜色：低于阈值为绿色，高于阈值为红色
    p_colors = ['#2ecc71' if val < p_value_threshold else '#e74c3c' for val in p_values]
    
    # 为P值图表创建数据项，包含颜色信息
    p_items = []
    for i, p_val in enumerate(p_values):
        p_items.append({
            'value': p_val,
            'itemStyle': {
                'color': p_colors[i]
            }
        })
    
    # P值条形图配置 - 横向显示
    p_option = {
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        'grid': {
            'left': '5%',
            'right': '4%',
            'bottom': '3%',
            'top': '5%',
            'containLabel': True
        },
        'yAxis': {
            'type': 'value',
            'name': 'P值',
            'nameLocation': 'center',
            'nameGap': 40,
            'splitLine': {
                'show': True
            }
        },
        'xAxis': {
            'type': 'category',
            'data': features,
            'axisLabel': {
                'interval': 0,
                'rotate': 45
            }
        },
        'series': [{
            'name': 'P值',
            'type': 'bar',
            'data': p_items,
            'label': {
                'show': True,
                'position': 'top',
                'formatter': '{c0}'
            },
            'markLine': {
                'symbol': ['none', 'none'],
                'data': [
                    {'yAxis': p_value_threshold, 'lineStyle': {'type': 'dashed', 'color': '#e74c3c'}, 'label': {'formatter': str(round(p_value_threshold, 2))}}
                ]
            }
        }]
    }
    
    # 为保持API兼容性，返回一个空选项作为第一个元素
    empty_f_option = {}
    
    return empty_f_option, p_option

# 创建相关性条形统计图
def create_correlation_bar_chart(corr_data, correlation_threshold=0.5):
    """
    创建相关性条形统计图
    
    参数:
        corr_data: 包含Feature和Correlation列的DataFrame
        correlation_threshold: 相关性阈值
        
    返回:
        dict: ECharts配置项字典
    """
    # 按相关性绝对值排序，但保留原始相关性值
    features = corr_data['Feature'].tolist()
    correlations = corr_data['Correlation'].tolist()
    
    # 生成数据源，排序和生成颜色
    data = []
    for i in range(len(features)):
        data.append({
            'name': features[i],
            'value': correlations[i]
        })
    
    # 按相关性绝对值排序
    data.sort(key=lambda x: abs(x['value']), reverse=True)
    
    # 提取排序后的特征和相关性数据
    features = [item['name'] for item in data]
    correlations = [item['value'] for item in data]
    
    # 设置条形颜色：正相关为红色，负相关为蓝色
    colors = ['#c23531' if val >= 0 else '#3498db' for val in correlations]
    
    # 创建ECharts选项
    option = {
        'title': {
            'text': '特征与目标变量的相关性',
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '3%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'value',
            'name': '相关系数',
            'nameLocation': 'center',
            'nameGap': 30,
            'min': -1,
            'max': 1,
            'splitLine': {
                'show': True
            },
            'axisLine': {
                'show': True
            }
        },
        'yAxis': {
            'type': 'category',
            'data': features,
            'axisLabel': {
                'interval': 0,
                'formatter': {
                    'function': 'function(value) { if(value.length > 15) return value.substring(0,12) + "..."; return value; }'
                }
            }
        },
        'series': [{
            'name': '相关性',
            'type': 'bar',
            'data': correlations,
            'itemStyle': {
                'color': {
                    'function': 'function(params) { return colors[params.dataIndex]; }'
                }
            },
            'label': {
                'show': True,
                'position': 'right',
                'formatter': {
                    'function': 'function(params) { return params.value.toFixed(4); }'
                }
            },
            'markLine': {
                'symbol': ['none', 'none'],
                'data': [
                    {'xAxis': correlation_threshold, 'lineStyle': {'type': 'dashed', 'color': '#c23531'}, 'label': {'formatter': '+'+str(round(correlation_threshold, 2))}},
                    {'xAxis': -correlation_threshold, 'lineStyle': {'type': 'dashed', 'color': '#3498db'}, 'label': {'formatter': str(round(-correlation_threshold, 2))}}
                ]
            }
        }]
    }
    
    return option

