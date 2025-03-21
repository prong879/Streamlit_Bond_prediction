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
def select_features(df, correlation_threshold=0.5, vif_threshold=10.0, p_value_threshold=0.05):
    """
    特征筛选函数 - 并行评估各筛选方法，取交集
    
    Args:
        df: 输入数据
        correlation_threshold: 相关性阈值
        vif_threshold: VIF阈值
        p_value_threshold: P值阈值
        
    Returns:
        dict: 包含筛选结果和数据的字典
    """
    try:
        # 确保数据是数值类型
        numeric_df = df.select_dtypes(include=[np.number])
        
        # 1. 相关性分析 - 独立评估所有特征
        corr_matrix = numeric_df.corr(numeric_only=True)
        # 计算与目标变量的相关性，保留原始值（不取绝对值）
        target_corr = corr_matrix['Close']
        # 按相关性绝对值排序，但保留原始相关性值
        target_corr_sorted = target_corr[abs(target_corr).sort_values(ascending=False).index]
        # 根据阈值筛选特征（使用绝对值判断）
        high_correlation_features = target_corr_sorted[abs(target_corr_sorted) > correlation_threshold].index.tolist()
        
        # 准备相关性数据用于展示
        corr_data = pd.DataFrame({
            'Feature': target_corr_sorted.index,
            'Correlation': target_corr_sorted.values
        })
            
        # 2. VIF分析 - 独立评估所有特征
        # 基于所有数值特征进行VIF分析，包括目标变量
        vif_features = numeric_df.columns.tolist()
        low_vif_features = []
        vif_data = pd.DataFrame()
        vif_warnings = []  # 收集VIF分析过程中的警告信息
        
        # 加入特征数量验证检查
        if len(vif_features) > 1:  # 确保至少有两个特征
            X = numeric_df[vif_features].copy()
            
            # 加入非空检查
            if not X.empty and X.shape[1] > 0:
                # 检查是否存在已知的高度相关特征
                if 'MA20' in X.columns and 'Upper_Band' in X.columns and 'Lower_Band' in X.columns:
                    X = X.drop(['Upper_Band', 'Lower_Band'], axis=1, errors='ignore')
                    vif_warnings.append("布林带指标（Upper_Band、Lower_Band）与MA20存在完全共线性关系")
                
                # 添加常数项
                X = sm.add_constant(X)
                vif_data = pd.DataFrame()
                vif_data["Feature"] = X.columns
                vif_values = []
                
                # 用于收集完全共线性的特征
                collinear_features = []
                
                for i in range(X.shape[1]):
                    try:
                        # 计算VIF值
                        r_squared_i = sm.OLS(
                            X.iloc[:, i],
                            X.iloc[:, list(range(i)) + list(range(i+1, X.shape[1]))],
                            missing='drop'
                        ).fit().rsquared
                        
                        # 处理极端情况
                        if r_squared_i > 0.999:
                            vif_i = float('inf')
                            collinear_features.append(X.columns[i])
                        else:
                            vif_i = 1.0 / (1.0 - r_squared_i)
                            
                        # 处理数值异常
                        if not np.isfinite(vif_i) or vif_i > 1e6:
                            vif_i = float('inf')
                            if X.columns[i] not in collinear_features:
                                collinear_features.append(X.columns[i])
                            
                    except Exception as e:
                        vif_warnings.append(f"计算特征 '{X.columns[i]}' 的VIF值时出错: {str(e)}")
                        vif_i = float('inf')
                        if X.columns[i] not in collinear_features:
                            collinear_features.append(X.columns[i])
                    
                    vif_values.append(vif_i)
                
                vif_data["VIF"] = vif_values
                vif_data = vif_data[vif_data["Feature"] != "const"]  # 移除常数项
                vif_data = vif_data.sort_values("VIF", ascending=False)
                
                # 获取VIF低于阈值的特征
                low_vif_features = vif_data[vif_data["VIF"] < vif_threshold]["Feature"].tolist()
            else:
                low_vif_features = vif_features
        else:
            low_vif_features = vif_features
        
        # 3. 统计显著性分析 - 独立评估所有特征
        significant_features = []
        sig_data = pd.DataFrame()
        
        # 使用所有数值特征
        X_sig_features = numeric_df.columns.tolist()
        
        if len(X_sig_features) > 0:
            X = numeric_df[X_sig_features]
            y = numeric_df['Close']
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
                        X_feature = X[[feature]]
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
                    'P值': p_values.values,
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
        if 'Close' not in selected_features:
            selected_features.append('Close')
        
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
            'selected_features': ['Close'] + [col for col in df.columns if col != 'Close'][:5] if 'Close' in df.columns else df.columns.tolist()[:6]
        }

def create_correlation_bar_chart(corr_data, threshold):
    """
    创建相关性条形图
    
    Args:
        corr_data: 包含特征和相关性数据的DataFrame
        threshold: 相关性阈值
    """
    # 准备数据
    features = corr_data['Feature'].tolist()
    correlations = corr_data['Correlation'].tolist()
    
    # 创建相关性条形图ECharts配置
    option = {
        'title': {
            'text': '特征与目标变量的相关性',
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            },
            'formatter': "function(params) { return params[0].name + ': ' + params[0].value.toFixed(4); }"
        },
        'grid': {
            'left': '5%',
            'right': '10%',
            'bottom': '15%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'category',
            'data': features,
            'name': '特征',
            'axisLabel': {
                'interval': 0,
                'rotate': 45
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '相关系数',
            'min': -1,
            'max': 1,
            'interval': 0.2
        },
        'series': [
            {
                'name': '相关性',
                'type': 'bar',
                'data': correlations,
                'itemStyle': {
                    'color': "function(params) { return params.value >= 0 ? '#5470c6' : '#ee6666'; }"
                }
            }
        ],
        'markLine': {
            'data': [
                {
                    'yAxis': threshold,
                    'lineStyle': {
                        'color': '#91cc75',
                        'type': 'dashed'
                    },
                    'label': {
                        'formatter': f'阈值: +{threshold}',
                        'position': 'end'
                    }
                },
                {
                    'yAxis': -threshold,
                    'lineStyle': {
                        'color': '#91cc75',
                        'type': 'dashed'
                    },
                    'label': {
                        'formatter': f'阈值: -{threshold}',
                        'position': 'end'
                    }
                }
            ]
        }
    }
    
    # 使用streamlit-echarts渲染图表
    st_echarts(option, height="400px")

def create_vif_bar_chart(vif_data, threshold):
    """
    创建VIF条形图
    
    Args:
        vif_data: 包含特征和VIF数据的DataFrame
        threshold: VIF阈值
    """
    # 准备数据
    features = vif_data['Feature'].tolist()
    # 处理无穷大值，将其替换为一个较大的数值
    vif_values = []
    for vif in vif_data['VIF'].tolist():
        if np.isinf(vif) or vif > 1e6:
            vif_values.append(1e6)  # 使用1e6代替无穷大
        else:
            vif_values.append(float(vif))
    
    # 创建VIF条形图ECharts配置
    option = {
        'title': {
            'text': '特征的VIF值（值越大表示共线性越严重）',
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            },
            'formatter': "function(params) { return params[0].name + ': ' + (params[0].value >= 1e6 ? '∞' : params[0].value.toFixed(2)); }"
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '3%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'value',
            'name': 'VIF值',
            'axisLabel': {
                'formatter': "function(value) { return value >= 1e6 ? '∞' : value.toFixed(2); }"
            }
        },
        'yAxis': {
            'type': 'category',
            'data': features,
            'name': '特征',
            'axisLabel': {
                'interval': 0,
                'rotate': 0
            }
        },
        'series': [
            {
                'name': 'VIF',
                'type': 'bar',
                'data': vif_values,
                'itemStyle': {
                    'color': '#91cc75'
                }
            }
        ],
        'markLine': {
            'data': [
                {
                    'xAxis': threshold,
                    'lineStyle': {
                        'color': '#ff0000',
                        'type': 'dashed'
                    },
                    'label': {
                        'formatter': f'阈值: {threshold}',
                        'position': 'end'
                    }
                }
            ]
        }
    }
    
    # 使用streamlit-echarts渲染图表
    st_echarts(option, height=f"{max(400, len(features) * 30)}px")

def create_significance_charts(sig_data, p_value_threshold):
    """
    创建统计显著性图表（F值和P值）
    
    Args:
        sig_data: 包含特征、F值和P值数据的DataFrame
        p_value_threshold: P值阈值
    """
    # 准备数据
    features = sig_data['Feature'].tolist()
    # 处理F值，确保不超过显示限制
    f_scores = []
    for f_value in sig_data['F值'].tolist():
        if f_value > 1e6:
            f_scores.append(1e6)
        else:
            f_scores.append(float(f_value))
    p_values = sig_data['P值'].tolist()
    
    # F值图表配置
    f_score_option = {
        'title': {
            'text': '特征的F值（值越大表示特征越重要）',
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            },
            'formatter': "function(params) { return params[0].name + ': ' + (params[0].value >= 1e6 ? '> 1e6' : params[0].value.toFixed(2)); }"
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '3%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'value',
            'name': 'F值',
            'axisLabel': {
                'formatter': "function(value) { return value >= 1e6 ? '> 1e6' : value.toFixed(2); }"
            }
        },
        'yAxis': {
            'type': 'category',
            'data': features,
            'name': '特征',
            'axisLabel': {
                'interval': 0,
                'rotate': 0
            }
        },
        'series': [
            {
                'name': 'F值',
                'type': 'bar',
                'data': f_scores,
                'itemStyle': {
                    'color': '#fac858'
                }
            }
        ]
    }
    
    # P值图表配置
    p_value_option = {
        'title': {
            'text': '特征的P值（值越小表示越显著）',
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
            'name': 'P值',
            'axisLabel': {
                'formatter': '{value}'
            }
        },
        'yAxis': {
            'type': 'category',
            'data': features,
            'name': '特征',
            'axisLabel': {
                'interval': 0,
                'rotate': 0
            }
        },
        'series': [
            {
                'name': 'P值',
                'type': 'bar',
                'data': p_values,
                'itemStyle': {
                    'color': '#ee6666'
                }
            }
        ],
        'markLine': {
            'data': [
                {
                    'xAxis': p_value_threshold,
                    'lineStyle': {
                        'color': '#ff0000',
                        'type': 'dashed'
                    },
                    'label': {
                        'formatter': f'阈值: {p_value_threshold}',
                        'position': 'end'
                    }
                }
            ]
        }
    }
    
    # 使用streamlit-echarts渲染图表
    st_echarts(f_score_option, height=f"{max(400, len(features) * 30)}px")
    st_echarts(p_value_option, height=f"{max(400, len(features) * 30)}px")

def create_correlation_heatmap(corr_matrix):
    """
    创建相关性热力图
    
    Args:
        corr_matrix: 相关性矩阵DataFrame
    """
    # 准备数据
    features = corr_matrix.columns.tolist()
    data = []
    
    # 转换数据格式为ECharts所需的格式
    for i in range(len(features)):
        for j in range(len(features)):
            value = corr_matrix.iloc[i, j]
            data.append([i, j, round(float(value), 4)])
    
    # 创建相关性热力图ECharts配置
    option = {
        'title': {
            'text': '特征相关性热力图',
            'left': 'center'
        },
        'tooltip': {
            'position': 'top',
            'formatter': {
                'type': 'function',
                'function': "function(params) { return features[params.data[1]] + ' vs ' + features[params.data[0]] + ': ' + params.data[2].toFixed(4); }"
            }
        },
        'grid': {
            'height': '70%',
            'top': '10%'
        },
        'xAxis': {
            'type': 'category',
            'data': features,
            'splitArea': {
                'show': True
            },
            'axisLabel': {
                'interval': 0,
                'rotate': 45
            }
        },
        'yAxis': {
            'type': 'category',
            'data': features,
            'splitArea': {
                'show': True
            }
        },
        'visualMap': {
            'min': -1,
            'max': 1,
            'calculable': True,
            'orient': 'horizontal',
            'left': 'center',
            'bottom': '0%',
            'inRange': {
                'color': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
            }
        },
        'series': [{
            'name': '相关性',
            'type': 'heatmap',
            'data': data,
            'label': {
                'show': True,
                'formatter': {
                    'type': 'function',
                    'function': "function(params) { return params.data[2].toFixed(4); }"
                }
            },
            'emphasis': {
                'itemStyle': {
                    'shadowBlur': 10,
                    'shadowColor': 'rgba(0, 0, 0, 0.5)'
                }
            }
        }]
    }
    
    # 使用streamlit-echarts渲染图表
    st_echarts(option, height=f"{max(400, len(features) * 30)}px") 