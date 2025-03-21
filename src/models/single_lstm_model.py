"""
单层LSTM模型模块 - 定义和训练单层LSTM模型
"""
import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt

class SingleLSTM(nn.Module):
    """
    单层LSTM模型类
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化单层LSTM模型
        
        参数:
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        """
        super(SingleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1  # 单层LSTM

        self.lstm = nn.LSTM(input_dim, hidden_dim, self.num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)  # 使用较小的dropout
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播
        
        参数:
        x: 输入数据
        
        返回:
        模型输出
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])  # 应用dropout
        out = self.fc(out)
        return out

def train_single_lstm_model(x_train, y_train, input_dim, hidden_dim=64, output_dim=1, 
                    num_epochs=100, learning_rate=0.01, weight_decay=1e-5, patience=15, output_dir=None):
    """
    训练单层LSTM模型
    
    参数:
    x_train: 训练集特征
    y_train: 训练集标签
    input_dim: 输入特征维度
    hidden_dim: 隐藏层维度
    output_dim: 输出维度
    num_epochs: 训练轮数
    learning_rate: 学习率
    weight_decay: L2正则化参数
    patience: 早停耐心值
    output_dir: 输出目录
    
    返回:
    训练好的模型和训练历史
    """
    # 转换为PyTorch张量
    x_train_tensor = torch.from_numpy(x_train).float() if not isinstance(x_train, torch.Tensor) else x_train
    y_train_tensor = torch.from_numpy(y_train).float() if not isinstance(y_train, torch.Tensor) else y_train
    
    # 创建模型
    model = SingleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # 定义损失函数和优化器
    criterion = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 训练历史
    hist = np.zeros(num_epochs)
    start_time = time.time()
    
    # 早停机制
    best_loss = float('inf')
    counter = 0
    early_stop = False
    best_model_path = os.path.join(output_dir, 'single_lstm_best_model.pth') if output_dir else 'single_lstm_best_model.pth'
    
    # 训练循环
    for t in range(num_epochs):
        # 前向传播
        y_train_pred = model(x_train_tensor)
        
        # 计算损失
        loss = criterion(y_train_pred, y_train_tensor)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] = loss.item()
        
        # 反向传播和优化
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        # 早停检查
        if loss.item() < best_loss:
            best_loss = loss.item()
            counter = 0
            # 保存最佳模型
            try:
                torch.save(model.state_dict(), best_model_path)
                print(f"成功保存模型到: {best_model_path}")
            except Exception as e:
                print(f"保存模型时出错: {str(e)}")
                print(f"将尝试使用绝对路径保存")
                try:
                    # 尝试使用绝对路径保存
                    abs_path = os.path.abspath(os.path.join(os.path.dirname(best_model_path), f"model_{t}.pth"))
                    torch.save(model.state_dict(), abs_path)
                    best_model_path = abs_path  # 更新路径
                    print(f"成功使用绝对路径保存模型: {abs_path}")
                except Exception as e2:
                    print(f"使用绝对路径保存也失败: {str(e2)}")
                    print("将继续训练但不保存模型")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {t}")
                # 加载最佳模型
                try:
                    model.load_state_dict(torch.load(best_model_path))
                    print(f"成功加载最佳模型")
                except Exception as e:
                    print(f"加载最佳模型时出错: {str(e)}")
                    print("将使用当前模型继续")
                # 更新hist数组，截断未使用的部分
                hist = hist[:t+1]
    
    training_time = time.time() - start_time
    print("Training time: {}".format(training_time))
    print(f"Best loss: {best_loss:.6f}")
    
    return model, hist, training_time, best_loss 