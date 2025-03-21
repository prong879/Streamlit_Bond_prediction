"""
模型模块初始化文件
"""
from src.models.dual_lstm_model import DualLSTM, train_dual_lstm_model
from src.models.single_lstm_model import SingleLSTM, train_single_lstm_model

# 为了向后兼容，保留原来的LSTM类名
LSTM = DualLSTM
train_lstm_model = train_dual_lstm_model
