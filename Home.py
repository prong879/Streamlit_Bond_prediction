"""
主页面
股票价格预测系统的主页面
"""
import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import torch  # 导入PyTorch，用于解决兼容性问题

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

from src.utils.session import init_session_state, get_state, set_state, update_status

# 设置页面标题和图标
st.set_page_config(
    page_title="股票价格预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
init_session_state()

# 页面标题
st.title("📈 股票价格预测系统")

# 欢迎信息
st.markdown("""
## 欢迎使用股票价格预测系统
本系统使用机器学习和深度学习方法，帮助您分析和预测股票价格趋势。通过简单的交互界面，您可以轻松上传数据、训练模型并可视化预测结果。
""")

# 显示系统功能说明
st.header("系统功能")

col1, col2 = st.columns(2)

with col1:
    st.subheader("数据查看 📊")
    st.markdown("""
    - 加载和预览股票数据
    - 计算超过20种技术指标
    - 分析特征相关性
    - 可视化数据和技术指标
    - 数据导出功能
    """)
    
    st.subheader("模型训练 🔬")
    st.markdown("""
    - 特征选择和工程
    - 数据预处理和归一化
    - LSTM模型训练
    - 实时训练进度和损失可视化
    - 模型参数保存
    """)

with col2:
    st.subheader("模型评估 📈")
    st.markdown("""
    - 多指标模型性能评估
    - 预测结果可视化
    - 误差分析
    - 残差分析
    - 未来价格预测
    """)
    
    st.subheader("系统特点 🌟")
    st.markdown("""
    - 直观易用的界面
    - 可视化分析工具
    - 灵活的参数配置
    - 模型训练过程监控
    - 完整的评估报告
    """)

# 使用说明
st.header("使用说明")
st.markdown("""
1. **数据加载**: 在"数据查看"页面上传CSV格式的股票数据，或使用内置的示例数据
2. **特征工程**: 系统自动计算技术指标，您可以在"数据查看"页面分析特征相关性
3. **模型训练**: 在"模型训练"页面选择输入特征和目标变量，配置模型参数并训练模型
4. **模型评估**: 在"模型评估"页面查看模型性能指标和预测结果，分析预测误差
5. **未来预测**: 基于训练好的模型预测未来价格走势
""")

# 项目信息
st.header("项目信息")
st.markdown("""
### 技术栈
- **Python 3.8+**: 编程语言
- **PyTorch**: 深度学习框架
- **Streamlit**: Web应用框架
- **Plotly**: 交互式数据可视化
- **Pandas & NumPy**: 数据处理和分析
- **scikit-learn**: 机器学习工具

### 主要功能
- 数据加载与预处理
- 技术指标计算
- 特征相关性分析
- LSTM深度学习模型训练
- 模型性能评估
- 股价趋势预测
""")

# 显示系统状态
st.header("系统状态")

# 获取当前系统状态
status = update_status()

# 创建状态指标
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("数据状态", status["数据状态"])
    
with col2:
    st.metric("已选特征", status["已选特征"])
    
with col3:
    st.metric("模型状态", status["模型状态"])

# 底部信息
st.markdown("""
---
### 开始使用
请使用上方导航菜单选择功能模块，开始您的股票预测分析之旅！
""")

# 版权信息
st.markdown("""
<div style="text-align: center; margin-top: 30px; color: gray; font-size: 0.8em;">
    © 2025 股票价格预测系统 | 基于LSTM深度学习
</div>
""", unsafe_allow_html=True) 