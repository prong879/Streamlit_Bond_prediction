# LSTM-Pytorch 时间序列分析与预测系统

这是一个基于Streamlit和PyTorch的时间序列分析与预测系统，集成了ARIMA和LSTM两种主流时间序列预测模型。

## 系统特点

- 直观友好的Web界面
- 数据可视化与探索
- ARIMA模型分析与预测
- LSTM神经网络深度学习
- 模型评估与比较

## 技术栈

- **前端框架**: Streamlit
- **数据处理**: Pandas, NumPy
- **可视化**: Streamlit原生图表
- **机器学习**: Scikit-learn, Statsmodels
- **深度学习**: PyTorch
- **时间序列分析**: ARIMA, LSTM

## 主要功能

1. **数据加载与预处理**
   - 支持CSV文件上传
   - 数据清洗与异常值处理
   - 时间序列特征工程

2. **数据可视化**
   - 时间序列趋势图
   - 相关性分析
   - 分布统计

3. **ARIMA模型**
   - 平稳性检验
   - 差分处理
   - 参数优化
   - 模型诊断
   - 预测分析

4. **LSTM模型**
   - 序列数据准备
   - 模型结构配置
   - 训练过程可视化
   - 模型评估
   - 预测分析

5. **模型比较**
   - 预测精度对比
   - 计算性能对比

## 最近更新

- 使用Streamlit原生图表替代matplotlib和plotly图表
- 优化ARIMA模型分析流程
- 改进数据可视化效果

## 安装指南

1. 克隆仓库:
```bash
git clone https://github.com/yourusername/LSTM-Pytorch.git
cd LSTM-Pytorch
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 运行应用:
```bash
streamlit run src/app.py
```

## 使用方法

1. 在首页上传CSV数据文件
2. 在"数据查看"页面检查数据并进行预处理
3. 在"模型训练"页面选择并训练ARIMA或LSTM模型
4. 查看预测结果和模型评估指标

## 项目结构

```
LSTM-Pytorch/
├── data/             # 示例数据文件
├── models/           # 保存的模型
├── src/              # 源代码
│   ├── app.py        # 主应用入口
│   ├── web/          # Web界面
│   │   ├── pages/    # 页面模块
│   │   └── utils/    # 工具函数
├── notebooks/        # Jupyter笔记本
└── requirements.txt  # 依赖包
```
