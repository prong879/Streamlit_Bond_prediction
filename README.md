# 股票价格预测系统 - Web应用

## 项目概述
本项目是一个基于Streamlit框架开发的股票价格预测Web应用，提供了直观的图形界面来执行股票预测的完整流程，包括数据加载、特征工程、模型训练和结果分析等。本系统通过LSTM和ARIMA等时间序列分析模型，为投资者提供专业的股票趋势预测工具。

## 项目结构
```
.
├── Home.py                     # 主页面
├── pages/                      # 多页面组件
│   ├── 1_DataView.py           # 数据查看与分析页面
│   └── 2_ModelTraining.py      # 模型训练页面
├── src/                        # 源代码目录
│   ├── models/                 # 模型实现
│   │   ├── lstm_model.py       # LSTM模型实现
│   │   └── arima_model.py      # ARIMA模型实现
│   └── utils/                  # 工具函数
│       ├── session.py          # 会话状态管理
│       └── visualization.py    # 可视化工具
├── data/                       # 数据目录
│   ├── rlData.csv              # 实际分析数据
│   └── example_stock_data.csv  # 示例数据
├── requirements.txt            # 项目依赖
├── cmd_main_old.py             # 命令行版本（功能完善，作为对照）
├── cmd_README.md               # 命令行版本文档
├──
```

## 功能模块

### 1. 主页面 (Home)
主页面提供系统概述和导航入口，包括:
- 系统功能介绍
- 使用流程指南
- 技术栈说明
- 实时系统状态监控（数据状态、特征选择、模型状态）

### 2. 数据查看页面 (DataView)
数据查看页面用于数据加载、预处理和特征分析。

#### 主要功能：
- **数据加载**：支持上传CSV文件或使用内置示例数据
- **数据预览**：显示数据前几行和基本统计信息
- **技术指标计算**：自动计算22种常用技术指标：
  - 均线指标：MA5、MA10、MA20等
  - 移动平均线差异：MA5_MA10_Diff、MA10_MA20_Diff
  - 相对强弱指标：RSI
  - 布林带指标：Upper_Band、Lower_Band、BB_Width、BB_Position
  - MACD指标：MACD、MACD_Signal、MACD_Hist
  - 随机振荡器：Stoch_K、Stoch_D
  - 平均方向指数：ADX
  - 威廉指标：Williams_R
  - 顺势指标：CCI
  - 成交量相关指标：Volume、Volume_Change
  - 价格变化指标：Price_Change
  - 平均真实波幅：ATR
  - 能量潮指标：OBV
  - 资金流量指数：MFI
  - 变动率指标：ROC
- **特征相关性矩阵**：使用ECharts交互式热力图显示特征之间的相关性
- **股票K线图**：显示K线图、收盘价和移动平均线的交互式图表
- **成交量分析**：显示股票成交量走势图
- **数据导出**：支持导出原始数据和计算的技术指标数据

### 3. 模型训练页面 (ModelTraining)
模型训练页面用于特征选择、模型参数配置和训练执行。

#### 主要功能：
- **特征选择**：
  - 相关性筛选
  - VIF(方差膨胀因子)筛选
  - 统计显著性筛选
  - 特征筛选结果可视化
- **模型参数配置**：
  - LSTM模型超参数设置（隐藏层大小、层数、Dropout率等）
  - ARIMA模型参数设置（p, d, q参数）
  - Prophet模型参数设置（季节性配置等）
- **训练控制**：
  - 实时训练进度显示
  - 训练/验证损失曲线可视化
  - 早停控制
- **模型评估**：
  - MSE、RMSE、MAE等指标计算
  - 训练结果可视化
  - 模型保存功能

## 技术实现细节

### 关键技术点
1. **会话状态管理**：
   - 使用Streamlit的session_state机制管理各页面间的数据共享
   - 通过自定义的`session.py`模块实现状态的统一管理
   - 所有页面共享会话状态，确保数据和模型在不同页面间的一致性

2. **模型实现**：
   - LSTM模型：基于PyTorch实现，支持自定义层数、隐藏层大小等参数
   - ARIMA模型：基于statsmodels实现，包括模型参数优化、残差分析等功能

3. **数据可视化**：
   - 使用streamlit-echarts实现交互式图表
   - K线图与成交量联动
   - 相关性热力图
   - 训练进度和损失可视化

4. **特征工程**：
   - 技术指标自动计算
   - 多重筛选标准（相关性、VIF、统计显著性）
   - 特征重要性可视化

### ECharts热力图配置
热力图是基于ECharts实现的交互式数据可视化组件，具体实现如下：
```py
# 调用热力图函数
correlation_heatmap_option = create_correlation_heatmap(corr_matrix, high_correlation_features)
```

```py

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
            # 这里不需要加格式控制语句，默认显示的就是对应值，写了反而容易出错
            'position': 'top', 
        },
        # 绘图区域
        'grid': {
            'top': '0', # 上面接顶
            'bottom': '10%', # 下面给变量名留空
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
```



## 使用说明

### 安装与启动
```bash
# 克隆仓库
git clone <repository-url>

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run Home.py
```

### 基本使用流程
1. **数据加载**：在"数据查看"页面上传CSV格式的股票数据，或使用内置的示例数据
2. **特征工程**：系统自动计算技术指标，分析特征相关性
3. **模型训练**：在"模型训练"页面选择输入特征，配置模型参数并训练模型
4. **模型评估**：查看模型性能指标和预测结果
5. **保存模型**：将训练好的模型保存供后续使用

## 更新日志

### v2.0.1
- 基于Streamlit进行Web版本的重构
- 调整模型训练页面布局为2栏1:2的比例
- 完成了模型训练页面的基本界面设计
- 应该是实现了lstm的训练功能，用了模块化的模型引入
- 可自己调整设定lstm模型的参数

### v2.0.2
- 整理了项目结构，将Web版本独立出来

### v2.0.3
- 继续完善arima的功能
- 数据预处理完成
- arima中平稳性检验、正态检验、描述性统计表、时序图、分布直方图（带有正态拟合曲线）、qq图已经能正常显示

### v2.0.4
- 修改了特征筛选界面的布局逻辑（考虑到Streamlit流式运行的原理）
- 数据预览界面
  - 增加了导入数据的变量名识别、升序设置
  - 增加了对K线图数据的检查，在数据不完整时显示警告信息
  - 当OHLC数据不完整但至少有收盘价和日期数据时，会显示一个简单的收盘价折线图
  - 在没有任何价格数据时提供明确的错误信息
- LSTM界面
  - 正在修复lstm模型中特征筛选展开框显示热力图、条形统计图等代码
  - 热力图正常，统计显著性条形图正常，之前有概率不出现可能是因为第一次打开页面，未进行筛选，所有没有数据传入其中。增加了下相关图的显示按钮，保证图成功绘制
- ARIMA界面
  - arima部分左侧设置了各信息检测结果框，右侧是相应的各种可视化图。


## 计划功能
- 添加模型评估专用页面，提供更详细的性能分析
- 实现模型对比功能，支持不同参数和模型类型的性能比较
- 添加数据预处理工具，如异常值处理、缺失值填充等
- 增加用户认证和数据保存功能
- 实现模型预测结果的导出和报告生成功能
- 添加更多模型选项（Prophet、Transformer等）

## 技术栈
- **前端框架**：Streamlit
- **数据处理**：Pandas, NumPy
- **数据可视化**：ECharts (通过streamlit-echarts)
- **技术指标计算**：ta库
- **模型框架**：PyTorch, statsmodels
- **统计分析**：statsmodels
- **机器学习工具**：scikit-learn

## 贡献指南
欢迎为本项目贡献代码或提出建议。如要贡献代码，请：
1. Fork项目仓库
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证
[项目许可证信息] 