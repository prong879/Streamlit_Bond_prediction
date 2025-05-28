# 基于Streamlit的LSTM+ARIMA多模型时序预测系统

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/prong879/Streamlit_Bond_prediction)



## 项目概述
本项目是一个基于Streamlit框架开发的时序预测Web应用，提供了直观的图形界面来执行时序预测的完整流程，包括数据加载、特征工程、模型训练和结果分析等。本系统采取模块化设计，可后期引入更多模型，实现多模型预测对比。可用于股票预测等典型场景。

## 项目结构
```
.
├── Home.py                     # 主页面
├── pages/                      # 多页面组件
│   ├── 1_DataView.py           # 数据查看与分析页面
│   ├── 2_ModelTraining.py      # 模型训练页面
│   └── 3_ModelEvaluation.py    # 模型评估页面
├── src/                        # 源代码目录
│   ├── models/                 # 模型实现
│   │   ├── lstm_model.py       # LSTM模型实现（支持GPU加速）
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
├── gpu_test.py                 # GPU加速测试脚本
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

### 4. 模型评估页面 (ModelEvaluation)
模型评估页面用于对训练完成的模型进行详细性能分析、对比和报告生成。

#### 主要功能：
- **模型状态概览**：快速查看已训练模型及其基本性能指标。
- **模型性能对比**：
    - 多模型性能指标表格对比。
    - 基于ECharts的雷达图和柱状图进行多维度可视化对比。
    - 自动推荐最佳模型。
- **预测结果分析**：
    - 实际值与多个模型预测值的对比折线图。
    - 预测准确性散点图 (预测 vs 实际)，包含R²值。
    - 支持数据长度不一致时的对齐策略选择。
- **误差分析** (暂时隐藏，后续版本开放):
    - 残差时间序列图。
    - 误差分布直方图。
    - 详细的残差统计信息。
- **模型诊断** (暂时隐藏，后续版本开放):
    - 模型稳定性测试。
    - 参数敏感性分析。
- **详细评估报告**：
    - **报告配置**：用户可选择报告包含的章节 (执行摘要、性能指标、预测分析、建议结论等)。
    - **多格式导出**：支持HTML、Markdown和JSON三种格式的报告导出。
    - **报告预览**：在页面直接预览生成的报告内容。
    - **数据导出**：一键导出评估过程中的关键数据为JSON文件。

## 技术实现细节

### 关键技术点
1. **会话状态管理**：
   - 使用Streamlit的session_state机制管理各页面间的数据共享
   - 通过自定义的`session.py`模块实现状态的统一管理
   - 所有页面共享会话状态，确保数据和模型在不同页面间的一致性

2. **模型实现**：
   - LSTM模型：基于PyTorch实现，支持自定义层数、隐藏层大小等参数
     - 支持GPU加速，自动检测并使用可用的CUDA设备
     - 训练过程中自动将模型和数据迁移到GPU上
     - 保存与加载模型时自动处理设备兼容性
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

5. **GPU加速**：
   - 自动检测CUDA是否可用，并选择适当的设备（GPU/CPU）
   - 使用PyTorch的CUDA支持加速深度学习模型
   - 模型的训练、验证和预测过程均可获得GPU加速
   - 测试结果显示在RTX 3050上可获得约1.1x-10x的加速比（视数据集大小和模型复杂度而定）

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

### LSTM模型GPU加速

LSTM模型已经优化为自动利用GPU加速训练和预测。以下是关键实现部分：

```python
# 检测可用设备
def get_device():
    """
    检测并返回可用的设备（GPU/CPU）
    
    返回:
        device: PyTorch设备对象
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# 训练过程中的设备使用
def train_lstm_model(...):
    # 获取设备
    device = get_device()
    
    # 将数据移至设备
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    
    # 将模型移至设备
    model = LSTMModel(...).to(device)
    
    # 训练和评估
    ...
```

## 使用说明

### 安装与启动
```bash
# 克隆仓库
git clone https://github.com/prong879/Streamlit_Bond_prediction.git

# 安装依赖，包含CUDA支持的PyTorch版本
pip install -r requirements.txt

# 测试GPU支持
python gpu_test.py

# 启动应用
streamlit run Home.py
```

### GPU加速使用说明
本项目支持GPU加速，可以大幅提升LSTM模型的训练速度。如果您的系统配置了兼容的NVIDIA GPU和CUDA环境，系统将自动检测并使用它：

1. 系统会在训练页面显示CUDA和GPU信息
2. LSTM模型训练和预测过程会自动利用GPU加速
3. 如果没有检测到GPU，系统将自动切换到CPU模式

要验证GPU加速是否正常工作，您可以运行以下命令：
```bash
python gpu_test.py
```

如果需要安装支持CUDA的PyTorch，可以参考[PyTorch官方安装指南](https://pytorch.org/get-started/locally/)选择适合您系统的安装命令。

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
  - 增加了CUDA加速支持
  - 正在修复lstm模型中特征筛选展开框显示热力图、条形统计图等代码
  - 热力图正常，统计显著性条形图正常，之前有概率不出现可能是因为第一次打开页面，未进行筛选，所有没有数据传入其中。增加了下相关图的显示按钮，保证图成功绘制
- ARIMA界面
  - arima部分左侧设置了各信息检测结果框，右侧是相应的各种可视化图。
  - 设置了自动检测最优参数模块，下方用户也可自定义需要的模型参数
- 侧边栏
  - 能根据ARIMA or LSTM模型训练完成情况，显示"LSTM和ARIMA模型均已训练完成" or "LSTM模型已训练完成" or "ARIMA模型已训练完成"
  - 迁移了命令行版本的动态预测方法
  - 拓展训练模块，实现多次训练（由于arima预测添加了随机种子）
  - 把因差分导致的前几个值缺失以致拟合值为0的情况解决
  - 需要增加一张还原为原序列后的预测与真实值折线图


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