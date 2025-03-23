# -*- coding: utf-8 -*-
"""
数据查看页面 (DataView Page)
此模块提供股票数据的可视化和分析功能

主要功能:
1. 数据加载与预览
2. 技术指标计算
3. 相关性分析
4. K线图展示
5. 数据导出

技术栈:
- streamlit: Web应用框架
- pandas: 数据处理和分析
- ta: 技术指标计算
- streamlit_echarts: 图表可视化
"""
import streamlit as st  # Streamlit用于构建Web应用界面
import pandas as pd    # Pandas用于数据处理和分析
import numpy as np     # Numpy用于数学计算
from datetime import datetime  # 导入datetime用于记录时间戳
import os  # 操作系统接口
import ta  # 技术分析库，用于计算技术指标
from streamlit_echarts import st_echarts  # ECharts图表组件
from src.utils.session import get_state, set_state  # 状态管理工具
import torch  # 导入PyTorch，用于解决兼容性问题
import re  # 导入正则表达式模块，用于列名匹配

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 设置页面标题
st.title("📊 数据查看")

def normalize_column_names(df):
    """
    标准化股票数据的列名，处理常见的变体形式
    
    参数:
        df (DataFrame): 原始数据框
        
    返回:
        DataFrame: 列名标准化后的数据框
        dict: 列名的映射关系
    """
    # 标准列名
    standard_columns = {
        'Date': ['date', 'time', 'datetime', 'timestamp', 'trade_date', 'trading_date'],
        'Open': ['open', 'open_price', 'opening', 'first', 'first_price'],
        'High': ['high', 'high_price', 'highest', 'max', 'maximum', 'highest_price'],
        'Low': ['low', 'low_price', 'lowest', 'min', 'minimum', 'lowest_price'],
        'Close': ['close', 'close_price', 'closing', 'last', 'last_price', 'close/last', 'adj_close', 'adjusted_close'],
        'Volume': ['volume', 'vol', 'quantity', 'turnover', 'trade_volume', 'trading_volume']
    }
    
    # 创建映射字典
    column_mapping = {}
    rename_info = []
    
    # 获取当前列名的小写形式
    lowercase_columns = {col.lower(): col for col in df.columns}
    processed_columns = set()  # 添加一个集合记录已处理的列
    
    # 遍历标准列名及其变体
    for standard, variants in standard_columns.items():
        # 如果标准列名已存在，跳过
        if standard in df.columns:
            continue
        
        # 检查变体是否存在
        for variant in variants:
            # 精确匹配
            if variant in lowercase_columns:
                original_name = lowercase_columns[variant]
                if original_name not in processed_columns:  # 检查是否已处理
                    column_mapping[original_name] = standard
                    rename_info.append(f"'{original_name}' → '{standard}'")
                    processed_columns.add(original_name)  # 标记为已处理
                break
                
            # 部分匹配（比如包含特殊字符或空格的情况）
            for col in lowercase_columns.values():
                if col in processed_columns:  # 跳过已处理的列
                    continue
                # 使用正则表达式处理特殊情况，如 "close/last", "adj. close" 等
                pattern = r'\b' + re.escape(variant) + r'\b'
                if re.search(pattern, col.lower()) or variant in col.lower().replace(" ", "").replace("_", "").replace("-", ""):
                    column_mapping[col] = standard
                    rename_info.append(f"'{col}' → '{standard}'")
                    processed_columns.add(col)  # 标记为已处理
                    break
    
    # 重命名列
    if column_mapping:
        df = df.rename(columns=column_mapping)
        
    return df, rename_info

def calculate_ma(df, periods=[5, 10, 20, 30]):
    """
    计算移动平均线
    
    参数:
        df (DataFrame): 包含股票价格数据的DataFrame
        periods (list): 移动平均线的周期列表，默认为[5, 10, 20, 30]
        
    返回:
        DataFrame: 包含移动平均线的DataFrame，结果保留两位小数
    """
    df_ma = df.copy()
    for period in periods:
        # 计算移动平均线并保留两位小数
        df_ma[f'MA{period}'] = df['Close'].rolling(window=period).mean().round(3)
    return df_ma

def calculate_technical_indicators(df):
    """
    计算技术指标
    
    功能:
    1. 基础价格指标: 移动平均线、价格变化率
    2. 趋势指标: MACD、ADX
    3. 动量指标: RSI、随机指标、威廉指标
    4. 波动性指标: 布林带、ATR
    5. 成交量指标: OBV、MFI
    
    参数:
        df (DataFrame): 原始股票数据，需包含OHLCV数据
        
    返回:
        DataFrame: 包含所有技术指标的DataFrame
    """
    # 确保数据中有必要的列
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.warning(f"数据中缺少以下列: {missing_cols}，将仅使用可用的列计算技术指标")
    
    # 创建特征DataFrame
    features = pd.DataFrame(index=df.index)
    
    # 添加原始价格数据
    features['Close'] = df['Close']
    
    # 如果有成交量数据，添加成交量
    if 'Volume' in df.columns:
        features['Volume'] = df['Volume']
        # 成交量变化率
        features['Volume_Change'] = df['Volume'].pct_change()
    
    # 价格变化
    features['Price_Change'] = df['Close'].pct_change()
    
    # 移动平均线
    features['MA5'] = df['Close'].rolling(window=5).mean()
    features['MA10'] = df['Close'].rolling(window=10).mean()
    features['MA20'] = df['Close'].rolling(window=20).mean()
    
    # 移动平均线差异
    features['MA5_MA10_Diff'] = features['MA5'] - features['MA10']
    features['MA10_MA20_Diff'] = features['MA10'] - features['MA20']
    
    # 如果有高低价数据，计算更多指标
    if 'High' in df.columns and 'Low' in df.columns:
        # 使用 ta 库计算 RSI
        try:
            features['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        except Exception as e:
            st.error(f"RSI 计算失败: {e}")
            # 如果 ta 库计算失败，使用自定义函数计算 RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            features['RSI'] = 100 - (100 / (1 + rs))
        
        # 布林带
        try:
            bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
            features['Upper_Band'] = bollinger.bollinger_hband()
            features['Lower_Band'] = bollinger.bollinger_lband()
            features['BB_Width'] = bollinger.bollinger_wband()
            features['BB_Position'] = (df['Close'] - features['Lower_Band']) / (features['Upper_Band'] - features['Lower_Band'])
        except Exception as e:
            st.error(f"布林带计算失败: {e}")
            # 如果 ta 库计算失败，使用自定义函数计算布林带
            features['MA20_std'] = df['Close'].rolling(window=20).std()
            features['Upper_Band'] = features['MA20'] + (features['MA20_std'] * 2)
            features['Lower_Band'] = features['MA20'] - (features['MA20_std'] * 2)
            features['BB_Width'] = (features['Upper_Band'] - features['Lower_Band']) / features['MA20']
            features['BB_Position'] = (df['Close'] - features['Lower_Band']) / (features['Upper_Band'] - features['Lower_Band'])
        
        # 计算MACD
        try:
            macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
            features['MACD'] = macd.macd()
            features['MACD_Signal'] = macd.macd_signal()
            features['MACD_Hist'] = macd.macd_diff()
        except Exception as e:
            st.error(f"MACD 计算失败: {e}")
            # 如果 ta 库计算失败，使用自定义函数计算 MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            features['MACD'] = exp1 - exp2
            features['MACD_Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()
            features['MACD_Hist'] = features['MACD'] - features['MACD_Signal']
        
        # 添加更多技术指标
        try:
            # 随机振荡器
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
            features['Stoch_K'] = stoch.stoch()
            features['Stoch_D'] = stoch.stoch_signal()
            
            # 平均方向指数
            adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
            features['ADX'] = adx.adx()
            
            # 威廉指标
            features['Williams_R'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close'], lbp=14).williams_r()
            
            # 顺势指标
            features['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
            
            # ATR - 平均真实波幅
            features['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
            
            # OBV - 能量潮指标
            features['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # 资金流量指数
            features['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
            
            # ROC - 变动率指标
            features['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()
        except Exception as e:
            st.error(f"额外技术指标计算失败: {e}")
    
    # 删除NaN值
    features = features.dropna()
    
    return features

def load_example_data():
    """
    加载示例数据
    
    功能:
    - 从data/example目录加载示例股票数据
    - 转换日期格式
    
    返回:
        DataFrame or None: 成功返回DataFrame，失败返回None
    """
    try:
        # 从示例数据文件夹加载数据
        data_path = os.path.join("data", "example", "stock_data.csv")
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"加载示例数据失败：{str(e)}")
        return None

def create_correlation_heatmap(df):
    """
    创建相关性热力图配置
    
    功能:
    - 计算特征间的相关性矩阵
    - 根据与收盘价的相关性对特征进行排序
    - 配置ECharts热力图样式
    - 支持交互式数据展示
    
    参数:
        df (DataFrame): 包含技术指标的DataFrame
        
    返回:
        dict: ECharts配置项字典
    """
    # 计算相关性矩阵
    corr_matrix = df.corr().round(2)
    
    # 计算相关性矩阵的最小值和最大值
    corr_min = corr_matrix.min().min()
    corr_max = corr_matrix.max().max()
    
    # 根据与Close（收盘价）的相关性排序特征
    if 'Close' in corr_matrix.columns:
        close_correlations = corr_matrix['Close']  # 不使用abs()，保留正负号
        sorted_features = close_correlations.sort_values(ascending=False).index  # 从高到低排序
        corr_matrix = corr_matrix.loc[sorted_features, sorted_features]
    
    # 获取排序后的特征名称和相关性数据
    features = list(corr_matrix.columns)
    corr_data = []
    
    # 构建热力图数据
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            value = corr_matrix.iloc[i, j]
            corr_data.append([j, i, float(value)])
    
    # 计算特征相关性
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    
    # 保存特征列表到session state
    st.session_state['feature_list'] = numeric_cols.tolist()
    
    # 创建相关性热力图ECharts配置
    option = {
        "tooltip": {
            "position": "top",  # 提示框位置：顶部
        },
        "grid": {
            "top": "0%",       # 上边距
            "left": "150px",      # 左边距，用于放置较长的特征名称
            "bottom": "100px"     
        },
        "dataset": {
            "source": corr_data  # 数据源：相关性矩阵数据
        },
        "xAxis": {
            "type": "category",     # 坐标轴类型：类别型
            "data": features,       # 坐标轴数据：特征名称
            "splitArea": {
                "show": True       # 显示分隔区域
            },
            "axisLabel": {
                "rotate": 90       # 标签旋转90度，避免重叠
            }
        },
        "yAxis": {
            "type": "category",     # 坐标轴类型：类别型
            "data": features,       # 坐标轴数据：特征名称
            "splitArea": {
                "show": True       # 显示分隔区域
            }
        },
        "visualMap": {
            "min": float(corr_min),  # 使用实际最小值
            "max": float(corr_max),  # 使用实际最大值
            "calculable": True,    # 是否显示拖拽手柄
            "orient": "vertical",  # 垂直布局
            "left": "0",          # 位置：左侧
            "bottom": "center",    # 位置：垂直居中
            "inRange": {
                "color": ["#195696", "#ffffff", "#ae172a"]  # 颜色范围：蓝-白-红
            }
        },
        "series": [{
            "name": "相关系数",
            "type": "heatmap",    # 图表类型：热力图
            "data": corr_data,    # 数据
            "label": {
                "show": True,     # 显示数值标签
                "formatter": {    # 格式化标签，保留两位小数
                    "type": "function",
                    "function": "function(params) { return params.data[2].toFixed(2); }"
                }
            },
            "emphasis": {
                "itemStyle": {    # 鼠标悬停效果
                    "shadowBlur": 10,
                    "shadowColor": "rgba(0, 0, 0, 0.5)"
                }
            }
        }]
    }
    
    return option

# 创建K线图和成交量联动图表
def create_echarts_kline_volume(df, selected_mas=[]):
    """
    创建K线图和成交量联动图表
    
    功能:
    - K线图展示股票价格走势
    - 成交量柱状图
    - 支持多个移动平均线叠加显示
    - 支持图表联动和缩放
    
    参数:
        df (DataFrame): 股票数据
        selected_mas (list): 选中的移动平均线列表
        
    返回:
        dict: ECharts配置项字典
    """
    # 确保数据按日期排序（从旧到新）
    if 'Date' in df.columns:
        df = df.sort_values(by='Date')
    
    # 转换日期格式
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # 检查是否具备完整的OHLC数据
    has_ohlc = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close'])
    has_volume = 'Volume' in df.columns
    
    # 准备K线数据 [开盘价, 收盘价, 最低价, 最高价]
    if has_ohlc:
        kline_data = [[round(float(o), 3), round(float(c), 3), round(float(l), 3), round(float(h), 3)] for o, c, l, h in 
                    zip(df['Open'], df['Close'], df['Low'], df['High'])]
    else:
        # 如果缺少OHLC数据，则只显示收盘价
        close_values = df['Close'].values
        kline_data = [[float(c), float(c), float(c), float(c)] for c in close_values]
    
    # 准备移动平均线数据
    ma_series = []
    ma_colors = {
        'MA5': '#FF4B4B',   # 5日均线：红色
        'MA10': '#00B4D8',  # 10日均线：蓝色
        'MA20': '#2ECC71',  # 20日均线：绿色
        'MA30': '#9B59B6'   # 30日均线：紫色
    }
    
    # 仅当有完整OHLC数据时才计算移动平均线
    if has_ohlc:
        df_ma = calculate_ma(df)
        
        # 配置每条均线的样式
        for ma_name in selected_mas:
            ma_data = df_ma[ma_name].fillna('').tolist()
            ma_series.append({
                "name": ma_name,
                "type": "line",         # 图表类型：线图
                "xAxisIndex": 0,        # 使用第一个X轴
                "yAxisIndex": 0,        # 使用第一个Y轴
                "data": ma_data,
                "smooth": True,         # 平滑曲线
                "symbolSize": 3,        # 数据点大小
                "symbol": "circle",     # 数据点形状
                "showSymbol": False,    # 默认不显示数据点
                "lineStyle": {
                    "width": 2,
                    "color": ma_colors[ma_name]
                },
                "itemStyle": {
                    "color": ma_colors[ma_name]
                }
            })
    
    # 计算每日涨跌情况（如果可能）
    if has_ohlc:
        df['price_change'] = df['Close'] - df['Open']
    else:
        df['price_change'] = df['Close'].diff()
    
    # 准备成交量数据，根据涨跌设置颜色
    volume_data = []
    if has_volume:
        for i in range(len(df)):
            color = "#FF4B4B" if df['price_change'].iloc[i] >= 0 else "#2ECC71"  # 涨：红色，跌：绿色
            volume_data.append({
                "value": float(df['Volume'].iloc[i]),
                "itemStyle": {
                    "color": color
                }
            })
    
    # 创建ECharts配置：股票日K线及均线图、成交量图
    option = {
        "title": [{
            "text": "股票日K线及均线图",  # 主标题
            "left": "center",        # 水平居中
            "top": "0%"             # 距顶部距离
        }],
        "tooltip": {
            "trigger": "axis",       # 触发类型：坐标轴触发
            "axisPointer": {
                "type": "cross"      # 指示器类型：十字准星
            },
            "backgroundColor": "rgba(245, 245, 245, 0.8)",
            "borderWidth": 1,
            "borderColor": "#ccc",
            "padding": 10,
            "textStyle": {
                "color": "#000",
            }
        },
        "legend": {
            "data": ["K线", "收盘价"] + selected_mas,  # 图例项
            "top": "30px"                             # 图例位置
        },
        "axisPointer": {
            "link": {
                "xAxisIndex": "all"   # 联动所有x轴
            }
        },
        "grid": [{
            "left": "7%",           # 主图网格
            "right": "0%",
            "top": "15%",
            "height": "50%"          # 主图高度占比
        }],
        "xAxis": [{
            "type": "category",      # 主图X轴
            "data": dates,
            "scale": True,
            "boundaryGap": True,    # 修改为True，允许坐标轴两边留白
            "axisLine": {"onZero": False},
            "splitLine": {"show": False},
            "splitNumber": 20,
            "min": "dataMin",
            "max": "dataMax",
            "axisPointer": {
                "z": 100
            }
        }],
        "yAxis": [{
            "scale": True,           # 主图Y轴
            "splitArea": {
                "show": True         # 显示分隔区域
            }
        }],
        "dataZoom": [
            {
                "type": "inside",    # 内置型数据区域缩放组件
                "xAxisIndex": [0],    # 控制x轴
                "start": 10,          # 数据窗口范围的起始百分比
                "end": 100           # 数据窗口范围的结束百分比
            },
            {
                "show": True,        # 滑动条型数据区域缩放组件
                "xAxisIndex": [0],
                "type": "slider",
                "bottom": "5%",
                "start": 10,
                "end": 100
            }
        ],
        "series": [
            {
                "name": "K线",
                "type": "candlestick",  # 图表类型：K线图
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": kline_data,
                "itemStyle": {
                    "color": "#FF4B4B",  # 上涨颜色
                    "color0": "#2ECC71"  # 下跌颜色
                }
            },
            {
                "name": "收盘价",
                "type": "line",         # 图表类型：线图
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": df['Close'].round(3).tolist(),  # 先round后tolist
                "smooth": True,         # 平滑曲线
                "symbolSize": 3,
                "symbol": "circle",
                "showSymbol": False,
                "lineStyle": {
                    "width": 1,
                    "color": "#ff9900"  # 收盘价线颜色
                },
                "itemStyle": {
                    "color": "#ff9900"
                },
                "opacity": 0.7
            }
        ]
    }
    
    # 添加移动平均线系列
    option["series"].extend(ma_series)
    
    # 只有当存在成交量数据时，才添加成交量图
    if has_volume:
        # 添加成交量的标题
        option["title"].append({
            "text": "成交量",         # 副标题（成交量）
            "left": "center",
            "top": "70%"            # 位于主图下方
        })
        
        # 添加成交量的网格
        option["grid"].append({
            "left": "7%",           # 成交量网格
            "right": "0%",
            "top": "75%",
            "height": "15%"          # 成交量图高度占比
        })
        
        # 添加成交量的X轴
        option["xAxis"].append({
            "type": "category",      # 成交量X轴
            "gridIndex": 1,
            "data": dates,
            "scale": True,
            "boundaryGap": True,    # 修改为True，允许坐标轴两边留白
            "axisLine": {"onZero": False},
            "splitLine": {"show": False},
            "axisLabel": {"show": False},
            "axisTick": {"show": False},
            "axisPointer": {
                "label": {"show": False}
            }
        })
        
        # 添加成交量的Y轴
        option["yAxis"].append({
            "scale": True,           # 成交量Y轴
            "gridIndex": 1,
            "splitNumber": 2,
            "axisLabel": {"show": True},
            "axisLine": {"show": True},
            "axisTick": {"show": True},
            "splitLine": {"show": True}
        })
        
        # 更新数据缩放组件以包含成交量
        for dz in option["dataZoom"]:
            dz["xAxisIndex"] = [0, 1]
        
        # 添加成交量系列
        option["series"].append({
            "name": "成交量",
            "type": "bar",          # 图表类型：柱状图
            "xAxisIndex": 1,
            "yAxisIndex": 1,
            "data": volume_data
        })
    
    return option

# 获取侧边栏配置
data_source = get_state("data_source", "上传数据")
chart_theme = get_state("chart_theme", "plotly_white")

# 数据加载部分
st.header("数据加载")

if data_source == "上传数据":
    uploaded_file = st.file_uploader(
        "上传CSV文件",
        type=['csv'],
        help="请上传包含日期、开盘价、最高价、最低价、收盘价、成交量的CSV文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取CSV文件
            df = pd.read_csv(uploaded_file)
            
            # 标准化列名
            df, rename_info = normalize_column_names(df)
            
            # 如果有列名被重命名，显示信息
            if rename_info:
                st.info(f"已自动标准化以下列名: {', '.join(rename_info)}")
            
            # 处理日期列
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # 存储到session state
            set_state('raw_data', df)
            # 记录数据加载时间戳
            st.session_state['data_load_timestamp'] = datetime.now()
        except Exception as e:
            st.error(f"数据加载失败：{str(e)}")
else:
    df = load_example_data()
    if df is not None:
        # 标准化样例数据的列名
        df, rename_info = normalize_column_names(df)
        if rename_info:
            st.info(f"已自动标准化以下列名: {', '.join(rename_info)}")
        
        set_state('raw_data', df)
        # 记录数据加载时间戳

# 获取当前数据
df = get_state('raw_data')
if df is None:
    st.warning("请先加载数据")
    st.stop()

# 显示数据预览
st.header("数据预览")
st.dataframe(df.head(), hide_index=True)

# 显示基本统计信息
st.header("基本统计信息")
st.dataframe(df.describe(), hide_index=True)

# 计算技术指标
with st.spinner("正在计算技术指标..."):
    tech_indicators = calculate_technical_indicators(df)
    # 保存技术指标数据到session state
    st.session_state['tech_indicators'] = tech_indicators

# 显示计算出的技术指标

st.header("技术指标数据预览")
# 添加一个多选框，让用户选择要查看的指标
all_indicators = tech_indicators.columns.tolist()
default_indicators = all_indicators[:5]  # 默认显示前5个指标
selected_indicators = st.multiselect(
    "选择要查看的指标",
    options=all_indicators,
    default=default_indicators
)

if selected_indicators:
    st.dataframe(tech_indicators[selected_indicators].head(10))
else:
    st.dataframe(tech_indicators.head(10))

# 显示相关性矩阵
st.header("特征相关性矩阵")

# 创建选择框，让用户选择要包含在相关性矩阵中的指标
all_potential_indicators = [
    'Close', 'MA5', 'MA10', 'MA20', 'Lower_Band', 'Upper_Band',
    'MACD_Signal', 'RSI', 'MACD', 'BB_Position', 'CCI', 'Stoch_D',
    'Williams_R', 'Stoch_K', 'MA10_MA20_Diff', 'MA5_MA10_Diff',
    'Volume', 'ADX', 'Price_Change', 'MACD_Hist', 'Volume_Change',
    'BB_Width'
]

# 过滤掉不存在于tech_indicators中的指标
available_columns = tech_indicators.columns.tolist()
default_indicators = [indicator for indicator in all_potential_indicators if indicator in available_columns]

selected_corr_indicators = st.multiselect(
    "选择要包含在相关性矩阵中的指标",
    options=available_columns,
    default=default_indicators
)

if selected_corr_indicators:
    # 计算并显示相关性矩阵
    corr_df = tech_indicators[selected_corr_indicators]
    
    # 使用streamlit_echarts渲染相关性热力图
    option = create_correlation_heatmap(corr_df)
    
    # 添加导出图表功能
    option.update({
        "toolbox": {
            "show": True,
            "feature": {
                "saveAsImage": {
                    "show": True,
                    "title": "保存为图片",
                    "type": "png",
                    "pixelRatio": 2
                }
            },
            "right": "0%",
            "top": "3%"
        }
    })
    
    st_echarts(options=option, height="600px")

# 使用联动的K线图和成交量图
st.header("股票走势与成交量分析")

# 检查OHLC数据是否完整
required_cols_ohlc = ['Open', 'High', 'Low', 'Close']
missing_cols_ohlc = [col for col in required_cols_ohlc if col not in df.columns]

if missing_cols_ohlc:
    st.warning(f"缺少K线图所需的列: {', '.join(missing_cols_ohlc)}。无法绘制K线图，需要完整的OHLC数据。")
    
    # 如果至少有收盘价数据，可以显示折线图
    if 'Close' in df.columns and 'Date' in df.columns:
        st.info("已检测到收盘价数据，将显示收盘价折线图。")
        
        # 确保数据按日期排序（从旧到新）
        df_sorted = df.sort_values(by='Date')
        
        # 创建简单的收盘价折线图配置
        dates = df_sorted['Date'].dt.strftime('%Y-%m-%d').tolist()
        close_data = df_sorted['Close'].round(3).tolist()
        
        line_option = {
            "title": {
                "text": "收盘价走势图",
                "left": "center"
            },
            "tooltip": {
                "trigger": "axis"
            },
            "toolbox": {
                "show": True,
                "feature": {
                    "saveAsImage": {
                        "show": True,
                        "title": "保存为图片",
                        "type": "png"
                    }
                },
                "right": "0%"
            },
            "xAxis": {
                "type": "category",
                "data": dates,
                "axisLabel": {
                    "rotate": 45
                }
            },
            "yAxis": {
                "type": "value"
            },
            "series": [{
                "name": "收盘价",
                "type": "line",
                "data": close_data,
                "smooth": True,
                "itemStyle": {
                    "color": "#ff9900"
                }
            }],
            "dataZoom": [{
                "type": "inside",
                "start": 0,
                "end": 100
            }, {
                "type": "slider",
                "start": 0,
                "end": 100
            }]
        }
        
        st_echarts(options=line_option, height="400px")
    else:
        st.error("无法显示任何价格图表，请确保数据中至少包含收盘价(Close)和日期(Date)列。")
else:
    # 检查日期列
    if 'Date' not in df.columns:
        st.error("缺少日期列(Date)，无法绘制时间序列图表。")
    else:
        # 添加导出图表选项
        export_option = {
            "toolbox": {
                "show": True,
                "feature": {
                    "saveAsImage": {
                        "show": True,
                        "title": "保存为图片",
                        "type": "png",
                        "pixelRatio": 2
                    }
                },
                "right": "0%",
                "top": "3%"
            }
        }

        # 选择合适的均线（仅显示可用的均线）
        available_mas = []
        for ma in ['MA5', 'MA10', 'MA20', 'MA30']:
            if ma in tech_indicators.columns:
                available_mas.append(ma)

        # 合并导出选项到原有选项
        combined_option = create_echarts_kline_volume(df, available_mas)
        combined_option.update(export_option)

        # 显示图表
        st_echarts(options=combined_option, height="500px")

# 添加数据下载按钮
st.header("数据导出")
col_raw, col_tech = st.columns(2)

with col_raw:
    if st.button("导出原始数据为CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="点击下载原始数据",
            data=csv,
            file_name="stock_data.csv",
            mime="text/csv"
        )

with col_tech:
    if st.button("导出技术指标数据为CSV"):
        tech_csv = tech_indicators.to_csv(index=False)
        st.download_button(
            label="点击下载技术指标数据",
            data=tech_csv,
            file_name="technical_indicators.csv",
            mime="text/csv"
        ) 