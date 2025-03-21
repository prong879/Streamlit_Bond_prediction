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

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 设置页面标题
st.title("📊 数据查看")

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
    # 转换日期格式
    dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    
    # 准备K线数据 [开盘价, 收盘价, 最低价, 最高价]
    kline_data = [[round(float(o), 3), round(float(c), 3), round(float(l), 3), round(float(h), 3)] for o, c, l, h in 
                zip(df['Open'], df['Close'], df['Low'], df['High'])]
    
    # 准备移动平均线数据
    ma_series = []
    ma_colors = {
        'MA5': '#FF4B4B',   # 5日均线：红色
        'MA10': '#00B4D8',  # 10日均线：蓝色
        'MA20': '#2ECC71',  # 20日均线：绿色
        'MA30': '#9B59B6'   # 30日均线：紫色
    }
    
    # 计算移动平均线
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
    
    # 计算每日涨跌情况
    df['price_change'] = df['Close'] - df['Open']
    
    # 准备成交量数据，根据涨跌设置颜色
    volume_data = []
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
        }, {
            "text": "成交量",         # 副标题（成交量）
            "left": "center",
            "top": "70%"            # 位于主图下方
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
        }, {
            "left": "7%",           # 成交量网格
            "right": "0%",
            "top": "75%",
            "height": "15%"          # 成交量图高度占比
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
        }, {
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
        }],
        "yAxis": [{
            "scale": True,           # 主图Y轴
            "splitArea": {
                "show": True         # 显示分隔区域
            }
        }, {
            "scale": True,           # 成交量Y轴
            "gridIndex": 1,
            "splitNumber": 2,
            "axisLabel": {"show": True},
            "axisLine": {"show": True},
            "axisTick": {"show": True},
            "splitLine": {"show": True}
        }],
        "dataZoom": [
            {
                "type": "inside",    # 内置型数据区域缩放组件
                "xAxisIndex": [0, 1], # 控制两个x轴
                "start": 10,          # 数据窗口范围的起始百分比
                "end": 100           # 数据窗口范围的结束百分比
            },
            {
                "show": True,        # 滑动条型数据区域缩放组件
                "xAxisIndex": [0, 1],
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
            },
            {
                "name": "成交量",
                "type": "bar",          # 图表类型：柱状图
                "xAxisIndex": 1,
                "yAxisIndex": 1,
                "data": volume_data
            }
        ]
    }
    
    # 添加移动平均线系列
    option["series"].extend(ma_series)
    
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
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            set_state('raw_data', df)
            # 记录数据加载时间戳
            st.session_state['data_load_timestamp'] = datetime.now()
        except Exception as e:
            st.error(f"数据加载失败：{str(e)}")
else:
    df = load_example_data()
    if df is not None:
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
default_indicators = [
    'Close', 'MA5', 'MA10', 'MA20', 'Lower_Band', 'Upper_Band',
    'MACD_Signal', 'RSI', 'MACD', 'BB_Position', 'CCI', 'Stoch_D',
    'Williams_R', 'Stoch_K', 'MA10_MA20_Diff', 'MA5_MA10_Diff',
    'Volume', 'ADX', 'Price_Change', 'MACD_Hist', 'Volume_Change',
    'BB_Width'
]

selected_corr_indicators = st.multiselect(
    "选择要包含在相关性矩阵中的指标",
    options=tech_indicators.columns.tolist(),
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

# 合并导出选项到原有选项
combined_option = create_echarts_kline_volume(df, ['MA5', 'MA10', 'MA20', 'MA30'])
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