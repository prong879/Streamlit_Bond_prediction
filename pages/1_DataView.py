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
from src.utils.data_processing import fix_datetime_for_arrow, safe_dataframe_display, normalize_column_names  # 数据处理工具
from src.utils.chart_utils import create_correlation_heatmap, create_echarts_kline_volume, calculate_ma  # 图表工具
import torch  # 导入PyTorch，用于解决兼容性问题
import re  # 导入正则表达式模块，用于列名匹配

# 修复PyTorch与Streamlit的兼容性问题
torch.classes.__path__ = []

# 设置页面标题
st.title("📊 数据查看")

# 页面布局和数据处理逻辑开始

def sort_by_date(df):
    """
    确保数据按日期升序排列（由远及近）
    
    参数:
        df (DataFrame): 包含日期列的数据框
        
    返回:
        DataFrame: 按日期排序后的数据框
    """
    # 检查是否有Date列
    if 'Date' in df.columns:
        # 确保Date列是日期时间格式
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception as e:
                st.warning(f"日期格式转换失败: {e}，将保持原始排序")
                return df
        
        # 按日期升序排列
        df = df.sort_values(by='Date', ascending=True)
        st.info("数据已按日期从过去到现在（升序）排列")
    
    return df

# 移动平均线计算已移至 src/utils/chart_utils.py

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
        
        # 确保数据按日期升序排列
        df = df.sort_values(by='Date', ascending=True)
        
        return df
    except Exception as e:
        st.error(f"加载示例数据失败：{str(e)}")
        return None

# 移动平均线计算已移至 src/utils/chart_utils.py

# K线图和成交量图函数已移至 src/utils/chart_utils.py

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
            
            # 确保数据按日期升序排序
            df = sort_by_date(df)
            
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
        
        # 确保数据按日期升序排序
        df = sort_by_date(df)
        
        set_state('raw_data', df)
        # 记录数据加载时间戳

# 获取当前数据
df = get_state('raw_data')
if df is None:
    st.warning("请先加载数据")
    st.stop()

# 显示数据预览
st.header("数据预览")
# 使用安全的数据显示函数
safe_dataframe_display(df.head(), hide_index=True)

# 显示基本统计信息
st.header("基本统计信息")
safe_dataframe_display(df.describe(), hide_index=True)

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
    safe_dataframe_display(tech_indicators[selected_indicators].head(10))
else:
    safe_dataframe_display(tech_indicators.head(10))

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
    option = create_correlation_heatmap(corr_df, selected_corr_indicators)
    
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
        
        # 假设数据已经由sort_by_date函数进行了排序，这里不需要再次排序
        # 创建简单的收盘价折线图配置
        dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
        close_data = df['Close'].round(3).tolist()
        
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
        available_mas = ['MA5', 'MA10', 'MA20', 'MA30']

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
        # 修复时间戳数据后导出
        df_export = fix_datetime_for_arrow(df)
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="点击下载原始数据",
            data=csv,
            file_name="stock_data.csv",
            mime="text/csv"
        )

with col_tech:
    if st.button("导出技术指标数据为CSV"):
        # 修复时间戳数据后导出
        tech_export = fix_datetime_for_arrow(tech_indicators)
        tech_csv = tech_export.to_csv(index=False)
        st.download_button(
            label="点击下载技术指标数据",
            data=tech_csv,
            file_name="technical_indicators.csv",
            mime="text/csv"
        ) 