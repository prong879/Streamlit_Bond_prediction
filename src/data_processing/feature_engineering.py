"""
特征工程模块 - 计算技术指标
"""
import pandas as pd
import numpy as np
import ta  # 使用 ta 库替代 talib

def add_technical_indicators(df):
    """
    添加技术指标作为特征
    
    参数:
    df: 包含价格数据的DataFrame
    
    返回:
    添加了技术指标的DataFrame
    """
    # 确保数据中有必要的列
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"警告: 数据中缺少以下列: {missing_cols}")
        print("将仅使用可用的列计算技术指标")
    
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
            print(f"RSI 计算失败: {e}")
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
            print(f"布林带计算失败: {e}")
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
            print(f"MACD 计算失败: {e}")
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
        except Exception as e:
            print(f"额外技术指标计算失败: {e}")
    
    # 删除NaN值
    features.dropna(inplace=True)
    
    return features

def prepare_data_for_model(features, scaler=None, feature_range=(-1, 1)):
    """
    对特征数据进行归一化处理
    
    参数:
    features: 特征DataFrame
    scaler: 已有的缩放器（如果为None则创建新的）
    feature_range: 缩放范围
    
    返回:
    缩放后的特征数据和缩放器
    """
    from sklearn.preprocessing import MinMaxScaler
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_features = pd.DataFrame(
            scaler.fit_transform(features),
            columns=features.columns,
            index=features.index
        )
    else:
        scaled_features = pd.DataFrame(
            scaler.transform(features),
            columns=features.columns,
            index=features.index
        )
    
    return scaled_features, scaler

def split_data(stock, lookback):
    """
    将数据分割为训练集和测试集
    
    参数:
    stock: 股票数据
    lookback: 观察窗口大小
    
    返回:
    x_train, y_train, x_test, y_test
    """
    data_raw = stock.to_numpy()
    data = []

    # 创建时间序列数据
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)  # 只预测收盘价

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, 0].reshape(-1, 1)  # 只预测收盘价

    return [x_train, y_train, x_test, y_test] 