# 本项目基于开源项目拓展

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ta  # 使用 ta 库替代 talib
import time
import os

# 设置中文字体支持
import matplotlib
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# 定义自定义配色方案
def set_custom_style():
    """设置自定义的图表样式和配色方案"""
    # 设置Seaborn样式
    sns.set_style("whitegrid", {
        'grid.linestyle': '--',
        'grid.color': '#E0E0E0',
        'axes.edgecolor': '#303030',
        'axes.linewidth': 1.5
    })
    
    # 设置Matplotlib参数
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.edgecolor'] = '#303030'
    
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 自定义颜色映射
    # 价格图颜色
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#1F77B4',  # 蓝色
        '#FF7F0E',  # 橙色
        '#2CA02C',  # 绿色
        '#D62728',  # 红色
        '#9467BD',  # 紫色
        '#8C564B',  # 棕色
        '#E377C2',  # 粉色
        '#7F7F7F',  # 灰色
        '#BCBD22',  # 黄绿色
        '#17BECF'   # 青色
    ])
    
    # 创建自定义热力图颜色映射
    colors = ["#053061", "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", 
              "#FFFFFF", "#FDDBC7", "#F4A582", "#D6604D", "#B2182B", "#67001F"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    
    return custom_cmap

# 设置字体
try:
    # 设置Times New Roman字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    print("Successfully set Times New Roman font")
except Exception as e:
    print(f"Error setting font: {e}")
    print("Will use default font")

# 设置自定义样式
custom_cmap = set_custom_style()

# 辅助函数：设置matplotlib图表的字体和样式
def set_plot_font(ax, title=None, xlabel=None, ylabel=None, title_size=16, label_size=14, legend=True):
    """
    设置matplotlib图表的字体和样式
    
    参数:
    ax: matplotlib轴对象
    title: 图表标题
    xlabel: x轴标签
    ylabel: y轴标签
    title_size: 标题字体大小
    label_size: 轴标签字体大小
    legend: 是否显示图例
    """
    if title:
        ax.set_title(title, fontsize=title_size, fontweight='bold', fontfamily='Times New Roman')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size, fontweight='bold', fontfamily='Times New Roman')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size, fontweight='bold', fontfamily='Times New Roman')
    
    # 设置刻度标签字体
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_fontfamily('Times New Roman')
    for label in ax.get_yticklabels():
        label.set_fontsize(10)
        label.set_fontfamily('Times New Roman')
    
    # 设置图例字体
    if legend and ax.get_legend() is not None:
        for text in ax.get_legend().get_texts():
            text.set_fontsize(12)
            text.set_fontfamily('Times New Roman')
        
        # 美化图例
        ax.get_legend().get_frame().set_facecolor('#F8F8F8')
        ax.get_legend().get_frame().set_edgecolor('#303030')
    
    # 美化轴线
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#303030')
    
    # 设置网格线
    ax.grid(True, linestyle='--', alpha=0.7, color='#E0E0E0')
    
    # 设置刻度
    ax.tick_params(direction='out', length=6, width=1.5, colors='#303030')

filepath = 'data/rlData.csv'
data = pd.read_csv(filepath)
data = data.sort_values('Date')
print(data.head())
print(data.shape)

# 创建output文件夹（如果不存在）
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 绘制原始股票价格走势图
plt.figure(figsize=(15, 9), facecolor='white')
ax = plt.gca()

# 添加价格线 - 保持较粗的线宽
plt.plot(data.index, data['Close'], linewidth=2.5, color='#1F77B4', label='Close Price')

# 添加移动平均线 - 使用与原序列相同的线宽
ma5 = data['Close'].rolling(window=5).mean()
ma10 = data['Close'].rolling(window=10).mean()
ma20 = data['Close'].rolling(window=20).mean()
ma30 = data['Close'].rolling(window=30).mean()
plt.plot(data.index, ma5, linewidth=2.5, color='black', label='M5')
plt.plot(data.index, ma10, linewidth=2.5, color='#FFD700', label='M10')  # 黄色
plt.plot(data.index, ma20, linewidth=2.5, color='#FF7F0E', label='M20')  # 橘色
plt.plot(data.index, ma30, linewidth=2.5, color='#2CA02C', label='M30')  # 绿色

# 设置x轴刻度
plt.xticks(range(0, data.shape[0], 20), data['Date'].iloc[::20], rotation=45)

# 添加标题和标签
plt.title("Historical Stock Price Trend", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel("Date", fontsize=18, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel("Price (USD)", fontsize=18, fontweight='bold', fontfamily='Times New Roman')

# 添加图例
plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 美化图表
set_plot_font(ax)

# 添加边框
plt.box(True)

# 调整布局
plt.tight_layout()

# 保存原始股票价格走势图
plt.savefig(os.path.join(output_dir, 'original_stock_price.png'), dpi=300, bbox_inches='tight')

plt.show()

# 1.特征工程 - 增强版
# 添加技术指标作为特征
def add_technical_indicators(df):
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

# 应用特征工程
try:
    price_features = add_technical_indicators(data)
    print("Successfully added technical indicators, feature count:", price_features.shape[1])
    print(price_features.columns.tolist())
except Exception as e:
    print("Error adding technical indicators:", str(e))
    # 如果特征工程失败，回退到只使用收盘价
    price_features = data[['Close']].copy()

# 显示部分特征的相关性
plt.figure(figsize=(14, 12), facecolor='white')

# 计算相关性矩阵
correlation_matrix = price_features.corr()

# 绘制热力图
heatmap = sns.heatmap(
    correlation_matrix, 
    annot=True,                  # 显示数值
    fmt='.2f',                   # 数值格式
    cmap=custom_cmap,            # 使用自定义颜色映射
    linewidths=0.5,              # 网格线宽度
    annot_kws={"size": 10},      # 注释文本大小
    cbar_kws={"shrink": 0.8}     # 颜色条大小
)

# 设置标题和字体
ax = plt.gca()
set_plot_font(ax, title='Feature Correlation Matrix Analysis')

# 调整布局
plt.tight_layout()

# 保存相关性矩阵图
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')

plt.show()

# 特征选择 - 基于相关性和多重共线性分析
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. 基于相关性的特征选择
# 计算与目标变量(Close)的相关性
target_correlations = abs(correlation_matrix['Close']).sort_values(ascending=False)
print("\n特征与目标变量的相关性排名:")
for feature, corr in target_correlations.items():
    print(f"{feature}: {corr:.4f}")

# 选择相关性高于阈值的特征
correlation_threshold = 0.5
high_correlation_features = target_correlations[target_correlations > correlation_threshold].index.tolist()
print(f"\n相关性高于{correlation_threshold}的特征: {high_correlation_features}")

# 2. 多重共线性分析 - 计算VIF (Variance Inflation Factor)
# 创建一个没有目标变量的特征子集
X = price_features.drop('Close', axis=1)
# 添加常数项
X_with_const = sm.add_constant(X)

# 计算VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X_with_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
vif_data = vif_data.sort_values("VIF", ascending=False)
print("\nVIF分析结果 (VIF > 10表示存在严重的多重共线性):")
print(vif_data)

# 移除VIF过高的特征(通常VIF>10表示严重的多重共线性)
vif_threshold = 10
high_vif_features = vif_data[vif_data["VIF"] > vif_threshold]["Feature"].tolist()
if 'const' in high_vif_features:
    high_vif_features.remove('const')  # 移除常数项
print(f"\n多重共线性严重的特征 (VIF > {vif_threshold}): {high_vif_features}")

# 3. 基于统计显著性的特征选择
# 使用f_regression评估特征的统计显著性
X = price_features.drop('Close', axis=1).values
y = price_features['Close'].values
f_selector = SelectKBest(f_regression, k='all')
f_selector.fit(X, y)

# 获取每个特征的p值和F值
f_scores = pd.DataFrame()
f_scores["Feature"] = price_features.drop('Close', axis=1).columns
f_scores["F Score"] = f_selector.scores_
f_scores["P Value"] = f_selector.pvalues_
f_scores = f_scores.sort_values("F Score", ascending=False)
print("\n特征的F检验结果:")
print(f_scores)

# 选择统计显著的特征(p值<0.05)
significant_features = f_scores[f_scores["P Value"] < 0.05]["Feature"].tolist()
print(f"\n统计显著的特征 (P < 0.05): {significant_features}")

# 4. 综合以上分析，选择最终的特征集
# 从高相关性特征中移除多重共线性严重的特征
selected_features = [f for f in high_correlation_features if f not in high_vif_features]
# 确保所有统计显著的特征都被包含
for feature in significant_features:
    if feature not in selected_features and feature != 'Close':
        selected_features.append(feature)
# 确保目标变量在特征集中
if 'Close' not in selected_features:
    selected_features.append('Close')

print(f"\n最终选择的特征集: {selected_features}")
print(f"特征数量从 {price_features.shape[1]} 减少到 {len(selected_features)}")

# 使用选定的特征子集
price_features_selected = price_features[selected_features]

# 可视化选定特征的相关性矩阵
plt.figure(figsize=(12, 10), facecolor='white')
correlation_matrix_selected = price_features_selected.corr()
heatmap = sns.heatmap(
    correlation_matrix_selected, 
    annot=True,
    fmt='.2f',
    cmap=custom_cmap,
    linewidths=0.5,
    annot_kws={"size": 10},
    cbar_kws={"shrink": 0.8}
)
ax = plt.gca()
set_plot_font(ax, title='Selected Features Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'selected_features_correlation.png'), dpi=300, bbox_inches='tight')
plt.show()

# 保存特征选择结果到文本文件
with open(os.path.join(output_dir, 'feature_selection_results.txt'), 'w', encoding='utf-8') as f:
    f.write("特征选择分析结果\n")
    f.write("=" * 50 + "\n\n")
    
    f.write("1. 与目标变量的相关性排名:\n")
    for feature, corr in target_correlations.items():
        f.write(f"{feature}: {corr:.4f}\n")
    
    f.write("\n2. 多重共线性分析 (VIF):\n")
    for _, row in vif_data.iterrows():
        f.write(f"{row['Feature']}: {row['VIF']:.4f}\n")
    
    f.write("\n3. 特征的统计显著性:\n")
    for _, row in f_scores.iterrows():
        f.write(f"{row['Feature']}: F={row['F Score']:.4f}, P={row['P Value']:.6f}\n")
    
    f.write("\n4. 最终选择的特征集:\n")
    for feature in selected_features:
        f.write(f"- {feature}\n")
    
    f.write(f"\n特征数量从 {price_features.shape[1]} 减少到 {len(selected_features)}\n")

from sklearn.preprocessing import MinMaxScaler
# 进行不同的数据缩放，将数据缩放到-1和1之间
scaler = MinMaxScaler(feature_range=(-1, 1))
# 对选定的特征进行缩放
price_features_scaled = pd.DataFrame(
    scaler.fit_transform(price_features_selected),
    columns=price_features_selected.columns,
    index=price_features_selected.index
)
print(price_features_scaled.shape)

# 2.数据集制作 - 修改以支持多特征
# 今天的多个特征预测明天的收盘价
# lookback表示观察的跨度
def split_data(stock, lookback):
    data_raw = stock.to_numpy()
    data = []

    # you can free play（seq_length）
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, 0].reshape(-1, 1)  # 只预测收盘价

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, 0].reshape(-1, 1)  # 只预测收盘价

    return [x_train, y_train, x_test, y_test]

lookback = 20
x_train, y_train, x_test, y_test = split_data(price_features_scaled, lookback)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)


# 注意：pytorch的nn.LSTM input shape=(seq_length, batch_size, input_size)
# 3.模型构建 —— LSTM - 修改以支持多特征输入

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
y_train_lstm = torch.from_numpy(y_train).float()
y_test_lstm = torch.from_numpy(y_test).float()
y_train_gru = torch.from_numpy(y_train).float()
y_test_gru = torch.from_numpy(y_test).float()

# 输入的维度为特征数量
input_dim = x_train.shape[2]  # 动态获取特征维度
print(f"模型输入维度: {input_dim}")
# 隐藏层特征的维度
hidden_dim = 32  # 从64降低到32，减少模型复杂度
# 循环的layers
num_layers = 1   # 从2层降低到1层，简化模型结构
# 预测后一天的收盘价
output_dim = 1
num_epochs = 100


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)  # 从0.2增加到0.3，增强正则化
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out[:, -1, :])  # 应用dropout
        out = self.fc(out)
        return out



model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
criterion = torch.nn.MSELoss()
# 添加L2正则化，权重衰减参数为1e-5
weight_decay = 1e-5
optimiser = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)

# 4.模型训练
import time

# 创建output文件夹（如果不存在）已在文件开头定义
# output_dir = 'output'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

# 添加早停机制
patience = 15
best_loss = float('inf')
counter = 0
early_stop = False
best_model_path = os.path.join(output_dir, 'best_model.pth')

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
    # 早停检查
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), best_model_path)
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {t}")
            # 加载最佳模型
            model.load_state_dict(torch.load(best_model_path))
            # 更新hist数组，截断未使用的部分
            hist = hist[:t+1]
            break

training_time = time.time() - start_time
print("Training time: {}".format(training_time))
print(f"Best loss: {best_loss:.6f}")

# 5.模型结果可视化
# 注意：我们只预测收盘价，所以需要创建一个临时DataFrame来反向转换
# 创建临时DataFrame，列数与选定的特征数量一致
temp_df = pd.DataFrame(np.zeros((len(y_train_pred), len(selected_features))))
temp_df[temp_df.columns[0]] = y_train_pred.detach().numpy().flatten()
predict = pd.DataFrame(scaler.inverse_transform(temp_df)[:, 0])

temp_df = pd.DataFrame(np.zeros((len(y_train_lstm), len(selected_features))))
temp_df[temp_df.columns[0]] = y_train_lstm.detach().numpy().flatten()
original = pd.DataFrame(scaler.inverse_transform(temp_df)[:, 0])

import seaborn as sns
sns.set_style("darkgrid")

# 创建一个更美观的图表
fig = plt.figure(figsize=(18, 8), facecolor='white')
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

# 第一个子图：预测结果对比
ax1 = plt.subplot(gs[0])
# 绘制实际数据
sns.lineplot(x=original.index, y=original[0], label="Actual Data", color='#1F77B4', 
             linewidth=2.5, ax=ax1)
# 绘制预测数据
sns.lineplot(x=predict.index, y=predict[0], label="LSTM Prediction", color='#FF7F0E', 
             linewidth=2, ax=ax1, linestyle='--')

# 添加预测误差区域
ax1.fill_between(predict.index, 
                 original[0], 
                 predict[0], 
                 color='#FF7F0E', 
                 alpha=0.2, 
                 label='Prediction Error')

# 设置图表样式
set_plot_font(ax1, title='LSTM Stock Price Prediction Results', xlabel='Trading Days', ylabel='Price (USD)')

# 添加网格线
ax1.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax1.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

# 第二个子图：训练损失曲线
ax2 = plt.subplot(gs[1])
# 绘制损失曲线
sns.lineplot(x=range(len(hist)), y=hist, color='#2CA02C', linewidth=2.5, ax=ax2)

# 添加平滑曲线
window_size = 5
if len(hist) > window_size:
    smoothed_hist = np.convolve(hist, np.ones(window_size)/window_size, mode='valid')
    sns.lineplot(x=range(window_size-1, len(hist)), y=smoothed_hist, 
                 color='#D62728', linewidth=2, ax=ax2, label='Smoothed Loss')

# 设置图表样式
set_plot_font(ax2, title='Training Loss Curve', xlabel='Epochs', ylabel='Loss Value (MSE)')

# 添加网格线
ax2.grid(True, linestyle='--', alpha=0.7)

# 添加最终损失值标注
final_loss = hist[-1]
ax2.annotate(f'Final Loss: {final_loss:.4f}',
             xy=(len(hist)-1, final_loss),
             xytext=(len(hist)*0.7, final_loss*1.5),
             arrowprops=dict(facecolor='#9467BD', shrink=0.05, width=1.5, headwidth=8),
             fontsize=12, fontweight='bold', 
             fontproperties=font_manager.FontProperties(family='Times New Roman'))

# 添加图例
if len(hist) > window_size:
    ax2.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

# 调整布局
plt.tight_layout()

# 保存matplotlib图像到output文件夹
plt.savefig(os.path.join(output_dir, 'training_visualization.png'), dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 6.模型验证
# print(x_test[-1])
import math, time
from sklearn.metrics import mean_squared_error

# 模型验证
model.eval()
with torch.no_grad():
    test_predict = model(x_test)
    
# 反归一化 - 修复维度不匹配问题
# 创建临时DataFrame来反向转换
temp_df = pd.DataFrame(np.zeros((len(test_predict), len(selected_features))))
temp_df[temp_df.columns[0]] = test_predict.detach().cpu().numpy().flatten()
test_predict = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)

temp_df = pd.DataFrame(np.zeros((len(y_test), len(selected_features))))
temp_df[temp_df.columns[0]] = y_test.flatten()  # y_test已经是numpy数组，不需要cpu()
y_test_orig = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)

temp_df = pd.DataFrame(np.zeros((len(y_train_pred), len(selected_features))))
temp_df[temp_df.columns[0]] = y_train_pred.detach().cpu().numpy().flatten()
train_predict = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)

temp_df = pd.DataFrame(np.zeros((len(y_train), len(selected_features))))
temp_df[temp_df.columns[0]] = y_train.flatten()  # y_train已经是numpy数组，不需要cpu()
y_train_orig = scaler.inverse_transform(temp_df)[:, 0].reshape(-1, 1)

# 计算RMSE
train_score = np.sqrt(mean_squared_error(y_train_orig, train_predict))
test_score = np.sqrt(mean_squared_error(y_test_orig, test_predict))
print(f'Training Set RMSE: {train_score:.4f}')
print(f'Test Set RMSE: {test_score:.4f}')

# 添加方向准确率评估
def direction_accuracy(y_true, y_pred):
    # 计算实际和预测的方向变化
    y_true_direction = np.sign(np.diff(y_true.flatten()))
    y_pred_direction = np.sign(np.diff(y_pred.flatten()))
    
    # 计算方向准确率
    correct_direction = np.sum(y_true_direction == y_pred_direction)
    return correct_direction / len(y_true_direction) * 100

train_direction_acc = direction_accuracy(y_train_orig, train_predict)
test_direction_acc = direction_accuracy(y_test_orig, test_predict)
print(f'Training Set Direction Accuracy: {train_direction_acc:.2f}%')
print(f'Test Set Direction Accuracy: {test_direction_acc:.2f}%')

# 保存评估结果到文本文件
with open(os.path.join(output_dir, 'model_evaluation.txt'), 'w') as f:
    f.write(f'Training Set RMSE: {train_score:.4f}\n')
    f.write(f'Test Set RMSE: {test_score:.4f}\n')
    f.write(f'Training Time: {training_time:.4f} seconds\n')
    f.write(f'Training Set Direction Accuracy: {train_direction_acc:.2f}%\n')
    f.write(f'Test Set Direction Accuracy: {test_direction_acc:.2f}%\n')

# 创建验证结果可视化
fig = plt.figure(figsize=(16, 10), facecolor='white')
gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

# 第一个子图：训练集和测试集预测结果
ax1 = plt.subplot(gs[0])

# 创建日期索引
train_dates = price_features.index[lookback:lookback+len(y_train_orig)]
test_dates = price_features.index[lookback+len(y_train_orig):lookback+len(y_train_orig)+len(y_test_orig)]

# 绘制原始数据
sns.lineplot(x=price_features.index[lookback:], y=price_features['Close'][lookback:], 
             label="Actual Stock Price", color='#1F77B4', linewidth=2, ax=ax1)

# 绘制训练集预测
sns.lineplot(x=train_dates, y=train_predict.flatten(), 
             label="Training Set Prediction", color='#2CA02C', linewidth=1.5, ax=ax1)

# 绘制测试集预测
sns.lineplot(x=test_dates, y=test_predict.flatten(), 
             label="Test Set Prediction", color='#FF7F0E', linewidth=1.5, ax=ax1)

# 添加训练集预测的误差范围
train_std = np.std(y_train_orig.flatten() - train_predict.flatten())
ax1.fill_between(train_dates, 
                 train_predict.flatten() - train_std, 
                 train_predict.flatten() + train_std, 
                 color='#2CA02C', alpha=0.2, label='Training Error Range (±1σ)')

# 添加测试集预测的误差范围
test_std = np.std(y_test_orig.flatten() - test_predict.flatten())
ax1.fill_between(test_dates, 
                 test_predict.flatten() - test_std, 
                 test_predict.flatten() + test_std, 
                 color='#FF7F0E', alpha=0.2, label='Test Error Range (±1σ)')

# 添加训练集和测试集分隔线
split_date = train_dates[-1]
ax1.axvline(x=split_date, color='#D62728', linestyle='--', linewidth=2, label='Train/Test Split')

# 设置图表样式
set_plot_font(ax1, title='LSTM Model Training and Test Set Prediction Results', xlabel='Date', ylabel='Stock Price (USD)')

# 添加RMSE标注
ax1.annotate(f'Training RMSE: {train_score:.4f}',
             xy=(train_dates[len(train_dates)//4], max(train_predict.flatten())*0.95),
             xytext=(train_dates[len(train_dates)//4], max(train_predict.flatten())*0.95),
             fontsize=12, fontweight='bold', 
             fontproperties=font_manager.FontProperties(family='Times New Roman'),
             bbox=dict(boxstyle="round,pad=0.3", fc="#D8BFD8", ec="black", alpha=0.8))

ax1.annotate(f'Test RMSE: {test_score:.4f}',
             xy=(test_dates[len(test_dates)//4], max(test_predict.flatten())*0.90),
             xytext=(test_dates[len(test_dates)//4], max(test_predict.flatten())*0.90),
             fontsize=12, fontweight='bold', 
             fontproperties=font_manager.FontProperties(family='Times New Roman'),
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFD700", ec="black", alpha=0.8))

# 第二个子图：预测误差分析
ax2 = plt.subplot(gs[1])

# 计算误差
train_error = y_train_orig.flatten() - train_predict.flatten()
test_error = y_test_orig.flatten() - test_predict.flatten()

# 绘制训练集误差
sns.lineplot(x=train_dates, y=train_error, 
             label="Training Error", color='#2CA02C', linewidth=1.5, ax=ax2)

# 绘制测试集误差
sns.lineplot(x=test_dates, y=test_error, 
             label="Test Error", color='#FF7F0E', linewidth=1.5, ax=ax2)

# 添加误差标准差范围线
ax2.axhline(y=train_std, color='#2CA02C', linestyle='--', alpha=0.7, label=f'Train Error σ: {train_std:.2f}')
ax2.axhline(y=-train_std, color='#2CA02C', linestyle='--', alpha=0.7)
ax2.axhline(y=test_std, color='#FF7F0E', linestyle='--', alpha=0.7, label=f'Test Error σ: {test_std:.2f}')
ax2.axhline(y=-test_std, color='#FF7F0E', linestyle='--', alpha=0.7)

# 添加零线
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

# 添加训练集和测试集分隔线
ax2.axvline(x=split_date, color='#D62728', linestyle='--', linewidth=2)

# 设置图表样式
set_plot_font(ax2, title='Prediction Error Analysis', xlabel='Date', ylabel='Error Value (USD)')

# 添加网格线
ax2.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax2.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.8, shadow=True)

# 调整布局
plt.tight_layout()

# 保存matplotlib图像到output文件夹
plt.savefig(os.path.join(output_dir, 'validation_results.png'), dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 创建预测结果对比图
plt.figure(figsize=(16, 8), facecolor='white')

# 创建日期索引
all_dates = price_features.index[lookback:]

# 创建预测数据框架
train_plot = np.full((len(price_features), 1), np.nan)
train_plot[lookback:lookback+len(train_predict)] = train_predict

test_plot = np.full((len(price_features), 1), np.nan)
test_plot[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict

# 创建误差范围数据
train_upper = np.full((len(price_features), 1), np.nan)
train_lower = np.full((len(price_features), 1), np.nan)
train_upper[lookback:lookback+len(train_predict)] = train_predict + train_std
train_lower[lookback:lookback+len(train_predict)] = train_predict - train_std

test_upper = np.full((len(price_features), 1), np.nan)
test_lower = np.full((len(price_features), 1), np.nan)
test_upper[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict + test_std
test_lower[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)] = test_predict - test_std

# 绘制原始数据
plt.plot(all_dates, price_features['Close'][lookback:], 
         label='Actual Price', color='#1F77B4', linewidth=2.5)

# 绘制训练集预测
plt.plot(price_features.index, train_plot, 
         label='Training Prediction', color='#2CA02C', linewidth=1.5)

# 绘制测试集预测
plt.plot(price_features.index, test_plot, 
         label='Test Prediction', color='#FF7F0E', linewidth=1.5)

# 添加误差范围区域
plt.fill_between(price_features.index, train_lower.flatten(), train_upper.flatten(), 
                 color='#2CA02C', alpha=0.2, label='Training Error Range (±1σ)')
plt.fill_between(price_features.index, test_lower.flatten(), test_upper.flatten(), 
                 color='#FF7F0E', alpha=0.2, label='Test Error Range (±1σ)')

# 添加训练集和测试集分隔线
split_idx = lookback + len(train_predict)
plt.axvline(x=price_features.index[split_idx], color='#D62728', 
            linestyle='--', linewidth=2, label='Train/Test Split')

# 设置标题和标签
plt.title('Multi-feature LSTM Stock Price Prediction with Error Ranges', 
          fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.xlabel('Date', fontsize=16, fontweight='bold', fontfamily='Times New Roman')
plt.ylabel('Price (USD)', fontsize=16, fontweight='bold', fontfamily='Times New Roman')

# 添加RMSE和误差标准差标注
plt.annotate(f'Training RMSE: {train_score:.4f}, σ: {train_std:.2f}',
             xy=(price_features.index[lookback+len(train_predict)//4], 
                 max(price_features['Close'][lookback:lookback+len(train_predict)])*0.95),
             xytext=(price_features.index[lookback+len(train_predict)//4], 
                    max(price_features['Close'][lookback:lookback+len(train_predict)])*0.95),
             fontsize=12, fontweight='bold', 
             fontproperties=font_manager.FontProperties(family='Times New Roman'),
             bbox=dict(boxstyle="round,pad=0.3", fc="#D8BFD8", ec="black", alpha=0.8))

plt.annotate(f'Test RMSE: {test_score:.4f}, σ: {test_std:.2f}',
             xy=(price_features.index[lookback+len(train_predict)+len(test_predict)//4], 
                 max(price_features['Close'][lookback+len(train_predict):])*0.90),
             xytext=(price_features.index[lookback+len(train_predict)+len(test_predict)//4], 
                    max(price_features['Close'][lookback+len(train_predict):])*0.90),
             fontsize=12, fontweight='bold', 
             fontproperties=font_manager.FontProperties(family='Times New Roman'),
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFD700", ec="black", alpha=0.8))

# 调整布局
plt.tight_layout()

# 保存matplotlib图像到output文件夹
plt.savefig(os.path.join(output_dir, 'prediction_results.png'), dpi=300, bbox_inches='tight')

# 显示图像
plt.show()

# 特征重要性分析 - 可选
try:
    # 使用简单的相关性分析评估特征重要性
    feature_importance = abs(correlation_matrix['Close']).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    feature_importance.drop('Close').plot(kind='bar')
    ax = plt.gca()
    set_plot_font(ax, title='Feature Importance (Correlation with Closing Price)', xlabel='Features', ylabel='Absolute Correlation')
    plt.tight_layout()
    
    # 保存特征重要性图像到output文件夹
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # 显示图像
    plt.show()
    plt.close()
    
    # 将特征重要性排名，保存到文本文件
    with open(os.path.join(output_dir, 'feature_importance.txt'), 'w', encoding='utf-8') as f:
        f.write("Feature Importance Ranking:\n")
        for i, (feature, importance) in enumerate(feature_importance.items()):
            if feature != 'Close':
                f.write(f"{i}. {feature}: {importance:.4f}\n")
                print(f"{i}. {feature}: {importance:.4f}")
except Exception as e:
    print("特征重要性分析失败:", str(e))
