"""
特征选择模块 - 基于相关性、多重共线性和统计显著性进行特征选择
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def select_features(price_features, output_dir=None, correlation_threshold=0.5, vif_threshold=10):
    """
    综合特征选择函数，基于相关性、多重共线性和统计显著性
    
    参数:
    price_features: 包含所有特征的DataFrame
    output_dir: 输出目录，用于保存特征选择结果
    correlation_threshold: 相关性阈值
    vif_threshold: VIF阈值，用于多重共线性检测
    
    返回:
    选择后的特征DataFrame
    """
    # 计算相关性矩阵
    correlation_matrix = price_features.corr()
    
    # 1. 基于相关性的特征选择
    # 计算与目标变量(Close)的相关性
    target_correlations = abs(correlation_matrix['Close']).sort_values(ascending=False)
    print("\n特征与目标变量的相关性排名:")
    for feature, corr in target_correlations.items():
        print(f"{feature}: {corr:.4f}")

    # 选择相关性高于阈值的特征
    high_correlation_features = target_correlations[target_correlations > correlation_threshold].index.tolist()
    print(f"\n相关性高于{correlation_threshold}的特征: {high_correlation_features}")

    # 2. 多重共线性分析 - 计算VIF (Variance Inflation Factor)
    # 创建一个没有目标变量的特征子集
    X = price_features.drop('Close', axis=1)
    # 添加常数项
    X_with_const = sm.add_constant(X)

    # 计算VIF，添加异常处理
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_values = []
    
    for i in range(X_with_const.shape[1]):
        try:
            # 计算VIF值，处理可能的除零错误
            r_squared_i = sm.OLS(X_with_const.iloc[:, i], X_with_const.iloc[:, list(range(i)) + list(range(i+1, X_with_const.shape[1]))]).fit().rsquared
            vif_i = 1.0 / (1.0 - r_squared_i) if r_squared_i < 0.999 else float('inf')
            vif_values.append(vif_i)
        except Exception as e:
            print(f"计算特征 '{X_with_const.columns[i]}' 的VIF时出错: {e}")
            vif_values.append(float('inf'))  # 将错误情况标记为无穷大
    
    vif_data["VIF"] = vif_values
    vif_data = vif_data.sort_values("VIF", ascending=False)
    print("\nVIF分析结果 (VIF > 10表示存在严重的多重共线性):")
    print(vif_data)

    # 移除VIF过高的特征(通常VIF>10表示严重的多重共线性)
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
    
    # 如果提供了输出目录，保存特征选择结果
    if output_dir:
        save_feature_selection_results(output_dir, target_correlations, vif_data, f_scores, selected_features, price_features.shape[1])
    
    return price_features_selected, correlation_matrix, selected_features

def save_feature_selection_results(output_dir, target_correlations, vif_data, f_scores, selected_features, original_feature_count):
    """
    保存特征选择结果到文本文件
    
    参数:
    output_dir: 输出目录
    target_correlations: 与目标变量的相关性
    vif_data: VIF分析结果
    f_scores: F检验结果
    selected_features: 选择的特征列表
    original_feature_count: 原始特征数量
    """
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
        
        f.write(f"\n特征数量从 {original_feature_count} 减少到 {len(selected_features)}\n") 