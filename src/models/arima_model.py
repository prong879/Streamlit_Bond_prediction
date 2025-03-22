"""
ARIMA模型工具模块 - 实现ARIMA时间序列分析的各项功能

该模块包含了实现ARIMA模型分析所需的各种函数:
- 平稳性检验（ADF检验）
- 随机性检验（Ljung-Box检验）
- 正态性检验（JB检验、QQ图）
- 自相关/偏自相关分析
- 模型参数优化（BIC/AIC）
- 模型拟合与预测
"""
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.api import qqplot
from scipy import stats
import io
import base64
from statsmodels.stats.stattools import durbin_watson

def generate_descriptive_statistics(series):
    """
    生成描述性统计表，包含N、mean、p50、sd、min、max、skewness、kurtosis等指标
    
    参数:
    series: 时间序列数据
    
    返回:
    pd.DataFrame: 描述性统计表
    dict: 正态性检验结果 (Jarque-Bera检验)
    """
    # 获取基本统计量
    n = len(series)
    mean = series.mean()
    median = series.median()
    std = series.std()
    min_val = series.min()
    max_val = series.max()
    skew = stats.skew(series.dropna())
    kurt = stats.kurtosis(series.dropna())
    
    # 创建统计表
    stats_df = pd.DataFrame({
        'VARIABLES': [series.name if series.name else 'Series'],
        'N': [n],
        'mean': [mean],
        'p50': [median],
        'sd': [std],
        'min': [min_val],
        'max': [max_val],
        'skewness': [skew],
        'kurtosis': [kurt]
    })
    
    # 进行Jarque-Bera正态性检验
    jb_stat, jb_pvalue = stats.jarque_bera(series.dropna())
    normality_test = {
        'statistic': jb_stat,
        'p_value': jb_pvalue,
        'is_normal': jb_pvalue > 0.05
    }
    
    return stats_df, normality_test

def check_stationarity(series):
    """
    检查时间序列的平稳性
    
    参数:
    series: 时间序列数据
    
    返回:
    dict: 平稳性检验结果
    bool: 是否平稳
    pd.DataFrame: Streamlit可绘制的数据
    """
    # 检查数据类型，确保是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 如果是日期时间类型，转换为时间戳
        if pd.api.types.is_datetime64_any_dtype(series):
            series = (series - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        else:
            raise TypeError(f"无法处理类型为 {series.dtype} 的数据，需要数值类型数据")
    
    # 确保数据中没有缺失值
    series = series.dropna()
    
    # 执行ADF检验
    try:
        result = adfuller(series)
        
        # 创建结果字典
        results = {
            'ADF统计量': result[0],
            'p值': result[1],
            '临界值': {
                '1%': result[4]['1%'],
                '5%': result[4]['5%'],
                '10%': result[4]['10%']
            }
        }
        
        # 确定是否平稳
        is_stationary = results['p值'] < 0.05
    except Exception as e:
        # 如果ADF检验失败，返回默认结果
        st.error(f"ADF检验失败: {str(e)}")
        results = {
            'ADF统计量': float('nan'),
            'p值': float('nan'),
            '临界值': {
                '1%': float('nan'),
                '5%': float('nan'),
                '10%': float('nan')
            }
        }
        is_stationary = False
    
    # 计算移动平均线和标准差
    rolling_mean = series.rolling(window=12).mean()
    rolling_std = series.rolling(window=12).std()
    
    # 准备数据用于Streamlit绘图
    if isinstance(series, pd.Series):
        df = pd.DataFrame({
            '原始数据': series,
            '移动平均': rolling_mean,
            '移动标准差': rolling_std
        })
    else:
        # 如果输入不是Series，转换为DataFrame
        index = range(len(series))
        df = pd.DataFrame({
            '原始数据': series,
            '移动平均': rolling_mean,
            '移动标准差': rolling_std
        }, index=index)
    
    return results, is_stationary, df

def diff_series(series, diff_order=1, log_diff=False):
    """
    对时间序列进行差分处理，支持普通差分和对数差分
    
    参数:
    series: 时间序列数据
    diff_order: 差分阶数
    log_diff: 是否进行对数差分
    
    返回:
    diff_data: 差分后的序列
    pd.DataFrame: Streamlit可绘制的数据
    """
    # 检查数据类型，确保是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 如果是日期时间类型，转换为时间戳
        if pd.api.types.is_datetime64_any_dtype(series):
            series = (series - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        else:
            raise TypeError(f"无法处理类型为 {series.dtype} 的数据，需要数值类型数据")
    
    # 如果是对数差分，先取对数再差分
    if log_diff:
        if (series <= 0).any():
            # 如果数据包含非正值，给出警告并使用普通差分
            log_diff = False
            diff_data = series.diff(diff_order).dropna()
            diff_type = f"{diff_order}阶普通差分"
        else:
            # 进行对数差分
            log_series = np.log(series)
            diff_data = log_series.diff(diff_order).dropna()
            diff_type = f"{diff_order}阶对数差分"
    else:
        # 进行普通差分
        diff_data = series.diff(diff_order).dropna()
        diff_type = f"{diff_order}阶普通差分"
    
    # 准备用于Streamlit绘图的数据
    if isinstance(series, pd.Series):
        # 如果原始数据有日期索引，保留它
        index_series = series.index
        index_diff = diff_data.index
        
        # 创建对比图表的数据框
        # 创建含 None 的列表，长度为差分阶数
        none_list = [0] * diff_order  # 使用0代替None
        df = pd.DataFrame({
            '原始序列': pd.Series(series.values, index=index_series),
            f'{diff_type}序列': pd.Series(none_list + diff_data.values.tolist(), index=index_series)
        })
    else:
        # 如果没有日期索引，使用默认的数字索引
        index = range(len(series))
        diff_index = range(diff_order, len(series))
        
        # 创建对比图表的数据框
        # 使用0代替None，确保JSON序列化不会出错
        df = pd.DataFrame({
            '原始序列': series,
            f'{diff_type}序列': [0] * diff_order + diff_data.tolist()
        }, index=index)
    
    return diff_data, df

def check_white_noise(series):
    """
    白噪声检验 (Ljung-Box检验)
    
    参数:
    series: 时间序列数据
    
    返回:
    pd.DataFrame: 检验结果，包含Q统计量和对应p值
    bool: 是否为白噪声
    """
    # 执行Ljung-Box检验，计算滞后阶数为1到10的Q统计量
    result = acorr_ljungbox(series, lags=range(1, 11))
    
    # 整理结果
    lb_df = pd.DataFrame({
        '滞后阶数': range(1, 11),
        'Q统计量': result['lb_stat'].values,
        'p值': result['lb_pvalue'].values
    })
    
    # 判断是否为白噪声（如果任何p值小于0.05，则不是白噪声）
    is_white_noise = not (lb_df['p值'] < 0.05).any()
    
    return lb_df, is_white_noise

def analyze_acf_pacf(series):
    """
    分析时间序列的自相关函数和偏自相关函数
    
    参数:
    series: 时间序列数据
    
    返回:
    acf_values: 自相关系数
    pacf_values: 偏自相关系数
    dict: 包含可用于Streamlit绘图的数据
    """
    # 计算ACF和PACF值
    acf_values = acf(series, nlags=40)
    pacf_values = pacf(series, nlags=40)
    
    # 计算置信区间
    conf_level = 1.96 / np.sqrt(len(series))
    
    # 准备ACF数据
    acf_df = pd.DataFrame({
        'ACF': acf_values,
        '上限': [conf_level] * len(acf_values),
        '下限': [-conf_level] * len(acf_values)
    })
    
    # 准备PACF数据
    pacf_df = pd.DataFrame({
        'PACF': pacf_values,
        '上限': [conf_level] * len(pacf_values),
        '下限': [-conf_level] * len(pacf_values)
    })
    
    # 将数据打包到字典中
    chart_data = {
        'acf': acf_df,
        'pacf': pacf_df
    }
    
    return acf_values, pacf_values, chart_data

def find_best_arima_params(timeseries, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3), 
                         P_range=None, D_range=None, Q_range=None, s=None, criterion='aic'):
    """
    寻找最佳ARIMA或SARIMA模型参数
    
    参数:
    timeseries: 时间序列数据
    p_range: AR阶数的范围（如range(0, 3)表示搜索0,1,2）
    d_range: 差分阶数的范围
    q_range: MA阶数的范围
    P_range: 季节性AR阶数的范围（非None时使用SARIMA模型）
    D_range: 季节性差分阶数的范围
    Q_range: 季节性MA阶数的范围
    s: 季节性周期
    criterion: 使用的信息准则（'aic'或'bic'）
    
    返回:
    best_params: 最佳参数组合 (p,d,q) 或 (p,d,q,P,D,Q,s)
    """
    results = []
    seasonal = P_range is not None and D_range is not None and Q_range is not None and s is not None
    
    # 根据是否包含季节性参数决定搜索范围
    if seasonal:
        # 搜索SARIMA模型
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    for P in P_range:
                        for D in D_range:
                            for Q in Q_range:
                                # 跳过p=0且q=0且P=0且Q=0的情况
                                if p == 0 and q == 0 and P == 0 and Q == 0:
                                    continue
                                
                                try:
                                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                                    model = SARIMAX(
                                        timeseries, 
                                        order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    model_fit = model.fit(disp=False)
                                    
                                    # 记录AIC/BIC值
                                    results.append({
                                        'order': (p, d, q, P, D, Q, s),
                                        'AIC': model_fit.aic,
                                        'BIC': model_fit.bic,
                                        'success': True
                                    })
                                    
                                except Exception as e:
                                    results.append({
                                        'order': (p, d, q, P, D, Q, s),
                                        'AIC': np.nan,
                                        'BIC': np.nan,
                                        'success': False,
                                        'error': str(e)
                                    })
    else:
        # 搜索普通ARIMA模型
        for p in p_range:
            for d in d_range:
                for q in q_range:
                    # 跳过p=0且q=0的情况
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        # 创建并拟合ARIMA模型
                        model = ARIMA(timeseries, order=(p, d, q))
                        model_fit = model.fit()
                        
                        # 记录AIC和BIC值
                        results.append({
                            'order': (p, d, q),
                            'AIC': model_fit.aic,
                            'BIC': model_fit.bic,
                            'success': True
                        })
                        
                    except Exception as e:
                        results.append({
                            'order': (p, d, q),
                            'AIC': np.nan,
                            'BIC': np.nan,
                            'success': False,
                            'error': str(e)
                        })
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    
    # 找出AIC或BIC最小的参数组合
    successful_results = results_df[results_df['success']]
    if not successful_results.empty:
        if criterion.lower() == 'aic':
            best_params = successful_results.loc[successful_results['AIC'].idxmin()]['order']
        else:  # 使用BIC
            best_params = successful_results.loc[successful_results['BIC'].idxmin()]['order']
    else:
        # 默认值
        best_params = (1, 1, 1) if not seasonal else (1, 1, 1, 1, 0, 1, s)
    
    return best_params

def fit_arima_model(timeseries, order):
    """
    拟合ARIMA或SARIMA模型
    
    参数:
    timeseries: 时间序列数据
    order: ARIMA模型的阶数 (p, d, q) 或SARIMA模型的阶数 (p, d, q, P, D, Q, s)
    
    返回:
    model_fit: 拟合后的模型
    model_summary: 模型摘要信息
    """
    try:
        # 确保原始数据为一维数组
        if isinstance(timeseries, pd.DataFrame):
            timeseries = timeseries.values.flatten()
        
        # 根据order的长度判断是ARIMA还是SARIMA
        if len(order) == 3:
            # 拟合ARIMA模型
            model = ARIMA(timeseries, order=order)
            model_fit = model.fit()
        else:
            # 拟合SARIMA模型
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            p, d, q, P, D, Q, s = order
            model = SARIMAX(
                timeseries, 
                order=(p, d, q), 
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
        
        # 获取模型摘要
        model_summary = {
            'AIC': model_fit.aic,
            'BIC': model_fit.bic,
            'params': model_fit.params.to_dict()
        }
        
        return model_fit, model_summary
    
    except Exception as e:
        st.error(f"ARIMA模型拟合失败: {str(e)}")
        # 如果失败，尝试使用更简单的模型
        try:
            model = ARIMA(timeseries, order=(0, 1, 1))
            model_fit = model.fit()
            
            model_summary = {
                'AIC': model_fit.aic,
                'BIC': model_fit.bic,
                'params': model_fit.params.to_dict(),
                'fallback': True
            }
            
            return model_fit, model_summary
        
        except Exception as e2:
            st.error(f"备选ARIMA模型拟合也失败: {str(e2)}")
            return None, None

def check_residuals(model):
    """
    分析ARIMA模型残差
    
    参数:
    model: 拟合后的ARIMA模型
    
    返回:
    dict: 残差诊断结果
    dict: 包含Streamlit可绘制的数据
    """
    if model is None:
        return None, None
    
    # 获取残差
    residuals = model.resid
    
    # Durbin-Watson检验（自相关性）
    dw_stat = durbin_watson(residuals)
    if dw_stat < 1.5:
        dw_conclusion = "存在正自相关"
    elif dw_stat > 2.5:
        dw_conclusion = "存在负自相关"
    else:
        dw_conclusion = "无自相关"
    
    # Jarque-Bera检验（正态性）
    jb_test = stats.jarque_bera(residuals)
    jb_stat = jb_test[0]
    jb_pvalue = jb_test[1]
    jb_conclusion = "正态分布" if jb_pvalue > 0.05 else "非正态分布"
    
    # Ljung-Box检验（白噪声检验）
    lb_test = acorr_ljungbox(residuals, lags=[10])
    lb_pvalue = lb_test['lb_pvalue'].values[0]
    lb_conclusion = "残差为白噪声" if lb_pvalue > 0.05 else "残差不是白噪声"
    
    # 创建残差诊断结果字典
    diagnostics = {
        'Durbin-Watson检验': {
            '统计量': dw_stat,
            '结论': dw_conclusion
        },
        'Jarque-Bera检验': {
            '统计量': jb_stat,
            'p值': jb_pvalue,
            '结论': jb_conclusion
        },
        'Ljung-Box检验': {
            'p值': lb_pvalue,
            '结论': lb_conclusion
        }
    }
    
    # 准备Streamlit图表数据
    # 1. 残差时间序列
    residual_series_df = pd.DataFrame({
        '残差': residuals
    })
    
    # 2. 计算ACF值用于自相关图
    acf_values = acf(residuals, nlags=20)
    acf_df = pd.DataFrame({
        'ACF': acf_values
    })
    
    # 准备置信区间数据
    conf_level = 1.96 / np.sqrt(len(residuals))
    confidence_df = pd.DataFrame({
        '上限': [conf_level] * len(acf_values),
        '下限': [-conf_level] * len(acf_values)
    })
    
    # 为QQ图准备数据
    theoretical_quantiles = np.random.normal(0, 1, len(residuals))
    theoretical_quantiles.sort()
    sample_quantiles = np.sort(residuals)
    
    qq_df = pd.DataFrame({
        '理论分位数': theoretical_quantiles,
        '样本分位数': sample_quantiles
    })
    
    # 将所有图表数据打包到一个字典中
    chart_data = {
        'residual_series': residual_series_df,
        'acf': acf_df,
        'confidence': confidence_df,
        'qq': qq_df
    }
    
    return diagnostics, chart_data

def forecast_arima(model, train_data, steps=10):
    """
    使用ARIMA或SARIMA模型进行预测
    
    参数:
    model: 拟合的ARIMA或SARIMA模型
    train_data: 用于训练的数据（用于获取日期索引）
    steps: 预测步数
    
    返回:
    forecast_results: 预测结果字典
    forecast_df: 包含Streamlit可绘制的预测数据的DataFrame
    """
    if model is None:
        return None, None
    
    # 获取原始数据日期索引（如果有）
    if isinstance(train_data, pd.Series) and isinstance(train_data.index, pd.DatetimeIndex):
        has_date_index = True
        date_index = train_data.index
    else:
        has_date_index = False
    
    # 进行预测
    forecast = model.forecast(steps=steps)
    
    # 构建预测结果的DataFrame
    if has_date_index:
        # 如果有日期索引，继续延伸日期
        last_date = date_index[-1]
        if hasattr(date_index, 'freq') and date_index.freq is not None:
            freq = date_index.freq
        else:
            # 尝试推断频率
            freq = pd.infer_freq(date_index)
            if freq is None:
                # 如果无法推断，使用日频率
                freq = 'D'
        
        # 创建预测日期
        forecast_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        # 创建包含预测结果的DataFrame
        forecast_df = pd.DataFrame({
            '历史数据': pd.Series(train_data.values, index=date_index),
            '预测值': pd.Series(forecast.predicted_mean.values, index=forecast_dates),
            '95%置信区间下限': pd.Series(forecast.conf_int().iloc[:, 0].values, index=forecast_dates),
            '95%置信区间上限': pd.Series(forecast.conf_int().iloc[:, 1].values, index=forecast_dates)
        })
    else:
        # 如果没有日期索引，使用数字索引
        train_index = range(len(train_data))
        forecast_index = range(len(train_data), len(train_data) + steps)
        
        # 创建包含预测结果的DataFrame
        forecast_df = pd.DataFrame({
            '历史数据': pd.Series(train_data, index=train_index),
            '预测值': pd.Series(forecast.predicted_mean.values, index=forecast_index),
            '95%置信区间下限': pd.Series(forecast.conf_int().iloc[:, 0].values, index=forecast_index),
            '95%置信区间上限': pd.Series(forecast.conf_int().iloc[:, 1].values, index=forecast_index)
        })
    
    # 返回预测结果
    forecast_results = {
        'forecast_mean': forecast.predicted_mean,
        'lower_ci': forecast.conf_int().iloc[:, 0],
        'upper_ci': forecast.conf_int().iloc[:, 1]
    }
    
    return forecast_results, forecast_df

def plot_to_base64(fig):
    """
    将matplotlib图形转换为base64编码的字符串
    
    参数:
    fig: matplotlib图形
    
    返回:
    base64编码的图像字符串
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

def evaluate_arima_model(test_data, forecast_values, train_data=None):
    """
    评估ARIMA模型性能
    
    参数:
    test_data: 测试集数据
    forecast_values: 模型预测值
    train_data: 训练集数据（用于计算AIC和BIC，可选）
    
    返回:
    metrics: 评估指标字典
    """
    if test_data is None or forecast_values is None or len(test_data) == 0:
        return None
    
    # 确保数据类型一致，并转换为numpy数组
    if isinstance(test_data, pd.Series):
        test_data = test_data.values
    if isinstance(forecast_values, pd.Series):
        forecast_values = forecast_values.values
    
    # 如果长度不同，截断到相同长度
    min_len = min(len(test_data), len(forecast_values))
    test_data = test_data[:min_len]
    forecast_values = forecast_values[:min_len]
    
    # 计算评估指标
    # MSE - 均方误差
    mse = np.mean((forecast_values - test_data) ** 2)
    # RMSE - 均方根误差
    rmse = np.sqrt(mse)
    # MAE - 平均绝对误差
    mae = np.mean(np.abs(forecast_values - test_data))
    
    # MAPE - 平均绝对百分比误差
    # 避免除以零
    mape = np.mean(np.abs((test_data - forecast_values) / np.where(test_data != 0, test_data, 1))) * 100
    
    # 创建评估指标字典
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    # 如果提供了训练数据和拟合模型，添加AIC和BIC
    if train_data is not None and hasattr(train_data, 'model'):
        metrics['AIC'] = train_data.model.aic
        metrics['BIC'] = train_data.model.bic
    
    return metrics

# 逆差分
def inverse_diff(original_data, diff_df, d=1, log_diff=False):
    """
    对差分序列进行逆差分操作，还原为原始数据尺度
    
    参数:
    original_data: 原始时间序列（用于获取初始值）
    diff_df: 包含差分数据的DataFrame
    d: 差分阶数
    log_diff: 是否进行对数逆差分
    
    返回:
    pd.DataFrame: 逆差分后的DataFrame
    """
    # 创建结果的副本
    result_df = diff_df.copy()
    
    # 获取原始序列的前d个值
    if isinstance(original_data, pd.Series):
        init_values = original_data[:d].values
    else:
        init_values = np.array(original_data[:d])
    
    # 确保有足够的初始值
    if len(init_values) < d:
        raise ValueError(f"原始数据长度不足，需要至少{d}个初始值进行逆差分")
    
    # 对每个列进行逆差分
    for col in result_df.columns:
        # 跳过非数值列
        if not pd.api.types.is_numeric_dtype(result_df[col]):
            continue
        
        # 获取当前列的数据
        series = result_df[col].dropna()
        
        # 执行逆差分
        if d > 0:
            # 针对普通差分
            if not log_diff:
                # 一阶差分的逆操作
                if d == 1:
                    value = init_values[0]
                    undiff_values = []
                    
                    for val in series:
                        if not pd.isna(val):
                            value = value + val
                            undiff_values.append(value)
                    
                    # 更新DataFrame中的值
                    result_df.loc[series.index, col] = undiff_values
                
                # 二阶差分的逆操作
                elif d == 2:
                    value1 = init_values[0]
                    value2 = init_values[1]
                    undiff_values = []
                    
                    for val in series:
                        if not pd.isna(val):
                            new_value = val + 2 * value2 - value1
                            undiff_values.append(new_value)
                            value1 = value2
                            value2 = new_value
                    
                    # 更新DataFrame中的值
                    result_df.loc[series.index, col] = undiff_values
            
            # 针对对数差分
            else:
                value = init_values[0]
                undiff_values = []
                
                if d == 1:
                    for val in series:
                        if not pd.isna(val):
                            # 对数差分的逆操作: e^(log(prev) + diff)
                            value = np.exp(np.log(value) + val)
                            undiff_values.append(value)
                else:
                    # 多阶对数差分的逆操作更复杂，可能需要更专业的实现
                    st.warning("多阶对数差分的逆变换可能不准确")
                    
                    value1 = init_values[0]
                    value2 = init_values[1]
                    
                    for val in series:
                        if not pd.isna(val):
                            # 这是简化的实现，可能需要改进
                            diff1 = np.log(value2) - np.log(value1)
                            log_value = np.log(value2) + diff1 + val
                            new_value = np.exp(log_value)
                            undiff_values.append(new_value)
                            value1 = value2
                            value2 = new_value
                
                # 更新DataFrame中的值
                result_df.loc[series.index, col] = undiff_values
    
    return result_df

def create_timeseries_chart(df, title='时间序列图', series_names=None):
    """
    创建时间序列折线图的echarts选项
    
    参数:
    df: DataFrame，包含要绘制的时间序列数据
    title: 图表标题
    series_names: 要绘制的列名列表，如果为None则绘制所有列
    
    返回:
    dict: echarts图表选项
    """
    # 如果输入是Series，转换为DataFrame
    if isinstance(df, pd.Series):
        df = pd.DataFrame({df.name if df.name else "值": df})
    
    # 如果没有指定要绘制的列，则使用所有列
    if series_names is None:
        series_names = df.columns.tolist()
    
    # 检查df的索引类型，决定x轴类型
    is_date_index = isinstance(df.index, pd.DatetimeIndex)
    
    # 准备x轴数据
    if is_date_index:
        # 根据数据长度选择合适的日期格式
        if len(df) > 365:  # 超过一年的数据
            date_format = '%Y-%m'  # 年-月
        elif len(df) > 31:  # 超过一个月的数据
            date_format = '%m-%d'  # 月-日
        else:
            date_format = '%m-%d %H:%M'  # 月-日 时:分
        
        x_data = df.index.strftime(date_format).tolist()
        x_axis_type = 'category'
    else:
        # 尝试将索引转换为日期时间类型
        try:
            # 检查索引是否可以转换为日期时间
            date_index = pd.to_datetime(df.index)
            # 如果可以，使用转换后的日期索引
            if len(df) > 365:  # 超过一年的数据
                date_format = '%Y-%m'  # 年-月
            elif len(df) > 31:  # 超过一个月的数据
                date_format = '%m-%d'  # 月-日
            else:
                date_format = '%m-%d %H:%M'  # 月-日 时:分
            
            x_data = date_index.strftime(date_format).tolist()
            x_axis_type = 'category'
        except:
            # 如果无法转换，使用数字索引
            x_data = list(range(len(df)))
            x_axis_type = 'value'
    
    # 准备系列数据
    series = []
    for name in series_names:
        if name in df.columns:
            # 确保数据是数值类型
            series_data = df[name]
            if not pd.api.types.is_numeric_dtype(series_data):
                if pd.api.types.is_datetime64_any_dtype(series_data):
                    # 将日期时间转换为时间戳
                    series_data = (series_data - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
                else:
                    # 跳过非数值类型的列
                    continue
            
            # 处理系列配置
            series_item = {
                'name': name,
                'type': 'line',
                'data': series_data.fillna("").tolist(),  # 填充NaN为空字符串
                'showSymbol': False,  # 隐藏线上的点，使图表更干净
                'smooth': True,       # 使线条平滑
            }
            series.append(series_item)
    
    # 创建echarts选项
    option = {
        'title': {
            'text': title,
            'left': 'center'
        },
        'tooltip': {
            "position": "bottom",
            'trigger': 'axis'
        },
        'legend': {
            'data': [name for name in series_names if name in df.columns],
            'top': 'bottom'
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '13%',
            'containLabel': True
        },
        'xAxis': {
            'type': x_axis_type,
            'data': x_data,
            'boundaryGap': False,
            'axisLabel': {
                'rotate': 45 if len(x_data) > 10 else 0  # 当数据点较多时旋转标签
            }
        },
        'yAxis': {
            'type': 'value',
            'scale': True,
            'name': name if len(series_names) == 1 else ''
        },
        'series': series,
        'dataZoom': [
            {
                'type': 'slider',
                'show': len(df) > 30,  # 当数据点较多时显示缩放控件
                'start': 0,
                'end': 100
            }
        ]
    }
    
    return option

# 创建ECharts分布直方图函数，带有正态拟合线
def create_histogram_chart(series, title='分布直方图', bins=30):
    """
    创建直方图的echarts选项
    
    参数:
    series: Series，要绘制直方图的数据
    title: 图表标题
    bins: 直方图的分箱数
    
    返回:
    dict: echarts图表选项
    """
    # 确保数据是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 如果是日期时间类型，转换为时间戳
        if pd.api.types.is_datetime64_any_dtype(series):
            series = (series - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        else:
            raise TypeError(f"无法处理类型为 {series.dtype} 的数据，需要数值类型数据")
    
    # 移除缺失值
    series = series.dropna()
    
    # 计算直方图
    hist, bin_edges = np.histogram(series, bins=bins)
    
    # 准备x轴标签（使用分箱中点）
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]
    
    # 计算正态分布拟合曲线
    # 获取数据的均值和标准差
    mu = series.mean()
    sigma = series.std()
    
    # 计算理论正态分布的Y值
    # 首先计算每个箱子中心的概率密度函数值
    pdf_values = stats.norm.pdf(bin_centers, mu, sigma)
    
    # 将PDF值转换为与直方图对应的频数
    # 需要乘以数据总数和箱宽度来与直方图高度匹配
    bin_width = bin_edges[1] - bin_edges[0]
    pdf_heights = pdf_values * len(series) * bin_width
    
    # 创建echarts选项
    option = {
        'title': {
            'text': title,
            'left': 'center'
        },
        'tooltip': {
            "position": "bottom",
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '8%',
            'containLabel': True
        },
        'xAxis': {
            'type': 'category',
            'data': bin_labels,
            'axisTick': {
                'alignWithLabel': True
            },
            'axisLabel': {
                'rotate': 45,
                'interval': 'auto'
            }
        },
        'yAxis': {
            'type': 'value',
            'name': '频数'
        },
        'series': [
            {
                'name': '频数',
                'type': 'bar',
                'data': hist.tolist(),
                'barWidth': '90%'
            },
            {
                'name': '正态分布拟合',
                'type': 'line',
                'smooth': True,
                'data': list(zip(range(len(bin_centers)), pdf_heights.tolist())),
                'showSymbol': False,
                'lineStyle': {
                    'color': '#FF0000',
                    'width': 2
                },
                'z': 10
            }
        ]
    }
    
    return option

def create_qq_plot(series, title='Q-Q图'):
    """
    创建QQ图的echarts选项
    
    参数:
    series: Series，要绘制QQ图的数据
    title: 图表标题
    
    返回:
    dict: echarts图表选项
    """
    # 确保数据是数值类型
    if not pd.api.types.is_numeric_dtype(series):
        # 如果是日期时间类型，转换为时间戳
        if pd.api.types.is_datetime64_any_dtype(series):
            series = (series - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        else:
            raise TypeError(f"无法处理类型为 {series.dtype} 的数据，需要数值类型数据")
    
    # 移除缺失值和无穷大值
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # 如果数据量太少，无法绘制QQ图
    if len(series) < 3:
        raise ValueError("数据点不足，无法绘制QQ图，至少需要3个数据点")

    # 检查数据是否有足够的变异性
    if series.std() == 0:
        raise ValueError("数据没有变异性(标准差为0)，无法绘制QQ图")
    
    try:
        # 准备QQ图数据
        # 排序数据
        sorted_data = np.sort(series)
        n = len(sorted_data)
        
        # 计算理论分位数
        # 生成分位点
        p = np.arange(1, n + 1) / (n + 1)
        theoretical_quantiles = stats.norm.ppf(p)
        
        # 创建散点图数据
        data = list(zip(theoretical_quantiles.tolist(), sorted_data.tolist()))
        
        # 计算理论正态分布线的起点和终点
        min_x = min(theoretical_quantiles)
        max_x = max(theoretical_quantiles)
        min_y = np.mean(sorted_data) + np.std(sorted_data) * min_x
        max_y = np.mean(sorted_data) + np.std(sorted_data) * max_x
        
        # 创建echarts选项
        option = {
            'title': {
                'text': title,
                'left': 'center'
            },
            'tooltip': {
                "position": "bottom",
                'trigger': 'item'
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '8%',
                'containLabel': True
            },
            'xAxis': {
                'type': 'value',
                'name': '理论分位数',
                'scale': True
            },
            'yAxis': {
                'type': 'value',
                'name': '样本分位数',
                'scale': True
            },
            'series': [
                {
                    'name': 'QQ图',
                    'type': 'scatter',
                    'data': data,
                    'itemStyle': {
                        'opacity': 0.7
                    }
                },
                {
                    'name': '理论正态线',
                    'type': 'line',
                    'showSymbol': False,
                    'data': [[min_x, min_y], [max_x, max_y]],
                    'lineStyle': {
                        'color': '#FF0000',
                        'width': 2
                    }
                }
            ]
        }
        
        return option
    except Exception as e:
        # 捕获所有可能的异常，包括计算理论分位数时可能出现的问题
        raise ValueError(f"生成QQ图时出错: {str(e)}") 