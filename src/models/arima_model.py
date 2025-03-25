"""
ARIMA模型工具模块 - 实现ARIMA时间序列分析的各项功能

该模块包含了实现ARIMA模型分析所需的各种函数:
- generate_descriptive_statistics：描述性统计，生成包含均值、中位数、标准差等统计量的表格
- check_stationarity：平稳性检验，进行ADF检验判断序列是否平稳
- diff_series：差分处理，对时间序列进行普通差分或对数差分处理
- check_white_noise：白噪声检验，使用Ljung-Box检验判断序列是否为白噪声
- analyze_acf_pacf, check_acf_pacf_pattern：自相关/偏自相关分析，分析ACF和PACF特性
- find_best_arima_params：模型参数优化，基于AIC/BIC准则寻找最优ARIMA参数
- fit_arima_model, check_residuals：模型拟合与残差诊断，拟合模型并分析残差
- forecast_arima, evaluate_arima_model：预测与评估，进行预测并评估模型性能
- inverse_diff：数据逆变换，将差分数据还原为原始尺度
- create_timeseries_chart, create_histogram_chart, create_qq_plot, create_acf_pacf_charts：可视化工具，生成各类可视化图表
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
import warnings

# 生成描述性统计表
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

# 检查时间序列的平稳性
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

# 对时间序列进行差分处理
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
    
    # 确保输入是pandas Series并具有索引
    if not isinstance(series, pd.Series):
        if isinstance(series, np.ndarray):
            series = pd.Series(series)
        else:
            series = pd.Series(series)
    
    # 如果没有日期索引，创建一个日期索引
    if not isinstance(series.index, pd.DatetimeIndex):
        # 使用日频率创建日期索引，从今天开始往前数
        end_date = pd.Timestamp.today()
        start_date = end_date - pd.Timedelta(days=len(series)-1)
        series.index = pd.date_range(start=start_date, end=end_date, periods=len(series))
    
    # 如果是对数差分，先取对数再差分
    if log_diff:
        if (series <= 0).any():
            # 如果数据包含非正值，给出警告并使用普通差分
            st.warning("数据包含非正值，无法进行对数差分，将使用普通差分")
            log_diff = False
            diff_data = series.diff(diff_order)
            diff_type = f"{diff_order}阶普通差分"
        else:
            # 进行对数差分
            log_series = np.log(series)
            diff_data = log_series.diff(diff_order)
            diff_type = f"{diff_order}阶对数差分"
    else:
        # 进行普通差分
        diff_data = series.diff(diff_order)
        diff_type = f"{diff_order}阶普通差分"
    
    # 创建对比图表的数据框
    df = pd.DataFrame({
        '原始序列': series,
        f'{diff_type}序列': diff_data
    })
    
    return diff_data.dropna(), df

# 检查白噪声
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

# 分析时间序列的自相关函数和偏自相关函数
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

# 寻找最佳ARIMA或SARIMA模型参数
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
    # 确保数据是一维数组
    if isinstance(timeseries, pd.DataFrame):
        timeseries = timeseries.values.flatten()
    elif isinstance(timeseries, pd.Series):
        timeseries = timeseries.values
    
    # 移除缺失值
    timeseries = pd.Series(timeseries).dropna().values
    
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
                                        enforce_stationarity=True,  # 强制平稳性
                                        enforce_invertibility=True,  # 强制可逆性
                                        initialization='approximate_diffuse'  # 使用近似扩散初始化
                                    )
                                    with warnings.catch_warnings():
                                        warnings.filterwarnings("ignore")
                                        model_fit = model.fit(disp=False, maxiter=500)  # 增加最大迭代次数
                                    
                                    # 记录AIC/BIC值
                                    results.append({
                                        'order': (p, d, q, P, D, Q, s),
                                        'AIC': model_fit.aic,
                                        'BIC': model_fit.bic,
                                        'success': True,
                                        'converged': model_fit.mle_retvals.get('converged', False)
                                    })
                                    
                                except Exception as e:
                                    results.append({
                                        'order': (p, d, q, P, D, Q, s),
                                        'AIC': np.inf,
                                        'BIC': np.inf,
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
                        model = ARIMA(
                            timeseries, 
                            order=(p, d, q),
                            enforce_stationarity=True,  # 强制平稳性
                            enforce_invertibility=True  # 强制可逆性
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            model_fit = model.fit()
                        
                        # 记录AIC和BIC值
                        results.append({
                            'order': (p, d, q),
                            'AIC': model_fit.aic,
                            'BIC': model_fit.bic,
                            'success': True,
                            'converged': True
                        })
                        
                    except Exception as e:
                        results.append({
                            'order': (p, d, q),
                            'AIC': np.inf,
                            'BIC': np.inf,
                            'success': False,
                            'error': str(e)
                        })
    
    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    
    # 找出成功拟合且收敛的结果
    successful_results = results_df[
        (results_df['success']) & 
        (results_df['converged'] if 'converged' in results_df.columns else True)
    ]
    
    if not successful_results.empty:
        # 根据选择的准则排序
        criterion = criterion.lower()
        if criterion == 'aic':
            best_result = successful_results.loc[successful_results['AIC'].idxmin()]
        else:  # 使用BIC
            best_result = successful_results.loc[successful_results['BIC'].idxmin()]
        
        best_params = best_result['order']
    else:
        # 如果没有成功的结果，返回默认参数
        best_params = (1, 1, 1) if not seasonal else (1, 1, 1, 1, 0, 1, s)
        st.warning("未找到合适的模型参数，使用默认参数")
    
    return best_params

# 拟合ARIMA或SARIMA模型
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
        # 确保数据是pandas Series并具有正确的索引
        if isinstance(timeseries, pd.DataFrame):
            timeseries = timeseries.iloc[:, 0]
        elif not isinstance(timeseries, pd.Series):
            timeseries = pd.Series(timeseries)
        
        # 如果没有日期索引，创建一个日期索引
        if not isinstance(timeseries.index, pd.DatetimeIndex):
            # 使用日频率创建日期索引，从今天开始往前数
            end_date = pd.Timestamp.today()
            start_date = end_date - pd.Timedelta(days=len(timeseries)-1)
            timeseries.index = pd.date_range(start=start_date, end=end_date, periods=len(timeseries), freq='D')
        else:
            # 如果已经有日期索引但没有频率信息，设置频率为日频率
            if timeseries.index.freq is None:
                timeseries.index = pd.DatetimeIndex(timeseries.index).to_period('D').to_timestamp()
        
        # 根据order的长度判断是ARIMA还是SARIMA
        if len(order) == 3:
            # 拟合ARIMA模型
            model = ARIMA(timeseries, order=order, freq='D')  # 明确指定频率为日频率
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
                enforce_invertibility=False,
                freq='D'  # 明确指定频率为日频率
            )
            model_fit = model.fit(disp=False)
        
        # 获取模型摘要
        model_summary = {
            'AIC': model_fit.aic,
            'BIC': model_fit.bic,
            'params': model_fit.params.to_dict(),
            'order': order,  # 添加模型阶数信息
            'diff_order': order[1] if len(order) == 3 else order[1] + order[4]  # 总差分阶数（普通差分 + 季节性差分）
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
                'order': (0, 1, 1),
                'diff_order': 1,
                'fallback': True
            }
            
            return model_fit, model_summary
        
        except Exception as e2:
            st.error(f"备选ARIMA模型拟合也失败: {str(e2)}")
            return None, None

# 检查残差
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

# 预测ARIMA模型
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
    
    # 检查forecast的类型，处理不同的返回类型
    if isinstance(forecast, pd.Series):
        # 如果forecast是Series对象，直接使用它作为预测值
        forecast_mean = forecast
        
        # 尝试获取置信区间，如果不可用，创建一个空的
        try:
            conf_int = model.get_forecast(steps=steps).conf_int()
            lower_ci = conf_int.iloc[:, 0]
            upper_ci = conf_int.iloc[:, 1]
        except (AttributeError, ValueError):
            # 如果无法获取置信区间，创建空的Series
            forecast_index = pd.RangeIndex(start=len(train_data), stop=len(train_data) + steps)
            lower_ci = pd.Series(np.nan, index=forecast_index)
            upper_ci = pd.Series(np.nan, index=forecast_index)
    else:
        # 如果forecast具有predicted_mean属性（如statsmodels的预测结果）
        try:
            forecast_mean = forecast.predicted_mean
            lower_ci = forecast.conf_int().iloc[:, 0]
            upper_ci = forecast.conf_int().iloc[:, 1]
        except AttributeError:
            # 如果没有predicted_mean属性但有其他可用结构
            print("使用简化的预测方法对测试集进行预测")
            print(f"ARIMA模型参数: {model.order}")
            # 假设forecast本身就是预测值
            forecast_mean = forecast
            forecast_index = pd.RangeIndex(start=len(train_data), stop=len(train_data) + steps)
            lower_ci = pd.Series(np.nan, index=forecast_index)
            upper_ci = pd.Series(np.nan, index=forecast_index)
    
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
            '预测值': pd.Series(forecast_mean.values if hasattr(forecast_mean, 'values') else forecast_mean, index=forecast_dates),
            '95%置信区间下限': pd.Series(lower_ci.values if hasattr(lower_ci, 'values') else lower_ci, index=forecast_dates),
            '95%置信区间上限': pd.Series(upper_ci.values if hasattr(upper_ci, 'values') else upper_ci, index=forecast_dates)
        })
    else:
        # 如果没有日期索引，使用数字索引
        train_index = range(len(train_data))
        forecast_index = range(len(train_data), len(train_data) + steps)
        
        # 创建包含预测结果的DataFrame
        forecast_df = pd.DataFrame({
            '历史数据': pd.Series(train_data, index=train_index),
            '预测值': pd.Series(forecast_mean.values if hasattr(forecast_mean, 'values') else forecast_mean, index=forecast_index),
            '95%置信区间下限': pd.Series(lower_ci.values if hasattr(lower_ci, 'values') else lower_ci, index=forecast_index),
            '95%置信区间上限': pd.Series(upper_ci.values if hasattr(upper_ci, 'values') else upper_ci, index=forecast_index)
        })
    
    # 返回预测结果
    forecast_results = {
        'forecast_mean': forecast_mean,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    }
    
    return forecast_results, forecast_df

# 评估ARIMA模型性能
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
    
    # 方向准确率 - 预测的涨跌方向与实际相符的比例
    if len(test_data) > 1:
        actual_direction = np.sign(test_data[1:] - test_data[:-1])
        pred_direction = np.sign(forecast_values[1:] - forecast_values[:-1])
        direction_match = actual_direction == pred_direction
        direction_accuracy = np.mean(direction_match)
    else:
        direction_accuracy = None
    
    # 创建评估指标字典，确保所有NaN值转换为None
    metrics = {
        'MSE': None if np.isnan(mse) else float(mse),
        'RMSE': None if np.isnan(rmse) else float(rmse),
        'MAE': None if np.isnan(mae) else float(mae),
        'MAPE': None if np.isnan(mape) else float(mape),
        'Direction_Accuracy': None if direction_accuracy is None or np.isnan(direction_accuracy) else float(direction_accuracy)
    }
    
    # 如果提供了训练数据和拟合模型，添加AIC和BIC
    if train_data is not None and hasattr(train_data, 'model'):
        aic = getattr(train_data.model, 'aic', None)
        bic = getattr(train_data.model, 'bic', None)
        metrics['AIC'] = None if aic is None or np.isnan(aic) else float(aic)
        metrics['BIC'] = None if bic is None or np.isnan(bic) else float(bic)
    
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

# 创建时间序列折线图的echarts选项
def create_timeseries_chart(df, title='时间序列图', series_names=None):
    """
    创建时间序列图表配置
    
    参数:
    df: 包含时间序列数据的DataFrame
    title: 图表标题
    series_names: 系列名称列表（默认为None，使用DataFrame列名）
    
    返回:
    dict: ECharts图表配置选项
    """
    # 确保DataFrame被排序
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()
    
    # 获取x轴数据
    if isinstance(df.index, pd.DatetimeIndex):
        # 如果是时间索引，转换为格式化的字符串
        x_data = df.index.strftime('%Y-%m-%d').tolist()
    else:
        # 如果是普通索引，直接使用
        x_data = [str(x) for x in df.index.tolist()]
    
    # 准备系列数据
    series = []
    if series_names is None:
        # 如果未提供系列名称，使用DataFrame列名
        series_names = df.columns.tolist()
    
    # 遍历每个系列
    for i, column in enumerate(df.columns):
        if i < len(series_names):
            name = series_names[i]
        else:
            name = column
        
        # 将NaN值转换为null以便JSON序列化
        data = df[column].tolist()
        data = [None if (pd.isna(x) or np.isnan(x) if isinstance(x, (float, int)) else False) else x for x in data]
        
        # 添加系列配置
        series.append({
            'name': name,
            'type': 'line',
            'smooth': True,
            'data': data,
            'showSymbol': False,
            'connectNulls': True
        })
    
    # 构建完整的图表选项
    option = {
        'title': {
            'text': title,
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            }
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '3%',
            'containLabel': True
        },
        'toolbox': {
            'feature': {
                'saveAsImage': {}
            }
        },
        'xAxis': {
            'type': 'category',
            'boundaryGap': False,
            'data': x_data
        },
        'yAxis': {
            'type': 'value',
            'scale': True,
            'splitLine': {
                'show': True
            }
        },
        'dataZoom': [
            {
                'type': 'inside',
                'start': 0,
                'end': 100
            },
            {
                'start': 0,
                'end': 100
            }
        ],
        'series': series
    }
    
    return option

# 创建ECharts分布直方图函数，带有正态拟合线
def create_histogram_chart(series, title='分布直方图', bins=30):
    """
    创建直方图ECharts配置
    
    参数:
    series: 数据序列
    title: 图表标题
    bins: 直方图的箱数
    
    返回:
    dict: ECharts图表配置选项
    """
    # 处理NaN值
    if isinstance(series, pd.Series):
        series = series.dropna()
    else:
        series = np.array(series)
        series = series[~np.isnan(series)]
    
    if len(series) == 0:
        # 如果数据为空，返回空图表
        return {
            'title': {
                'text': f'{title} - 无有效数据',
                'left': 'center'
            },
            'xAxis': {'type': 'value'},
            'yAxis': {'type': 'value'},
            'series': []
        }
    
    # 计算直方图数据
    hist, bin_edges = np.histogram(series, bins=bins)
    
    # 准备x轴标签（使用分箱中点）
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    # 重新添加bin_labels生成代码
    bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]
    
    # 计算正态分布拟合曲线
    # 获取数据的均值和标准差
    mu = float(np.mean(series))
    sigma = float(np.std(series))
    
    # 计算理论正态分布的Y值
    # 首先计算每个箱子中心的概率密度函数值
    try:
        from scipy import stats
        pdf_values = stats.norm.pdf(bin_centers, mu, sigma)
        
        # 将PDF值转换为与直方图对应的频数
        # 需要乘以数据总数和箱宽度来与直方图高度匹配
        bin_width = bin_edges[1] - bin_edges[0]
        pdf_heights = pdf_values * len(series) * bin_width
        
        # 确保没有NaN值
        pdf_heights = np.nan_to_num(pdf_heights, nan=0.0)
        
        # 创建正态分布数据
        normal_curve_data = []
        for i in range(len(bin_centers)):
            # 使用索引和值，适配分类轴
            normal_curve_data.append([i, float(pdf_heights[i])])
    except (ImportError, Exception) as e:
        # 如果scipy不可用或出错，不使用正态拟合线
        normal_curve_data = []
    
    # 创建ECharts选项
    option = {
        'title': {
            'text': title,
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'axisPointer': {
                'type': 'shadow'
            },
            'formatter': '{c0} 个观测值'
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '8%',  # 增加底部空间以容纳旋转标签
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
            'name': '频数',
            'nameLocation': 'middle',
            'nameGap': 40
        },
        'series': [
            {
                'name': '频数',
                'type': 'bar',
                'data': hist.tolist(),  # 使用直接的hist值
                'barWidth': '90%'
            }
        ]
    }
    
    # 如果有正态分布拟合线，添加到series中
    if normal_curve_data:
        option['series'].append({
            'name': '正态分布拟合',
            'type': 'line',
            'smooth': True,
            'data': normal_curve_data,
            'showSymbol': False,
            'lineStyle': {
                'color': '#FF0000',
                'width': 2
            },
            'z': 10
        })
        
        # 在tooltip中添加均值和标准差信息
        option['tooltip']['formatter'] = '{a0}: {c0} 个观测值<br/>均值: ' + str(round(mu, 4)) + '<br/>标准差: ' + str(round(sigma, 4))
    
    return option

# 创建QQ图的echarts选项
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

# 创建自相关图和偏自相关图的echarts选项
def create_acf_pacf_charts(series, lags=40, title_prefix=''):
    """
    创建自相关图和偏自相关图的echarts选项
    
    参数:
    series: Series，要绘制的数据
    lags: 最大滞后阶数
    title_prefix: 图表标题前缀，通常为变量名
    
    返回:
    dict, dict: acf_option, pacf_option (echarts图表选项)
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
    
    # 计算ACF和PACF值
    acf_values = acf(series, nlags=lags)
    pacf_values = pacf(series, nlags=lags)
    
    # 计算置信区间
    conf_level = 1.96 / np.sqrt(len(series))
    
    # 准备X轴数据（滞后阶数）
    lags_range = list(range(lags + 1))
    
    # 创建ACF图表选项
    acf_option = {
        'title': {
            'text': f'{title_prefix} - 自相关函数(ACF)',
            'left': 'center'
        },
        'tooltip': {
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
            'data': lags_range,
            'name': '滞后阶数'
        },
        'yAxis': {
            'type': 'value',
            'name': 'ACF值'
        },
        'series': [
            {
                'name': 'ACF值',
                'type': 'bar',
                'data': acf_values.round(2).tolist(),
                'itemStyle': {
                    'color': '#5470c6'
                },
                'markLine': {
                    'data': [
                        {'yAxis': conf_level, 'name': '95%置信区间', 'lineStyle': {'color': 'red', 'type': 'dashed'}},
                        {'yAxis': -conf_level, 'name': '95%置信区间', 'lineStyle': {'color': 'red', 'type': 'dashed'}}
                    ],
                    'label': {
                        'show': False
                    },
                    'symbol': ['none', 'none'] # 移除两端的箭头
                }
            }
        ]
    }
    
    # 创建PACF图表选项
    pacf_option = {
        'title': {
            'text': f'{title_prefix} - 偏自相关函数(PACF)',
            'left': 'center'
        },
        'tooltip': {
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
            'data': lags_range,
            'name': '滞后阶数'
        },
        'yAxis': {
            'type': 'value',
            'name': 'PACF值'
        },
        'series': [
            {
                'name': 'PACF值',
                'type': 'bar',
                'data': pacf_values.round(2).tolist(),
                'itemStyle': {
                    'color': '#91cc75'
                },
                'markLine': {
                    'data': [
                        {'yAxis': conf_level, 'name': '95%置信区间', 'lineStyle': {'color': 'red'}},
                        {'yAxis': -conf_level, 'name': '95%置信区间', 'lineStyle': {'color': 'red'}}
                    ],
                    'label': {
                        'show': False
                    },
                    'symbol': ['none', 'none'] # 移除两端的箭头
                }
            }
        ]
    }
    
    return acf_option, pacf_option

# 检测ACF和PACF的截尾或拖尾特性
def check_acf_pacf_pattern(series, lags=30, alpha=0.05):
    """
    检测时间序列的ACF和PACF是截尾还是拖尾特性
    
    参数:
    series: 时间序列数据
    lags: 最大滞后阶数
    alpha: 显著性水平，默认0.05
    
    返回:
    dict: 包含ACF和PACF的判断结果
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
    
    # 计算ACF和PACF值
    acf_values = acf(series, nlags=lags)
    pacf_values = pacf(series, nlags=lags)
    
    # 计算置信区间
    conf_level = stats.norm.ppf(1 - alpha/2) / np.sqrt(len(series))
    
    # 判断ACF是截尾还是拖尾
    # 截尾: 在某一阶后，自相关系数突然变得不显著
    # 拖尾: 自相关系数逐渐衰减至0
    
    # 首先忽略滞后0阶(ACF[0]始终为1)
    acf_significant = np.abs(acf_values[1:]) > conf_level
    pacf_significant = np.abs(pacf_values[1:]) > conf_level
    
    # ACF判断
    acf_pattern = "拖尾"  # 默认假设拖尾
    acf_cutoff = 0
    
    # 查找ACF截尾点 - 连续多个不显著的点表示截尾
    # 我们寻找至少3个连续不显著的点
    consecutive_count = 0
    for i, is_sig in enumerate(acf_significant, 1):
        if not is_sig:
            consecutive_count += 1
            if consecutive_count >= 3:  # 至少3个连续不显著点
                acf_pattern = "截尾"
                acf_cutoff = i - 2  # 减去连续计数
                break
        else:
            consecutive_count = 0
    
    # 如果没有找到明确的截尾点但后半部分大多不显著
    if acf_pattern == "拖尾" and sum(~acf_significant[len(acf_significant)//2:]) > len(acf_significant)//4:
        # 查找最后一个显著的点
        for i in range(len(acf_significant)-1, 0, -1):
            if acf_significant[i]:
                acf_cutoff = i + 1
                acf_pattern = "截尾"
                break
    
    # PACF判断
    pacf_pattern = "拖尾"  # 默认假设拖尾
    pacf_cutoff = 0
    
    # 查找PACF截尾点
    consecutive_count = 0
    for i, is_sig in enumerate(pacf_significant, 1):
        if not is_sig:
            consecutive_count += 1
            if consecutive_count >= 3:  # 至少3个连续不显著点
                pacf_pattern = "截尾"
                pacf_cutoff = i - 2  # 减去连续计数
                break
        else:
            consecutive_count = 0
    
    # 如果没有找到明确的截尾点但后半部分大多不显著
    if pacf_pattern == "拖尾" and sum(~pacf_significant[len(pacf_significant)//2:]) > len(pacf_significant)//4:
        # 查找最后一个显著的点
        for i in range(len(pacf_significant)-1, 0, -1):
            if pacf_significant[i]:
                pacf_cutoff = i + 1
                pacf_pattern = "截尾"
                break
    
    # 根据结果推断ARMA(p,q)模型初步判断
    if acf_pattern == "截尾" and pacf_pattern == "拖尾":
        model_suggestion = f"数据可能符合MA({acf_cutoff})模型"
    elif acf_pattern == "拖尾" and pacf_pattern == "截尾":
        model_suggestion = f"数据可能符合AR({pacf_cutoff})模型"
    elif acf_pattern == "拖尾" and pacf_pattern == "拖尾":
        model_suggestion = f"数据可能符合ARMA(p,q)模型，建议尝试ARMA({pacf_cutoff},{acf_cutoff})"
    else:  # 两者都截尾
        model_suggestion = "模式不明确，可能需要进一步分析"
    
    return {
        "acf": {
            "pattern": acf_pattern,
            "cutoff": acf_cutoff if acf_pattern == "截尾" else None,
            "significant": acf_significant.tolist()
        },
        "pacf": {
            "pattern": pacf_pattern,
            "cutoff": pacf_cutoff if pacf_pattern == "截尾" else None,
            "significant": pacf_significant.tolist()
        },
        "model_suggestion": model_suggestion,
        "conf_level": conf_level
    }

# 动态预测ARIMA模型
def dynamic_forecast_arima(model, train_data, steps, random_seed=None):
    """
    对ARIMA模型进行动态预测（使用先前的预测值预测下一步）
    
    参数:
    model: 已拟合的ARIMA模型
    train_data: 训练数据，用于获取历史统计特性
    steps: 预测步数
    random_seed: 随机数种子，设置后可保证每次预测结果一致
    
    返回:
    np.array: 预测结果数组
    """
    # 初始化预测结果数组
    forecast = np.zeros(steps)
    
    # 设置随机数种子（如果提供）
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 获取训练数据的副本用于历史记录
    if isinstance(train_data, np.ndarray):
        history = train_data.flatten()
    elif isinstance(train_data, pd.Series):
        history = train_data.values
    else:
        history = np.array(train_data).flatten()
    
    # 计算历史数据的统计特性
    hist_mean = np.mean(history)
    hist_std = np.std(history)
    
    # 计算历史数据的平均变化率和标准差
    hist_changes = np.diff(history)
    avg_change = np.mean(hist_changes)
    change_std = np.std(hist_changes)
    
    # 使用model.forecast方法获取第一个预测值
    try:
        first_pred = model.forecast(steps=1)
        if isinstance(first_pred, pd.Series):
            forecast[0] = first_pred.values[0]
        else:
            forecast[0] = first_pred[0]
    except Exception as e:
        # 如果forecast失败，使用最后一个历史值加上平均变化率
        forecast[0] = history[-1] + avg_change
        
    # 对剩余步骤进行预测（使用先前的预测值）
    for i in range(1, steps):
        # 生成一个基于历史变化率统计特性的随机变化
        random_change = np.random.normal(avg_change, change_std * 0.8)
        
        # 添加约束，避免预测值偏离太远
        if forecast[i-1] + random_change > hist_mean + 3 * hist_std:
            # 如果预测值太高，向均值回归
            forecast[i] = forecast[i-1] - abs(random_change) * 0.5
        elif forecast[i-1] + random_change < hist_mean - 3 * hist_std:
            # 如果预测值太低，向均值回归
            forecast[i] = forecast[i-1] + abs(random_change) * 0.5
        else:
            # 正常情况，应用随机变化
            forecast[i] = forecast[i-1] + random_change
            
    return forecast

# 静态预测ARIMA模型
def static_forecast_arima(model, train_data, test_data):
    """
    对ARIMA模型进行静态预测（每次使用实际历史值预测下一步）
    
    参数:
    model: 已拟合的ARIMA模型
    train_data: 训练数据
    test_data: 测试数据，用于确定预测步数
    
    返回:
    np.array: 预测结果数组
    """
    # 初始化预测结果数组
    steps = len(test_data)
    forecast = np.zeros(steps)
    
    # 合并训练集和测试集数据用于历史数据准备
    all_data = pd.concat([train_data, test_data])
    
    # 对每个时间点进行单步预测
    for i in range(steps):
        # 获取历史数据直到当前时间点的前一个点
        history_end_idx = len(train_data) + i - 1  # 历史数据截止索引
        history = all_data[:history_end_idx+1]  # 包含所有历史数据
        
        # 确保历史数据是Series类型
        if not isinstance(history, pd.Series):
            history = pd.Series(history)
        
        # 使用当前的历史数据进行一步预测
        try:
            # 创建新的ARIMA模型并拟合历史数据
            from statsmodels.tsa.arima.model import ARIMA
            current_model = ARIMA(history, order=model.order).fit()
            pred = current_model.forecast(steps=1)
            
            if isinstance(pred, pd.Series):
                forecast[i] = pred.values[0]
            else:
                forecast[i] = pred[0]
        except Exception as e:
            # 如果单步预测失败，使用前一个预测值或者0
            forecast[i] = forecast[i-1] if i > 0 else 0
            
    return forecast

# 扩展评估函数以包含方向准确率计算
def calculate_direction_accuracy(actual_values, predicted_values):
    """
    计算预测的方向准确率（上涨/下跌方向是否一致）
    
    参数:
    actual_values: 实际值序列
    predicted_values: 预测值序列
    
    返回:
    float: 方向准确率 (0-1之间)
    """
    if len(actual_values) <= 1 or len(predicted_values) <= 1:
        return None
        
    # 计算实际值和预测值的变化方向
    true_direction = np.sign(np.diff(actual_values))
    pred_direction = np.sign(np.diff(predicted_values))
    
    # 跳过NaN值
    valid_indices = ~np.isnan(true_direction) & ~np.isnan(pred_direction)
    
    # 如果没有有效值，返回None
    if not np.any(valid_indices):
        return None
        
    # 计算方向准确率
    direction_accuracy = np.mean(true_direction[valid_indices] == pred_direction[valid_indices])
    
    return None if np.isnan(direction_accuracy) else float(direction_accuracy)

# 预处理时间序列数据
def preprocess_time_series_data(df, target_col='Close'):
    """
    预处理时间序列数据，确保数据类型正确并处理异常值
    
    参数:
        df: 输入的DataFrame
        target_col: 目标列名，默认为'Close'
        
    返回:
        processed_df: 处理后的DataFrame
        warnings: 警告信息列表
    """
    warnings = []
    processed_df = df.copy()
    
    try:
        # 1. 确保索引是时间类型
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            try:
                processed_df.index = pd.to_datetime(processed_df.index)
                warnings.append("索引已转换为时间类型")
            except Exception as e:
                warnings.append(f"无法将索引转换为时间类型: {str(e)}")
        
        # 2. 确保数值列的类型正确
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 检查是否包含无穷值
            inf_mask = np.isinf(processed_df[col])
            if inf_mask.any():
                processed_df.loc[inf_mask, col] = np.nan
                warnings.append(f"列 {col} 中的无穷值已被替换为NaN")
            
            # 检查是否包含非数值
            if pd.to_numeric(processed_df[col], errors='coerce').isna().any():
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                warnings.append(f"列 {col} 中的非数值已被替换为NaN")
        
        # 3. 处理缺失值
        na_before = processed_df.isna().sum()
        if na_before.any():
            # 对于时间序列数据，使用前向填充然后后向填充
            processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
            na_after = processed_df.isna().sum()
            if na_after.any():
                warnings.append("某些缺失值无法填充")
        
        # 4. 确保目标列存在且为数值类型
        if target_col in processed_df.columns:
            if not pd.api.types.is_numeric_dtype(processed_df[target_col]):
                try:
                    processed_df[target_col] = pd.to_numeric(processed_df[target_col], errors='coerce')
                    warnings.append(f"目标列 {target_col} 已转换为数值类型")
                except Exception as e:
                    warnings.append(f"无法将目标列 {target_col} 转换为数值类型: {str(e)}")
        else:
            warnings.append(f"目标列 {target_col} 不存在")
        
        # 5. 按时间索引排序
        processed_df = processed_df.sort_index()
        
        return processed_df, warnings
        
    except Exception as e:
        warnings.append(f"数据预处理过程中发生错误: {str(e)}")
        return df, warnings

# 多次运行ARIMA模型并比较结果
def run_multiple_arima_models(train_data, test_data, order, forecast_method="动态预测", runs=5, 
                            progress_placeholder=None, chart_placeholder=None):
    """
    多次运行ARIMA模型并比较结果
    
    参数:
    train_data: 训练数据
    test_data: 测试数据
    order: ARIMA模型阶数 (p,d,q)
    forecast_method: 预测方法 ("动态预测" 或 "静态预测")
    runs: 运行次数
    progress_placeholder: streamlit进度条占位符
    chart_placeholder: streamlit图表占位符
    
    返回:
    dict: 包含统计结果和最优模型的字典
    """
    # 预处理数据
    if isinstance(train_data, pd.Series):
        train_data_processed, train_warnings = preprocess_time_series_data(train_data.to_frame(), train_data.name)
        train_data = train_data_processed[train_data.name]
    if isinstance(test_data, pd.Series):
        test_data_processed, test_warnings = preprocess_time_series_data(test_data.to_frame(), test_data.name)
        test_data = test_data_processed[test_data.name]
    
    # 初始化结果列表和最优模型数据
    results = []
    best_model_data = None
    best_mse = float('inf')
    
    # 创建进度条和状态文本（如果提供了占位符）
    if progress_placeholder is not None:
        with progress_placeholder.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info(f"准备进行{runs}次ARIMA模型训练...")
    else:
        progress_bar = None
        status_text = None
    
    # 创建用于存储每次运行结果的DataFrame
    metrics_df = pd.DataFrame(columns=['运行次数', 'MSE', 'RMSE', 'MAE', '方向准确率'])
    
    # 多次运行模型
    for run in range(runs):
        # 为每次运行设置不同的随机种子
        random_seed = run + 1
        
        try:
            # 拟合ARIMA模型
            model, _ = fit_arima_model(train_data, order)
            
            if model is None:
                # 如果模型拟合失败，记录失败信息并继续下一次
                results.append({
                    'run': run + 1,
                    'status': 'failed',
                    'error': 'Model fitting failed',
                    'metrics': {
                        'MSE': None,
                        'RMSE': None,
                        'MAE': None,
                        'Direction_Accuracy': None
                    }
                })
                continue
            
            # 进行预测
            if forecast_method == "动态预测":
                test_pred = dynamic_forecast_arima(model, train_data, len(test_data), random_seed)
            else:
                test_pred = static_forecast_arima(model, train_data, test_data)
            
            # 计算评估指标
            mse = np.mean((test_pred - test_data) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(test_pred - test_data))
            direction_accuracy = calculate_direction_accuracy(test_data, test_pred)
            
            # 如果方向准确率为None，设置为0
            if direction_accuracy is None:
                direction_accuracy = 0
            
            # 保存本次运行的结果
            run_result = {
                'run': run + 1,
                'status': 'success',
                'seed': random_seed,
                'model': model,
                'predictions': test_pred,
                'metrics': {
                    'MSE': float(mse),
                    'RMSE': float(rmse),
                    'MAE': float(mae),
                    'Direction_Accuracy': float(direction_accuracy)
                }
            }
            
            results.append(run_result)
            
            # 检查是否为最优模型
            if mse < best_mse:
                best_mse = mse
                best_model_data = run_result
            
            # 更新metrics_df
            metrics_df.loc[run] = [run + 1, mse, rmse, mae, direction_accuracy]
            
            # 更新图表（如果提供了占位符）
            if chart_placeholder is not None:
                with chart_placeholder:
                    # 创建指标对比图表
                    metrics_chart_option = create_metrics_comparison_chart(
                        {'results': results, 'best_model': best_model_data},
                        'MSE'
                    )
                    st.write("模型评估指标对比")
                    st_echarts(options=metrics_chart_option, height="400px")
            
            # 更新进度
            if progress_bar is not None:
                progress = (run + 1) / runs
                progress_bar.progress(progress)
                status_text.info(f"完成第 {run + 1}/{runs} 次运行，当前最佳MSE: {best_mse:.4f}")
                
        except Exception as e:
            # 记录错误并继续下一次运行
            results.append({
                'run': run + 1,
                'status': 'error',
                'error': str(e),
                'metrics': {
                    'MSE': None, 
                    'RMSE': None, 
                    'MAE': None, 
                    'Direction_Accuracy': None
                }
            })
            
            if status_text is not None:
                status_text.error(f"第 {run + 1}/{runs} 次运行出错: {str(e)}")
    
    # 计算成功运行的数量
    successful_runs = sum(1 for r in results if r['status'] == 'success')
    
    # 如果没有成功的运行，返回空结果
    if successful_runs == 0:
        return {
            'status': 'failed',
            'message': f"所有 {runs} 次运行均失败",
            'runs': runs,
            'successful_runs': 0,
            'results': results,
            'best_model': None,
            'statistics': None,
            'metrics_df': metrics_df  # 添加metrics_df到返回结果
        }
    
    # 提取成功运行的指标
    successful_metrics = [r['metrics'] for r in results if r['status'] == 'success']
    
    # 计算统计数据
    statistics = {
        'MSE': {
            'mean': np.mean([m['MSE'] for m in successful_metrics]),
            'std': np.std([m['MSE'] for m in successful_metrics]),
            'min': np.min([m['MSE'] for m in successful_metrics]),
            'max': np.max([m['MSE'] for m in successful_metrics])
        },
        'RMSE': {
            'mean': np.mean([m['RMSE'] for m in successful_metrics]),
            'std': np.std([m['RMSE'] for m in successful_metrics]),
            'min': np.min([m['RMSE'] for m in successful_metrics]),
            'max': np.max([m['RMSE'] for m in successful_metrics])
        },
        'MAE': {
            'mean': np.mean([m['MAE'] for m in successful_metrics]),
            'std': np.std([m['MAE'] for m in successful_metrics]),
            'min': np.min([m['MAE'] for m in successful_metrics]),
            'max': np.max([m['MAE'] for m in successful_metrics])
        },
        'Direction_Accuracy': {
            'mean': np.mean([m['Direction_Accuracy'] for m in successful_metrics]),
            'std': np.std([m['Direction_Accuracy'] for m in successful_metrics]),
            'min': np.min([m['Direction_Accuracy'] for m in successful_metrics]),
            'max': np.max([m['Direction_Accuracy'] for m in successful_metrics])
        }
    }
    
    # 清理进度显示（如果有）
    if progress_bar is not None:
        progress_bar.empty()
    if status_text is not None:
        status_text.success(f"成功完成 {successful_runs}/{runs} 次运行")
    
    # 返回结果
    return {
        'status': 'success',
        'message': f"成功完成 {successful_runs}/{runs} 次运行",
        'runs': runs,
        'successful_runs': successful_runs,
        'results': results,
        'best_model': best_model_data,
        'statistics': statistics,
        'metrics_df': metrics_df  # 添加metrics_df到返回结果
    }

# 创建多次运行ARIMA模型的指标对比图
def create_metrics_comparison_chart(multiple_runs_result, metric_name='MSE'):
    """
    创建多次运行ARIMA模型的指标对比图
    
    参数:
    multiple_runs_result: run_multiple_arima_models函数的返回结果
    metric_name: 要比较的指标名称('MSE', 'RMSE', 'MAE', 'Direction_Accuracy')
    
    返回:
    dict: ECharts图表配置
    """
    # 提取成功运行的结果
    successful_results = [r for r in multiple_runs_result['results'] if r['status'] == 'success']
    
    if not successful_results:
        # 如果没有成功的运行，返回空图表
        return {
            'title': {'text': f'没有成功的ARIMA模型运行 - {metric_name}对比'},
            'xAxis': {'type': 'category', 'data': []},
            'yAxis': {'type': 'value'},
            'series': []
        }
    
    # 提取运行序号和对应的指标值
    run_numbers = [r['run'] for r in successful_results]
    metric_values = [r['metrics'][metric_name] for r in successful_results]
    
    # 获取最优模型的运行序号
    best_run = multiple_runs_result['best_model']['run']
    
    # 准备图表数据
    series_data = []
    for i, (run, value) in enumerate(zip(run_numbers, metric_values)):
        # 特殊标记最优模型
        item_style = {'color': '#5470c6'}  # 默认颜色
        if run == best_run:
            item_style = {'color': '#91cc75'}  # 最优模型使用绿色
        
        series_data.append({
            'value': value,
            'itemStyle': item_style
        })
    
    # 计算平均值
    mean_value = np.mean(metric_values)
    
    # 创建图表配置
    chart_option = {
        'title': {
            'text': f'ARIMA模型多次运行 {metric_name} 对比',
            'left': 'center'
        },
        'tooltip': {
            'trigger': 'axis',
            'formatter': f'运行 {{b}}<br/>{metric_name}: {{c}}'
        },
        'toolbox': {
            'feature': {
                'saveAsImage': {}
            }
        },
        'xAxis': {
            'type': 'category',
            'data': run_numbers,
            'name': '运行序号'
        },
        'yAxis': {
            'type': 'value',
            'name': metric_name
        },
        'series': [
            {
                'data': series_data,
                'type': 'bar',
                'name': metric_name
            }
        ],
        'markLine': {
            'data': [
                {'yAxis': mean_value, 'name': '平均值'}
            ],
            'lineStyle': {
                'color': '#ff7f50',
                'type': 'dashed'
            },
            'label': {
                'formatter': f'平均: {mean_value:.4f}'
            }
        }
    }
    
    return chart_option

# 创建指标统计对比表格数据
def create_metrics_statistics_table(multiple_runs_result):
    """
    创建指标统计对比表格数据
    
    参数:
    multiple_runs_result: run_multiple_arima_models函数的返回结果
    
    返回:
    pd.DataFrame: 包含统计数据的DataFrame
    """
    # 检查是否有统计数据
    if multiple_runs_result['status'] == 'failed' or not multiple_runs_result['statistics']:
        return pd.DataFrame()
    
    # 提取统计数据
    stats = multiple_runs_result['statistics']
    
    # 创建DataFrame
    data = {
        '指标': ['MSE', 'RMSE', 'MAE', '方向准确率'],
        '平均值': [
            stats['MSE']['mean'],
            stats['RMSE']['mean'],
            stats['MAE']['mean'],
            stats['Direction_Accuracy']['mean']
        ],
        '标准差': [
            stats['MSE']['std'],
            stats['RMSE']['std'],
            stats['MAE']['std'],
            stats['Direction_Accuracy']['std']
        ],
        '最小值': [
            stats['MSE']['min'],
            stats['RMSE']['min'],
            stats['MAE']['min'],
            stats['Direction_Accuracy']['min']
        ],
        '最大值': [
            stats['MSE']['max'],
            stats['RMSE']['max'],
            stats['MAE']['max'],
            stats['Direction_Accuracy']['max']
        ]
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('指标')
    
    # 格式化为字符串，保留4位小数
    for col in df.columns:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")
    
    return df 

# 准备ARIMA预测结果图表
def prepare_arima_charts(model, train_data, test_data, test_pred, residuals=None):
    """
    准备ARIMA模型的预测结果图表、残差图表和残差分布图表
    
    参数:
    model: 拟合的ARIMA模型
    train_data: 训练数据
    test_data: 测试数据
    test_pred: 测试集预测值
    residuals: 模型残差，如果为None则使用model.resid
    
    返回:
    dict: 包含预测结果图表、残差图表和残差分布图表的字典
    """
    # 使用模型的残差或者提供的残差
    if residuals is None and model is not None:
        residuals = model.resid
    
    # 创建结果DataFrame
    train_pred = None
    if model is not None:
        train_pred = model.fittedvalues
    
    results_df = pd.DataFrame({
        '实际值': pd.concat([train_data, test_data]),
        '训练集拟合值': pd.concat([train_pred, pd.Series(np.full(len(test_data), np.nan), index=test_data.index)]) if train_pred is not None else None,
        '测试集预测值': pd.concat([pd.Series(np.full(len(train_data), np.nan), index=train_data.index), pd.Series(test_pred, index=test_data.index)])
    })
    
    # 预测结果图表
    prediction_chart = create_timeseries_chart(
        results_df,
        title='ARIMA预测结果对比',
        series_names=['实际值', '训练集拟合值', '测试集预测值']
    )
    
    # 残差图表
    residuals_chart = None
    residuals_hist = None
    if residuals is not None:
        residuals_df = pd.DataFrame({'残差': residuals})
        residuals_chart = create_timeseries_chart(
            residuals_df,
            title='ARIMA模型残差'
        )
        
        # 残差分布图表
        residuals_hist = create_histogram_chart(
            residuals,
            title='残差分布直方图'
        )
    
    return {
        'prediction_chart': prediction_chart,
        'residuals_chart': residuals_chart,
        'residuals_hist': residuals_hist
    }

def calculate_statistics(metrics_list):
    """
    计算指标列表的统计数据
    
    参数:
    metrics_list: 指标字典列表，每个字典包含 'MSE', 'RMSE', 'MAE', 'Direction_Accuracy' 等键
    
    返回:
    dict: 统计数据字典
    """
    # 筛选出有效的指标
    valid_metrics = [m for m in metrics_list if m and all(key in m for key in ['MSE', 'RMSE', 'MAE', 'Direction_Accuracy'])]
    
    if not valid_metrics:
        return None
    
    # 提取各项指标
    mse_values = [m['MSE'] for m in valid_metrics if m['MSE'] is not None]
    rmse_values = [m['RMSE'] for m in valid_metrics if m['RMSE'] is not None]
    mae_values = [m['MAE'] for m in valid_metrics if m['MAE'] is not None]
    dir_acc_values = [m['Direction_Accuracy'] for m in valid_metrics if m['Direction_Accuracy'] is not None]
    
    # 计算统计量
    statistics = {}
    
    if mse_values:
        statistics['MSE'] = {
            'mean': float(np.mean(mse_values)),
            'std': float(np.std(mse_values)),
            'min': float(np.min(mse_values)),
            'max': float(np.max(mse_values))
        }
    
    if rmse_values:
        statistics['RMSE'] = {
            'mean': float(np.mean(rmse_values)),
            'std': float(np.std(rmse_values)),
            'min': float(np.min(rmse_values)),
            'max': float(np.max(rmse_values))
        }
    
    if mae_values:
        statistics['MAE'] = {
            'mean': float(np.mean(mae_values)),
            'std': float(np.std(mae_values)),
            'min': float(np.min(mae_values)),
            'max': float(np.max(mae_values))
        }
    
    if dir_acc_values:
        statistics['Direction_Accuracy'] = {
            'mean': float(np.mean(dir_acc_values)),
            'std': float(np.std(dir_acc_values)),
            'min': float(np.min(dir_acc_values)),
            'max': float(np.max(dir_acc_values))
        }
    
    return statistics