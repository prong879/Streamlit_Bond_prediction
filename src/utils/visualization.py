"""
可视化工具
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    """
    可视化工具类，用于生成各种图表
    """
    
    @staticmethod
    def plot_stock_price(data: pd.DataFrame, indicators: List[str] = None, title: str = "股票价格走势") -> go.Figure:
        """
        绘制股票价格K线图和技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            indicators: 要显示的技术指标列表
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        fig = go.Figure()
        
        # 添加K线图
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="K线"
        ))
        
        # 添加技术指标
        if indicators:
            if 'MA' in indicators:
                # 添加移动平均线
                for period in [5, 10, 20]:
                    ma = data['Close'].rolling(window=period).mean()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=ma,
                        name=f'MA{period}',
                        line=dict(width=1)
                    ))
            
            if 'BOLL' in indicators:
                # 添加布林带
                ma20 = data['Close'].rolling(window=20).mean()
                std20 = data['Close'].rolling(window=20).std()
                upper = ma20 + 2 * std20
                lower = ma20 - 2 * std20
                
                fig.add_trace(go.Scatter(
                    x=data.index, y=upper,
                    name='布林上轨',
                    line=dict(width=1, dash='dash')
                ))
                fig.add_trace(go.Scatter(
                    x=data.index, y=ma20,
                    name='布林中轨',
                    line=dict(width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=data.index, y=lower,
                    name='布林下轨',
                    line=dict(width=1, dash='dash')
                ))
        
        fig.update_layout(
            title=title,
            yaxis_title='价格',
            xaxis_title='日期',
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    @staticmethod
    def plot_volume(data: pd.DataFrame, title: str = "成交量") -> go.Figure:
        """
        绘制成交量柱状图
        
        Args:
            data: 包含OHLCV数据的DataFrame
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        # 计算涨跌
        data = data.copy()
        data['Price_Change'] = data['Close'] - data['Open']
        
        fig = go.Figure()
        
        # 添加上涨成交量（红色）
        fig.add_trace(go.Bar(
            x=data[data['Price_Change'] >= 0].index,
            y=data[data['Price_Change'] >= 0]['Volume'],
            name='上涨成交量',
            marker_color='red'
        ))
        
        # 添加下跌成交量（绿色）
        fig.add_trace(go.Bar(
            x=data[data['Price_Change'] < 0].index,
            y=data[data['Price_Change'] < 0]['Volume'],
            name='下跌成交量',
            marker_color='green'
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title='成交量',
            xaxis_title='日期',
            template='plotly_white',
            barmode='stack',
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_technical_indicators(data: pd.DataFrame, indicators: List[str]) -> Dict[str, go.Figure]:
        """
        绘制技术指标图表
        
        Args:
            data: 包含技术指标数据的DataFrame
            indicators: 要显示的技术指标列表
            
        Returns:
            包含各个技术指标图表的字典
        """
        figs = {}
        
        if 'MACD' in indicators:
            # 绘制MACD
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=data.index, y=data['MACD'],
                name='MACD',
                line=dict(width=1)
            ))
            fig_macd.add_trace(go.Scatter(
                x=data.index, y=data['MACD_Signal'],
                name='信号线',
                line=dict(width=1)
            ))
            fig_macd.add_trace(go.Bar(
                x=data.index, y=data['MACD_Hist'],
                name='MACD柱状'
            ))
            fig_macd.update_layout(
                title='MACD指标',
                template='plotly_white',
                showlegend=True
            )
            figs['MACD'] = fig_macd
        
        if 'RSI' in indicators:
            # 绘制RSI
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=data.index, y=data['RSI'],
                name='RSI',
                line=dict(width=1)
            ))
            # 添加超买超卖线
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖")
            fig_rsi.update_layout(
                title='RSI指标',
                yaxis_title='RSI值',
                template='plotly_white',
                showlegend=True
            )
            figs['RSI'] = fig_rsi
        
        if 'KDJ' in indicators:
            # 绘制KDJ
            fig_kdj = go.Figure()
            fig_kdj.add_trace(go.Scatter(
                x=data.index, y=data['Stoch_K'],
                name='K值',
                line=dict(width=1)
            ))
            fig_kdj.add_trace(go.Scatter(
                x=data.index, y=data['Stoch_D'],
                name='D值',
                line=dict(width=1)
            ))
            fig_kdj.update_layout(
                title='KDJ指标',
                template='plotly_white',
                showlegend=True
            )
            figs['KDJ'] = fig_kdj
        
        if 'CCI' in indicators:
            # 绘制CCI
            fig_cci = go.Figure()
            fig_cci.add_trace(go.Scatter(
                x=data.index, y=data['CCI'],
                name='CCI',
                line=dict(width=1)
            ))
            fig_cci.add_hline(y=100, line_dash="dash", line_color="red")
            fig_cci.add_hline(y=-100, line_dash="dash", line_color="green")
            fig_cci.update_layout(
                title='CCI指标',
                template='plotly_white',
                showlegend=True
            )
            figs['CCI'] = fig_cci
        
        if 'ADX' in indicators:
            # 绘制ADX
            fig_adx = go.Figure()
            fig_adx.add_trace(go.Scatter(
                x=data.index, y=data['ADX'],
                name='ADX',
                line=dict(width=1)
            ))
            fig_adx.update_layout(
                title='ADX指标',
                template='plotly_white',
                showlegend=True
            )
            figs['ADX'] = fig_adx
        
        if 'Williams_R' in indicators:
            # 绘制威廉指标
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Scatter(
                x=data.index, y=data['Williams_R'],
                name='威廉指标',
                line=dict(width=1)
            ))
            fig_wr.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="超买")
            fig_wr.add_hline(y=-80, line_dash="dash", line_color="green", annotation_text="超卖")
            fig_wr.update_layout(
                title='威廉指标',
                template='plotly_white',
                showlegend=True
            )
            figs['Williams_R'] = fig_wr
        
        return figs
    
    @staticmethod
    def plot_correlation_matrix(data: pd.DataFrame, title: str = "特征相关性矩阵") -> go.Figure:
        """
        绘制相关性矩阵热力图
        
        Args:
            data: 特征数据DataFrame
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        corr = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], title: str = "训练历史") -> go.Figure:
        """
        绘制训练历史曲线
        
        Args:
            history: 包含训练历史数据的字典
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        fig = go.Figure()
        
        if 'train_loss' in history:
            fig.add_trace(go.Scatter(
                y=history['train_loss'],
                name='训练损失',
                mode='lines'
            ))
            
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                name='验证损失',
                mode='lines'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='轮次',
            yaxis_title='损失',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_prediction_results(actual: np.ndarray, predicted: np.ndarray, 
                              title: str = "预测结果对比") -> go.Figure:
        """
        绘制预测结果对比图
        
        Args:
            actual: 实际值数组
            predicted: 预测值数组
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=actual,
            name='实际值',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            y=predicted,
            name='预测值',
            mode='lines'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='时间',
            yaxis_title='价格',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                            title: str = "模型性能对比") -> go.Figure:
        """
        绘制模型性能对比图
        
        Args:
            results: 包含各模型性能指标的字典
            title: 图表标题
            
        Returns:
            plotly图表对象
        """
        models = list(results.keys())
        metrics = list(results[models[0]].keys())
        
        fig = go.Figure(data=[
            go.Bar(name=metric,
                  x=models,
                  y=[results[model][metric] for model in models])
            for metric in metrics
        ])
        
        fig.update_layout(
            title=title,
            barmode='group',
            template='plotly_white'
        )
        
        return fig 