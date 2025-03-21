"""
会话状态管理工具
用于管理应用的会话状态
"""
import streamlit as st
from typing import Any, Dict, Optional

def init_session_state():
    """
    初始化会话状态
    """
    # 数据相关
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []
    
    # 图表设置
    if 'chart_theme' not in st.session_state:
        st.session_state.chart_theme = 'plotly_white'
    
    # 模型相关
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None
    
    if 'training_history' not in st.session_state:
        st.session_state.training_history = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

def get_state(key, default=None):
    """
    获取会话状态值
    
    Args:
        key: 状态键名
        default: 默认值，当键不存在时返回
    
    Returns:
        会话状态值
    """
    return st.session_state.get(key, default)

def set_state(key, value):
    """
    设置会话状态值
    
    Args:
        key: 状态键名
        value: 状态值
    """
    st.session_state[key] = value

def update_states(states_dict):
    """
    批量更新会话状态
    
    Args:
        states_dict: 包含状态键值对的字典
    """
    for key, value in states_dict.items():
        set_state(key, value)

def clear_states(keys):
    """
    清除指定的会话状态
    
    Args:
        keys: 要清除的状态键列表
    """
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def clear_all_states():
    """
    清除所有会话状态
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def update_status():
    """
    更新系统状态信息
    
    Returns:
        dict: 包含系统各项状态的字典
    """
    # 获取数据状态
    if get_state('raw_data') is not None:
        data_status = "已加载"
    else:
        data_status = "未加载"
    
    # 获取特征状态
    features = get_state('selected_features', [])
    feature_count = len(features)
    
    # 获取模型状态
    if get_state('trained_model') is not None:
        model_status = "已训练"
    else:
        model_status = "未训练"
    
    return {
        "数据状态": data_status,
        "已选特征": f"{feature_count}个",
        "模型状态": model_status
    } 