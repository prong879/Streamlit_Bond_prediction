"""
-*- coding: utf-8 -*-

@Author : hf_lcx
@Time : 2025/04/18
@File : 探查常量表.py
@Description : 通过关键词交互式查询 CT_SystemConst 表，以帮助推断常量代码的含义。
"""

import pandas as pd
from sqlalchemy import create_engine
import time

# 计算函数运行时间
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} 函数的执行时间为:{execution_time:.2f} 秒")
        return result
    return wrapper

@timing_decorator
def query_ct_systemconst_by_keyword(keyword, search_field='MS', limit=100):
    """
    根据关键词查询 CT_SystemConst 表
    :param keyword: 要搜索的关键词
    :param search_field: 要搜索的字段 ('MS' 或 'LBMC')
    :param limit: 返回的最大行数
    """
    engine = create_engine("mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb")
    
    if search_field not in ['MS', 'LBMC']: # 基本的安全检查
        raise ValueError("Invalid search_field")

    param_keyword = f"%{keyword}%" 
    
    # PyMySQL 使用 %s 作为占位符, SQLAlchemy 会处理 :name 到 %s 的转换
    # 我们显式使用字典传递参数给 params
    sql_query = f"SELECT LB, LBMC, DM, MS, IVALUE FROM CT_SystemConst WHERE {search_field} LIKE %(keyword_like)s LIMIT %(limit_val)s"
    
    print(f"\nExecuting SQL: {sql_query}")
    print(f"With params: keyword_like='{param_keyword}', limit_val={limit}")

    df = pd.read_sql(sql_query, engine, params={"keyword_like": param_keyword, "limit_val": int(limit)})
    
    return df

if __name__ == "__main__":
    print("开始批量探查 CT_SystemConst 表...")

    # 定义要探查的目标列表
    # 每个元素是一个元组: (描述, 关键词, 搜索字段)
    exploration_targets = [
        ("探测 SecuMarket (证券市场) LB 值", "市场", "MS"),
        ("探测 SecuMarket (证券市场) LB 值 - 备选", "交易所", "MS"),
        ("探测 SecuMarket (证券市场) LB 值 - LBMC", "交易所", "LBMC"), # 尝试LBMC
        ("探测 SecuCategory (证券类别) LB 值", "类别", "MS"),
        ("探测 SecuCategory (证券类别) LB 值 - 备选", "品种", "MS"),
        ("探测 SecuCategory (证券类别) LB 值 - LBMC", "证券类别", "LBMC"),
        ("探测 ListedSector (上市板块) LB 值", "板块", "MS"),
        ("探测 ListedSector (上市板块) LB 值 - LBMC", "上市板块", "LBMC"),
        ("探测 ListedState (上市状态) LB 值", "状态", "MS"),
        ("探测 ListedState (上市状态) LB 值 - 备选", "上市状态", "MS"), # 更精确的关键词
        ("探测 ListedState (上市状态) LB 值 - LBMC", "上市状态", "LBMC")
    ]

    for description, keyword, search_field in exploration_targets:
        print(f"\n==================================================================================")
        print(f"探查任务: {description}")
        print(f"关键词: '{keyword}', 搜索字段: {search_field}")
        print(f"==================================================================================")
        
        try:
            const_data_df = query_ct_systemconst_by_keyword(keyword, search_field=search_field)
            if not const_data_df.empty:
                print(f"\n找到包含 '{keyword}' 的记录 (字段: {search_field}, 最多显示100条):")
                print(const_data_df.to_string(index=False, max_colwidth=None))
            else:
                print("未找到相关记录。")
        except Exception as e:
            print(f"查询 CT_SystemConst 时出错 (关键词: {keyword}, 字段: {search_field}): {str(e)}")
        print("\n") # 在每个探查任务后添加空行以便区分

    print("批量探查 CT_SystemConst 表结束。") 