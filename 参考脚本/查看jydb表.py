"""
-*- coding: utf-8 -*-

@Author : hf_lcx
@Time : 2025/04/16
@File : 查看jydb表.py
@Description : 连接jydb数据库并查看所有表信息
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
        print("{} 函数的执行时间为:{:.2f} 秒".format(func.__name__, execution_time))
        return result
    return wrapper

@timing_decorator
def get_all_tables():
    """
    连接到jydb数据库并获取所有表列表
    """
    # 创建数据库连接引擎
    engine = create_engine("mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb")
    
    # 查询所有表
    sql = "SHOW TABLES"
    df = pd.read_sql(sql, engine)
    
    return df

@timing_decorator
def show_table_info(table_name):
    """
    显示指定表的结构信息
    """
    engine = create_engine("mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb")
    
    # 查询表结构
    sql = f"DESCRIBE {table_name}"
    df = pd.read_sql(sql, engine)
    
    return df

@timing_decorator
def count_table_rows(table_name):
    """
    统计表中的行数
    """
    engine = create_engine("mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb")
    
    # 统计行数
    sql = f"SELECT COUNT(*) FROM {table_name}"
    result = pd.read_sql(sql, engine)
    
    return result.iloc[0, 0]

if __name__ == "__main__":
    # 获取所有表
    tables_df = get_all_tables()
    print(f"jydb数据库中共有 {len(tables_df)} 个表")
    
    # 显示所有表名
    print("\n表列表:")
    for idx, table in enumerate(tables_df['Tables_in_jydb']):
        print(f"{idx+1}. {table}")
    
    # 计算不同前缀表的数量
    prefixes = {'lc': 0, 'qt': 0, 'mf': 0, 'other': 0}
    for table in tables_df['Tables_in_jydb']:
        if table.startswith('lc'):
            prefixes['lc'] += 1
        elif table.startswith('qt'):
            prefixes['qt'] += 1
        elif table.startswith('mf'):
            prefixes['mf'] += 1
        else:
            prefixes['other'] += 1
    
    print("\n表前缀统计:")
    print(f"lc (公司数据): {prefixes['lc']}个表")
    print(f"qt (日交易数据): {prefixes['qt']}个表")
    print(f"mf (基金数据): {prefixes['mf']}个表")
    print(f"其他: {prefixes['other']}个表")
    
    # 询问是否查看特定表的详细信息
    check_details = input("\n是否查看特定表的详细信息？(y/n): ")
    if check_details.lower() == 'y':
        table_name = input("请输入要查看的表名: ")
        try:
            # 显示表结构
            table_info = show_table_info(table_name)
            print(f"\n{table_name} 表结构:")
            print(table_info)
            
            # 获取表行数
            try:
                row_count = count_table_rows(table_name)
                print(f"\n{table_name} 表共有 {row_count} 行数据")
            except Exception as e:
                print(f"获取行数时出错: {str(e)}")
        except Exception as e:
            print(f"获取表信息时出错: {str(e)}") 