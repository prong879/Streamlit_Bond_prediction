"""
-*- coding: utf-8 -*-

@Author : hf_lcx
@Time : 2025/04/16
@File : icbc_performance_downloader.py
@Description : 下载工商银行(601398)的qt_performance数据并支持增量更新保存为pkl格式
"""

from sqlalchemy import create_engine
import pandas as pd
import os
import time
import re
import glob
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings("ignore")

# 设置参数
STOCK_CODE = "601398"  # 工商银行股票代码
STOCK_NAME = "工商银行"
YEARS_TO_FETCH = 1  # 设置要获取的年份数量，可以根据需要修改
current_date = datetime.now()
start_date = current_date - timedelta(days=YEARS_TO_FETCH*365)
START_DATE_STR = start_date.strftime('%Y-%m-%d')  # 使用标准日期格式
print(f"设置数据获取范围: {START_DATE_STR} 至今")
print(f"股票: {STOCK_NAME}({STOCK_CODE})")

# 每个文件的最大行数
MAX_ROWS_PER_FILE = 1000000  # 每个文件最多存储100万行数据
CSV_SAMPLE_ROWS = 5  # CSV样本数据行数

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

# 进度条显示函数
def display_progress(current, total, bar_length=50):
    """显示下载进度条
    
    Args:
        current: 当前进度
        total: 总进度
        bar_length: 进度条长度
    """
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write('\r进度: [{0}] {1}%  ({2}/{3})'.format(
        arrow + spaces, 
        int(round(percent * 100)), 
        current, 
        total))
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')

def filter_files_by_pattern(file_list, pattern):
    """
    根据正则表达式模式过滤文件名
    
    Args:
        file_list: 文件名列表
        pattern: 正则表达式模式
        
    Returns:
        匹配的文件名列表
    """
    regex = re.compile(f"{pattern}\.pkl$")
    return [file for file in file_list if regex.search(file)]

def save_csv_sample(df, file_path, sample_rows=CSV_SAMPLE_ROWS):
    """
    保存数据的CSV样本，并删除已有的CSV文件
    
    Args:
        df: 要保存的DataFrame
        file_path: 保存路径
        sample_rows: 样本行数
    """
    # 删除已有的CSV文件
    csv_pattern = os.path.join(os.path.dirname(file_path), f"{os.path.basename(file_path).split('_')[0]}_{os.path.basename(file_path).split('_')[1]}*.csv")
    existing_csv_files = glob.glob(csv_pattern)
    for csv_file in existing_csv_files:
        try:
            os.remove(csv_file)
            print(f"已删除现有CSV文件: {csv_file}")
        except Exception as e:
            print(f"删除文件 {csv_file} 时出错: {str(e)}")
    
    # 保存CSV样本
    sample_df = df.tail(sample_rows)  # 获取最近的几条数据
    sample_df.to_csv(file_path, index=False)
    print(f"已保存CSV样本数据 ({len(sample_df)} 行) 至: {file_path}")

class ICBCDataDownloader:
    def __init__(self, data_dir):
        """
        初始化下载器
        
        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir  # 数据存储路径
        self.table_name = "qt_performance"  # 要下载的表名
        self.stock_code = STOCK_CODE  # 工商银行股票代码
        self.result_prefix = f"{STOCK_NAME}_{STOCK_CODE}" # 结果文件前缀
        self.inner_code = None # 工商银行的内部编码
        
        # 数据库连接引擎
        print("尝试连接数据库...")
        try:
            self.engine = create_engine("mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb")
            # 测试连接
            test_result = pd.read_sql("SELECT 1", self.engine)
            print("数据库连接成功！", test_result)
            
            # 获取内部编码并检查表结构
            self.inner_code = self.get_inner_code()
            self.check_table_structure()
        except Exception as e:
            print(f"数据库连接失败: {str(e)}")
            raise
    
    def check_table_structure(self):
        """检查表结构，确认日期和JSID字段"""
        try:
            sql_desc = f"DESCRIBE {self.table_name}"
            table_structure = pd.read_sql(sql_desc, self.engine)
            print("表结构:")
            print(table_structure)
            
            # 确认JSID字段存在
            if 'JSID' not in table_structure['Field'].values:
                print("警告: 表中没有JSID字段，增量更新功能可能无法正常工作")
            
            # 确认日期字段
            self.date_field = 'TradingDay'
            if self.date_field not in table_structure['Field'].values:
                # 尝试其他可能的日期字段名
                possible_date_fields = ['TradeDate', 'Date', 'ReportDate']
                for field in possible_date_fields:
                    if field in table_structure['Field'].values:
                        self.date_field = field
                        break
                else:
                    print("警告: 未找到日期字段，将使用JSID进行数据筛选")
                    self.date_field = None
            
            if self.date_field:
                print(f"使用日期字段: {self.date_field}")
            
        except Exception as e:
            print(f"检查表结构失败: {str(e)}")
            self.date_field = 'TradingDay'  # 默认使用TradingDay
        
    def create_data_dir(self):
        """创建数据存储目录"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建目录: {self.data_dir}")
        else:
            print(f"目录已存在: {self.data_dir}")
    
    def get_inner_code(self):
        """获取工商银行的InnerCode"""
        try:
            # 假设SecuMain表中存储了股票代码和InnerCode的对应关系
            sql = f"SELECT InnerCode FROM SecuMain WHERE SecuCode = '{self.stock_code}' AND SecuCategory = 1"
            print(f"执行查询: {sql}")
            start_time = time.time()
            result = pd.read_sql(sql, self.engine)
            end_time = time.time()
            print(f"查询耗时: {end_time - start_time:.2f} 秒")
            
            if result.empty:
                raise ValueError(f"未找到股票代码 {self.stock_code} 的InnerCode")
            
            inner_code = result.iloc[0, 0]
            print(f"工商银行(601398) InnerCode: {inner_code}")
            return inner_code
        except Exception as e:
            print(f"获取InnerCode失败: {str(e)}")
            
            # 如果无法通过SecuMain获取，尝试直接查询qt_performance表
            try:
                sql = f"SELECT DISTINCT InnerCode FROM {self.table_name} WHERE InnerCode IN (SELECT InnerCode FROM SecuMain WHERE SecuCode = '{self.stock_code}') LIMIT 1"
                print(f"尝试直接从{self.table_name}表获取InnerCode: {sql}")
                result = pd.read_sql(sql, self.engine)
                if not result.empty:
                    inner_code = result.iloc[0, 0]
                    print(f"工商银行(601398) InnerCode: {inner_code}")
                    return inner_code
            except Exception as inner_e:
                print(f"从{self.table_name}表获取InnerCode失败: {str(inner_e)}")
            
            raise ValueError(f"无法确定股票代码 {self.stock_code} 的InnerCode")
    
    def get_table_count(self, where_clause=""):
        """获取表的总行数，用于进度显示"""
        sql = f"SELECT COUNT(*) FROM {self.table_name} {where_clause}"
        print(f"执行查询: {sql}")
        try:
            start_time = time.time()
            result = pd.read_sql(sql, self.engine)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"获取行数查询耗时: {execution_time:.2f} 秒")
            return result.iloc[0, 0]
        except Exception as e:
            print(f"获取表行数失败: {str(e)}")
            raise
    
    @timing_decorator
    def get_data(self, last_jsid=None):
        """
        下载工商银行数据，支持增量更新
        
        Args:
            last_jsid: 上次更新的最大JSID，None表示全量下载
            
        Returns:
            DataFrame: 包含下载数据的DataFrame
        """
        try:
            # 构建查询条件
            where_conditions = [f"InnerCode = {self.inner_code}"]
            
            if last_jsid is not None:
                where_conditions.append(f"JSID > {last_jsid}")
            elif self.date_field:
                where_conditions.append(f"{self.date_field} >= '{START_DATE_STR}'")
            
            where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 获取符合条件的总行数
            print(f"正在获取符合条件 '{where_clause}' 的数据总行数...")
            total_rows = self.get_table_count(where_clause)
            print(f"符合条件的数据总行数: {total_rows}")
            
            if total_rows == 0:
                print(f"未找到 {STOCK_NAME}({self.stock_code}) 符合条件的数据")
                return pd.DataFrame()
            
            # 分批下载数据以避免内存问题
            all_data = []
            for offset in range(0, total_rows, MAX_ROWS_PER_FILE):
                # 构建完整查询
                limit_clause = f"LIMIT {MAX_ROWS_PER_FILE} OFFSET {offset}"
                sql = f"""
                SELECT * FROM {self.table_name} 
                {where_clause}
                ORDER BY JSID ASC
                {limit_clause}
                """
                
                print(f"下载第 {offset//MAX_ROWS_PER_FILE + 1} 批数据: {sql}")
                
                batch_start = time.time()
                df_batch = pd.read_sql(sql, self.engine)
                batch_end = time.time()
                
                # 如果第一批，显示数据样本
                if offset == 0:
                    print("数据样本:")
                    print(df_batch.head(2))
                
                print(f"获取第 {offset//MAX_ROWS_PER_FILE + 1} 批数据: {len(df_batch)} 行, 耗时 {batch_end - batch_start:.2f} 秒")
                
                all_data.append(df_batch)
                display_progress(min(offset + MAX_ROWS_PER_FILE, total_rows), total_rows)
                
                if len(df_batch) < MAX_ROWS_PER_FILE:  # 如果获取的数据少于限制，说明已经获取完毕
                    break
            
            if not all_data:
                return pd.DataFrame()
            
            # 合并所有数据
            df = pd.concat(all_data, ignore_index=True)
            print(f"下载完成，共 {len(df)} 行数据")
            
            return df
                
        except Exception as e:
            print(f"下载数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    @timing_decorator
    def save_data(self):
        """
        首次下载数据并保存为pickle文件
        
        Returns:
            DataFrame: 保存的数据
        """
        try:
            print("开始首次下载数据...")
            
            # 下载数据
            df = self.get_data()
            
            if df.empty:
                print("没有数据可保存")
                return pd.DataFrame()
            
            # 保存数据
            max_jsid = df['JSID'].max()
            file_path = os.path.join(self.data_dir, f"{self.result_prefix}_{max_jsid}.pkl")
            df.to_pickle(file_path)
            
            print(f"已保存数据至 {file_path}")
            print(f"保存了 {len(df)} 行数据")
            
            # 保存CSV样本文件
            csv_path = os.path.join(self.data_dir, f'{self.result_prefix}_{datetime.now().strftime("%Y%m%d")}_sample.csv')
            save_csv_sample(df, csv_path)
            
            # 显示数据概要
            self.show_data_summary(df)
            
            return df
        
        except Exception as e:
            print(f"下载数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def show_data_summary(self, df):
        """显示数据概要信息"""
        print("\n数据概要:")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print("数据日期范围:")
        if self.date_field and self.date_field in df.columns:
            print(f"开始日期: {df[self.date_field].min()}")
            print(f"结束日期: {df[self.date_field].max()}")
            print(f"共包含 {df[self.date_field].nunique()} 个交易日")
        else:
            print("未找到日期字段，无法显示日期范围")
    
    @timing_decorator
    def update_data(self):
        """
        更新数据，支持增量更新
        
        Returns:
            DataFrame: 更新后的数据
        """
        files = os.listdir(self.data_dir)
        file_lst = filter_files_by_pattern(files, f"{self.result_prefix}_\\d+")
        
        if not file_lst:
            print("未找到现有数据文件，执行首次下载...")
            return self.save_data()
        
        # 找到最新的文件
        latest_file = max(file_lst)
        file_path = os.path.join(self.data_dir, latest_file)
        print(f"找到最新文件: {file_path}")
        
        try:
            # 加载现有数据
            df_old = pd.read_pickle(file_path)
            max_jsid = df_old['JSID'].max()
            print(f"现有数据最大JSID: {max_jsid}")
            
            # 获取数据库中的最大JSID
            sql = f"SELECT MAX(JSID) FROM {self.table_name} WHERE InnerCode = {self.inner_code}"
            db_max_jsid = pd.read_sql(sql, self.engine).iloc[0, 0]
            print(f"数据库中{STOCK_NAME}({STOCK_CODE})最大JSID: {db_max_jsid}")
            
            if max_jsid >= db_max_jsid:
                print(f"数据已是最新，无需更新")
                return df_old
            
            # 下载新数据
            print("\n下载新数据:")
            df_new = self.get_data(max_jsid)
            
            if df_new.empty:
                print("没有新数据可更新")
                return df_old
            
            # 检查合并后的数据是否超过最大行数限制
            if len(df_old) + len(df_new) <= MAX_ROWS_PER_FILE:
                # 如果合并后不超过限制，则合并为一个文件
                df_combined = pd.concat([df_old, df_new], ignore_index=True)
                # 去重
                df_combined.drop_duplicates(subset=['JSID'], keep='last', inplace=True)
                
                # 保存为新文件
                new_max_jsid = df_combined['JSID'].max()
                new_file_path = os.path.join(self.data_dir, f"{self.result_prefix}_{new_max_jsid}.pkl")
                df_combined.to_pickle(new_file_path)
                
                # 删除旧文件
                os.remove(file_path)
                print(f"合并数据后保存至新文件: {new_file_path}")
                
                # 保存CSV样本文件
                csv_path = os.path.join(self.data_dir, f'{self.result_prefix}_{datetime.now().strftime("%Y%m%d")}_sample.csv')
                save_csv_sample(df_combined, csv_path)
                
                # 显示数据概要
                self.show_data_summary(df_combined)
                
                return df_combined
            else:
                # 如果合并后超过限制，则新数据单独保存为一个文件
                new_max_jsid = df_new['JSID'].max()
                new_file_path = os.path.join(self.data_dir, f"{self.result_prefix}_{new_max_jsid}.pkl")
                df_new.to_pickle(new_file_path)
                print(f"数据量较大，新数据单独保存至: {new_file_path}")
                
                # 保存CSV样本文件 - 只包含新数据的样本
                csv_path = os.path.join(self.data_dir, f'{self.result_prefix}_{datetime.now().strftime("%Y%m%d")}_新增_sample.csv')
                save_csv_sample(df_new, csv_path)
                
                # 显示数据概要
                self.show_data_summary(df_new)
                
                return df_new
        
        except Exception as e:
            print(f"更新数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == "__main__":
    # 设置程序开始时间
    program_start = time.time()
    print(f"程序开始执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建数据目录
    data_dir = os.path.join(os.getcwd(), 'data')
    
    # 下载并保存数据
    try:
        downloader = ICBCDataDownloader(data_dir)
        downloader.create_data_dir()
        
        # 更新数据
        df = downloader.update_data()
        print(f"数据更新完毕，共 {len(df)} 行")
        
    except KeyboardInterrupt:
        print("\n下载被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        program_end = time.time()
        program_duration = program_end - program_start
        print(f"\n程序总执行时间: {program_duration:.2f} 秒")
        print(f"程序结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 