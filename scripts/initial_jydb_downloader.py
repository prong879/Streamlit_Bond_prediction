"""
脚本说明:
本脚本用于从远程 jydb 数据库首次批量下载A股股票基本信息、行业信息、
日线行情数据以及沪深300指数行情数据，并存入本地 MySQL 数据库。
数据默认下载近5年。

在运行此脚本之前，请确保：
1. 您已经运行了 `scripts/create_target_tables.py` 来创建本地数据库表结构。
2. 您已经安装了必要的库: pymysql, sqlalchemy, pandas.
3. 更新下面的 JYDB_DATABASE_URI 和 LOCAL_MYSQL_DATABASE_URI 为您的实际连接信息。
4. 远程 jydb 数据库可访问，并且本地 MySQL 服务已启动且目标数据库已创建。
"""
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import time
import logging
import sys
import os

# --- 配置区域 ---
# 源数据库 (聚源 JYDB)
JYDB_DATABASE_URI = "mysql+pymysql://client:123456789@6.tcp.vip.cpolar.cn:12624/jydb" 
# 例如: "mysql+pymysql://user:pass@host:port/jydb"

# 目标数据库 (本地 MySQL)
LOCAL_MYSQL_DATABASE_URI = "sqlite:///data/bond_prediction_data.db" 
# 例如: "mysql+mysqlconnector://root:password@localhost:3306/bond_prediction_data"

# 数据下载时间范围 (年)
YEARS_TO_FETCH = 5

# 沪深300指数代码 (根据聚源数据库中的实际代码调整)
HS300_INDEX_CODE_JYDB = '000300' # 假设在聚源中是 '000300'，可能需要加后缀如 .SH
HS300_INDEX_CODE_LOCAL = '000300.SH' # 我们在本地存储的格式

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从 create_target_tables.py 导入表定义
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本的父目录 (即 scripts 目录)
scripts_dir = os.path.dirname(current_script_path)
# 获取 scripts 目录的父目录 (即项目根目录)
project_root = os.path.dirname(scripts_dir)
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts.create_target_tables import Stock, StockIndustry, SystemConstant, DailyQuote, IndexQuote

# 计算函数运行时间
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} 执行耗时: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

@timing_decorator
def fetch_a_share_stocks(jydb_engine):
    """从聚源数据库获取A股股票列表 (SecuMain)"""
    logger.info("开始获取A股股票列表...")
    # SecuCategory = 1 代表股票; SecuMarket IN (83, 90, 18) 代表上交所、深交所、北交所A股
    # ListedState = 1 代表上市 (具体值需根据CT_SystemConst确认，这里假设为1)
    # 您可能需要更精确的筛选条件
    query = """
    SELECT InnerCode, CompanyCode, SecuCode, ChiName, SecuAbbr, SecuMarket, SecuCategory, 
           ListedDate, ListedSector, ListedState, ISIN, XGRQ AS jydb_updated_time
    FROM SecuMain 
    WHERE SecuCategory = 1 AND SecuMarket IN (83, 90, 18) AND ListedState = 1 
    """ # TODO: 确认 ListedState = 1 是否正确代表A股的"上市"状态
    try:
        df_stocks = pd.read_sql(query, jydb_engine)
        logger.info(f"成功获取 {len(df_stocks)} 条A股股票基本信息。")
        # TODO: 使用 CT_SystemConst 转换代码为名称 (SecuMarketName, SecuCategoryName, etc.)
        # 这一步可以在存入本地前完成，或者在加载到本地后再通过JOIN更新
        return df_stocks
    except Exception as e:
        logger.error(f"获取A股列表失败: {e}")
        raise

@timing_decorator
def fetch_stock_industry(jydb_engine, company_codes):
    """根据公司代码列表获取行业信息 (LC_ExgIndustry)"""
    if not company_codes:
        logger.info("没有公司代码可供查询行业信息。")
        return pd.DataFrame()
    logger.info(f"开始获取 {len(company_codes)}家公司的行业信息...")
    # Standard = 18 代表申万行业分类 (2021版), 假设您需要这个标准
    # 您可能需要选择不同的标准或获取所有标准的
    # InfoPublDate 应选择最新的或特定日期的
    company_codes_str = ",".join(map(str, company_codes))
    query = f"""
    SELECT CompanyCode, InfoPublDate, Standard AS standard_code, Industry AS industry_id_from_source,
           FirstIndustryCode, FirstIndustryName, SecondIndustryCode, SecondIndustryName,
           ThirdIndustryCode, ThirdIndustryName, FourthIndustryCode, FourthIndustryName,
           IfPerformed AS if_performed_code, CancelDate, InsertTime AS jydb_insert_time, XGRQ AS jydb_updated_time
    FROM LC_ExgIndustry
    WHERE CompanyCode IN ({company_codes_str}) 
    AND Standard = 18 
    AND IfPerformed = 1 
    ORDER BY CompanyCode, InfoPublDate DESC
    """ # TODO: 确认 Standard=18, IfPerformed=1 是否为所需
          # 此查询可能对每个公司返回多条记录（如果历史上行业变更），需要后续处理取最新
    try:
        df_industry = pd.read_sql(query, jydb_engine)
        # 取每个公司最新的行业分类
        df_industry = df_industry.drop_duplicates(subset=['CompanyCode', 'standard_code'], keep='first')
        logger.info(f"成功获取 {len(df_industry)} 条行业信息。")
        # TODO: 使用 CT_SystemConst 转换 standard_code, if_performed_code 为名称
        return df_industry
    except Exception as e:
        logger.error(f"获取行业信息失败: {e}")
        raise

@timing_decorator
def fetch_daily_quotes(jydb_engine, inner_codes, start_date_str, end_date_str):
    """获取指定内部编码列表和日期范围的日行情数据 (QT_DailyQuote)"""
    if not inner_codes:
        logger.info("没有内部编码可供查询日行情。")
        return pd.DataFrame()
    logger.info(f"开始获取 {len(inner_codes)} 只股票从 {start_date_str} 到 {end_date_str} 的日行情...")
    
    all_quotes_df = pd.DataFrame()
    # 为避免SQL过长，可以分批查询inner_codes
    chunk_size = 50 # 每批查询50个股票
    for i in range(0, len(inner_codes), chunk_size):
        chunk_inner_codes = inner_codes[i:i+chunk_size]
        inner_codes_str = ",".join(map(str, chunk_inner_codes))
        query = f"""
        SELECT InnerCode AS inner_code, TradingDay AS trade_date, 
               OpenPrice AS open_price, HighPrice AS high_price, LowPrice AS low_price, 
               ClosePrice AS close_price, PrevClosePrice AS prev_close_price, 
               TurnoverVolume AS volume, TurnoverValue AS amount,
               TurnoverRate AS turnover_rate, PE AS pe_ratio_ttm, PB AS pb_ratio_lf
        FROM QT_DailyQuote
        WHERE InnerCode IN ({inner_codes_str}) 
        AND TradingDay >= '{start_date_str}' AND TradingDay <= '{end_date_str}'
        ORDER BY InnerCode, TradingDay ASC
        """
        try:
            logger.info(f"查询批次 {i//chunk_size + 1} (股票代码: {inner_codes_str[:100]}...)")
            df_quotes_chunk = pd.read_sql(query, jydb_engine)
            all_quotes_df = pd.concat([all_quotes_df, df_quotes_chunk], ignore_index=True)
            logger.info(f"批次 {i//chunk_size + 1} 获取 {len(df_quotes_chunk)} 条行情, 总计 {len(all_quotes_df)} 条")
        except Exception as e:
            logger.error(f"获取股票批次 {inner_codes_str[:50]}... 的日行情失败: {e}")
            # 可选择跳过此批次或终止
    
    logger.info(f"成功获取 {len(all_quotes_df)} 条日行情数据。")
    return all_quotes_df

@timing_decorator
def fetch_index_quotes(jydb_engine, index_code_jydb, start_date_str, end_date_str, index_code_local):
    """获取指定指数和日期范围的日行情数据"""
    logger.info(f"开始获取指数 {index_code_jydb} 从 {start_date_str} 到 {end_date_str} 的日行情...")
    # 假设指数行情表为 QT_IndexQuote，字段类似QT_DailyQuote但用IndexCode
    query = f"""
    SELECT '{index_code_local}' AS index_code, TradingDay AS trade_date, 
           OpenPrice AS open_price, HighPrice AS high_price, LowPrice AS low_price, 
           ClosePrice AS close_price, PrevClosePrice AS prev_close_price,
           TurnoverVolume AS volume, TurnoverValue AS amount
    FROM QT_IndexQuote  # TODO: 确认指数行情表名和字段名
    WHERE IndexCode = '{index_code_jydb}'  # TODO: 确认 IndexCode 字段名
    AND TradingDay >= '{start_date_str}' AND TradingDay <= '{end_date_str}'
    ORDER BY TradingDay ASC
    """
    try:
        df_index_quotes = pd.read_sql(query, jydb_engine)
        logger.info(f"成功获取 {len(df_index_quotes)} 条指数 {index_code_local} 行情数据。")
        return df_index_quotes
    except Exception as e:
        logger.error(f"获取指数 {index_code_jydb} 行情失败: {e}")
        raise

@timing_decorator
def fetch_system_constants(jydb_engine, lb_codes_to_fetch):
    """获取指定的系统常量 (CT_SystemConst)"""
    if not lb_codes_to_fetch:
        logger.info("没有指定要获取的常量分类编码 (LB codes)。")
        return pd.DataFrame()
    lb_codes_str = ",".join(map(str, lb_codes_to_fetch))
    logger.info(f"开始获取 LB codes IN ({lb_codes_str}) 的系统常量...")
    query = f"""
    SELECT ID AS id_from_source, LB AS lb_code, LBMC AS lbmc_name, MS AS description, 
           DM AS dm_code, CVALUE AS cvalue, IVALUE AS ivalue, XGRQ AS jydb_updated_time
    FROM CT_SystemConst
    WHERE LB IN ({lb_codes_str})
    """
    try:
        df_constants = pd.read_sql(query, jydb_engine)
        logger.info(f"成功获取 {len(df_constants)} 条系统常量。")
        return df_constants
    except Exception as e:
        logger.error(f"获取系统常量失败: {e}")
        raise

@timing_decorator
def save_df_to_mysql(df, table_name, local_engine, if_exists='append', index=False):
    """将DataFrame保存到本地MySQL数据库"""
    if df.empty:
        logger.info(f"数据框为空，无需保存到表 {table_name}。")
        return
    logger.info(f"开始将 {len(df)} 条数据保存到本地MySQL表: {table_name} (策略: {if_exists})...")
    try:
        # 为避免 "MySQL server has gone away" 或大数据量插入问题，可以考虑分块插入
        # 但 pandas to_sql 对于 sqlalchemy 引擎本身有chunksize参数，但对于某些驱动可能不完美
        # 对于初始化，如果数据量巨大，直接 to_sql 可能较慢或出问题
        df.to_sql(name=table_name, con=local_engine, if_exists=if_exists, index=index, chunksize=10000) # chunksize 视情况调整
        logger.info(f"成功保存数据到 {table_name}。")
    except Exception as e:
        logger.error(f"保存数据到表 {table_name} 失败: {e}")
        # 可以考虑更细致的错误处理，例如记录哪些数据失败了
        raise

def main():
    logger.info("--- 开始初始化本地数据库数据 ---")
    
    # 检查 JYDB_DATABASE_URI 是否包含默认占位符
    # 对于 LOCAL_MYSQL_DATABASE_URI，由于我们使用SQLite，它总是指向一个文件路径，
    # 所以我们主要关心 JYDB_DATABASE_URI 是否已正确配置。
    if "client:123456789" not in JYDB_DATABASE_URI or "jydb" not in JYDB_DATABASE_URI.lower():
        logger.error("错误: 请先在脚本中正确配置源数据库 JYDB_DATABASE_URI！")
        logger.error("例如: 'mysql+pymysql://user:pass@host:port/jydb'")
        logger.error("脚本将不会执行。")
        return
    
    # 检查 LOCAL_MYSQL_DATABASE_URI 是否是SQLite路径 (这是预期的)
    if not LOCAL_MYSQL_DATABASE_URI.startswith("sqlite:///"):
        logger.warning(f"警告: LOCAL_MYSQL_DATABASE_URI ('{LOCAL_MYSQL_DATABASE_URI}')看起来不是一个SQLite连接字符串。")
        logger.warning("脚本将继续执行，但请确保这是您期望的本地数据库配置。")

    try:
        jydb_engine = create_engine(JYDB_DATABASE_URI)
        local_engine = create_engine(LOCAL_MYSQL_DATABASE_URI)

        # 测试数据库连接
        with jydb_engine.connect() as conn_jydb, local_engine.connect() as conn_local:
            logger.info("远程 JYDB 和本地 MySQL 数据库连接成功！")

        # 0. 定义日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=YEARS_TO_FETCH * 365)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        logger.info(f"数据下载时间范围: {start_date_str} 到 {end_date_str}")

        # 1. 获取并保存 SystemConstants
        # LB=201 (交易所), LB=1177 (证券类别), LB=207 (上市板块), LB=1176 (上市状态)
        # LB=1081 (行业划分标准), LB=999 (是否执行等通用布尔型常量，需确认JYDB中具体值)
        lb_codes_for_constants = [201, 1177, 207, 1176, 1081] # TODO: 添加其他需要的LB
        df_system_constants = fetch_system_constants(jydb_engine, lb_codes_for_constants)
        # 对于初始化，通常先清空再插入
        # save_df_to_mysql(df_system_constants, SystemConstant.__tablename__, local_engine, if_exists='replace')
        # 这里使用append，假设create_target_tables.py已创建空表，并且此脚本只运行一次进行初始化
        # 如果要重复运行，需要处理主键冲突，'replace'会删表重建，可能不是最佳选择
        # 更稳妥的做法是，先DELETE符合LB条件的数据，再INSERT，或者使用数据库的UPSERT
        logger.info(f"对于 {SystemConstant.__tablename__}，建议初始化时使用 'replace' 或先手动清空，或实现UPSERT逻辑。")
        logger.info("此处暂时跳过 system_constants 的保存，请根据实际情况处理。")
        # TODO: 实现 system_constants 的安全保存逻辑 (upsert 或 delete then insert)

        # 2. 获取A股股票列表
        df_stocks_raw = fetch_a_share_stocks(jydb_engine)
        if df_stocks_raw.empty:
            logger.info("未获取到A股股票信息，程序终止。")
            return
        
        # 重命名DataFrame的列以匹配目标表结构
        df_stocks_to_save = df_stocks_raw.rename(columns={
            'InnerCode': 'inner_code',
            'CompanyCode': 'company_code',
            'SecuCode': 'secu_code',
            'ChiName': 'chi_name',
            'SecuAbbr': 'secu_abbr',
            'SecuMarket': 'secu_market_code',
            'SecuCategory': 'secu_category_code',
            'ListedDate': 'listed_date',
            'ListedSector': 'listed_sector_code',
            'ListedState': 'listed_state_code',
            'ISIN': 'isin_code'
            # 'jydb_updated_time' 在查询中已经通过 AS 关键字重命名了，所以这里不需要
        })
        
        # TODO: 数据转换 - 将代码转换为名称 (例如，SecuMarketCode -> SecuMarketName)
        # 这个转换依赖于 df_system_constants，可以在此进行或在保存后通过SQL更新
        # 为简化，这里先保存原始代码，名称转换留作后续步骤或展示层处理
        save_df_to_mysql(df_stocks_to_save, Stock.__tablename__, local_engine, if_exists='replace') 
                                                                                         

        all_inner_codes = df_stocks_to_save['inner_code'].unique().tolist() # 使用重命名后的列名
        all_company_codes = df_stocks_to_save['company_code'].unique().tolist() # 使用重命名后的列名

        # 3. 获取并保存行业信息
        df_industry_raw = fetch_stock_industry(jydb_engine, all_company_codes)
        # TODO: 数据转换 - 行业标准名称等
        # df_industry_to_save = df_industry_raw.rename(columns={...})
        # 需要将 CompanyCode 和 InnerCode 关联起来，因为 StockIndustry 表有 inner_code 外键
        # 这里假设 LC_ExgIndustry 的 CompanyCode 可以直接关联到 SecuMain 的 CompanyCode
        # 我们需要 df_stocks_raw 中的 inner_code 来填充 StockIndustry.inner_code
        if not df_industry_raw.empty:
            # 与重命名后的 stocks DataFrame 合并
            df_industry_merged = pd.merge(df_industry_raw, df_stocks_to_save[['company_code', 'inner_code']].drop_duplicates(),
                                           left_on='CompanyCode', right_on='company_code', how='left')
            
            # 如果合并后同时存在 CompanyCode 和 company_code，则删除原始的 CompanyCode
            # 因为 company_code (小写) 是从 df_stocks_to_save 合并过来的，是我们需要的标准列名
            if 'CompanyCode' in df_industry_merged.columns and 'company_code' in df_industry_merged.columns:
                df_industry_merged = df_industry_merged.drop(columns=['CompanyCode'])
            
            # 重命名行业DataFrame的列以匹配目标表结构
            df_industry_to_save_final = df_industry_merged.rename(columns={
                # 'CompanyCode': 'company_code', # company_code 应该已经从 merge 导入且正确
                'InfoPublDate': 'info_publ_date',
                'Standard': 'standard_code', # 在源数据中是 Standard
                'Industry': 'industry_id_from_source', # 在源数据中是 Industry
                'FirstIndustryCode': 'first_industry_code',
                'FirstIndustryName': 'first_industry_name',
                'SecondIndustryCode': 'second_industry_code',
                'SecondIndustryName': 'second_industry_name',
                'ThirdIndustryCode': 'third_industry_code',
                'ThirdIndustryName': 'third_industry_name',
                'FourthIndustryCode': 'fourth_industry_code',
                'FourthIndustryName': 'fourth_industry_name',
                'IfPerformed': 'if_performed_code', # 在源数据中是 IfPerformed
                'CancelDate': 'cancel_date',
                'InsertTime': 'jydb_insert_time',
                # 'jydb_updated_time' 和 'inner_code' 已经在合并和查询中处理或名称正确
                # 注意：源查询 LC_ExgIndustry 中的 Standard, Industry, IfPerformed 字段名
                # 在 df_industry_raw 中就是 Standard, Industry, IfPerformed
            })
            
            # 确保 inner_code 列存在且正确 (合并后应该已经有了)
            if 'inner_code' not in df_industry_to_save_final.columns:
                logger.error("错误: 合并行业数据后未能找到 'inner_code' 列。")
            else:
                save_df_to_mysql(df_industry_to_save_final, StockIndustry.__tablename__, local_engine, if_exists='replace')

        # 4. 获取并保存日行情数据
        df_daily_quotes = fetch_daily_quotes(jydb_engine, all_inner_codes, start_date_str, end_date_str)
        save_df_to_mysql(df_daily_quotes, DailyQuote.__tablename__, local_engine, if_exists='replace')

        # 5. 获取并保存沪深300指数行情
        df_hs300_quotes = fetch_index_quotes(jydb_engine, HS300_INDEX_CODE_JYDB, start_date_str, end_date_str, HS300_INDEX_CODE_LOCAL)
        save_df_to_mysql(df_hs300_quotes, IndexQuote.__tablename__, local_engine, if_exists='replace')

        logger.info("--- 初始化本地数据库数据完成 ---")

    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {e}", exc_info=True)

if __name__ == "__main__":
    main() 