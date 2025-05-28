"""
脚本说明:
本脚本使用 SQLAlchemy 定义并创建目标 MySQL 数据库中的表结构。
这些表用于存储处理后的股票基本信息、行业信息、行情数据以及相关的系统常量。

在运行此脚本之前，请确保：
1. 您已经安装了 mysql-connector-python 和 SQLAlchemy:
   pip install mysql-connector-python sqlalchemy
2. 您的 MySQL 服务器正在运行。
3. 您已经创建了目标数据库 (例如，名为 'bond_prediction_data')。
4. 您已经更新了下面的 DATABASE_URI，使其包含正确的数据库连接信息。
"""
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Date, TIMESTAMP, ForeignKey, BigInteger, func, Numeric
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import UniqueConstraint, PrimaryKeyConstraint

# --- 配置数据库连接 ---
# 请将下面的字符串替换为您的实际 MySQL 数据库连接信息
# 格式: "mysql+mysqlconnector://USER:PASSWORD@HOST:PORT/DATABASE_NAME"
DATABASE_URI = "sqlite:///data/bond_prediction_data.db"
# 例如: "mysql+mysqlconnector://root:password@localhost:3306/bond_prediction_data"

Base = declarative_base()

class Stock(Base):
    __tablename__ = "stocks"

    inner_code = Column(Integer, primary_key=True, comment="证券内部编码 (源自 SecuMain.InnerCode)")
    company_code = Column(Integer, index=True, comment="公司代码 (源自 SecuMain.CompanyCode)")
    secu_code = Column(String(30), nullable=False, index=True, comment="证券代码 (源自 SecuMain.SecuCode)")
    chi_name = Column(String(200), nullable=False, comment="中文名称 (源自 SecuMain.ChiName)")
    secu_abbr = Column(String(100), nullable=False, comment="证券简称 (源自 SecuMain.SecuAbbr)")
    
    secu_market_code = Column(Integer, comment="原始证券市场代码 (源自 SecuMain.SecuMarket)")
    secu_market_name = Column(String(100), comment="证券市场名称 (通过 CT_SystemConst.LB=201 转换)")
    
    secu_category_code = Column(Integer, comment="原始证券类别代码 (源自 SecuMain.SecuCategory)")
    secu_category_name = Column(String(100), comment="证券类别名称 (通过 CT_SystemConst.LB=1177 转换, 如 A股)")
    
    listed_date = Column(Date, comment="上市日期 (源自 SecuMain.ListedDate)")
    
    listed_sector_code = Column(Integer, comment="原始上市板块代码 (源自 SecuMain.ListedSector)")
    listed_sector_name = Column(String(100), comment="上市板块名称 (通过 CT_SystemConst.LB=207 转换)")
    
    listed_state_code = Column(Integer, comment="原始上市状态代码 (源自 SecuMain.ListedState)")
    listed_state_name = Column(String(100), comment="上市状态名称 (通过 CT_SystemConst.LB=1176 转换)")
    
    isin_code = Column(String(20), comment="ISIN代码 (源自 SecuMain.ISIN)")
    
    jydb_updated_time = Column(DateTime, nullable=False, comment="聚源数据库中记录的更新时间 (源自 SecuMain.XGRQ)")
    
    etl_created_at = Column(TIMESTAMP, server_default=func.now(), comment="ETL创建时间戳")
    etl_updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="ETL更新时间戳")

    industries = relationship("StockIndustry", back_populates="stock")
    daily_quotes = relationship("DailyQuote", back_populates="stock")

    def __repr__(self):
        return f"<Stock(inner_code={self.inner_code}, secu_code='{self.secu_code}', secu_abbr='{self.secu_abbr}')>"

class StockIndustry(Base):
    __tablename__ = "stock_industry"

    company_code = Column(Integer, comment="公司代码 (源自 LC_ExgIndustry.CompanyCode)")
    info_publ_date = Column(Date, comment="信息发布日期 (源自 LC_ExgIndustry.InfoPublDate)")
    standard_code = Column(Integer, comment="行业划分标准代码 (源自 LC_ExgIndustry.Standard)")
    industry_id_from_source = Column(Integer, comment="源行业ID (源自 LC_ExgIndustry.Industry)")

    inner_code = Column(Integer, ForeignKey('stocks.inner_code'), nullable=False, index=True, comment="证券内部编码, 关联 stocks 表")
    
    standard_name = Column(String(255), comment="行业划分标准名称 (通过 CT_SystemConst.LB=1081 等转换)")
    
    first_industry_code = Column(String(20), comment="一级行业代码 (源自 LC_ExgIndustry)")
    first_industry_name = Column(String(100), comment="一级行业名称 (源自 LC_ExgIndustry)")
    second_industry_code = Column(String(20), comment="二级行业代码 (源自 LC_ExgIndustry)")
    second_industry_name = Column(String(100), comment="二级行业名称 (源自 LC_ExgIndustry)")
    third_industry_code = Column(String(20), comment="三级行业代码 (源自 LC_ExgIndustry)")
    third_industry_name = Column(String(100), comment="三级行业名称 (源自 LC_ExgIndustry)")
    fourth_industry_code = Column(String(20), comment="四级行业代码 (源自 LC_ExgIndustry)")
    fourth_industry_name = Column(String(100), comment="四级行业名称 (源自 LC_ExgIndustry)")
    
    if_performed_code = Column(Integer, comment="是否执行代码 (源自 LC_ExgIndustry.IfPerformed)")
    if_performed_name = Column(String(50), comment="是否执行名称 (通过 CT_SystemConst 转换)")
    
    cancel_date = Column(Date, comment="取消日期 (源自 LC_ExgIndustry.CancelDate)")
    
    jydb_insert_time = Column(DateTime, comment="聚源数据库记录插入时间 (源自 LC_ExgIndustry.InsertTime)")
    jydb_updated_time = Column(DateTime, nullable=False, comment="聚源数据库记录修改日期 (源自 LC_ExgIndustry.XGRQ)")
    
    etl_created_at = Column(TIMESTAMP, server_default=func.now(), comment="ETL创建时间戳")
    etl_updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="ETL更新时间戳")

    stock = relationship("Stock", back_populates="industries")
    
    __table_args__ = (
        PrimaryKeyConstraint('company_code', 'info_publ_date', 'standard_code', 'industry_id_from_source', name='pk_stock_industry'),
    )

    def __repr__(self):
        return f"<StockIndustry(company_code={self.company_code}, standard_code={self.standard_code}, first_industry_name='{self.first_industry_name}')>"

class SystemConstant(Base):
    __tablename__ = "system_constants"

    lb_code = Column(Integer, comment="常量分类编码 (源自 CT_SystemConst.LB)")
    dm_code = Column(Integer, comment="常量代码 (源自 CT_SystemConst.DM)")
    
    id_from_source = Column(BigInteger, nullable=False, unique=True, comment="源表ID (源自 CT_SystemConst.ID)") # Should be unique
    lbmc_name = Column(String(50), nullable=False, comment="常量分类名称 (源自 CT_SystemConst.LBMC)")
    description = Column(String(300), comment="常量描述 (源自 CT_SystemConst.MS)")
    cvalue = Column(String(2000), comment="字符值 (源自 CT_SystemConst.CVALUE)")
    ivalue = Column(Integer, comment="整型值, 用于区分层级等 (源自 CT_SystemConst.IVALUE)")
    
    jydb_updated_time = Column(DateTime, nullable=False, comment="聚源数据库记录修改日期 (源自 CT_SystemConst.XGRQ)")
    etl_created_at = Column(TIMESTAMP, server_default=func.now(), comment="ETL创建时间戳")

    __table_args__ = (
        PrimaryKeyConstraint('lb_code', 'dm_code', name='pk_system_constant'),
    )
    
    def __repr__(self):
        return f"<SystemConstant(lb_code={self.lb_code}, dm_code={self.dm_code}, description='{self.description}')>"

class DailyQuote(Base):
    __tablename__ = "daily_quotes"

    inner_code = Column(Integer, ForeignKey('stocks.inner_code'), comment="证券内部编码")
    trade_date = Column(Date, comment="交易日期")
    
    open_price = Column(Numeric(10, 2), comment="开盘价")
    high_price = Column(Numeric(10, 2), comment="最高价")
    low_price = Column(Numeric(10, 2), comment="最低价")
    close_price = Column(Numeric(10, 2), comment="收盘价")
    prev_close_price = Column(Numeric(10, 2), comment="昨收盘价 (源自 QT_DailyQuote.PrevClosePrice)")
    
    volume = Column(BigInteger, comment="成交量 (股, 源自 QT_DailyQuote.TurnoverVolume)")
    amount = Column(Numeric(18, 2), comment="成交额 (元, 源自 QT_DailyQuote.TurnoverValue)")
    
    turnover_rate = Column(Numeric(8, 4), comment="换手率 (源自 QT_DailyQuote.TurnoverRate)")
    pe_ratio_ttm = Column(Numeric(10, 2), comment="市盈率TTM (源自 QT_DailyQuote.PE)") # Assuming TTM
    pb_ratio_lf = Column(Numeric(10, 2), comment="市净率LF (源自 QT_DailyQuote.PB)") # Assuming LF

    etl_created_at = Column(TIMESTAMP, server_default=func.now(), comment="ETL创建时间戳")
    etl_updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="ETL更新时间戳")

    stock = relationship("Stock", back_populates="daily_quotes")

    __table_args__ = (
        PrimaryKeyConstraint('inner_code', 'trade_date', name='pk_daily_quote'),
        # Index('ix_daily_quote_trade_date', 'trade_date'), # Optional: if querying by date a lot
    )

    def __repr__(self):
        return f"<DailyQuote(inner_code={self.inner_code}, trade_date={self.trade_date}, close_price={self.close_price})>"

class IndexQuote(Base):
    __tablename__ = "index_quotes"

    index_code = Column(String(20), comment="指数代码 (如 000300.SH)")
    trade_date = Column(Date, comment="交易日期")

    open_price = Column(Numeric(10, 2), comment="开盘价")
    high_price = Column(Numeric(10, 2), comment="最高价")
    low_price = Column(Numeric(10, 2), comment="最低价")
    close_price = Column(Numeric(10, 2), comment="收盘价")
    prev_close_price = Column(Numeric(10, 2), comment="昨收盘价")
    
    volume = Column(BigInteger, comment="成交量")
    amount = Column(Numeric(18, 2), comment="成交额")
    
    etl_created_at = Column(TIMESTAMP, server_default=func.now(), comment="ETL创建时间戳")
    etl_updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="ETL更新时间戳")

    __table_args__ = (
        PrimaryKeyConstraint('index_code', 'trade_date', name='pk_index_quote'),
    )

    def __repr__(self):
        return f"<IndexQuote(index_code='{self.index_code}', trade_date={self.trade_date}, close_price={self.close_price})>"

def create_tables(engine):
    """在数据库中创建所有定义的表"""
    Base.metadata.create_all(engine)
    print("成功创建表结构 (如果它们尚不存在)。")

if __name__ == "__main__":
    # 检查 DATABASE_URI 是否已配置
    if "your_user:your_password" in DATABASE_URI or "your_database_name" in DATABASE_URI:
        print("错误: 请先在脚本中配置 DATABASE_URI！")
        print("脚本将不会执行。请更新 DATABASE_URI 并重新运行。")
    # 对于SQLite，我们不需要检查 "your_user:your_password" 或 "your_database_name"
    # 只需要确保DATABASE_URI不是默认的MySQL示例即可。
    # 更准确的检查可能是检查它是否仍然包含 "mysql+mysqlconnector" 和 "your_"
    elif "mysql+mysqlconnector" in DATABASE_URI and ("your_user" in DATABASE_URI or "your_database_name" in DATABASE_URI):
        print("错误: 检测到默认的MySQL连接字符串，但配置应为SQLite。")
        print("请确保 DATABASE_URI 已正确设置为SQLite路径，例如 'sqlite:///data/bond_prediction_data.db'。")
    else:
        try:
            engine = create_engine(DATABASE_URI)
            # 测试连接
            with engine.connect() as connection:
                print("数据库连接成功！")
            
            # 创建表
            create_tables(engine)
            
            print("\n请检查您的 MySQL 数据库，确认表 'stocks', 'stock_industry', 'system_constants', 'daily_quotes', 和 'index_quotes' 已成功创建。")
            
        except Exception as e:
            print(f"连接到数据库或创建表时发生错误: {e}")
            print("请检查您的 DATABASE_URI 配置和 MySQL 服务器状态。") 