"""
PostgreSQL database connector for saving preprocessed NASA log data
Handles table creation, data insertion, and connection management
"""

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from typing import Optional, Literal
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.config import config


class PostgreSQLConnector:
    """Handle PostgreSQL database operations for NASA log data"""
    
    def __init__(self):
        """Initialize database connector"""
        self.engine = None
        self.schema = config.POSTGRES_SCHEMA
        
    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Validate config
            is_valid, error = config.validate()
            if not is_valid:
                print(f"❌ Configuration error: {error}")
                return False
            
            # Create SQLAlchemy engine
            connection_string = config.get_connection_string()
            self.engine = create_engine(
                connection_string,
                poolclass=NullPool,  # Disable pooling for simplicity
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print(f"✓ Connected to PostgreSQL: {config.POSTGRES_DB}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            return False
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        try:
            with self.engine.connect() as conn:
                # Create schema if it doesn't exist
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
                conn.commit()
                
                # Table for parsed logs
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.parsed_logs (
                        id SERIAL PRIMARY KEY,
                        split VARCHAR(10) NOT NULL,
                        host VARCHAR(255),
                        timestamp VARCHAR(50),
                        request TEXT,
                        status INTEGER,
                        bytes BIGINT,
                        datetime TIMESTAMP WITH TIME ZONE,
                        method VARCHAR(20),
                        url TEXT,
                        version VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Table for 1-minute time series
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.timeseries_1m (
                        id SERIAL PRIMARY KEY,
                        split VARCHAR(10) NOT NULL,
                        datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                        hits INTEGER,
                        total_bytes BIGINT,
                        unique_hosts INTEGER,
                        unique_urls INTEGER,
                        unique_methods INTEGER,
                        unique_versions INTEGER,
                        min_bytes BIGINT,
                        max_bytes BIGINT,
                        mean_bytes DOUBLE PRECISION,
                        std_bytes DOUBLE PRECISION,
                        median_bytes DOUBLE PRECISION,
                        count_success INTEGER,
                        count_redirect_301 INTEGER,
                        count_redirect_302 INTEGER,
                        count_cache_304 INTEGER,
                        count_client_err INTEGER,
                        count_server_err INTEGER,
                        hits_lag_1 INTEGER,
                        hits_lag_5 INTEGER,
                        hits_lag_10 INTEGER,
                        bytes_lag_1 BIGINT,
                        bytes_lag_5 BIGINT,
                        bytes_lag_10 BIGINT,
                        hits_roll_mean_5 DOUBLE PRECISION,
                        hits_roll_std_5 DOUBLE PRECISION,
                        hits_roll_mean_10 DOUBLE PRECISION,
                        bytes_roll_mean_5 DOUBLE PRECISION,
                        bytes_roll_mean_10 DOUBLE PRECISION,
                        success_rate DOUBLE PRECISION,
                        cache_304_rate DOUBLE PRECISION,
                        redirect_rate DOUBLE PRECISION,
                        error_rate DOUBLE PRECISION,
                        error_rate_lag_1 DOUBLE PRECISION,
                        redirect_rate_lag_1 DOUBLE PRECISION,
                        cache_rate_lag_1 DOUBLE PRECISION,
                        hour INTEGER,
                        day_of_week INTEGER,
                        is_weekend INTEGER,
                        sin_hour DOUBLE PRECISION,
                        cos_hour DOUBLE PRECISION,
                        sin_dow DOUBLE PRECISION,
                        cos_dow DOUBLE PRECISION,
                        hits_diff_1 INTEGER,
                        bytes_diff_1 BIGINT,
                        hits_pct_change_1 DOUBLE PRECISION,
                        bytes_pct_change_1 DOUBLE PRECISION,
                        bytes_per_hit DOUBLE PRECISION,
                        is_gap INTEGER,
                        time_gap_sec DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Table for 5-minute time series
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.timeseries_5m (
                        id SERIAL PRIMARY KEY,
                        split VARCHAR(10) NOT NULL,
                        datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                        hits INTEGER,
                        total_bytes BIGINT,
                        unique_hosts INTEGER,
                        unique_urls INTEGER,
                        unique_methods INTEGER,
                        unique_versions INTEGER,
                        min_bytes BIGINT,
                        max_bytes BIGINT,
                        mean_bytes DOUBLE PRECISION,
                        std_bytes DOUBLE PRECISION,
                        median_bytes DOUBLE PRECISION,
                        count_success INTEGER,
                        count_redirect_301 INTEGER,
                        count_redirect_302 INTEGER,
                        count_cache_304 INTEGER,
                        count_client_err INTEGER,
                        count_server_err INTEGER,
                        hits_lag_1 INTEGER,
                        hits_lag_3 INTEGER,
                        hits_lag_6 INTEGER,
                        bytes_lag_1 BIGINT,
                        bytes_lag_3 BIGINT,
                        bytes_lag_6 BIGINT,
                        hits_roll_mean_3 DOUBLE PRECISION,
                        hits_roll_std_3 DOUBLE PRECISION,
                        hits_roll_mean_6 DOUBLE PRECISION,
                        bytes_roll_mean_3 DOUBLE PRECISION,
                        bytes_roll_mean_6 DOUBLE PRECISION,
                        success_rate DOUBLE PRECISION,
                        redirect_301_rate DOUBLE PRECISION,
                        redirect_302_rate DOUBLE PRECISION,
                        cache_304_rate DOUBLE PRECISION,
                        client_error_rate DOUBLE PRECISION,
                        server_error_rate DOUBLE PRECISION,
                        redirect_rate DOUBLE PRECISION,
                        error_rate DOUBLE PRECISION,
                        error_rate_lag_1 DOUBLE PRECISION,
                        redirect_rate_lag_1 DOUBLE PRECISION,
                        cache_rate_lag_1 DOUBLE PRECISION,
                        hour INTEGER,
                        day_of_week INTEGER,
                        is_weekend INTEGER,
                        sin_hour DOUBLE PRECISION,
                        cos_hour DOUBLE PRECISION,
                        sin_dow DOUBLE PRECISION,
                        cos_dow DOUBLE PRECISION,
                        hits_diff_1 INTEGER,
                        bytes_diff_1 BIGINT,
                        hits_pct_change_1 DOUBLE PRECISION,
                        bytes_pct_change_1 DOUBLE PRECISION,
                        bytes_per_hit DOUBLE PRECISION,
                        is_gap INTEGER,
                        time_gap_sec DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Table for 15-minute time series
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.schema}.timeseries_15m (
                        id SERIAL PRIMARY KEY,
                        split VARCHAR(10) NOT NULL,
                        datetime TIMESTAMP WITH TIME ZONE NOT NULL,
                        hits INTEGER,
                        total_bytes BIGINT,
                        unique_hosts INTEGER,
                        unique_urls INTEGER,
                        unique_methods INTEGER,
                        unique_versions INTEGER,
                        min_bytes BIGINT,
                        max_bytes BIGINT,
                        mean_bytes DOUBLE PRECISION,
                        std_bytes DOUBLE PRECISION,
                        median_bytes DOUBLE PRECISION,
                        count_success INTEGER,
                        count_redirect_301 INTEGER,
                        count_redirect_302 INTEGER,
                        count_cache_304 INTEGER,
                        count_client_err INTEGER,
                        count_server_err INTEGER,
                        hits_lag_1 INTEGER,
                        hits_lag_2 INTEGER,
                        hits_lag_4 INTEGER,
                        bytes_lag_1 BIGINT,
                        bytes_lag_2 BIGINT,
                        bytes_lag_4 BIGINT,
                        hits_roll_mean_2 DOUBLE PRECISION,
                        hits_roll_std_2 DOUBLE PRECISION,
                        hits_roll_mean_4 DOUBLE PRECISION,
                        hits_roll_std_4 DOUBLE PRECISION,
                        bytes_roll_mean_2 DOUBLE PRECISION,
                        bytes_roll_mean_4 DOUBLE PRECISION,
                        bytes_roll_std_2 DOUBLE PRECISION,
                        bytes_roll_std_4 DOUBLE PRECISION,
                        success_rate DOUBLE PRECISION,
                        redirect_301_rate DOUBLE PRECISION,
                        redirect_302_rate DOUBLE PRECISION,
                        cache_304_rate DOUBLE PRECISION,
                        client_error_rate DOUBLE PRECISION,
                        server_error_rate DOUBLE PRECISION,
                        redirect_rate DOUBLE PRECISION,
                        error_rate DOUBLE PRECISION,
                        error_rate_lag_1 DOUBLE PRECISION,
                        redirect_rate_lag_1 DOUBLE PRECISION,
                        cache_rate_lag_1 DOUBLE PRECISION,
                        hour INTEGER,
                        day_of_week INTEGER,
                        is_weekend INTEGER,
                        sin_hour DOUBLE PRECISION,
                        cos_hour DOUBLE PRECISION,
                        sin_dow DOUBLE PRECISION,
                        cos_dow DOUBLE PRECISION,
                        hits_diff_1 INTEGER,
                        bytes_diff_1 BIGINT,
                        hits_pct_change_1 DOUBLE PRECISION,
                        bytes_pct_change_1 DOUBLE PRECISION,
                        bytes_per_hit DOUBLE PRECISION,
                        is_gap INTEGER,
                        time_gap_sec DOUBLE PRECISION,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Create indexes for better query performance
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_parsed_logs_split ON {self.schema}.parsed_logs(split)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_parsed_logs_datetime ON {self.schema}.parsed_logs(datetime)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_1m_split ON {self.schema}.timeseries_1m(split)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_1m_datetime ON {self.schema}.timeseries_1m(datetime)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_5m_split ON {self.schema}.timeseries_5m(split)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_5m_datetime ON {self.schema}.timeseries_5m(datetime)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_15m_split ON {self.schema}.timeseries_15m(split)"))
                conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_ts_15m_datetime ON {self.schema}.timeseries_15m(datetime)"))
                
                conn.commit()
            
            print("✓ Tables created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create tables: {e}")
            raise
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        split: Literal["train", "test"],
        if_exists: str = "append"
    ) -> bool:
        """
        Save DataFrame to PostgreSQL table
        
        Args:
            df: DataFrame to save
            table_name: Target table name (without schema prefix)
            split: Data split (train or test)
            if_exists: How to behave if table exists ('append', 'replace', 'fail')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Add split column if not present
            if 'split' not in df.columns:
                df = df.copy()
                df['split'] = split
            
            # Save to database
            full_table_name = f"{self.schema}.{table_name}"
            df.to_sql(
                table_name,
                self.engine,
                schema=self.schema,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            print(f"✓ Saved {len(df):,} rows to {full_table_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save to {table_name}: {e}")
            return False
    
    def clear_split_data(self, split: Literal["train", "test"]):
        """
        Clear all data for a specific split before re-importing
        
        Args:
            split: Data split to clear
        """
        try:
            with self.engine.connect() as conn:
                tables = ['parsed_logs', 'timeseries_1m', 'timeseries_5m', 'timeseries_15m']
                for table in tables:
                    conn.execute(text(f"DELETE FROM {self.schema}.{table} WHERE split = :split"), {"split": split})
                conn.commit()
            
            print(f"✓ Cleared existing {split} data")
            
        except Exception as e:
            print(f"❌ Failed to clear {split} data: {e}")
    
    def get_table_stats(self):
        """Print statistics for all tables"""
        try:
            with self.engine.connect() as conn:
                tables = ['parsed_logs', 'timeseries_1m', 'timeseries_5m', 'timeseries_15m']
                
                print("\nDatabase Statistics:")
                print("=" * 60)
                
                for table in tables:
                    result = conn.execute(text(f"""
                        SELECT 
                            split,
                            COUNT(*) as row_count,
                            MIN(datetime) as min_date,
                            MAX(datetime) as max_date
                        FROM {self.schema}.{table}
                        WHERE datetime IS NOT NULL
                        GROUP BY split
                        ORDER BY split
                    """))
                    
                    rows = result.fetchall()
                    if rows:
                        print(f"\n{table}:")
                        for row in rows:
                            print(f"  {row[0]}: {row[1]:,} rows | {row[2]} → {row[3]}")
                    else:
                        print(f"\n{table}: (empty)")
                
                print("=" * 60)
                
        except Exception as e:
            print(f"❌ Failed to get table stats: {e}")
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            print("✓ Database connection closed")


def test_connection():
    """Test PostgreSQL connection and configuration"""
    print("\n" + "=" * 60)
    print("Testing PostgreSQL Connection")
    print("=" * 60 + "\n")
    
    config.print_config()
    print()
    
    connector = PostgreSQLConnector()
    
    if connector.connect():
        print("✓ Connection test successful!")
        connector.close()
        return True
    else:
        print("❌ Connection test failed!")
        print("\nTroubleshooting:")
        print("  1. Check your .env file exists and has correct credentials")
        print("  2. Ensure PostgreSQL server is running")
        print("  3. Verify database exists")
        print("  4. Check network connectivity and firewall settings")
        return False


if __name__ == "__main__":
    test_connection()