import os
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """PostgreSQL database configuration"""
    
    # Priority 1: Use POSTGRES_URL if provided
    POSTGRES_URL = os.getenv('POSTGRES_URL', '')
    
    # Priority 2: Parse URL or use individual parameters
    if POSTGRES_URL:
        # Parse the connection URL
        parsed = urlparse(POSTGRES_URL)
        POSTGRES_HOST = parsed.hostname or '127.0.0.1'
        POSTGRES_PORT = parsed.port or 5432
        POSTGRES_DB = parsed.path.lstrip('/') if parsed.path else 'nasa_logs'
        POSTGRES_USER = parsed.username or 'postgres'
        POSTGRES_PASSWORD = parsed.password or ''
    else:
        # Use individual environment variables
        POSTGRES_HOST = os.getenv('POSTGRES_HOST', '127.0.0.1')
        POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', 5432))
        POSTGRES_DB = os.getenv('POSTGRES_DB', 'nasa_logs')
        POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
        POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # Connection pool settings
    POSTGRES_POOL_SIZE = int(os.getenv('POSTGRES_POOL_SIZE', 5))
    POSTGRES_MAX_OVERFLOW = int(os.getenv('POSTGRES_MAX_OVERFLOW', 10))
    
    # Schema
    POSTGRES_SCHEMA = os.getenv('POSTGRES_SCHEMA', 'public')
    
    # Enable/disable PostgreSQL saving
    SAVE_TO_POSTGRES = os.getenv('SAVE_TO_POSTGRES', 'true').lower() == 'true'
    
    @classmethod
    def get_connection_string(cls) -> str:
        """
        Generate SQLAlchemy connection string
        """
        # Use POSTGRES_URL if provided, otherwise construct from parts
        if cls.POSTGRES_URL:
            return cls.POSTGRES_URL
        
        return (
            f"postgresql+psycopg2://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}"
            f"@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"
        )
    
    @classmethod
    def get_psycopg2_params(cls) -> dict:
        """
        Generate psycopg2 connection parameters
        """
        return {
            'host': cls.POSTGRES_HOST,
            'port': cls.POSTGRES_PORT,
            'database': cls.POSTGRES_DB,
            'user': cls.POSTGRES_USER,
            'password': cls.POSTGRES_PASSWORD,
            'options': f'-c search_path={cls.POSTGRES_SCHEMA}',
        }
    
    @classmethod
    def validate(cls) -> tuple[bool, str]:
        """
        Validate configuration
        
        """
        if not cls.POSTGRES_USER:
            return False, "POSTGRES_USER not set"
        
        if not cls.POSTGRES_DB:
            return False, "POSTGRES_DB not set"
        
        return True, ""
    
    @classmethod
    def print_config(cls, hide_password: bool = True):
        """Print current configuration (for debugging)"""
        password = "***" if hide_password else cls.POSTGRES_PASSWORD
        
        print("PostgreSQL Configuration:")
        if cls.POSTGRES_URL:
            # Mask password in URL
            if hide_password and cls.POSTGRES_PASSWORD:
                masked_url = cls.POSTGRES_URL.replace(cls.POSTGRES_PASSWORD, "***")
                print(f"  Connection URL: {masked_url}")
            else:
                print(f"  Connection URL: {cls.POSTGRES_URL}")
        else:
            print(f"  Host: {cls.POSTGRES_HOST}")
            print(f"  Port: {cls.POSTGRES_PORT}")
            print(f"  Database: {cls.POSTGRES_DB}")
            print(f"  User: {cls.POSTGRES_USER}")
            print(f"  Password: {password}")
        print(f"  Schema: {cls.POSTGRES_SCHEMA}")
        print(f"  Save to PostgreSQL: {cls.SAVE_TO_POSTGRES}")


# Create global config instance
config = Config()