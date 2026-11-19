"""
Database connection and session management for PostgreSQL with SQLite fallback.
"""

import logging
import os
from pathlib import Path
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()
logger = logging.getLogger("Database")

Base = declarative_base()

def get_database_url() -> str:
    """Build database connection URL from environment variables."""
    explicit_url = os.getenv("DATABASE_URL")
    if explicit_url:
        if explicit_url.startswith("postgresql://"):
            return explicit_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return explicit_url

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "crowd_monitoring")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

def create_db_engine():
    """
    Attempt connection to PostgreSQL.
    Falls back to a local SQLite database if PostgreSQL is unavailable.
    """
    pg_url = get_database_url()
    try:
        engine = create_engine(pg_url, pool_pre_ping=True, connect_args={"connect_timeout": 3})
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info(f"Connected to PostgreSQL database at {pg_url.split('@')[-1]}")
        return engine, "postgresql"
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL ({e}). "
            f"Falling back to local SQLite database at 'crowd_monitoring.db'. "
            f"Ensure PostgreSQL is running and credentials in .env are correct."
        )
        sqlite_path = Path("crowd_monitoring.db").resolve()
        engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
        return engine, "sqlite"

engine, DB_DIALECT = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator:
    """FastAPI and app dependency for acquiring database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Create all tables in the configured database."""
    from src.storage import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    logger.info(f"Database schema initialized ({DB_DIALECT}).")
