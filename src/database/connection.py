"""
Database connection and session management.

This module handles database connection, session management, and initialization.
The design allows easy swapping between SQLite (development) and PostgreSQL (production).

Configuration is done via environment variables:
- DATABASE_URL: Full connection string (overrides other settings)
- DATABASE_TYPE: "sqlite" or "postgresql" (default: sqlite)
- DATABASE_PATH: Path for SQLite database file (default: data/stock_analyst.db)
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get the database URL from environment or build from components.

    Returns:
        Database connection URL

    Supports:
    - SQLite: sqlite:///path/to/db.sqlite
    - PostgreSQL: postgresql://user:pass@host:port/dbname

    To switch to PostgreSQL in the future, simply set:
        DATABASE_URL=postgresql://user:password@localhost:5432/stock_analyst
    """
    # Check for explicit DATABASE_URL first
    if url := os.environ.get("DATABASE_URL"):
        return url

    db_type = os.environ.get("DATABASE_TYPE", "sqlite").lower()

    if db_type == "sqlite":
        # Get database path, default to data directory
        db_path = os.environ.get(
            "DATABASE_PATH",
            str(Path(__file__).parent.parent.parent / "data" / "stock_analyst.db"),
        )
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{db_path}"

    elif db_type == "postgresql":
        # Build PostgreSQL URL from components
        host = os.environ.get("DATABASE_HOST", "localhost")
        port = os.environ.get("DATABASE_PORT", "5432")
        user = os.environ.get("DATABASE_USER", "postgres")
        password = os.environ.get("DATABASE_PASSWORD", "")
        dbname = os.environ.get("DATABASE_NAME", "stock_analyst")
        return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    else:
        raise ValueError(f"Unsupported database type: {db_type}")


def get_engine(url: Optional[str] = None, **kwargs) -> Engine:
    """
    Create SQLAlchemy engine.

    Args:
        url: Database URL (uses get_database_url() if not provided)
        **kwargs: Additional engine arguments

    Returns:
        SQLAlchemy Engine
    """
    db_url = url or get_database_url()

    # Default engine arguments
    engine_args = {
        "echo": os.environ.get("DATABASE_ECHO", "false").lower() == "true",
    }

    # SQLite-specific settings
    if db_url.startswith("sqlite"):
        engine_args["connect_args"] = {"check_same_thread": False}

    # PostgreSQL-specific settings
    elif db_url.startswith("postgresql"):
        engine_args["pool_size"] = int(os.environ.get("DATABASE_POOL_SIZE", "5"))
        engine_args["max_overflow"] = int(os.environ.get("DATABASE_MAX_OVERFLOW", "10"))
        engine_args["pool_pre_ping"] = True  # Check connection health

    engine_args.update(kwargs)

    engine = create_engine(db_url, **engine_args)

    # Enable SQLite foreign keys
    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    logger.info(f"Database engine created for: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    return engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """
    Create a session factory.

    Args:
        engine: SQLAlchemy engine (creates new if not provided)

    Returns:
        Session factory
    """
    if engine is None:
        engine = get_engine()

    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def init_database(engine: Optional[Engine] = None) -> None:
    """
    Initialize database tables.

    Creates all tables defined in the models if they don't exist.

    Args:
        engine: SQLAlchemy engine (creates new if not provided)
    """
    from database.models import Base

    if engine is None:
        engine = get_engine()

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized")


def drop_all_tables(engine: Optional[Engine] = None) -> None:
    """
    Drop all database tables.

    WARNING: This will delete all data!

    Args:
        engine: SQLAlchemy engine (creates new if not provided)
    """
    from database.models import Base

    if engine is None:
        engine = get_engine()

    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


class DatabaseSession:
    """
    Database session context manager.

    Usage:
        with DatabaseSession() as session:
            trades = session.query(TradeModel).all()

    Or as async context manager (for future async support):
        async with DatabaseSession() as session:
            trades = await session.execute(select(TradeModel))
    """

    _session_factory: Optional[sessionmaker] = None

    def __init__(self, session_factory: Optional[sessionmaker] = None):
        """
        Initialize session manager.

        Args:
            session_factory: Session factory (uses global if not provided)
        """
        if session_factory:
            self._factory = session_factory
        else:
            if DatabaseSession._session_factory is None:
                DatabaseSession._session_factory = get_session_factory()
            self._factory = DatabaseSession._session_factory

        self._session: Optional[Session] = None

    def __enter__(self) -> Session:
        """Enter context, create session."""
        self._session = self._factory()
        return self._session

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context, handle commit/rollback."""
        if self._session is None:
            return

        try:
            if exc_type is None:
                self._session.commit()
            else:
                self._session.rollback()
                logger.error(f"Session rolled back due to: {exc_val}")
        finally:
            self._session.close()
            self._session = None

    async def __aenter__(self) -> Session:
        """Async enter (for future async support)."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async exit (for future async support)."""
        self.__exit__(exc_type, exc_val, exc_tb)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Get a database session as a context manager.

    Usage:
        with get_db_session() as session:
            trades = session.query(TradeModel).all()

    Yields:
        SQLAlchemy Session
    """
    with DatabaseSession() as session:
        yield session


# =============================================================================
# Utility Functions
# =============================================================================


def reset_database(engine: Optional[Engine] = None) -> None:
    """
    Reset database by dropping and recreating all tables.

    WARNING: This will delete all data!

    Args:
        engine: SQLAlchemy engine (creates new if not provided)
    """
    if engine is None:
        engine = get_engine()

    drop_all_tables(engine)
    init_database(engine)
    logger.info("Database reset complete")


def get_database_info() -> dict:
    """
    Get information about the current database configuration.

    Returns:
        Dictionary with database info
    """
    url = get_database_url()

    # Parse URL for info (hide password)
    if "@" in url:
        # PostgreSQL or similar with credentials
        protocol, rest = url.split("://", 1)
        if "@" in rest:
            creds, host_part = rest.rsplit("@", 1)
            user = creds.split(":")[0] if ":" in creds else creds
            safe_url = f"{protocol}://{user}:***@{host_part}"
        else:
            safe_url = url
    else:
        safe_url = url

    return {
        "url": safe_url,
        "type": "sqlite" if url.startswith("sqlite") else "postgresql",
        "echo": os.environ.get("DATABASE_ECHO", "false").lower() == "true",
    }
