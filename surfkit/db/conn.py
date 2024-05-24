import logging
import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from surfkit.config import AGENTSEA_DB_DIR, DB_NAME

from .models import Base

logger = logging.getLogger(__name__)

DB_TYPE = os.environ.get("DB_TYPE", "sqlite")


def get_pg_conn() -> Engine:
    # Helper function to get environment variable with fallback
    def get_env_var(key: str) -> str:
        task_key = f"SURFKIT_{key}"
        value = os.environ.get(task_key)
        if value is None:
            value = os.environ.get(key)
            if value is None:
                raise ValueError(f"${key} must be set")
        return value

    # Retrieve environment variables with fallbacks
    db_user = get_env_var("DB_USER")
    db_pass = get_env_var("DB_PASS")
    db_host = get_env_var("DB_HOST")
    db_name = get_env_var("DB_NAME")

    logging.debug(f"connecting to db on postgres host '{db_host}' with db '{db_name}'")
    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}/{db_name}",
        client_encoding="utf8",
    )

    return engine


def get_sqlite_conn() -> Engine:
    db_path = os.path.join(AGENTSEA_DB_DIR, DB_NAME)
    logger.debug(f"connecting to local sqlite db {db_path}")
    os.makedirs(AGENTSEA_DB_DIR, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    return engine


if DB_TYPE == "postgres":
    engine = get_pg_conn()
else:
    engine = get_sqlite_conn()
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(bind=engine)


class WithDB:
    @staticmethod
    def get_db():
        """Get a database connection

        Example:
            ```
            for session in self.get_db():
                session.add(foo)
            ```
        """
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()


def get_db():
    """Get a database connection

    Example:
        ```
        for session in get_db():
            session.add(foo)
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
