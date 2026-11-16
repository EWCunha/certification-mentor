from sqlite3 import Connection as SQLiteConnection
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session
from pathlib import Path
from evals.schemas import Base


DB_PATH = Path(__file__).resolve().parent / "eval.db"


def _enable_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragma(
        dbapi_connection: SQLiteConnection, _connection_record: object
    ) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def engine_for(path: Path = DB_PATH) -> Engine:
    engine = create_engine(f"sqlite:///{path}")
    _enable_foreign_keys(engine)
    return engine


def create_database(path: Path = DB_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine_for(path))
    return path


def session_for(path: Path = DB_PATH) -> Session:
    return Session(engine_for(path))


if __name__ == "__main__":
    print(create_database())
