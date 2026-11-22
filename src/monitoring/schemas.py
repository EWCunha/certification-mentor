from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime


class Base(DeclarativeBase):
    pass


class Log(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str]
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now)
    message: Mapped[str]
