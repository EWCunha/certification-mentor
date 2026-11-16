from sqlalchemy import ForeignKey
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Domain(Base):
    __tablename__ = "domains"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    weight: Mapped[int]
    cca_f_domain: Mapped[str]

    task_statements: Mapped[list["TaskStatement"]] = relationship(
        back_populates="domain"
    )
    principles: Mapped[list["Principle"]] = relationship(back_populates="domain")
    questions: Mapped[list["Question"]] = relationship(back_populates="domain")


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    description: Mapped[str]

    questions: Mapped[list["Question"]] = relationship(back_populates="scenario")


class TaskStatement(Base):
    __tablename__ = "task_statements"

    code: Mapped[str] = mapped_column(primary_key=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"), index=True)
    title: Mapped[str]

    domain: Mapped[Domain] = relationship(back_populates="task_statements")
    questions: Mapped[list["Question"]] = relationship(back_populates="task_statement")


class Principle(Base):
    __tablename__ = "principles"

    code: Mapped[str] = mapped_column(primary_key=True)
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"), index=True)
    rule: Mapped[str]
    trap: Mapped[str]

    domain: Mapped[Domain] = relationship(back_populates="principles")
    questions: Mapped[list["Question"]] = relationship(back_populates="principle")


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[str] = mapped_column(primary_key=True)
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id"))
    domain_id: Mapped[int] = mapped_column(ForeignKey("domains.id"), index=True)
    task_code: Mapped[str] = mapped_column(
        ForeignKey("task_statements.code"), index=True
    )
    principle_code: Mapped[str] = mapped_column(ForeignKey("principles.code"))
    stem: Mapped[str]
    explanation: Mapped[str]
    multi_response: Mapped[bool]
    from_exam_guide: Mapped[bool] = mapped_column(index=True)
    contrast_note: Mapped[str | None]

    scenario: Mapped[Scenario] = relationship(back_populates="questions")
    domain: Mapped[Domain] = relationship(back_populates="questions")
    task_statement: Mapped[TaskStatement] = relationship(back_populates="questions")
    principle: Mapped[Principle] = relationship(back_populates="questions")
    choices: Mapped[list["Choice"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )
    relevance_labels: Mapped[list["Relevance"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict:
        data = {
            column.key: getattr(self, column.key)
            for column in inspect(self).mapper.column_attrs
        }
        for relation in inspect(self).mapper.relationships:
            if relation.direction.name != "MANYTOONE":
                continue
            related = getattr(self, relation.key)
            if related is None:
                data[relation.key] = None
                continue
            data[relation.key] = {
                column.key: getattr(related, column.key)
                for column in inspect(related).mapper.column_attrs
            }
        return data


class Choice(Base):
    __tablename__ = "choices"

    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(primary_key=True)
    body: Mapped[str]
    is_correct: Mapped[bool]

    question: Mapped[Question] = relationship(back_populates="choices")


class Relevance(Base):
    __tablename__ = "relevance"

    question_id: Mapped[str] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), primary_key=True
    )
    document_source_id: Mapped[str] = mapped_column(primary_key=True)
    grade: Mapped[int]

    question: Mapped[Question] = relationship(back_populates="relevance_labels")
