from sqlalchemy import select
from psycopg import Connection
from evals.schemas import Question, Relevance
from evals.database import session_for


def is_hit(result: list[str], correct_item: str) -> list[int]:
    return [int(item == correct_item) for item in result]


def get_questions() -> tuple[list[Question], list[Question]]:
    session = session_for()
    exam_guide_questions = session.execute(
        select(Question, Relevance.document_source_id)
        .join(Relevance, Relevance.question_id == Question.id)
        .where(Question.from_exam_guide.is_(True))
    ).all()

    other_questions = session.execute(
        select(Question, Relevance.document_source_id)
        .join(Relevance, Relevance.question_id == Question.id)
        .where(Question.from_exam_guide.is_(False))
    ).all()

    exam_guide_questions = [
        {**question.to_dict(), "document_source_id": relevance}
        for question, relevance in exam_guide_questions
    ]

    other_questions = [
        {**question.to_dict(), "document_source_id": relevance}
        for question, relevance in other_questions
    ]

    session.close()

    return exam_guide_questions, other_questions


def get_document(source_id: str, connection: Connection) -> str:
    query_string = """
        SELECT source_id, title, url, cca_f_domain, text
        FROM documents
        WHERE source_id = %s
        """

    rows = connection.execute(query_string, [source_id]).fetchone()
    return {
        "source_id": rows[0],
        "title": rows[1],
        "url": rows[2],
        "cca_f_domain": rows[3],
        "text": rows[4],
    }
