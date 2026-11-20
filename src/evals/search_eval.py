from dataclasses import dataclass
from agent.rag_vector import RAGPgVector
from evals.metrics import hit_rate, mrr
from evals.utils import is_hit, get_questions


@dataclass
class EvalItem:
    question_id: str
    hits: list[int]


def vector_search_eval(rag: RAGPgVector, max_results: int = 5) -> dict[str, float]:
    exam_guide_questions, other_questions = get_questions()

    exam_guide_evals = []
    for question in exam_guide_questions:
        results = rag.search(
            question["stem"],
            num_results=max_results,
            filter_dict={
                "cca_f_domain": question["domain"]["cca_f_domain"],
                "enabled": True,
            },
        )
        source_ids = [result["source_id"] for result in results]
        hits = is_hit(source_ids, question["document_source_id"])
        exam_guide_evals.append(EvalItem(question["id"], hits))

    other_evals = []
    for question in other_questions:
        results = rag.search(
            question["stem"],
            num_results=max_results,
            filter_dict={
                "cca_f_domain": question["domain"]["cca_f_domain"],
                "enabled": True,
            },
        )
        source_ids = [result["source_id"] for result in results]
        hits = is_hit(source_ids, question["document_source_id"])
        other_evals.append(EvalItem(question["id"], hits))

    all_exam_guide_hits = [eval.hits for eval in exam_guide_evals]
    all_other_hits = [eval.hits for eval in other_evals]

    return {
        "exam_guide": {
            "num_questions": len(exam_guide_evals),
            "hit_rate": hit_rate(all_exam_guide_hits),
            "mrr": mrr(all_exam_guide_hits),
        },
        "other": {
            "num_questions": len(other_evals),
            "hit_rate": hit_rate(all_other_hits),
            "mrr": mrr(all_other_hits),
        },
    }


if __name__ == "__main__":
    from openai import OpenAI
    from dotenv import load_dotenv
    from sentence_transformers import SentenceTransformer
    from psycopg import connect
    from config.config import CONFIG
    from helper.helpers import db_connection_string

    load_dotenv()

    embedder = SentenceTransformer(
        CONFIG.get("ingestion").get("memory").get("embedding_model")
    )
    conn = connect(db_connection_string())
    rag = RAGPgVector(
        embedder=embedder,
        conn=conn,
        llm_client=OpenAI(),
    )

    print(vector_search_eval(rag))
