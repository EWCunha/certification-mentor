from openai import OpenAI
from json import loads
from psycopg import connect
from agent.rag_vector import RAGPgVector
from evals.utils import get_questions, get_document
from helper.helpers import db_connection_string
from agent.agent import SYSTEM_PROMPT
from config.config import CONFIG


def rag_eval_generation_with_judge(
    rag: RAGPgVector, max_questions_to_evaluate: int = 30
):
    exam_guide_questions, other_questions = get_questions()
    documents = list(
        set(
            [
                question["document_source_id"]
                for question in exam_guide_questions + other_questions
            ]
        )
    )

    judge_system_prompt = """
    You are a judge that evaluates the quality of the generated answer. The answer must be regarding to a content within the document.
    The stem, the choices, the correct answer and the explanation are provided to you.
    Evaluate the following criteria:
    - The stem is answered by those documents. True if the stem is answered by those documents, false otherwise.
    - The marked option is right according to those documents. True if the marked option is right according to those documents, false otherwise.
    - The other options answer the question wrongly. That is, only the correct answer should be correct. True if the other options answer the question wrongly, false otherwise.
    - The explanation cites a fact that appears in those documents. True if the explanation cites a fact that appears in those documents, false otherwise.
    Give me the reasoning for your answer.
    
    Return true or false for each criterion in a JSON object. The reasoning should be provided in the "reasoning" key, with the reasoning behind each criterion.
    Examples:
        {
            "stem": false,
            "marked_option": true,
            "other_options": true,
            "explanation": false,
            "reasoning": {
                "stem": "The stem is not answered by those documents.",
                "marked_option": "The marked option is not right according to those documents.",
                "other_options": "The other options answer the question wrongly.",
                "explanation": "The explanation does not cite a fact that appears in those documents."
            }
        };
        {
            "stem": true,
            "marked_option": false,
            "other_options": false,
            "explanation": true,
            "reasoning": {
                "stem": "The stem is answered by those documents.",
                "marked_option": "The marked option is not right according to those documents.",
                "other_options": "At least one of the other options is actually correct.",
                "explanation": "The explanation cites a fact that appears in those documents."
            }
        };
        {
            "stem": true,
            "marked_option": true,
            "other_options": true,
            "explanation": true,
            "reasoning": {
                "stem": "The stem is answered by those documents.",
                "marked_option": "The marked option is right according to those documents.",
                "other_options": "The other options answer the question wrongly.",
                "explanation": "The explanation cites a fact that appears in those documents."
            }
        };
        {
            "stem": false,
            "marked_option": false,
            "other_options": false,
            "explanation": false,
            "reasoning": {
                "stem": "The stem is not answered by those documents.",
                "marked_option": "The marked option is not right according to those documents.",
                "other_options": "At least one of the other options is actually correct.",
                "explanation": "The explanation does not cite a fact that appears in those documents."
            }
        };
    """

    exam_categories = CONFIG.get("categories")
    total_amount_of_questions = sum(
        category["amount_of_questions"] for category in exam_categories
    )
    exam_categories_amount = len(exam_categories)

    rag_system_prompt = SYSTEM_PROMPT.format(
        total_amount_of_questions=total_amount_of_questions,
        exam_categories_amount=exam_categories_amount,
        exam_categories=exam_categories,
    )
    rag_system_prompt += """\n respond in the following format:
        {{
            "stem": "the stem of the question",
            "choices": ["choice 1", "choice 2", "choice 3", "choice 4"],
            "correct_answer": "the correct answer",
            "explanation": "the explanation of the question"
        }}
    """

    judge_llm = OpenAI()
    evals = _evaluate_questions(
        documents,
        rag,
        rag_system_prompt,
        judge_llm,
        judge_system_prompt,
        max_questions_to_evaluate,
    )

    metrics = {
        "stem": 0,
        "marked_option": 0,
        "other_options": 0,
        "explanation": 0,
    }
    for eval in evals:
        metrics["stem"] += eval["stem"]
        metrics["marked_option"] += eval["marked_option"]
        metrics["other_options"] += eval["other_options"]
        metrics["explanation"] += eval["explanation"]

        if eval["other_options"] == False:
            print(eval)

    print(f"Total questions: {len(evals)}")
    for key, value in metrics.items():
        print(f"{key}: {value} ({value / len(evals) * 100:.2f}%)")


def _evaluate_questions(
    documents: list[str],
    rag: RAGPgVector,
    rag_system_prompt: str,
    judge_llm: OpenAI,
    judge_system_prompt: str,
    max_questions_to_evaluate: int = 30,
) -> list[dict]:
    evals = []
    for document in documents:
        if len(evals) >= max_questions_to_evaluate:
            break

        document = get_document(document, rag.conn)
        context = rag.build_context(
            [document], fields=["cca_f_domain", "title", "text"]
        )
        message = [
            {"role": "system", "content": rag_system_prompt},
            {
                "role": "user",
                "content": f"Generate a question based on the following context: {context}",
            },
        ]
        response = rag.llm(message)
        generated_question = response.output_text

        messages = [
            {"role": "system", "content": judge_system_prompt},
            {"role": "user", "content": generated_question},
        ]

        response = judge_llm.responses.create(
            model="gpt-4o-mini",
            input=messages,
        )

        evaluation = response.output_text
        try:
            evals.append(loads(evaluation))
        except Exception as e:
            print(f"Error evaluating question: {e}")
            print(evaluation)

    return evals


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    rag = RAGPgVector(
        embedder=None, conn=connect(db_connection_string()), llm_client=OpenAI()
    )
    rag_eval_generation_with_judge(rag)
