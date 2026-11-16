from openai import OpenAI
from agent.rag_vector import RAGPgVector
from evals.utils import get_questions, get_document


def rag_eval_generation_with_judge(rag: RAGPgVector):
    exam_guide_questions, other_questions = get_questions()

    system_prompt = """
    You are a judge that evaluates the quality of the generated answer. The answer must be regarding to a content within the document.
    The stem, the choices, the correct answer and the explanation are provided to you.
    Evaluate the following criteria:
    - The stem is answered by those documents.
    - The marked option is right according to those documents.
    - The other options are wrong according to those documents.
    - The explanation cites a fact that appears in those documents.
    
    Return true or false for each criterion in a JSON object.
    Example:
    {
        "stem": true,
        "marked_option": true,
        "other_options": false,
        "explanation": true
    }
    """

    judge_llm = OpenAI()

    for stored_question in exam_guide_questions:
        document = get_document(stored_question["document_source_id"], rag.conn)
        context = rag.build_context([document])
        response = rag.llm(
            context + [{"role": "user", "content": f"context: {context}"}]
        )
        generated_question = response.output_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": generated_question},
        ]

        response = judge_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        evaluation = response.output_text
        print(evaluation)


def rag_eval_answer_in_context():
    exam_guide_questions, other_questions = get_questions()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
