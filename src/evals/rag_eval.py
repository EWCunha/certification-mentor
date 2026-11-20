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
    You grade one generated exam item against one source document. Use only the document text. Do not use outside knowledge.

    The user message has two parts:
    <document>
    ...page text...
    </document>
    <item>
    ...generated JSON with stem, choices, correct_answer, and explanation...
    </item>

    Score these four checks. Each value is true only when the document supports it.
    - stem: the document contains enough information to answer the stem. True if the document contains enough information to answer the stem, false otherwise.
    - marked_option: correct_answer is the right choice according to the document. True if the correct answer is the right choice according to the document, false otherwise.
    - distractors_are_wrong: true when every choice except correct_answer is a wrong answer according to the document. false only when one of those choices is also a right answer, or the document does not say enough to reject it. A wrong distractor is a passed check, so the value is true.
    - explanation: the explanation states a fact that appears in the document. True if the explanation states a fact that appears in the document, false otherwise.

    When your reason says a distractor is incorrect, distractors_are_wrong is true. Set distractors_are_wrong to false only when a distractor is also a right answer.

    For each check, quote the shortest span from the document that decides it. If the check is false, quote the item text that the document does not support and set the span to an empty string.

    Reply with one JSON object and no other text.

    Example 1. Document: "Project slash commands live in .claude/commands/ and are shared with the repo. ~/.claude/commands/ is for personal commands only. CLAUDE.md holds project instructions, not command definitions. There is no user-settings command file."
    The marked choice is ".claude/commands/ inside the repository." The others are "~/.claude/commands/", "CLAUDE.md", and "a user settings file." The explanation says project commands are shared with the repo.
    {
        "stem": true,
        "marked_option": true,
        "distractors_are_wrong": true,
        "explanation": true,
        "reasoning": {
            "stem": {"quote": "Project slash commands live in .claude/commands/", "why": "The page states where a shared slash command is stored."},
            "marked_option": {"quote": "Project slash commands live in .claude/commands/ and are shared with the repo.", "why": "That sentence is the marked choice."},
            "distractors_are_wrong": {"quote": "~/.claude/commands/ is for personal commands only. CLAUDE.md holds project instructions, not command definitions. There is no user-settings command file.", "why": "Those choices are wrong answers, so distractors_are_wrong is true."},
            "explanation": {"quote": "and are shared with the repo.", "why": "The explanation repeats that fact."}
        }
    }

    Example 2. Same document. The marked choice is "~/.claude/commands/". The other choices are "CLAUDE.md", "a user settings file", and "store it only as a personal command". None of them is .claude/commands/. The explanation says personal commands are committed with the repository.
    {
        "stem": true,
        "marked_option": false,
        "distractors_are_wrong": true,
        "explanation": false,
        "reasoning": {
            "stem": {"quote": "Project slash commands live in .claude/commands/", "why": "The page still answers where a shared command goes."},
            "marked_option": {"quote": "", "why": "The marked path is the personal directory. The page says that directory is not the shared location."},
            "distractors_are_wrong": {"quote": "~/.claude/commands/ is for personal commands only. CLAUDE.md holds project instructions, not command definitions. There is no user-settings command file.", "why": "Every unmarked choice is also a wrong answer, so distractors_are_wrong is true."},
            "explanation": {"quote": "", "why": "The explanation says personal commands are committed with the repo. The page says the opposite."}
        }
    }

    Example 3. Document: "Run Claude Code in CI with the -p flag so it prints the result and exits. The --print flag is the same option."
    The marked choice is "-p". Another choice is "--print". The explanation says -p is the non-interactive flag.
    {
        "stem": true,
        "marked_option": true,
        "distractors_are_wrong": false,
        "explanation": true,
        "reasoning": {
            "stem": {"quote": "Run Claude Code in CI with the -p flag", "why": "The page answers how to run Claude Code in CI."},
            "marked_option": {"quote": "Run Claude Code in CI with the -p flag so it prints the result and exits.", "why": "The marked flag is the one the page names."},
            "distractors_are_wrong": {"quote": "The --print flag is the same option.", "why": "--print is also a right answer, so distractors_are_wrong is false."},
            "explanation": {"quote": "so it prints the result and exits.", "why": "The explanation matches that sentence."}
        }
    }

    Example 4. Same CI document. The stem asks which Message Batches API setting saves 50% on overnight jobs. The marked choice is "batch the pre-merge check." The explanation says the Batches API returns in under one second.
    {
        "stem": false,
        "marked_option": false,
        "distractors_are_wrong": false,
        "explanation": false,
        "reasoning": {
            "stem": {"quote": "", "why": "The page is about the -p flag. It says nothing about the Message Batches API."},
            "marked_option": {"quote": "", "why": "The page never mentions batching a pre-merge check."},
            "distractors_are_wrong": {"quote": "", "why": "The page has no batching facts, so a distractor cannot be called wrong. distractors_are_wrong is false."},
            "explanation": {"quote": "", "why": "The page does not say the Batches API returns in under one second."}
        }
    }
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
        "distractors_are_wrong": 0,
        "explanation": 0,
    }
    for eval in evals:
        metrics["stem"] += eval["stem"]
        metrics["marked_option"] += eval["marked_option"]
        metrics["distractors_are_wrong"] += eval["distractors_are_wrong"]
        metrics["explanation"] += eval["explanation"]

        if eval["distractors_are_wrong"] == False:
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
            {
                "role": "user",
                "content": f"<document>\n{context}\n</document>\n<item>\n{generated_question}\n</item>",
            },
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
