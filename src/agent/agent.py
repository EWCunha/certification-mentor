from typing import Any, Optional
from logging import getLogger
from toyaikit.tools import Tools
from openai.types.responses import Response
from agent.rag_helper import RAGBase
from config.config import CONFIG
from state.state import State


SYSTEM_PROMPT = """
You are an expert in Anthropic's Claude Certified Architect - Foundations (CCAR-F). I need you to help the user study for the CCAR-F exam.
You will generate me a simulated exam question and answer based on the CCAR-F exam objectives. The exam has a total of {total_amount_of_questions} questions, 
separated into {exam_categories_amount} categories:
<exam_categories>
{exam_categories}
</exam_categories>

You will generate 1 questions with multiple choice answers per turn. After the user answers the question, you will tell if the answer is either CORRECT or INCORRECT. 
If the answer is INCORRECT, you will provide the correct answer and explanation as well as the reason why the given answer is incorrect. 
You will continue to generate questions and answers until the user has answered all the questions in the exam.

For the first question, just generate the question and choices. Do not provide the answer or explanation.
After that, you will use the following format for your responses:

<CORRECT or INCORRECT>
If INCORRECT:
    Correct answer: [correct answer]
    Explanation: [explanation]
    Reason why the given answer is incorrect: [reason why the given answer is incorrect]

<Next question number>: <next question number>
[choices]

After the user has answered all the questions, you will provide the total score and the percentage of questions answered correctly.
You will also provide the sources that the user can use to study the questions that they got wrong.

Your last response should be in the following format:
Result:
    [correct answers count] / {total_amount_of_questions} ([percentage of questions answered correctly])
    Per category:
        [category name]: {{
            Result: [number of questions answered correctly] / [number of questions in category] ([percentage of questions answered correctly in category])
            [question number answered wrong]: [source to study the question]
            [question number answered wrong]: [source to study the question]
            [question number answered wrong]: [source to study the question]
            ...
        }}


To help you keep track of the generated questions and answers, you can use the following tools:
<provided_tools>
- add_question(category: str, question: str, choices: list[str], correct_answer: str, explanation: str) -> int: adds a question to the list of questions and returns the question number
- add_answer(question_number: int, answer: str) -> bool: adds an answer to a generated question and returns True if the answer is correct, False otherwise
- add_reason(question_number: int, reason: str): adds a reason to a wrong answered question
- get_questions_count(category: str) -> int: returns the number of generated questions in a category
- get_all_questions() -> list[dict[str, Any]]: returns the list of all generated questions
- search(category: str, query: str) -> list[dict[str, Any]]: returns the context for a category and a query to help you generate the next question.
- exam_stats() -> dict[str, Any]: returns the statistics of the exam
- end_exam() -> None: ends the exam. This should be the last tool to be called and it will be the last response from you. 
</provided_tools>

The first user input should be something like "start" or "begin" to start the exam. If it is not something like that, you will ask something like "Shall we start the exam?"
If the user input is "end" or "stop", you should end the exam and provide the result.
"""


class Agent:
    rag: RAGBase
    system_prompt: str
    message_history: list[dict[str, Any]]
    tools: Tools

    def __init__(
        self,
        rag: RAGBase,
        *,
        system_prompt: Optional[str] = None,
        tools: Tools = Tools(),
    ):
        logger = getLogger(__name__)
        logger.debug("Initializing agent")

        self.rag = rag
        if system_prompt is None:
            exam_categories = CONFIG.get("categories")
            total_amount_of_questions = sum(
                category["amount_of_questions"] for category in exam_categories
            )
            exam_categories_amount = len(exam_categories)
            system_prompt = SYSTEM_PROMPT.format(
                total_amount_of_questions=total_amount_of_questions,
                exam_categories_amount=exam_categories_amount,
                exam_categories=exam_categories,
            )

        self.system_prompt = system_prompt
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        self.tools = tools

    def run(self, prompt: str) -> Response:
        logger = getLogger(__name__)
        logger.debug(f"Running agent with prompt: {prompt}")

        self.message_history.append({"role": "user", "content": prompt})
        response = self.rag.llm(self.message_history, self.tools)
        self.message_history.append({"role": "assistant", "content": response})
        return response

    def loop(self, state: State):
        logger = getLogger(__name__)
        logger.debug("Looping agent")

        is_first_turn = True
        should_skip_next_prompt = False
        while True:
            if not should_skip_next_prompt:
                if is_first_turn:
                    prompt = input(">>> ")
                    is_first_turn = False
                else:
                    prompt = input("Answer: ")
            else:
                should_skip_next_prompt = False

            self._add_user_message(prompt)
            logger.debug(f"Message history: {self.message_history}")
            response = self.rag.llm(self.message_history, self.tools.get_tools())
            self._add_assistant_message(response)
            for item in response.output:
                logger.debug(f"Item: {item}")
                if item.type == "function_call":
                    logger.info(f"Function call: {item}")
                    should_skip_next_prompt = True
                    result = self.tools.function_call(item)
                    logger.info(f"Function call result: {result}")

                    self._add_function_call_result(result)
                    if item.name == "end_exam":
                        state.end_exam()
                        break
                    elif item.name == "exam_stats":
                        should_skip_next_prompt = True

                elif item.type == "message":
                    should_skip_next_prompt = False
                    print(f"{item.content[0].text}\n")

            if state.is_exam_ended():
                break

    def _add_user_message(self, prompt: str) -> None:
        self.message_history.append({"role": "user", "content": prompt})

    def _add_assistant_message(self, response: Response) -> None:
        self.message_history.extend(response.output)

    def _add_function_call_result(self, result: Any) -> None:
        self.message_history.append(result)
