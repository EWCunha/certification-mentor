from typing import Any, Optional
from json import dumps
from uuid import uuid4
from logging import getLogger, Logger
from toyaikit.tools import Tools
from openai.types.responses import Response
from agent.rag_helper import RAGBase
from config.config import CONFIG
from state.state import State, Question
from monitoring.database import session_for
from monitoring.schemas import Log


SYSTEM_PROMPT = """
You are an expert in Anthropic's Claude Certified Architect - Foundations (CCAR-F). I need you to help the user 
study for the CCAR-F exam.
You will generate a simulated exam question and answers based on the CCAR-F exam objectives. The exam has a total of {total_amount_of_questions} questions, 
separated into {exam_categories_amount} categories:
<exam_categories>
{exam_categories}
</exam_categories>

Use `search` to search the RAG for documents related to the theme or category of the question.

You will generate 1 question with multiple choice answers (ideally 4 choices) per turn. Only one of the choices should be the correct answer. Provide the choices as a list of strings,
do NOT include the question header in the choices and do NOT include the choice letters.

If the user answers the question incorrectly, I'll forward to you the user's answer. You will then provide the reason why the given answer is incorrect.

If the user answers the question correctly, I'll just send you a message like "CORRECT".

After the user has answered all the questions, you will provide the total score and the percentage of questions answered correctly.
You will also provide the sources that the user can use to study the questions that they got wrong.

The first user input should be something like "start" or "begin" to start the exam. If it is not something like that, you will ask something like "Shall we start the exam?"

You don't need to send any text messages. Based on your function calls, I'll print what is necessary so that the user can go on with the exam.

You can ONLY end the exam when all {total_amount_of_questions} questions have been answered OR the user input is "end" or "stop".
"""

STATS_USER_PROMPT = """
Here are the results of the exam:
{stats}

The list/array of all questions is:
{questions}

This array is 0-indexed. To find the right index of the question, use the following formula:
question_index = question_number - 1

Give me a summary of the results. Additionally, provide the sources that the user can use to study the questions that they got wrong. Don't ask for any further input.
"""


class Agent:
    rag: RAGBase
    system_prompt: str
    message_history: list[dict[str, Any]]
    tools: Tools
    _total_amount_of_questions: int
    _session_id: str

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
            self._total_amount_of_questions = sum(
                category["amount_of_questions"] for category in exam_categories
            )

            if system_prompt is None:
                exam_categories_amount = len(exam_categories)
                system_prompt = SYSTEM_PROMPT.format(
                    total_amount_of_questions=self._total_amount_of_questions,
                    exam_categories_amount=exam_categories_amount,
                    exam_categories=exam_categories,
                )

        self.system_prompt = system_prompt
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        self.tools = tools
        self._session_id = ""

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

        started = False
        tools = self._tools()
        self._session_id = str(uuid4())
        while not state.is_exam_ended():
            if state.questions and not state.last_question().answered():
                question = state.last_question()
                # question.print()
                prompt = input("Answer: ")
                if len(prompt) == 1:
                    correct = state.add_answer(question.number, prompt)
                    self._log_message(
                        f"User answer: {prompt.upper()}. Correct: {correct}"
                    )
                    self._add_user_message(
                        dumps(
                            {
                                "user_answer": prompt.upper(),
                                "is_correct": correct,
                            }
                        )
                    )
                    if correct:
                        print("CORRECT\n\nGenerating next question...")
                else:
                    if prompt.lower() in ["end", "stop"]:
                        self._log_message(f"User input: {prompt.lower()}")
                        stats = state.exam_stats()
                        self._add_user_message(
                            self._parse_stats_message(stats, state.questions)
                        )
                        self._to_llm(tools, logger)
                        state.end_exam()
                        continue
                    else:
                        print(
                            "Invalid input. Please enter a letter for the choice or 'end' to stop the exam."
                        )
                        self._log_message(f"User input: {prompt.lower()}")
            elif not started:
                started = True
                self._add_user_message("start")
                self._log_message(f"User message: start")
            elif state.last_question_number() >= self._total_amount_of_questions:
                stats = state.exam_stats()
                self._add_user_message(
                    self._parse_stats_message(stats, state.questions)
                )
                self._to_llm(tools, logger)
                state.end_exam()
                self._log_message(f"Exam ended. Stats: {stats}")
                continue

            self._to_llm(tools, logger)

    def _to_llm(self, tools: list, logger: Logger) -> None:
        response = self.rag.llm(self.message_history, tools)
        self._log_message(f"Response: {response}")
        self._add_assistant_message(response)
        for item in response.output:
            logger.debug(f"Item: {item}")
            if item.type == "function_call":
                logger.debug(f"Function call: {item}")
                result = self.tools.function_call(item)
                logger.debug(f"Function call result: {result}")
                logger.info(
                    f"Function called {item.name}, with arguments {item.arguments}"
                )
                self._add_function_call_result(result)
                self._log_message(f"Function call: {item}. Result: {result}")
                if item.name == "end_exam":
                    return
            elif item.type == "message":
                if item.content[0].text == "":
                    self._add_user_message("continue")
                    self._log_message(f"User message: continue")
                    continue

                print(item.content[0].text)
                self._log_message(f"LLM message: {item.content[0].text}")

    def _add_user_message(self, prompt: str) -> None:
        self.message_history.append({"role": "user", "content": prompt})

    def _add_assistant_message(self, response: Response) -> None:
        self.message_history.extend(response.output)

    def _add_function_call_result(self, result: Any) -> None:
        self.message_history.append(result)

    def _tools(self) -> list:
        tools = self.tools.get_tools()
        tools_to_return = []
        for tool in tools:
            if not tool["name"] in ["add_answer", "end_exam", "_check_question_number"]:
                tools_to_return.append(tool)

        return tools_to_return

    def _parse_stats_message(self, stats: dict, questions: list[Question]) -> str:
        return STATS_USER_PROMPT.format(
            stats=stats,
            questions=questions,
        )

    def _log_message(self, message: str) -> None:
        with session_for() as session:
            session.add(Log(session_id=self._session_id, message=message))
            session.commit()
