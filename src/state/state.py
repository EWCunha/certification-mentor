from typing import Any
from json import loads, dumps
from dataclasses import dataclass
from logging import getLogger
from openai.types.responses import ResponseFunctionToolCall
from agent.index_interface import SearchIndex


@dataclass
class Question:
    number: int
    category: str
    question: str
    choices: list[str]
    correct_answer: str
    explanation: str
    given_answer: str = ""
    reason: str = ""

    def is_correct(self) -> bool:
        return self.given_answer == self.correct_answer

    def json(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "question": self.question,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "given_answer": self.given_answer,
            "reason": self.reason,
        }

    def print(self):
        print(f"{self.question}")
        for choice in self.choices:
            print(f"  {choice}")


class State:
    index: SearchIndex
    questions: list[Question] = []
    count_per_category: dict[str, int] = {}
    _next_question_number: int = 1
    _exam_ended: bool = False

    def __init__(self, index: SearchIndex):
        self.index = index

    # def make_call(self, call: ResponseFunctionToolCall) -> dict[str, Any]:
    #     logger = getLogger(__name__)
    #     logger.debug(f"Making call: {call.name}")
    #     logger.debug(f"Call arguments: {call.arguments}")

    #     args = loads(call.arguments)

    #     if call.name == "add_question":
    #         result = self.add_question(**args)
    #     elif call.name == "add_answer":
    #         result = self.add_answer(**args)
    #     elif call.name == "add_reason":
    #         result = self.add_reason(**args)
    #     elif call.name == "get_questions_count":
    #         result = self.get_questions_count(**args)
    #     elif call.name == "get_all_questions":
    #         result = self.get_all_questions()
    #     elif call.name == "exam_stats":
    #         result = self.exam_stats()
    #     elif call.name == "end_exam":
    #         self.end_exam()
    #         result = "Exam ended"
    #     elif call.name == "search":
    #         result = self.search(**args)
    #     else:
    #         raise ValueError(f"Unknown call: {call.name}")

    #     result_json = dumps(result, indent=2)

    #     return {
    #         "type": "function_call_output",
    #         "call_id": call.call_id,
    #         "output": result_json,
    #     }

    def add_question(
        self,
        category: str,
        question: str,
        choices: list[str],
        correct_answer: str,
        explanation: str,
    ) -> int:
        """
        Add a question to the state.

        Args:
            category (str): The category of the question.
            question (str): The question text/header only (NOT the choices!).
            choices (list[str]): The choices of the question.
            correct_answer (str): The correct answer. Give only the choice letter, not the choice text.
            explanation (str): The explanation of the question.

        Returns:
            int: The number of the question.
        """

        if isinstance(choices, str):
            choices = choices.split("\n")

        question_number = self._next_question_number
        self.questions.append(
            Question(
                question_number,
                category,
                question,
                choices,
                correct_answer,
                explanation,
            )
        )
        self._next_question_number += 1

        if category not in self.count_per_category:
            self.count_per_category[category] = 0

        self.count_per_category[category] += 1

        return question_number

    def add_answer(self, question_number: int, given_answer: str) -> bool:
        """
        Add an answer to a question.

        Args:
            question_number (int): The number of the question.
            given_answer (str): The given answer. Give only the choice letter, not the choice text.

        Returns:
            bool: True if the answer is correct, False otherwise.
        """

        self._check_question_number(question_number)

        self.questions[question_number - 1].given_answer = given_answer
        return self.questions[question_number - 1].is_correct()

    def add_reason(self, question_number: int, reason: str):
        """
        Add a reason to a wrong-answered question.

        Args:
            question_number (int): The number of the question.
            reason (str): The reason.
        """

        self._check_question_number(question_number)

        self.questions[question_number - 1].reason = reason

    def get_questions_count(self, category: str) -> int:
        """
        Get the number of questions in a category.

        Args:
            category (str): The category.

        Returns:
            int: The number of questions in the category.
        """

        return self.count_per_category[category]

    def get_all_questions(self) -> list[dict[str, Any]]:
        """
        Get all questions.

        Returns:
            list[dict[str, Any]]: The list of questions.
        """

        questions = []
        for question in self.questions:
            questions.append(question.json())

        return questions

    def exam_stats(self) -> dict[str, Any]:
        """
        Get the statistics of the exam.

        Returns:
            dict[str, Any]: The statistics of the exam.

        The statistics is a dictionary with the following structure:
        - overall: A dictionary with the following keys:
            - correct: The number of correct answers.
            - wrong (list[int]): The list of wrong-answered questions.
            - total (int): The total number of questions.
        - per_category (dict[str, dict[str, Any]]): A dictionary with the following structure:
            - category (str): The category.
                - correct (int): The number of correct answers.
                - wrong (list[int]): The list of wrong-answered questions.
                - total (int): The total number of questions.
        """

        stats = {"overall": {"correct": 0, "wrong": [], "total": 0}, "per_category": {}}
        for question in self.questions:
            if not question.category in stats["per_category"]:
                stats["per_category"][question.category] = {
                    "correct": 0,
                    "total": 0,
                    "wrong": [],
                }

            if question.is_correct():
                stats["per_category"][question.category]["correct"] += 1
                stats["overall"]["correct"] += 1
            else:
                stats["per_category"][question.category]["wrong"].append(
                    question.number
                )
                stats["overall"]["wrong"].append(question.number)

            stats["per_category"][question.category]["total"] += 1
            stats["overall"]["total"] += 1

        return stats

    def end_exam(self) -> None:
        logger = getLogger(__name__)
        logger.debug("Ending exam")
        self._exam_ended = True

    def is_exam_ended(self) -> bool:
        return self._exam_ended

    def search(self, category: str, query: str) -> list[dict[str, Any]]:
        return self.index.search(
            query,
            num_results=100,
            boost_dict={"text": 3.0},
            filter_dict={"cca_f_domain": category, "enabled": True},
        )

    def _check_question_number(self, question_number: int) -> None:
        if question_number >= self._next_question_number:
            raise ValueError(f"Question number {question_number} not found")
