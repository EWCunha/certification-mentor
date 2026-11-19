from typing import Any, Union, Optional
from json import dumps, loads
from string import ascii_uppercase
from random import shuffle
from dataclasses import dataclass
from logging import getLogger
from agent.rag_helper import RAGBase
from agent.rag_vector import RAGPgVector


@dataclass
class Question:
    number: int
    category: str
    stem: str
    choices: list[str]
    correct_answer: int
    explanation: str
    given_answer: int = -1
    reason: str = ""

    def is_correct(self) -> bool:
        return self.given_answer == self.correct_answer

    def json(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "stem": self.stem,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "given_answer_index": self.given_answer,
            "reason": self.reason,
        }

    def correct_answer_str(self) -> str:
        return ascii_uppercase[self.correct_answer]

    def answered(self) -> bool:
        return self.given_answer != -1

    def print(self):
        print(f"{self.number}: {self.stem}")
        for i, choice in enumerate(self.choices):
            print(f"  {ascii_uppercase[i]}: {choice}")

        print()

    def print_correct_answer_and_reason(self):
        print(
            f"Correct answer is: {self.correct_answer_str()}.\nWhy given answer is incorrect: {self.reason}\n"
        )

    def __str__(self) -> str:
        return dumps(self.json(), indent=2)


class State:
    rag: Union[RAGBase, RAGPgVector]
    questions: list[Question] = []
    count_per_category: dict[str, int] = {}
    _next_question_number: int = 1
    _exam_ended: bool = False

    def __init__(self, rag: Union[RAGBase, RAGPgVector]):
        self.rag = rag

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
        stem: str,
        choices: Union[list[str], str],
        correct_answer: int,
        explanation: str,
    ) -> int:
        """
        Add a question to the state. The choices are shuffled and the correct answer is corrected to the index of the correct answer in the shuffled choices.

        Args:
            category (str): The category of the question.
            stem (str): The question text/header only (NOT the choices!).
            choices (list[str]): List of the choices for the question.
            correct_answer (int): The correct answer index.
            explanation (str): The explanation of the question.

        Returns:
            int: The number of the question.
        """

        if isinstance(choices, str):
            choices_list = loads(choices.strip().removesuffix(","))
        else:
            choices_list = list(choices)

        correct_answer_str = choices_list[correct_answer]
        shuffle(choices_list)
        correct_answer = choices_list.index(correct_answer_str)

        question_number = self._next_question_number
        question = Question(
            question_number,
            category,
            stem,
            choices_list,
            correct_answer,
            explanation,
        )
        self.questions.append(question)

        getLogger(__name__).debug(f"Added question: {question.json()}")

        self._next_question_number += 1

        if category not in self.count_per_category:
            self.count_per_category[category] = 0

        self.count_per_category[category] += 1

        question.print()

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

        self.questions[question_number - 1].given_answer = ascii_uppercase.index(
            given_answer.upper()
        )
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

        self.questions[question_number - 1].print_correct_answer_and_reason()

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

    def last_question(self) -> Optional[Question]:
        """
        Get the last question.

        Returns:
            Optional[Question]: The last question. None if there are no questions.
        """

        if not self.questions:
            return None

        return self.questions[-1]

    def last_question_number(self) -> int:
        """
        Get the last question number.

        Returns:
            int: The last question number.
        """
        return self._next_question_number - 1

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
        """
        Ends the exam.

        """

        logger = getLogger(__name__)
        logger.debug("Ending exam")
        self._exam_ended = True

    def is_exam_ended(self) -> bool:
        """
        Check if the exam has ended.

        Returns:
            bool: True if the exam has ended, False otherwise.
        """

        return self._exam_ended

    def search(
        self, category: str, query: str, num_results: int = 20
    ) -> list[dict[str, Any]]:
        """
        Search the RAG for documents related to the query.

        Args:
            category (str): The category of the question.
            query (str): The query to search for.
            num_results (int): The number of results to return. Defaults to 20.

        Returns:
            list[dict[str, Any]]: The list of documents.
        """

        if isinstance(self.rag, RAGPgVector):
            return self.rag.search(
                query,
                num_results=num_results,
                filter_dict={"cca_f_domain": category, "enabled": True},
            )
        else:
            return self.rag.search(
                query,
                num_results=num_results,
                boost_dict={"text": 3.0},
                filter_dict={"cca_f_domain": category, "enabled": True},
            )

    def _check_question_number(self, question_number: int) -> None:
        if question_number >= self._next_question_number:
            raise ValueError(f"Question number {question_number} not found")
