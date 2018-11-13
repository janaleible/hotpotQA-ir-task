import json
from typing import List, Dict

from constants import *


class Question:

    def __init__(self, question: str, answer: str, type: str, level: str, context: List[List], supporting_facts: List[List]) -> None:

        self.question = question
        self.answer = answer
        self.level = level
        self.type = type

        self.gold_articles: List[str] = [fact[0] for fact in supporting_facts]
        self.context: Dict[str, List[str]] = {article[0]: article[1] for article in context}


class Dataset:

    def __init__(self, filename: str, max_questions: int = None) -> None:

        self.questions = []
        self._current_index = 0

        with open(filename, 'r') as file:
            json_data = json.load(file)

        for json_question in json_data:

            if max_questions is not None and len(self.questions) > max_questions:
                break

            question = Question(
                json_question['question'],
                json_question['answer'],
                json_question['type'],
                json_question['level'],
                json_question['context'],
                json_question['supporting_facts']
            )
            self.questions.append(question)

    def __iter__(self):
        return self

    def __next__(self):

        if self._current_index >= len(self.questions):
            self._current_index = 0
            raise StopIteration

        self._current_index += 1
        return self.questions[self._current_index - 1]


if __name__ == '__main__':

    training_set = Dataset(TRAINING_SET, max_questions=50)

    for question in training_set:
        print(question.question)
