from typing import List, Dict, Set
import json


class Question(object):

    def __init__(self, _id: str, question: str, answer: str, _type: str, level: str, context: List[List],
                 supporting_facts: List[List]) -> None:
        self.id = _id
        self.question = question
        self.answer = answer
        self.level = level
        self.type = _type

        self.gold_articles: Set[str] = set(fact[0] for fact in supporting_facts)
        self.context: Dict[str, List[str]] = {article[0]: article[1] for article in context}

    def to_json(self) -> Dict:
        return json.dumps({
            'id': self.id,
            'level': self.level,
            'type': self.type,
            'question': self.question,
            'answer': self.answer,
            'supporting_facts': self.gold_articles
        })

    def __repr__(self):
        return self.to_json()

    def __str__(self):
        return self.to_json()


class Dataset(object):

    def __init__(self) -> None:

        self.questions = []
        self._current_index = 0

    @staticmethod
    def from_file(filename: str, max_questions: int = None):

        dataset = Dataset()

        with open(filename, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        for json_question in json_data:

            if max_questions is not None and len(dataset) > max_questions:
                break

            question = Question(
                json_question['_id'],
                json_question['question'],
                json_question['answer'],
                json_question['type'],
                json_question['level'],
                json_question['context'],
                json_question['supporting_facts']
            )
            dataset.questions.append(question)

        return dataset

    def __iter__(self):
        return self

    def __next__(self):

        if self._current_index >= len(self.questions):
            self._current_index = 0
            raise StopIteration

        self._current_index += 1
        return self.questions[self._current_index - 1]

    def __len__(self):
        return len(self.questions)

    def get_treceval_reference(self) -> Dict[str, Dict[str, int]]:

        reference = {}
        # TODO: think about which id to use, currently uses titles as strings
        for question in self:
            relevant_docs = {}
            for document in question.gold_articles:
                relevant_docs[document] = 1
            reference[question.id] = relevant_docs

        return reference

    def find_by_id(self, _id) -> Question:

        candidates = list(filter(lambda question: question.id == _id, self.questions))

        if not len(candidates) == 1: raise IndexError
        return candidates[0]

    def filter_by_level(self, level: str) -> List[Question]:
        return list(filter(lambda question: question.level == level, self.questions))

    def filter_by_type(self, type: str) -> List[Question]:
        return list(filter(lambda question: question.type == type, self.questions))


