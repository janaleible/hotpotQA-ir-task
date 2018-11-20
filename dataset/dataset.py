from typing import List, Dict
from main_constants import *
import json


class Question(object):

    def __init__(self, _id: str, question: str, answer: str, _type: str, level: str, context: List[List],
                 supporting_facts: List[List]) -> None:
        self.id = _id
        self.question = question
        self.answer = answer
        self.level = level
        self.type = _type

        self.gold_articles: List[str] = [fact[0] for fact in supporting_facts]
        self.context: Dict[str, List[str]] = {article[0]: article[1] for article in context}


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


if __name__ == '__main__':

    os.makedirs(TRECEVAL_REFERENCE_DIR, exist_ok=True)

    for data_file, reference_file in zip([TRAINING_SET,             DEV_FULLWIKI_SET],
                                         [TRECEVAL_REFERENCE_TRAIN, TRECEVAL_REFERENCE_DEV]):
        data_set = Dataset.from_file(data_file)
        with open(reference_file, 'w') as file:
            json.dump(data_set.get_treceval_reference(), file, indent=True)
