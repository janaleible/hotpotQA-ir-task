import json
import sqlite3
from typing import Dict, List

import main_constants as constants
from retrieval.term.dataset import Dataset


class Run(dict):
    def add_ranking(self, _id: str, ranking: Dict[str, float]) -> None:
        self[_id] = ranking

    def update_ranking(self, question_id: str, document_title: str, score: float) -> None:
        # if not self.get(question_id, {}):
        if not question_id in self:
            self[question_id] = {}

        assert self[question_id].get(document_title, -1) == -1, f'Ranking already ' \
            f'exists {question_id}.{document_title}={self[question_id][document_title]}'

        self[question_id][document_title] = score

    def update_rankings(self, rankings: Dict[str, Dict[str, float]]) -> None:
        for _id in rankings.keys():
            if self.get(_id, {}):
                self[_id].update(rankings[_id])
            else:
                self[_id] = rankings[_id]

    def write_to_file(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self, file)

    def to_json(self, db, dataset_path) -> List[dict]:

        raise NotImplementedError("Really shouldn't be using this!")

        connection = sqlite3.connect(db)
        cursor = connection.cursor()

        questions = []

        dataset = Dataset.from_file(dataset_path)

        for question_id, ranking in self.items():

            context = []
            sorted_by_score = sorted(ranking.items(), key=lambda value: value[1], reverse=True)
            for rank in range(min(10, len(ranking))):
                (title, score) = sorted_by_score[rank]
                try:
                    (document_text,) = cursor.execute(f"SELECT document_text FROM features WHERE doc_title = ? LIMIT 1", [json.dumps(title)]).fetchone()
                except:
                    print(f'failed tp query db {db} with title {json.dumps(title)}, question_id {question_id}')
                    continue
                document_text = json.loads(document_text)
                article = [paragraph.split(constants.EOS.strip()) for paragraph in document_text.split(constants.EOP.strip())]
                article.insert(0, title)
                context.append(article)

            full_question = dataset.find_by_id(question_id)

            question = {
                '_id': full_question.id,
                'level': full_question.level,
                'type': full_question.type,
                'question': full_question.question,
                'context': context,
                'anwer': full_question.answer,
                'supporting_facts': full_question.supporting_facts
            }

            questions.append(question)

        return questions
