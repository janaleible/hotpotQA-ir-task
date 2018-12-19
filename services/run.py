import json
import sqlite3
from typing import Dict, List

from tqdm import tqdm

import main_constants as constants
from retrieval.term.dataset import Dataset
from services import helpers


class Run(dict):
    def add_ranking(self, _id: str, ranking: Dict[str, float]) -> None:
        self[_id] = ranking

    def update_ranking(self, question_id: str, document_title: str, score: float) -> None:
        if question_id not in self:
            self[question_id] = {}

        assert self[question_id].get(document_title, -1) == -1, f'Ranking already ' \
            f'exists {question_id}.{document_title}={self[question_id][document_title]}'

        self[question_id][document_title] = score

    def update_rankings(self, rankings: Dict[str, Dict[str, float]]) -> None:
        for _id in rankings.keys():
            if _id in self:
                self[_id].update(rankings[_id])
            else:
                self[_id] = rankings[_id]

    def write_to_file(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self, file)

    def to_json(self, db, dataset_path) -> List[dict]:
        helpers.log('Creating hotpot data.')
        dataset = Dataset.from_file(dataset_path)
        questions = []

        connection = sqlite3.connect(db)
        cursor = connection.cursor()
        doc_results = cursor.execute(f"SELECT DISTINCT doc_title, document_text FROM features").fetchall()
        title2text = {json.loads(doc_title): json.loads(doc_text) for (doc_title, doc_text) in doc_results}
        cursor.close()
        connection.close()
        helpers.log('Loaded title2text.')

        for question_id, ranking in tqdm(self.items()):
            context = []
            sorted_by_score = sorted(ranking.items(), key=lambda value: value[1], reverse=True)
            for rank in range(min(10, len(ranking))):
                (title, score) = sorted_by_score[rank]
                doc_text = title2text[title]
                article = [paragraph.split(constants.EOS.strip()) for paragraph in
                           doc_text.split(constants.EOP.strip())]
                article.insert(0, title)
                context.append(article)

            full_question = dataset.find_by_id(question_id)
            question = {
                '_id': full_question.id,
                'level': full_question.level,
                'type': full_question.type,
                'question': full_question.question,
                'context': context,
                'answer': full_question.answer,
                'supporting_facts': full_question.supporting_facts
            }
            questions.append(question)

        return questions
