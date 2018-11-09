import bz2
import glob
import json
import os
import re
from typing import List

data_dir = '../data/enwiki-20171001-pages-meta-current-withlinks-processed'


# word2index = defaultdict()

def remove_links(text: str):

    text.replace('</a>', ' ')
    text = re.sub('<a href=.*>', '', text)

    return text



data = {}
for file in glob.glob(os.path.join(data_dir, 'AA', '**')):

    for line in bz2.BZ2File(file):
        json_article = json.loads(line.decode('utf-8'))

        paragraphs = []
        char_count = 0
        paragraph_index = 0

        while char_count < 500 and paragraph_index < len(json_article['text']):
            plaintext = [remove_links(sentence) for sentence in json_article['text'][paragraph_index]]
            paragraphs.append(plaintext)
            char_count += sum(len(sentence) for sentence in plaintext)
            paragraph_index += 1

        data[json_article['title']] = paragraphs



with open('../data/preprocessed/AA.json', 'w') as data_file:
    json.dump(data, data_file)



