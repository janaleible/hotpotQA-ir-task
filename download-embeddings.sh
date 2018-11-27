#!/usr/bin/env bash

wget http://nlp.stanford.edu/data/glove.6B.zip &&
mv  glove.6B.zip ./data/embeddings/glove.6B.zip
unzip ./data/embeddings/glove.6B.zip -d ./data/embeddings/