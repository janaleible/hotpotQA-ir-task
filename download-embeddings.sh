#!/usr/bin/env bash

wget http://nlp.stanford.edu/data/glove.6B.zip > ./data/embeddings/
unzip ./data/embeddings/glove.6B.zip -d ./data/embeddings/