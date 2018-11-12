#!/usr/bin/env bash

mkdir -p data

# Wikipedia corpus
if [ ! -d "data/raw" ]; then
    wget --output-document data/raw.tar.bz2 https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-processed.tar.bz2
    bzip2 -d data/raw.tar.bz2
    tar -xvf data/raw.tar
    mv enwiki-20171001-pages-meta-current-withlinks-processed data/
    mv enwiki-20171001-pages-meta-current-withlinks-processed/ raw/
    rm data/raw.tar
fi

# HotpotQA data sets
if [ ! -d "data/hotpot" ]; then
    mkdir -p data/hotpot
    wget --output-document data/hotpot/train.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json
    wget --output-document data/hotpot/dev_distractor.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
    wget --output-document data/hotpot/dev_fullwiki.json http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json
fi
