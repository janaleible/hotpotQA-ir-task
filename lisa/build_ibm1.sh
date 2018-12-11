#!/usr/bin/env bash

#Specification of the job requirements for the batch system (number of nodes, expected runtime, etc)
#SBATCH --job-name=feature-extraction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

module load Python/3.6.3-foss-2017b

mkdir -p "$TMPDIR"/data
mkdir -p "$TMPDIR"/data/hotpot
cp -r $HOME/hotpotQA-ir-task/data/hotpot/ "$TMPDIR"/data/

mkdir -p "$TMPDIR"/models
cd $HOME/hotpotQA-ir-task
source venv/bin/activate
pip install -r requirements.txt
srun python3 retrieval/feature_extractors/IBM1FeatureExtractor.py &>> $HOME/log.log

mkdir -p $HOME/hotpotQA-ir-task/models
cp -r $TMPDIR/models $HOME/hotpotQA-ir-taks/models
