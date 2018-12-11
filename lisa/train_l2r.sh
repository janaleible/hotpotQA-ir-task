#!/bin/bash

#Specification of the job requirements for the batch system (number of nodes, expected runtime, etc)
#SBATCH --job-name=l2r
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared
#SBATCH --gres=gpu:1

#Load of modules needed to run your application
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.4
4-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

#Preparing input data (e.g. copying input data from your home folder to scratch, preprocessing)
mkdir -p "$TMPDIR"/data
mkdir -p "$TMPDIR"/data/l2r
mkdir -p "$TMPDIR"/data/index
mkdir -p "$TMPDIR"/data/maps
mkdir -p "$TMPDIR"/data/candidates
mkdir -p "$TMPDIR"/data/embeddings
mkdir -p "$TMPDIR"/data/trec_eval
cp -r $HOME/hotpotQA-ir-task/data/l2r/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/index/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/maps/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/candidates/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/embeddings/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/trec_eval/ "$TMPDIR"/data/
cp $HOME/hotpotQA-ir-task/index.xml "$TMPDIR"/index.xml

#Running your application
mkdir -p "$TMPDIR"/models
cd $HOME/hotpotQA-ir-task
source venv/bin/activate
pip install -r requirements.txt
srun python3 main_retrieval.py -g neural -m max_pool_bllr_pw &>> $HOME/log.log

#Aggergating output data (e.g. post-processing, copying data from scratch to your home)
mkdir -p $HOME/hotpotQA-ir-task/models
cp -r $TMPDIR/models $HOME/hotpotQA-ir-taks/models

