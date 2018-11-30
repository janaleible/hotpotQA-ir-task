#!/bin/bash

#Specification of the job requirements for the batch system (number of nodes, expected runtime, etc)
#SBATCH --job-name=example
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00
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
mkdir -p "$TMPDOR"/data/index
cp -r $HOME/hotpotQA-ir-task/data/l2r/ "$TMPDIR"/data/
cp -r $HOME/hotpotQA-ir-task/data/index/ "$TMPDIR"/data/
cp $HOME/hotpotQA-ir-task/index.xml "$TMPDIR"/index.xml

ls -l "$TMPDIR"/data/l2r &> $HOME/log.log

#Running your application
cd $HOME/hotpotQA-ir-task
source venv/bin/activate
pip install -r requirements.txt
srun python3 main_l2r.py -a train -e 15 &>> $HOME/log.log

#Aggergating output data (e.g. post-processing, copying data from scratch to your home)
mkdir -p $HOME/hotpotQA-ir-task/models
cp -r $TMPDIR/models $HOME/hotpotQA-ir-taks/models

