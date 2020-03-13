#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"
#SBATCH --output=slurm_%j.out

module purge
module load cuda/10.0.130
module load anaconda3/5.3.1
module load gcc/6.3.0

# Replace with your NetID
NETID=kll482
source activate /scratch/${NETID}/nlu/env

cd /scratch/${NETID}/nlu/code/transformers

GLUE_DIR=/scratch/${NETID}/nlu/data
TASK_NAME=BoolQ
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH

python ./examples/run_glue.py \
    --model_type roberta \
    --model_name_or_path roberta-base \
    --task_name ${TASK_NAME} \
    --do_train \
    --do_eval \
    --data_dir ${GLUE_DIR}/${TASK_NAME} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir /scratch/${NETID}/nlu/${TASK_NAME}_run/
