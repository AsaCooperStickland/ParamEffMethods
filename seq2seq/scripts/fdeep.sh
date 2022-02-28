#!/bin/bash
# This scripts trains bitfit method.
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1
#SBATCH --mem=14000  # memory in Mb
#SBATCH --exclude=charles[11-19]
#SBATCH --time=48:00:00

source ~/newconda/bin/activate survey
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3.
TASK=$1
EP=$2
LR=$3
MODEL=$4
BS_=$5
GA=$6
BS=$((BS_*GA))
GC=true
OUT_DIR=outputs/ftdp_${TASK}_${MODEL}_eps${EP}_lr${LR}_bs${BS}
mkdir -p $OUT_DIR
export PATH=/opt/cuda-11.0.1_460.32/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/opt/cuda-11.0.1_460.32/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_PATH=/opt/cuda-11.0.1_460.32/
export CXX=g++
CUDA_HOME=/opt/cuda-11.0.1_460.32/ deepspeed run_seq2seq.py --do_train --output_dir ${OUT_DIR} --do_eval --do_test --warmup_steps 500 --save_steps 100 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model average_metrics --greater_is_better true --evaluation_strategy epoch --save_strategy epoch --non_linearity gelu_new --split_validation_test --dataset_config_name en --eval_dataset_config_name en --test_dataset_config_name en --predict_with_generate --compute_memory --max_source_length 128 --deepspeed deepspeed/ds_config_zero2.json --task_name ${TASK} --eval_dataset_name ${TASK} --test_dataset_name ${TASK} --num_train_epochs ${EP} --learning_rate ${LR} --model_name_or_path ${MODEL} --tokenizer_name ${MODEL} --per_device_train_batch_size ${BS_} --per_device_eval_batch_size ${BS_} --gradient_accumulation_steps ${GA} &> $OUT_DIR/log.txt # --gradient_checkpointing ${GC}
#nvcc -V &> $OUT_DIR/log.txt
