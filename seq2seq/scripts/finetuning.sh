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
#SBATCH --exclude=charles[01-10]
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
N=$7
GC=false
if [ $N > 0.0 ]; then
    OUT_DIR=outputs/ft_${TASK}noise${N}_${MODEL}_eps${EP}_lr${LR}_bs${BS}
else
    OUT_DIR=outputs/ft_${TASK}_${MODEL}_eps${EP}_lr${LR}_bs${BS}
fi
mkdir -p $OUT_DIR
cat configs/bitfit.json > $OUT_DIR/settings.json
echo '"output_dir": "'${OUT_DIR}'",
"max_source_length": 128,
"noise_frac": "'${N}'",
"overwrite_cache": true,
"optim": "adafactor",
"task_name": "'${TASK}'",
"eval_dataset_name": "'${TASK}'",
"test_dataset_name": "'${TASK}'",
"num_train_epochs": '${EP}',
"learning_rate": '${LR}',
"model_name_or_path": "'${MODEL}'",
"tokenizer_name": "'${MODEL}'",
"per_device_train_batch_size": '${BS_}',
"per_device_eval_batch_size": '${BS_}',
"fp16": false,
"gradient_checkpointing": '${GC}',
"gradient_accumulation_steps": '${GA}'
}' >> $OUT_DIR/settings.json
python run_seq2seq.py $OUT_DIR/settings.json &> $OUT_DIR/log.txt
