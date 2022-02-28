#!/bin/bash
# This scripts trains adapters method.
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
D=$7
N=$8
GC=true
if [ $N > 0.0 ]; then
    OUT_DIR=outputs/ffngegluadapters${D}_${TASK}noise${N}_${MODEL}_eps${EP}_lr${LR}_bs${BS}
else
    OUT_DIR=outputs/ffngegluadapters${D}_${TASK}_${MODEL}_eps${EP}_lr${LR}_bs${BS}
fi
mkdir -p $OUT_DIR
cat configs/bitfit.json > $OUT_DIR/settings.json
echo '"add_layer_norm_before_adapter": false,
"add_layer_norm_after_adapter": false,
"adapter_config_name": "adapter",
"train_task_adapters": true,
"adapter_size": '${D}',
"non_linearity": "geglu",
"unfreeze_lm_head": false,
"unfreeze_layer_norms": true,
"output_dir": "'${OUT_DIR}'",
"max_source_length": 128,
"task_name": "'${TASK}'",
"noise_frac": "'${N}'",
"overwrite_cache": true,
"eval_dataset_name": "'${TASK}'",
"test_dataset_name": "'${TASK}'",
"num_train_epochs": '${EP}',
"learning_rate": '${LR}',
"model_name_or_path": "'${MODEL}'",
"tokenizer_name": "'${MODEL}'",
"per_device_train_batch_size": '${BS_}',
"per_device_eval_batch_size": '${BS_}',
"fp16": false,
"add_adapter_in_self_attention": false,
"gradient_accumulation_steps": '${GA}'
}' >> $OUT_DIR/settings.json
python run_seq2seq.py $OUT_DIR/settings.json &> $OUT_DIR/log.txt
