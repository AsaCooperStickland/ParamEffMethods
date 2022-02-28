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
python run_seq2seq.py configs/lhuc_after.json # &> outputs/lhuc_after/log.txt
python run_seq2seq.py configs/lhuc_b4.json # &> outputs/lhuc_after/log.txt
