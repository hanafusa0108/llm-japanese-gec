#!/bin/bash
#SBATCH -J gemma-3-1b-it-lr1e-4-predict       #ジョブ名。なんでもよい。
#SBATCH -p bacchus          #指定しなければデフォルトパーテーションはopus、[opus,luce,lepin,ausone,varuna,ganesa,budha,hestia]この中から選ぶ。
#SBATCH -o logs/gemma/%x_%j.log #標準出力先。この場合[ジョブ名]_[ジョブID].logというファイル名で保存される。
#SBATCH -t 365-00:00:00     #ジョブの実行上限時間を指定。days-hours:minutes:seconds
#SBATCH -c 4              #要求するCPUコア数指定。指定しなければ、1コアが割り当てられる。
#SBATCH --mem=128GB        #要求するCPUコア数指定。指定しなければ、1コアあたり1GBメモリが割り当てられる。
#SBATCH --gpus=1          #要求するGPUの枚数。指定しなければGPUは使えない。
#SBATCH --mail-type=ALL  # ジョブの開始や終了などメールを受け取ることができる。 値はBEGIN, END, FAIL, ALL。メール送信の際は#を1個を外すこと。
#SBATCH --mail-user=hanafusa@ai.cs.ehime-u.ac.jp   #メール送信先。メール送信の際は#を1個を外すこと。

MODEL=openai/gpt-oss-20b
CONFIG_PATH=configs/wikipedia.yaml
ADAPTER=./outputs/google/gemma-3-1b-it_lr0.0001/checkpoint-100
FILE_NAME=gemma-3-1b-it-lr1e-4

python -m scripts.predict.gpt_oss \
  --model $MODEL \
  --config $CONFIG_PATH \
  --adapter $ADAPTER \
  --file_name $FILE_NAME \