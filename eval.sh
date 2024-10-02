#!/bin/bash
# Dataset stat
python eval.py dataset_stat
# 定义所有要运行的组合
combinations=(
    "0 x"
    "1 x"
    "2 x"
    "3 x"
    "0 0"
    "0 1"
    "0 2"
)

for combo in "${combinations[@]}"; do
    read model_id ablation_id <<< "$combo"
    python eval.py generate_pres --model_id "$model_id" --thread_num 32 --ablation_id "$ablation_id"
    python eval.py postprocess_final_pptx
    python eval.py eval_experiment --model_id "$model_id" --ablation_id "$ablation_id"
    python eval.py get_fid --model_id "$model_id" --ablation_id "$ablation_id"
done
