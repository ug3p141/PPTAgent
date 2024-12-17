#!/bin/bash
# PPTCrew_with_gpt4o 463 500
# PPTCrew_retry_5 484 500
# PPTCrew_wo_Decoupling 12 500

# 输入数据
data=$(cat <<EOF
# PPTCrew-gpt-4o+gpt-4o+gpt-4o 489 500
# PPTCrew_wo_SchemaInduction 396 500
# PPTCrew_wo_HTML 373 500
# PPTCrew-Qwen2-VL+Qwen2-VL+Qwen2-VL 215 500
# PPTCrew_wo_LayoutInduction 455 500
# PPTCrew_wo_Structure 461 500
# PPTCrew-Qwen2.5+Qwen2.5+Qwen2-VL 475 500
EOF
)



# 遍历第一列并启动所有任务
echo "$data" | while read -r line; do
    param=$(echo "$line" | awk '{print $2}')
    python evals.py eval_experiment -s "$param" -p -t 1 -p &
done
