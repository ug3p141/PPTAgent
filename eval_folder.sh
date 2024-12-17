cat params.txt | while read -r line; do
    python evals.py eval_ppt -p "$line" &
done
