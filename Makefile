eval_folder:
	cat params.txt | while read -r line; do \
		python evals.py eval_ppt -p "$line" & \
	done
clean:
	rm -f final.*
describe_slide:
	ls
extract_prs:
	ls
