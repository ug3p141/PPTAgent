# PPTAgent

### Reproduce the evaluation

1. Requirements
```
pip install -r requirements.txt
sudo apt install libreoffice
# brew install libreoffice
sudo apt install poppler-utils
conda install -c conda-forge poppler
```

2. Reproduce the pptxs according the saved history files.
```
python rebuild.py rebuild_all --out_filename "final.pptx"
```

3. Parse the pptxs to images to prepare for evaluation.
```
python evals.py pptx2images
```

4. Evaluate the pptxs.
```
python evals.py eval_experiment -s 0 -j 0
```
