# PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides

> The code and data are coming soon...

We propose **PPTAgent**, a system for automatically generating presentations from documents. It follows a two-step process inspired by how people create slides, ensuring high-quality content, clear structure, and visually appealing design. To evaluate the generated presentations, we also introduce **PPTEval**, a framework that measures the quality of presentations in terms of content, design, and coherence.

## Distinctive Features✨
- Dynamically generate slides that incorporate both text and images.
- Leverage existing presentations as references without the need for prior annotation.
- Comprehensively evaluated the quality of presentations from multiple perspectives.

## PPTAgent🤖

PPTAgent generates presentations in two steps:
1. **Analyze**: Studies reference presentations to identify patterns in structure and content.
2. **Generate**: Creates outlines and completes slides with consistent and aligned formatting.

The workflow of PPTAgent is shown below:

![PPTAgent Workflow](resource/fig2.jpg)

## PPTEval⚖️

PPTEval evaluates presentations across three dimensions:
- **Content**: Check the accuracy and relevance of the slides.
- **Design**: Assesses the visual appeal and consistency.
- **Coherence**: Ensures the logical flow of ideas.

The workflow of PPTEval is shown below:

![PPTEval Workflow](resource/fig3.jpg)

## Case Study📈

- #### [Build Effective Agents](https://www.google.com/search?client=safari&rls=en&q=building+effective+agents&ie=UTF-8&oe=UTF-8)

<div style="display: flex; flex-wrap: wrap; gap: 10px;">

  <img src="resource/build_effective_agents/0001.jpg" alt="图片1" width="200"/>

  <img src="resource/build_effective_agents/0002.jpg" alt="图片2" width="200"/>

  <img src="resource/build_effective_agents/0003.jpg" alt="图片3" width="200"/>

  <img src="resource/build_effective_agents/0004.jpg" alt="图片4" width="200"/>

  <img src="resource/build_effective_agents/0005.jpg" alt="图片5" width="200"/>

  <img src="resource/build_effective_agents/0006.jpg" alt="图片6" width="200"/>

  <img src="resource/build_effective_agents/0007.jpg" alt="图片7" width="200"/>

  <img src="resource/build_effective_agents/0008.jpg" alt="图片8" width="200"/>

<img src="resource/build_effective_agents/0009.jpg" alt="图片8" width="200"/>

<img src="resource/build_effective_agents/0010.jpg" alt="图片8" width="200"/>

</div>

## Reproduce the evaluation🧪

1. Requirements
```sh
pip install -r requirements.txt
sudo apt install libreoffice
# brew install libreoffice
sudo apt install poppler-utils
# conda install -c conda-forge poppler
```

2. Reproduce the pptxs according the saved history files.
```sh
python rebuild.py rebuild_all --out_filename "final.pptx"
```

3. Parse the pptxs to images to prepare for evaluation.
```sh
python evals.py pptx2images
```

4. Evaluate the pptxs.
```sh
python evals.py eval_experiment -s 0 -j 0
```

## Citation🙏

If you find this project helpful, please use the following to cite it:
```bibtex
@misc{zheng2025pptagentgeneratingevaluatingpresentations,
      title={PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides},
      author={Hao Zheng and Xinyan Guan and Hao Kong and Jia Zheng and Hongyu Lin and Yaojie Lu and Ben He and Xianpei Han and Le Sun},
      year={2025},
      eprint={2501.03936},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.03936},
}
```
