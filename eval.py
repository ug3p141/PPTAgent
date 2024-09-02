# exp.1 fid score
# exp.2 mllm eval score
# exp.3 ablation study
# exp.4 llm study
import json
import os
import shutil
from itertools import product
from tempfile import TemporaryDirectory

import pytorch_fid.fid_score as fid

import llms
from agent import PPTAgent
from crawler import topics
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import (
    IMAGE_EXTENSIONS,
    app_config,
    filename_normalize,
    parse_pdf,
    pexists,
    pjoin,
    ppt_to_images,
)

fid.tqdm = lambda x: x


def exp1_0_llms(source_prs_folder, source_pdf_folder):
    if not (os.path.isdir(source_prs_folder) and os.path.isdir(source_pdf_folder)):
        return
    llm_models = [llms.gpt4o, llms.gpt4o_mini, llms.gemini, llms.qwen, llms.llama3]
    for llm in llm_models:
        llms.agent_model.set_model(llm)
        app_config.set_run_dir(
            f"experiments/exp1/{source_prs_folder.split('/')[-1]}/{source_pdf_folder.split('/')[-1]}/{llm.model}",
        )
        # 重新映射图片目录
        image_stats = json.load(
            open(pjoin(source_prs_folder, "image_stats.json"), "r"),
        )
        labler = ImageLabler(presentation)
        labler.apply_stats(image_stats)
        doc_json = json.load(
            open(pjoin(source_pdf_folder, "refined_doc.json"), "r"),
        )
        images = json.load(open(pjoin(source_pdf_folder, "images.json")))
        presentation = Presentation.from_file(pjoin(source_prs_folder, "source.pptx"))
        # 相同llm的此步是被缓存了的？没有，缓存一下吧
        slide_cluster = json.load(open(pjoin(source_prs_folder, "slide_cluster.json")))
        functional_keys, slide_cluster = (
            set(slide_cluster.pop("functional_keys")),
            slide_cluster,
        )
        PPTAgent(presentation, slide_cluster, images, 12, doc_json).work(
            functional_keys
        )


def exp1_1_fid(
    source_prs_file, generated_prs_file, batch_size=1, device="cuda", dims=2048
):
    with TemporaryDirectory() as temp_dir:
        source_folder = pjoin(temp_dir, "source")
        generated_folder = pjoin(temp_dir, "generated")
        os.makedirs(source_folder)
        os.makedirs(generated_folder)
        ppt_to_images(source_prs_file, source_folder)
        ppt_to_images(generated_prs_file, generated_folder)
        score = fid.calculate_fid_given_paths(
            [source_folder, generated_folder],
            batch_size=batch_size,
            device=device,
            dims=dims,
        )
        return score


# 1. 文本内容
# 2. 视觉表现力
# 3. 是否存在重叠
# 4. 是否具有视觉一致性
# 5. 是否完整
# 6. 呈现的内容与布局是否一致
def exp2_mllm_eval(source_prs_file, generated_prs_file):
    pass


def exp3_ablation_html(source_prs_file, generated_prs_file):
    pass


def exp3_ablation_cluster(source_prs_file, generated_prs_file):
    pass


if __name__ == "__main__":
    # 100 * 10
    for topic in topics:
        pptx_dir = pjoin("data", topic, "pptx")
        pdf_dir = pjoin("data", topic, "pdf")
        ppt_folders = [pjoin(pptx_dir, folder) for folder in os.listdir(pptx_dir)]
        pdf_folders = [pjoin(pdf_dir, folder) for folder in os.listdir(pdf_dir)]
        args = product(ppt_folders, pdf_folders)
