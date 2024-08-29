# exp.1 fid score
# exp.2 mllm eval score
# exp.3 ablation study
# exp.4 llm study
import hashlib
import json
import os
from tempfile import TemporaryDirectory

import pytorch_fid.fid_score as fid

from llms import agent_model, caption_model
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


def exp1_0_llms(
    source_prs_file, generated_prs_file, batch_size=1, device="cuda", dims=2048
):
    pass


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
def exp2_mllm_eval(source_prs_file, generated_prs_file):
    pass


def exp3_ablation_html(source_prs_file, generated_prs_file):
    pass


def exp3_ablation_cluster(source_prs_file, generated_prs_file):
    pass


def all_experiments():
    # llm 完成率 * fid_score * mllm

    session_id = pjoin(pdf_dir, prs_dir)
    app_config.set_session(session_id)

    # 1. 解析ppt

    # 2. 模板生成

    labler = ImageLabler(presentation)
    labler.caption_images()
    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    # duplicates = prs_dedup(presentation)
    del_idxs = eval(
        "[17, 17, 18, 18, 21, 28, 29, 30, 30, 31, 31, 31, 31, 31, 31, 31, 40, 43, 56]"
    )
    for slide_idx in del_idxs:
        slide = presentation.slides.pop(slide_idx)
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"))
    for err_idx, _ in presentation.error_history:
        os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
    assert len(presentation.slides) == len(
        [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
    )
    for i, slide in enumerate(presentation.slides):
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{i+1:04d}.jpg"),
        )
        slide.slide_idx = i + 1
    # caption_prompt = open("prompts/image_label/caption.txt").read()
    # images = {
    #     pjoin(images_dir, k): caption_model(caption_prompt, pjoin(images_dir, k))
    #     for k in os.listdir(images_dir) if k.split(".")[-1] in IMAGE_EXTENSIONS
    # }
    images = json.load(open("resource/image_caption.json"))

    functional_keys, slide_cluster = TemplateInducter(presentation).work()
    presentation = presentation.normalize()
    # TODO 效果还需要很大改进
    # labler.label_images(functional_keys, slide_cluster, images)

    # 3. 使用模板生成PPT
    # 重新安排shape idx方便后续调整
    agent_model.clear_history()
    PPTAgent(presentation, slide_cluster, text_content, images, num_pages).work()


def prepare_prs():
    for prs_file in crawled_prs:
        prs_dir = filename_normalize(prs_file[:10])


if __name__ == "__main__":
    # 100 * 10
    prepare_pdfs()
