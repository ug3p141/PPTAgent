import hashlib
import json
import os
from copy import deepcopy

import numpy as np

from agent import PPTAgent
from llms import agent_model, caption_model
from model_utils import text_embedding
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import (
    IMAGE_EXTENSIONS,
    app_config,
    pexists,
    pjoin,
    ppt_to_images,
    print,
    set_proxy,
)

# TODO 生成一个模板ppt的背景信息
# TODO 背景元素的识别


def ppt_gen(text_content: str, ppt_file: str, images_dir: str, num_pages: int = 12):
    session_id = hashlib.md5(
        (text_content + ppt_file + images_dir).encode()
    ).hexdigest()
    print(f"Session ID: {session_id}")
    app_config.set_session(session_id)

    # 1. 解析ppt
    presentation = Presentation.from_file(ppt_file)
    if len(presentation.error_history) > len(presentation.slides) // 3:
        raise ValueError(
            f"Too many errors (>25%) in the ppt: {presentation.error_history}"
        )
    if len(presentation.slides) < 6 or len(presentation.slides) > 100:
        raise ValueError("The number of effective slides should be between 6 and 100.")

    # 2. 模板生成

    labler = ImageLabler(presentation, app_config.RUN_DIR, app_config.IMAGE_DIR)
    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    del_idxs = eval(
        "[17, 17, 18, 18, 21, 28, 29, 30, 30, 31, 31, 31, 31, 31, 31, 31, 40, 43, 56]"
    )
    # del_idxs = []
    # pre_embedding = text_embedding(presentation.slides[0].to_text())
    # slide_idx = 1
    # while slide_idx < len(presentation.slides):
    #     cur_embedding = text_embedding(presentation.slides[slide_idx].to_text())
    #     if np.dot(pre_embedding, cur_embedding) / (np.linalg.norm(pre_embedding) * np.linalg.norm(cur_embedding)) > 0.8:
    #         del_idxs.append(slide_idx-1)
    #         presentation.slides.pop(slide_idx-1) # 去重
    #     else:
    #         slide_idx += 1
    #     pre_embedding = cur_embedding
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
    deepcopy(presentation).save(
        pjoin(app_config.RUN_DIR, "template.pptx"), layout_only=True
    )
    ppt_to_images(
        pjoin(app_config.RUN_DIR, "template.pptx"),
        pjoin(app_config.RUN_DIR, "template_images"),
    )
    functional_keys, slide_cluster = TemplateInducter(
        presentation, ppt_image_folder, pjoin(app_config.RUN_DIR, "template_images")
    ).work()
    presentation = presentation.normalize()
    # TODO 效果还需要很大改进
    # labler.label_images(functional_keys, slide_cluster, images)

    # 3. 使用模板生成PPT
    # 重新安排shape idx方便后续调整
    agent_model.clear_history()
    # 先文本，后图像
    PPTAgent(presentation, slide_cluster, text_content, images, num_pages).work(
        functional_keys
    )


if __name__ == "__main__":
    # set_proxy("http://124.16.138.148:7890")
    app_config.set_debug(True)
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        app_config.TEST_PPT,
        "resource/doc2slide_images",
    )
