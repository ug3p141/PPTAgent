import hashlib
import json
import os

import numpy as np

from agent import PPTAgent
from llms import caption_model
from model_utils import text_embedding
from multimodal import ImageLabler
from presentation import Presentation
from template_inducter import TemplateInducter
from utils import app_config, pjoin, print, set_proxy


def ppt_gen(pdf_markdown: str, ppt_file: str, images_dir: str, num_pages: int = 8):
    session_id = hashlib.md5(
        (pdf_markdown + ppt_file + images_dir + str(num_pages)).encode()
    ).hexdigest()
    print(f"Session ID: {session_id}")
    app_config.set_session(session_id)
    app_config.set_debug(True)

    # 1. 解析ppt
    presentation = Presentation.from_file(ppt_file)
    if len(presentation.error_history) > len(presentation.slides) // 3:
        raise ValueError(
            f"Too many errors (>25%) in the ppt: {presentation.error_history}"
        )
    if len(presentation.slides) < 6 or len(presentation.slides) > 100:
        raise ValueError("The number of effective slides should be between 6 and 100.")

    # 2. 模板生成

    labler = ImageLabler(presentation)
    labler.caption_images()
    slide_idx = 1
    # pre_embedding = text_embedding(presentation.slides[0].to_text())
    # del_idxs = []
    # while slide_idx < len(presentation.slides):
    #     cur_embedding = text_embedding(presentation.slides[slide_idx].to_text())
    #     if np.dot(pre_embedding, cur_embedding) / (np.linalg.norm(pre_embedding) * np.linalg.norm(cur_embedding)) > 0.8:
    #         del_idxs.append(slide_idx-1)
    #         presentation.slides.pop(slide_idx-1) # 去重
    #     else:
    #         slide_idx += 1
    #     pre_embedding = cur_embedding
    del_idxs = eval(
        "[17, 17, 18, 18, 21, 28, 29, 30, 30, 31, 31, 31, 31, 31, 31, 31, 40, 43, 56, 61]"
    )
    for slide_idx in del_idxs:
        presentation.slides.pop(slide_idx)

    for slide_idx in range(len(presentation.slides)):
        presentation.slides[slide_idx].slide_idx = slide_idx + 1
    # caption_prompt = open("prompts/image_label/caption.txt").read()
    # images = {
    #     pjoin(images_dir, k): caption_model(caption_prompt, pjoin(images_dir, k))
    #     for k in os.listdir(images_dir)
    # }
    images = json.load(open("resource/image_caption.json"))
    # 去掉对metadata keys的需求
    slide_cluster = TemplateInducter(presentation).work()
    # 其实这一步有点没必要了
    # labler.label_images()  # TODO 重写prompt size_match的原则？

    # 4. 使用模板生成PPT
    PPTAgent(presentation, slide_cluster, pdf_markdown, images, num_pages).work()


if __name__ == "__main__":
    # set_proxy("http://124.16.138.148:7890")
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        app_config.TEST_PPT,
        "resource/doc2ppt_images",
    )
