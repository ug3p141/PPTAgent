import hashlib
import json
import os

import numpy as np
from FlagEmbedding import BGEM3FlagModel

from agent import PPTAgent
from llms import caption_model
from multimodal import ImageLabler
from presentation import Presentation
from template_inducter import TemplateInducter
from utils import app_config, print, set_proxy

model = None


# 从lyj的ppt上来看，我觉得第一步就应该先去重
def ppt_gen(pdf_markdown: str, ppt_file: str, images_dir: str, num_pages: int = 8):
    global model
    # model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    session_id = hashlib.md5(
        (pdf_markdown + ppt_file + images_dir + str(num_pages)).encode()
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
    # slide_idx = 0
    # pre_embedding = model.encode(presentation.slides[slide_idx].to_text())
    # while slide_idx < len(presentation.slides):
    #     slide_idx += 1
    #     cur_embedding = model.encode(presentation.slides[slide_idx].to_text())
    #     if np.dot(pre_embedding, cur_embedding) / (np.linalg.norm(pre_embedding) * np.linalg.norm(cur_embedding)) > 0.8:
    #         presentation.slides.pop(slide_idx) # dedup
    #     else:
    #         slide_idx += 1
    # images = {pjoin(images_dir,k): caption_model(open('prompts/image_label/caption.txt').read()pjoin(images_dir,k)) for k in os.listdir(images_dir)}
    labler = ImageLabler(presentation).work()  # caption
    slide_cluster = TemplateInducter(presentation).work()
    labler.label_images()
    images = json.load(open("resource/image_caption.json"))

    # 4. 使用markdown及模板生成PPT
    PPTAgent(presentation, slide_cluster, pdf_markdown, images, num_pages).work()


if __name__ == "__main__":
    set_proxy("http://124.16.138.148:7890")
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        app_config.TEST_PPT,
        "resource/doc2ppt_images",
    )
