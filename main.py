import json
import os

from agent import PPTAgent
from llms import caption_image
from multimodal import ImageLabler
from presentation import Presentation
from template_inducter import TemplateInducter
from utils import app_config


def ppt_gen(pdf_markdown: str, ppt_file: str, images_dir: str, num_pages: int):
    # 1. 解析ppt
    presentation = Presentation.from_file(ppt_file)

    # 2. 图片标注
    # images = {pjoin(images_dir,k): caption_image(pjoin(images_dir,k)) for k in os.listdir(images_dir)}
    images = json.load(open("resource/image_caption.json"))
    ImageLabler(presentation).work()

    # 3. 模板生成
    slide_cluster = TemplateInducter(presentation).work()

    # 4. 使用markdown及模板生成PPT
    PPTAgent(presentation, slide_cluster, pdf_markdown, images, num_pages).work()


if __name__ == "__main__":
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        app_config.TEST_PPT,
        "data/readme/img/468319",
        5,
    )
