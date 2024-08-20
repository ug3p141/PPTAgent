import hashlib
import json
import os

import utils
from agent import PPTAgent
from llms import caption_image
from multimodal import ImageLabler
from presentation import Presentation
from template_inducter import TemplateInducter


def ppt_gen(pdf_markdown: str, ppt_file: str, images_dir: str, num_pages: int = 8):
    session_id = hashlib.md5(
        (pdf_markdown + ppt_file + images_dir + str(num_pages)).encode()
    ).hexdigest()
    print(f"Session ID: {session_id}")
    utils.app_config = utils.Config(session_id)
    # 1. 解析ppt
    presentation = Presentation.from_file(ppt_file)
    if len(presentation.error_history) > len(presentation.slides) // 3:
        raise ValueError(
            f"Too many errors (>25%) in the ppt: {presentation.error_history}"
        )
    if len(presentation.slides) < 6 or len(presentation.slides) > 50:
        raise ValueError("The number of effective slides should be between 6 and 50.")

    # 2. 图片标注
    # images = {pjoin(images_dir,k): caption_image(pjoin(images_dir,k)) for k in os.listdir(images_dir)}
    images = json.load(open("resource/image_caption.json"))
    ImageLabler(presentation).work()

    # 3. 模板生成
    slide_cluster = TemplateInducter(presentation).work()

    # 4. 使用markdown及模板生成PPT
    PPTAgent(presentation, slide_cluster, pdf_markdown, images, num_pages).work()


if __name__ == "__main__":
    utils.set_proxy("http://124.16.138.148:7890")
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        utils.app_config.TEST_PPT,
        "resource/doc2ppt_images",
    )
