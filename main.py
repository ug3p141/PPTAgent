import json

from apis import model_api
from multimodal import ImageLabler
from presentation import Presentation
from template_cluster import TemplateCluster
from utils import app_config, pjoin


def ppt_gen(pdf_markdown: str, ppt_file: str):
    # 1. 解析ppt
    presentation = Presentation.from_file(ppt_file)

    # 2. 图片标注
    ImageLabler(presentation, app_config.RUN_DIR).work()

    # 3. 模板生成
    slide_cluster = TemplateCluster(presentation, app_config.RUN_DIR).work()

    # 4. PPT内容生成

    # 5. 使用生成的内容及模板生成PPT


if __name__ == "__main__":
    pdf_markdown = "This is a test."
    ppt_gen(pdf_markdown, app_config.TEST_PPT)
