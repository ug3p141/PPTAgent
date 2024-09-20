import hashlib
import json
import os
from copy import deepcopy

import PIL

import llms
from agent import PPTAgent
from model_utils import prs_dedup
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


def ppt_gen(text_content: str, ppt_file: str, images_dir: str, num_pages: int = 8):
    session_id = hashlib.md5(
        (text_content + ppt_file + images_dir).encode()
    ).hexdigest()
    print(f"Session ID: {session_id}")
    app_config.set_session(session_id)

    llms.long_model = llms.gpt4o
    llms.caption_model = llms.gpt4o

    presentation = Presentation.from_file(ppt_file)
    if len(presentation.error_history) > len(presentation.slides) // 3:
        raise ValueError(
            f"Too many errors (>25%) in the ppt: {presentation.error_history}"
        )
    if len(presentation.slides) < 6 or len(presentation.slides) > 100:
        raise ValueError("The number of effective slides should be between 6 and 100.")

    labler = ImageLabler(presentation)
    labler.caption_images()
    labler.apply_stats()
    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    duplicates = prs_dedup(presentation, ppt_image_folder)
    for slide in duplicates:
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"))
    for err_idx, _ in presentation.error_history:
        os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
    assert len(presentation.slides) == len(
        [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
    )
    for i, slide in enumerate(presentation.slides, 1):
        slide.slide_idx = i
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
        )
    # caption_prompt = open("prompts/image_label/caption.txt").read()
    # images = {
    #     pjoin(images_dir, k): [
    #         caption_model(caption_prompt, [pjoin(images_dir, k)]),
    #         PIL.Image.open(pjoin(images_dir, k)).size,
    #     ]
    #     for k in os.listdir(images_dir)
    #     if k.split(".")[-1] in IMAGE_EXTENSIONS
    # }
    images = json.load(open("resource/image_caption.json"))
    deepcopy(presentation).normalize().save(
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

    doc_json = json.load(open(pjoin(app_config.RUN_DIR, "refined_doc.json"), "r"))

    PPTAgent(
        presentation, slide_cluster, images, num_pages, doc_json, functional_keys
    ).work()


if __name__ == "__main__":

    app_config.set_debug(False)
    ppt_gen(
        open("resource/DOC2PPT.md").read(),
        "data/topic/Artificial_Intelligence_and_its_Impact/pptx/SquadTactics/original.pptx",
        "resource/doc2ppt_images",
    )
