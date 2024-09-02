import asyncio
import json
import os
from collections import defaultdict
from copy import deepcopy

import aiohttp
from googlesearch import search

from llms import caption_model, get_refined_doc
from model_utils import prs_dedup
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import (
    IMAGE_EXTENSIONS,
    app_config,
    filename_normalize,
    parse_pdf,
    pjoin,
    ppt_to_images,
)

topics = [
    "Business Plan",
    "Marketing Strategy",
    "Project Management",
    "Financial Analysis",
    "Product Launch",
    "Company Overview",
    "Sales Report",
    "Training and Development",
    "Industry Trends",
    "Customer Experience",
    "Data Analysis",
    "Strategic Planning",
    "Risk Management",
    "Customer Relationship Management",
    "Supply Chain Management",
    "Human Resources",
    "IT Infrastructure",
    "Customer Support",
    "Research and Development",
    "Corporate Social Responsibility",
    "International Business",
    "Mergers and Acquisitions",
    "Real Estate",
    "Healthcare",
    "Education",
    "Technology",
]


def get_file_links(topic: str, num_results: int, filetype: str):
    query = f"{topic} filetype:{filetype}"
    return [
        {"title": url.split("/")[-1], "url": url}
        for url in search(query, num_results=num_results)
        if url.endswith(".pptx")
    ]


async def download_file(session, url, filepath):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, "wb") as f:
                    f.write(await response.read())
    except:
        return


async def crawl_datas():
    async with aiohttp.ClientSession() as session:
        files = [("pptx", 50), ("pdf", 10)]
        tasks = []
        for filetype, num_results in files:
            for topic in topics:
                links = get_file_links(topic, num_results, filetype)
                for link in links:
                    filepath = f"data/{filename_normalize(topic)}/{filetype}/{filename_normalize(link['title'])}"
                    tasks.append(download_file(session, link["url"], filepath))
        await asyncio.gather(*tasks)


def ppt_validate(presentation: Presentation):
    if len(presentation.slides) < 6 or len(presentation.slides) > 50:
        return False
    if len(presentation.error_history) > 0:
        return False
    layout_count = defaultdict(int)

    for slide in presentation.slides:
        layout_count[slide.layout_name] += 1
    if sum(layout_count.values()) / len(layout_count) < 3:
        return False

    return True


# 模板的主题要和ppt的主题接近


def prepare_pdf(filename: str, output_dir: str):
    text_content = parse_pdf(filename, output_dir)
    if len(text_content) < 500 or len(text_content) > 10000:
        return
    os.makedirs(output_dir)
    caption_prompt = open("prompts/image_label/caption.txt").read()
    image_stats = {}
    for image in os.listdir(pjoin(output_dir, "images")):
        if image.split(".")[-1] in IMAGE_EXTENSIONS:
            image_stats[image] = caption_model(
                caption_prompt, pjoin(output_dir, "images", image)
            )
    json.dump(
        image_stats,
        open(pjoin(output_dir, "image_caption.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )
    doc_json = get_refined_doc(text_content)
    json.dump(
        doc_json,
        open(pjoin(output_dir, "refined_doc.json"), "w"),
        indent=4,
        ensure_ascii=False,
    )


def prepare_ppt(filename: str):
    presentation = Presentation.from_file(filename)
    if not ppt_validate(presentation):
        os.remove(filename)
        return

    dedup_slides = prs_dedup(presentation)
    if len(dedup_slides) > len(presentation.slides) * 0.3:
        os.remove(filename)
        return

    ImageLabler(presentation).caption_images()
    # TODO: use internvl-72b
    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    for slide in dedup_slides:
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"))
    for i, slide in enumerate(presentation.slides):
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{i+1:04d}.jpg"),
        )
        slide.slide_idx = i + 1

    deepcopy(presentation).save(
        pjoin(app_config.RUN_DIR, "template.pptx"), layout_only=True
    )
    ppt_to_images(
        pjoin(app_config.RUN_DIR, "template.pptx"),
        pjoin(app_config.RUN_DIR, "template_images"),
    )
    presentation.normalize().save(pjoin(app_config.RUN_DIR, "source.pptx"))
    TemplateInducter(
        presentation, ppt_image_folder, pjoin(app_config.RUN_DIR, "template_images")
    ).work()


if __name__ == "__main__":
    os.makedirs("resource/crawler", exist_ok=True)
    asyncio.run(crawl_datas())
    ppt_files = os.listdir("data/pptx")
    for topic in topics:
        topic_dir = pjoin("data", topic)
        for ppt_file in ppt_files:
            app_config.set_rundir(
                pjoin(topic_dir, "pptx", filename_normalize(ppt_file[:10]))
            )
            prepare_ppt(
                pjoin(topic_dir, "pptx", ppt_file),
            )
        pdf_files = os.listdir(pjoin(topic_dir, "pdf"))
        for pdf_file in pdf_files:
            prepare_pdf(
                pjoin(topic_dir, "pdf", pdf_file),
                pjoin(topic_dir, "pdf", filename_normalize(pdf_file[:10])),
            )
