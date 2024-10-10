import asyncio
from concurrent.futures import ThreadPoolExecutor
import glob
import json
import os
import re
import shutil
import traceback
from collections import defaultdict
from itertools import product
from tempfile import TemporaryDirectory

import aiohttp
import func_argparse
import jsonlines
import PyPDF2
from googlesearch import search
from tqdm import tqdm

import llms
from model_utils import image_embedding, images_cosine_similarity, prs_dedup
from presentation import Presentation
from utils import (
    filename_normalize,
    parse_pdf,
    Config,
    pexists,
    pjoin,
    ppt_to_images,
    print,
)
app_config = Config()

topics = {
    "Artificial Intelligence and its Impact": [
        "The Evolution of Artificial Intelligence",
        "The Impact of Artificial Intelligence on Employment",
        "Artificial Intelligence in Healthcare",
        "The Ethics of Artificial Intelligence",
        "Artificial Intelligence in Financial Services",
        "The Role of Artificial Intelligence in Marketing",
    ],
    "Mental Health and Society": [
        "Mental Health Awareness in the Workplace",
        "The Impact of Social Media on Mental Health",
        "The Importance of Mental Health in Education",
        "The Impact of Mental Health on Physical Health",
        "The Role of Mental Health in Society",
        "The Impact of Mental Health on Academic Performance",
    ],
    "E-commerce and Digital Economy": [
        "The Rise of E-commerce",
        "The Rise of the Gig Economy",
        "The Future of Work in the Age of Automation",
        "The Future of Work: Remote and Hybrid Models",
        "The Impact of E-commerce on Traditional Retail",
        "The Role of E-commerce in Global Trade",
        "The Impact of E-commerce on Consumer Behavior",
    ],
    "Social Media and Cultural Influence": [
        "The Role of Social Media in Modern Marketing",
        "The Influence of Pop Culture on Society",
        "The Impact of Social Media on Mental Health",
        "The Role of Social Media in Social Change",
        "The Impact of Social Media on Privacy",
        "The Role of Social Media in Education",
        "The Impact of Social Media on Academic Performance",
        "The Role of Social Media in Politics",
        "The Impact of Social Media on Business",
    ],
    "Ethics, Leadership, and Society": [
        "Ethical Issues in Genetic Engineering",
        "The Role of Women in Leadership",
        "The Importance of Biodiversity",
        "The Role of Ethics in Business",
        "The Impact of Ethics on Society",
        "The Role of Ethics in Politics",
        "The Impact of Ethics on Academic Performance",
        "The Role of Ethics in Education",
    ],
}


def get_search_links(topic: str, num_results: int, filetype: str):
    query = f"{topic} filetype:{filetype}"
    return [
        {"title": url.split("/")[-1], "url": url}
        for url in search(query, num_results=num_results, lang="en", sleep_interval=1)
    ]


async def download_file(session, url, filepath):
    try:
        if pexists(filepath):
            return
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, "wb") as f:
                    f.write(await response.read())
    except:
        return


def get_file_links(jsonl_file: str, topics: list, num_results: int = 100):
    files = ["pptx", "pdf"]
    iter_tp_fp = product(topics, files)
    if pexists(jsonl_file):
        with jsonlines.open(jsonl_file) as reader:
            existed_tasks = list(reader)
        downloaded_tasks = set(
            [(task["topic"], task["filetype"]) for task in existed_tasks]
        )
        iter_tp_fp = [i for i in iter_tp_fp if i not in downloaded_tasks]
        existing_links = set([task["url"] for task in existed_tasks])
    else:
        existed_tasks = []

    with jsonlines.open(jsonl_file, mode="a") as writer:
        for topic, filetype in iter_tp_fp:
            print(f"crawling {topic}.{filetype}")
            try:
                links = get_search_links(topic, num_results, filetype)
            except Exception as e:
                print(f"crawled failed: {e}")
                exit()
            for link in links:
                filepath = f"data/subtopics/{filename_normalize(topic)}/{filetype}/{filename_normalize(link['title'])}"
                writer.write(
                    {
                        "filepath": filepath,
                        "url": link["url"],
                        "topic": topic,
                        "filetype": filetype,
                    }
                )
    print("File links saved to download_tasks.jsonl")


async def download_files(jsonl_file: str):
    async with aiohttp.ClientSession() as session:
        tasks = []
        with jsonlines.open(jsonl_file) as reader:
            for task in reader:
                tasks.append(download_file(session, task["url"], task["filepath"]))
        await asyncio.gather(*tasks)


def ppt_validate(presentation: Presentation):
    if len(presentation) < 8 or len(presentation) > 64:
        return False
    if len(presentation.error_history) > 5:
        return False
    layout_count = defaultdict(int)

    for slide in presentation.slides:
        layout_count[slide.slide_layout_name] += 1
    if sum(layout_count.values()) / len(layout_count) < 2:
        return False

    return True


def prepare_pdf(filename: str, output_dir: str):
    try:
        with open(filename, "rb") as f:
            num_pages = len(PyPDF2.PdfReader(f).pages)
    except:
        os.remove(filename)
        return
    if num_pages < 3 or num_pages > 30:
        os.remove(filename)
        return
    parse_pdf(filename, output_dir, "http://192.168.14.17:11223/convert")
    os.remove(filename)


markdown_clean_pattern = re.compile(r"!\[.*?\]\((.*?)\)")


def prepare_pdf_folder(pdf_folder: str):
    text_content = open(glob.glob(pdf_folder + "/*.md")[0]).read()

    if len(text_content) < 2048 or len(text_content) > 20480:
        shutil.rmtree(pdf_folder)
        return
    if not pexists(pjoin(pdf_folder, "image_caption.json")):
        images_embeddings = image_embedding(pdf_folder)
        images = [pjoin(pdf_folder, image) for image in images_embeddings]
        if len(images_embeddings) < 3:
            shutil.rmtree(pdf_folder)
            return
        similarity_matrix = images_cosine_similarity(list(images_embeddings.values()))
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > 0.85:
                    if pexists(images[i]):
                        os.remove(images[i])
                    break
        images = [image for image in images if pexists(image)]
        image_stats = {}
        caption_prompt = open("prompts/image_label/caption.txt").read()
        for image in images:
            image_stats[image] = llms.caption_model(caption_prompt, image)
        json.dump(
            image_stats,
            open(pjoin(pdf_folder, "image_caption.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )
    if not pexists(pjoin(pdf_folder, "refined_doc.json")):
        text_content = markdown_clean_pattern.sub("", text_content)
        doc_json = llms.get_refined_doc(text_content)
        json.dump(
            doc_json,
            open(pjoin(pdf_folder, "refined_doc.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )


def prepare_ppt(filename: str, output_dir: str):
    app_config.set_rundir(output_dir)
    try:
        presentation = Presentation.from_file(filename)
    except:
        os.remove(filename)
        app_config.remove_rundir()
        return
    if len(os.listdir(app_config.IMAGE_DIR)) // len(
        presentation.slides
    ) > 2 or not ppt_validate(presentation):
        os.remove(filename)
        app_config.remove_rundir()
        return

    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)

    duplicates = prs_dedup(presentation, ppt_image_folder)
    if len(duplicates) > len(presentation) * 0.3:
        os.remove(filename)
        app_config.remove_rundir()
        return

    for slide in duplicates:
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"))
    for err_idx, _ in presentation.error_history:
        os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
    assert len(presentation) == len(
        [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
    )
    for i, slide in enumerate(presentation.slides, 1):
        slide.slide_idx = i
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
        )

    presentation.save(pjoin(app_config.RUN_DIR, "source_standard.pptx"))
    normed_prs = presentation.normalize()
    normed_prs.save(pjoin(app_config.RUN_DIR, "template.pptx"), layout_only=True)
    ppt_to_images(
        pjoin(app_config.RUN_DIR, "template.pptx"),
        pjoin(app_config.RUN_DIR, "template_images"),
    )
    os.remove(pjoin(app_config.RUN_DIR, "template.pptx"))
    os.rename(filename, pjoin(app_config.RUN_DIR, "original.pptx"))


def download_data():
    subtopics = [item for sublist in topics.values() for item in sublist]
    get_file_links("data/crawl_links.jsonl", subtopics, 200)
    asyncio.run(download_files("data/crawl_links.jsonl"))
    print("PDF and PPTX files download finished")


def preprocess(file_type: str, limit: int = 20):
    num_files = len(glob.glob(f"data/subtopics/*/{file_type}/*"))
    progress_bar = tqdm(total=num_files, desc=f"Preprocessing {file_type} files")
    for topic, subtopics in topics.items():
        topic_dir = pjoin("data/topic", filename_normalize(topic), file_type)
        os.makedirs(topic_dir, exist_ok=True)
        for subtopic in subtopics:
            for root, dirs, files in os.walk(
                pjoin("data/subtopics", filename_normalize(subtopic))
            ):
                for file in files:
                    if not file_type in root.split("/"):
                        continue
                    progress_bar.update(1)
                    if len(os.listdir(topic_dir)) > limit:
                        continue
                    try:
                        if file_type == "pptx":
                            prepare_ppt(
                                pjoin(root, file),
                                pjoin(
                                    topic_dir,
                                    filename_normalize(file.rsplit(".", 1)[0]),
                                ),
                            )
                        elif file_type == "pdf":
                            prepare_pdf(
                                pjoin(root, file),
                                topic_dir,
                            )
                    except Exception as e:
                        print(f"preprocess {file} failed: {e}")
                        exit(-1)
        if len(os.listdir(topic_dir)) < 20:
            print(
                f"topic {topic} has only {len(os.listdir(topic_dir))} {file_type} files"
            )


def process_filetype(file_type: str, func: callable, thread_num: int = 10):
    folders = glob.glob(f"data/topic/*/{file_type}/*")
    progress_bar = tqdm(total=len(folders), desc=f"processing {file_type}")
    
    def process_folder(folder):
        try:
            func(folder)
        except Exception as e:
            print(f"process {file_type} folder {folder} failed: {e}")
            traceback.print_exc()
        finally:
            progress_bar.update(1)

    with ThreadPoolExecutor(thread_num) as executor:
        executor.map(process_folder, folders)

    progress_bar.close()


def data_stat(check_integrity: bool = False):
    for topic in glob.glob("data/topic/*"):
        for file_type in os.listdir(topic):
            if file_type not in ["pptx", "pdf"]:
                continue
            num_files = len(os.listdir(pjoin(topic, file_type)))
            print(f"{topic.split('/')[-1]}: {num_files} {file_type} files")
            if not check_integrity:
                continue
            if file_type == "pdf":
                for folder in glob.glob(pjoin(topic, file_type, "*")):
                    if not pexists(pjoin(folder, "image_caption.json")):
                        print(f"{folder} has no image_caption.json")
                    if not pexists(pjoin(folder, "refined_doc.json")):
                        print(f"{folder} has no refined_doc.json")
            if file_type == "pptx":
                for folder in glob.glob(pjoin(topic, file_type, "*")):
                    if not pexists(pjoin(folder, "source_standard.pptx")):
                        print(f"{folder} has no source_standard.pptx")


def postprocess_pdf():
    process_filetype("pdf", prepare_pdf_folder)


if __name__ == "__main__":
    app_config.DEBUG = True
    func_argparse.main(download_data, preprocess, postprocess_pdf, data_stat)
