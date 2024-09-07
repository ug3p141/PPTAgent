import asyncio
import glob
import json
import os
import shutil
import traceback
from collections import defaultdict
from copy import deepcopy
from itertools import product

import aiohttp
import jsonlines
import PyPDF2
from googlesearch import search
from tqdm import tqdm

from llms import caption_model, get_refined_doc
from model_utils import image_embedding, images_cosine_similarity, prs_dedup
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

topics = {
    "Artificial Intelligence and its Impact": [
        "The Evolution of Artificial Intelligence",  # 人工智能的演变
        "The Impact of Artificial Intelligence on Employment",  # 人工智能对就业的影响
        "Artificial Intelligence in Healthcare",  # 人工智能在医疗保健中的应用
        "The Ethics of Artificial Intelligence",  # 人工智能的伦理问题
        "Artificial Intelligence in Financial Services",  # 人工智能在金融服务中的应用
        "The Role of Artificial Intelligence in Marketing",  # 人工智能在营销中的作用
    ],
    "Renewable Energy and Environmental Impact": [
        "Renewable Energy Sources and Their Impact",  # 可再生能源及其影响
        "Climate Change: Causes and Effects",  # 气候变化：原因与影响
        "Renewable vs. Non-renewable Energy",  # 可再生能源与不可再生能源
        "The Role of Renewable Energy in Combating Climate Change",  # 可再生能源在应对气候变化中的作用
        "The Future of Renewable Energy",  # 可再生能源的未来
        "The Impact of Climate Change on Agriculture",  # 气候变化对农业的影响
    ],
    "Mental Health and Society": [
        "Mental Health Awareness in the Workplace",  # 职场中的心理健康意识
        "The Impact of Social Media on Mental Health",  # 社交媒体对心理健康的影响
        "The Importance of Mental Health in Education",  # 教育中的心理健康重要性
        "The Impact of Mental Health on Physical Health",  # 心理健康对身体健康的影响
        "The Role of Mental Health in Society",  # 心理健康在社会中的作用
        "The Impact of Mental Health on Academic Performance",  # 心理健康对学术表现的影响
    ],
    "Technological Advancements and Their Applications": [
        "Advances in Biotechnology and Their Applications",  # 生物技术的进展及其应用
        "The History of Quantum Computing",  # 量子计算的历史
        "The History and Future of 5G Technology",  # 5G技术的历史与未来
        "The Future of Space Exploration",  # 太空探索的未来
        "Blockchain Technology and Its Uses",  # 区块链技术及其应用
        "The Role of Virtual Reality in Education",  # 虚拟现实在教育中的作用
    ],
    "Cybersecurity and Digital Transformation": [
        "Cybersecurity Threats and Solutions",  # 网络安全威胁与解决方案
        "The Importance of Cyber Hygiene",  # 网络卫生的重要性
        "Digital Transformation in Healthcare",  # 医疗保健中的数字化转型
        "The Importance of Cybersecurity in Business",  # 网络安全在商业中的重要性
        "The Role of Cybersecurity in Society",  # 网络安全在社会中的作用
        "The Impact of Cybersecurity on Business",  # 网络安全对商业的影响
    ],
    "Globalization and Economic Development": [
        "Globalization and Its Impact on Developing Countries",  # 全球化及其对发展中国家的影响
        "The Role of Government in Economic Development",  # 政府在经济发展中的作用
        "The Impact of Globalization on Culture",  # 全球化对文化的影响
        "The Role of Innovation in Economic Growth",  # 创新在经济增长中的作用
        "The Impact of Globalization on Trade",  # 全球化对贸易的影响
    ],
    "Sustainability and Urban Planning": [
        "Modern Challenges in Urban Planning",  # 现代城市规划的挑战
        "Smart Cities: Opportunities and Challenges",  # 智慧城市：机遇与挑战
        "Green Architecture and Sustainable Building",  # 绿色建筑与可持续建筑
        "Sustainable Agriculture Practices",  # 可持续农业实践
        "Sustainable Tourism Practices",  # 可持续旅游实践
        "Water Conservation Strategies",  # 水资源保护策略
        "The Role of Urban Planning in Sustainable Development",  # 城市规划在可持续发展中的作用
    ],
    "E-commerce and Digital Economy": [
        "The Rise of E-commerce",  # 电子商务的崛起
        "The Rise of the Gig Economy",  # 零工经济的崛起
        "The Future of Work in the Age of Automation",  # 自动化时代的工作未来
        "The Future of Work: Remote and Hybrid Models",  # 工作的未来：远程与混合模式
        "The Impact of E-commerce on Traditional Retail",  # 电子商务对传统零售的影响
        "The Role of E-commerce in Global Trade",  # 电子商务在全球贸易中的作用
        "The Impact of E-commerce on Consumer Behavior",  # 电子商务对消费者行为的影响
    ],
    "Social Media and Cultural Influence": [
        "The Role of Social Media in Modern Marketing",  # 社交媒体在现代营销中的角色
        "The Influence of Pop Culture on Society",  # 流行文化对社会的影响
        "The Impact of Social Media on Mental Health",  # 社交媒体对心理健康的影响
        "The Role of Social Media in Social Change",  # 社交媒体在社会变革中的作用
        "The Impact of Social Media on Privacy",  # 社交媒体对隐私的影响
        "The Role of Social Media in Education",  # 社交媒体在教育中的作用
        "The Impact of Social Media on Academic Performance",  # 社交媒体对学术表现的影响
        "The Role of Social Media in Politics",  # 社交媒体在政治中的作用
        "The Impact of Social Media on Business",  # 社交媒体对商业的影响
    ],
    "Ethics, Leadership, and Society": [
        "Ethical Issues in Genetic Engineering",  # 基因工程中的伦理问题
        "The Role of Women in Leadership",  # 女性在领导中的角色
        "The Importance of Biodiversity",  # 生物多样性的重要性
        "The Role of Ethics in Business",  # 伦理在商业中的作用
        "The Impact of Ethics on Society",  # 伦理对社会的影响
        "The Role of Ethics in Politics",  # 伦理在政治中的作用
        "The Impact of Ethics on Academic Performance",  # 伦理对学术表现的影响
        "The Role of Ethics in Education",  # 伦理在教育中的作用
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
        # TODO  fix pptx download error
        if not filepath.endswith(".pptx"):
            return
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
            existing_tasks = list(reader)
        downloaded_tasks = set(
            [(task["topic"], task["filetype"]) for task in existing_tasks]
        )
        iter_tp_fp = [i for i in iter_tp_fp if i not in downloaded_tasks]
    else:
        existing_tasks = []

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
    if len(presentation.slides) < 12 or len(presentation.slides) > 64:
        return False
    if len(presentation.error_history) > 0:
        return False
    layout_count = defaultdict(int)

    for slide in presentation.slides:
        layout_count[slide.slide_layout_name] += 1
    if sum(layout_count.values()) / len(layout_count) < 2:
        return False

    return True


# 模板的主题要和ppt的主题接近


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
    parse_pdf(filename, output_dir, "http://192.168.14.11:11223/convert")
    os.remove(filename)


def prepare_pdf_folder(pdf_folder: str):
    text_content = open(glob.glob(pdf_folder + "/*.md")[0]).read()
    if len(text_content) < 2048 or len(text_content) > 20480:
        shutil.rmtree(pdf_folder)
        return
    # 获得的图片也应该进行去重
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
                        images.pop(images[i])
                    break
        image_stats = {}
        caption_prompt = open("prompts/image_label/caption.txt").read()
        for image in images:
            image_stats[image] = caption_model(caption_prompt, image)
        json.dump(
            image_stats,
            open(pjoin(pdf_folder, "image_caption.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )
    # elif not pexists(pjoin(pdf_folder, "refined_doc.json")):
    #     doc_json = get_refined_doc(text_content)
    #     json.dump(
    #         doc_json,
    #         open(pjoin(pdf_folder, "refined_doc.json"), "w"),
    #         indent=4,
    #         ensure_ascii=False,
    #     )


# image embedding similarity
def prepare_ppt(filename: str, output_dir: str):
    app_config.set_rundir(output_dir)
    try:
        presentation = Presentation.from_file(filename)
    except:
        os.remove(filename)
        app_config.remove_rundir()
        return
    if not ppt_validate(presentation):
        os.remove(filename)
        app_config.remove_rundir()
        return

    ppt_image_folder = pjoin(app_config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)

    duplicates = prs_dedup(presentation, ppt_image_folder)
    if len(duplicates) > len(presentation.slides) * 0.3:
        os.remove(filename)
        app_config.remove_rundir()
        return

    for slide in duplicates:
        os.remove(pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"))
    assert len(presentation.slides) == len(
        [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
    )
    for i, slide in enumerate(presentation.slides, 1):
        slide.slide_idx = i
        os.rename(
            pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
            pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
        )

    normed_prs = presentation.normalize()
    normed_prs.save(pjoin(app_config.RUN_DIR, "source.pptx"))
    normed_prs.save(pjoin(app_config.RUN_DIR, "template.pptx"), layout_only=True)
    ppt_to_images(
        pjoin(app_config.RUN_DIR, "template.pptx"),
        pjoin(app_config.RUN_DIR, "template_images"),
    )
    ImageLabler(presentation).caption_images()
    os.rename(filename, pjoin(app_config.RUN_DIR, "original.pptx"))
    # presentation = Presentation.from_file(pjoin(app_config.RUN_DIR, "source.pptx"))
    # functional_keys, slide_clusters = TemplateInducter(
    #     presentation, ppt_image_folder, pjoin(app_config.RUN_DIR, "template_images")
    # ).work()
    # if sum([len(cluster) for cluster in slide_clusters]) / len(presentation.slides) < 2:
    #     os.remove(filename)
    #     app_config.remove_rundir()
    #     return
    # os.remove(filename)


if __name__ == "__main__":
    app_config.DEBUG = True
    subtopics = [item for sublist in topics.values() for item in sublist]
    # get_file_links("data/crawl_links.jsonl", subtopics, 200)
    # asyncio.run(download_files("data/crawl_links.jsonl"))
    print("PDF and PPTX files download finished")
    num_files = sum(len(files) for _, _, files in os.walk("data/subtopics")) - 2
    progress_bar = tqdm(total=num_files, desc="Preprocessing pptx and pdf")
    for topic, subtopics in topics.items():
        for subtopic in subtopics:
            for root, dirs, files in os.walk(
                pjoin("data/subtopics", filename_normalize(subtopic))
            ):
                for file in files:
                    progress_bar.update(1)
                    try:
                        # if "pdf" in root.split("/"):
                        # prepare_pdf(
                        #     pjoin(root, file),
                        #     pjoin("data/topic", filename_normalize(topic), "pdf"),
                        # )
                        if "pptx" in root.split("/"):
                            prepare_ppt(
                                pjoin(root, file),
                                pjoin(
                                    "data/topic",
                                    filename_normalize(topic),
                                    "ppt",
                                    filename_normalize(file.rsplit(".", 1)[0]),
                                ),
                            )
                    except Exception as e:
                        print(f"File {file} encountered error: {e}")
                        print(traceback.format_exc())

    # pdf_folders = [
    #     pjoin(topic, pdf)
    #     for topic in glob.glob("data/topic/*/pdf")
    #     for pdf in os.listdir(topic)
    # ]
    # progress_bar = tqdm(total=len(pdf_folders), desc="Postprocessing pdfs")
    # for pdf_folder in pdf_folders:
    #     if pexists(pjoin(pdf_folder, "image_caption.json")):
    #         continue
    #     prepare_pdf_folder(pdf_folder)
    #     progress_bar.update(1)
