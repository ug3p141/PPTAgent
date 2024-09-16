import asyncio
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
    app_config,
    filename_normalize,
    parse_pdf,
    pexists,
    pjoin,
    ppt_to_images,
    print,
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
    # "Technological Advancements and Their Applications": [
    #     "Advances in Biotechnology and Their Applications",  # 生物技术的进展及其应用
    #     "The History of Quantum Computing",  # 量子计算的历史
    #     "The History and Future of 5G Technology",  # 5G技术的历史与未来
    #     "The Future of Space Exploration",  # 太空探索的未来
    #     "Blockchain Technology and Its Uses",  # 区块链技术及其应用
    #     "The Role of Virtual Reality in Education",  # 虚拟现实在教育中的作用
    # ],
    "Cybersecurity and Digital Transformation": [
        "Cybersecurity Threats and Solutions",  # 网络安全威胁与解决方案
        "The Importance of Cyber Hygiene",  # 网络卫生的重要性
        "Digital Transformation in Healthcare",  # 医疗保健中的数字化转型
        "The Importance of Cybersecurity in Business",  # 网络安全在商业中的重要性
        "The Role of Cybersecurity in Society",  # 网络安全在社会中的作用
        "The Impact of Cybersecurity on Business",  # 网络安全对商业的影响
    ],
    # "Globalization and Economic Development": [
    #     "Globalization and Its Impact on Developing Countries",  # 全球化及其对发展中国家的影响
    #     "The Role of Government in Economic Development",  # 政府在经济发展中的作用
    #     "The Impact of Globalization on Culture",  # 全球化对文化的影响
    #     "The Role of Innovation in Economic Growth",  # 创新在经济增长中的作用
    #     "The Impact of Globalization on Trade",  # 全球化对贸易的影响
    # ],
    # "Sustainability and Urban Planning": [
    #     "Modern Challenges in Urban Planning",  # 现代城市规划的挑战
    #     "Smart Cities: Opportunities and Challenges",  # 智慧城市：机遇与挑战
    #     "Green Architecture and Sustainable Building",  # 绿色建筑与可持续建筑
    #     "Sustainable Agriculture Practices",  # 可持续农业实践
    #     "Sustainable Tourism Practices",  # 可持续旅游实践
    #     "Water Conservation Strategies",  # 水资源保护策略
    #     "The Role of Urban Planning in Sustainable Development",  # 城市规划在可持续发展中的作用
    # ],
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
    if len(presentation.slides) < 8 or len(presentation.slides) > 64:
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
    parse_pdf(filename, output_dir, "http://192.168.14.11:11223/convert")
    os.remove(filename)


# clean the image, table and deleted elements in the markdown file
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
    if len(duplicates) > len(presentation.slides) * 0.3:
        os.remove(filename)
        app_config.remove_rundir()
        return

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

    # ? 不应该save normed pre 为source
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


def process_filetype(file_type: str, func: callable):
    folders = glob.glob(f"data/topic/*/{file_type}/*")
    progress_bar = tqdm(total=len(folders), desc=f"processing {file_type}")
    for folder in folders:
        progress_bar.update(1)
        try:
            func(folder)
        except Exception as e:
            print(f"prepare {file_type} folder {folder} failed: {e}")
            traceback.print_exc()


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
