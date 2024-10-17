import glob
import json
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from time import sleep, time

import jsonlines
import torch
from FlagEmbedding import BGEM3FlagModel
from marker.models import load_all_models
from tqdm import tqdm

import llms
from model_utils import (
    get_image_model,
    get_refined_doc,
    image_embedding,
    images_cosine_similarity,
    parse_pdf,
    prs_dedup,
)
from presentation import Presentation
from utils import Config, pexists, pjoin, ppt_to_images

text_models = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=2)
image_models = get_image_model(device=f"cuda:{3}")
marker_models = []
markdown_clean_pattern = re.compile(r"!\[.*?\]\((.*?)\)")


def older_than(filepath, seconds: int = 10):
    if not os.path.exists(filepath):
        return False
    file_creation_time = os.path.getctime(filepath)
    current_time = time()
    return seconds < (current_time - file_creation_time)


def prepare_pdf_folder(pdf_folder: str, idx: int):
    if not len(marker_models) > idx:
        model = load_all_models(
            device=f"cuda:{idx%torch.cuda.device_count()}", dtype=torch.float16
        )
        marker_models.append(model)
    else:
        model = marker_models[idx]
    if not older_than(pdf_folder + "/original.pdf"):
        return
    if not os.path.exists(pjoin(pdf_folder, "source.md")):
        text_content = parse_pdf(
            pdf_folder + "/original.pdf",
            pdf_folder,
            model,
        )
    else:
        text_content = open(pjoin(pdf_folder, "source.md")).read()

    if len(text_content) < 512 or len(text_content) > 32768:
        shutil.rmtree(pdf_folder)
        return
    if not pexists(pjoin(pdf_folder, "image_caption.jsonl")):
        images_embeddings = image_embedding(pdf_folder, *image_models)
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
            try:
                image_stats[image] = llms.caption_model(caption_prompt, image)
            except:
                os.remove(image)
        with jsonlines.open(
            pjoin(pdf_folder, "image_caption.jsonl"), mode="w"
        ) as writer:
            writer.write_all(image_stats)

    if not pexists(pjoin(pdf_folder, "refined_doc.json")):
        text_content = markdown_clean_pattern.sub("", text_content)
        doc_json = get_refined_doc(text_content)
        json.dump(
            doc_json,
            open(pjoin(pdf_folder, "refined_doc.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )


def prepare_ppt_folder(ppt_folder: str):
    if os.path.exists(ppt_folder + "/template_images"):
        return
    if not older_than(ppt_folder + "/original.pptx"):
        return
    config = Config(rundir=ppt_folder, debug=False)
    presentation = Presentation.from_file(ppt_folder + "/original.pptx", config=config)
    ppt_image_folder = pjoin(config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    duplicates = prs_dedup(presentation, text_models)
    if len(duplicates) > len(presentation) / 2:
        shutil.rmtree(ppt_folder)
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

    presentation.save(pjoin(ppt_folder, "source_standard.pptx"))
    normed_prs = presentation.normalize()
    normed_prs.save(pjoin(ppt_folder, "template.pptx"), layout_only=True)
    ppt_to_images(
        pjoin(ppt_folder, "template.pptx"),
        pjoin(ppt_folder, "template_images"),
    )
    os.remove(pjoin(ppt_folder, "template.pptx"))


def process_filetype(file_type: str, func: callable, thread_num: int = 32):
    folders = glob.glob(f"data/*/{file_type}/*")
    progress_bar = tqdm(total=len(folders), desc=f"processing {file_type}")

    def process_folder(folder, *args, **kwargs):
        try:
            func(folder, *args, **kwargs)
        except Exception as e:
            print(f"process {file_type} folder {folder} failed: {e}")
            traceback.print_exc()
        finally:
            progress_bar.update(1)

    with ThreadPoolExecutor(thread_num) as executor:
        list(executor.map(process_folder, folders, range(thread_num)))

    progress_bar.close()


def data_stat(check_integrity: bool = True):
    for topic in glob.glob("data/*/*/*"):
        for file_type in os.listdir(topic):
            if file_type not in ["pptx", "pdf"]:
                continue
            num_files = len(os.listdir(pjoin(topic, file_type)))
            print(f"{topic.split('/')[-1]}: {num_files} {file_type} files")
            if not check_integrity:
                continue
            if file_type == "pdf":
                for folder in glob.glob(pjoin(topic, file_type, "*")):
                    if not pexists(pjoin(folder, "image_caption.jsonl")):
                        print(f"{folder} has no image_caption.jsonl")
                    if not pexists(pjoin(folder, "refined_doc.json")):
                        print(f"{folder} has no refined_doc.json")
            if file_type == "pptx":
                for folder in glob.glob(pjoin(topic, file_type, "*")):
                    if not pexists(pjoin(folder, "source_standard.pptx")):
                        print(f"{folder} has no source_standard.pptx")


if __name__ == "__main__":
    while True:
        if sys.argv[1] == "prepare_ppt":
            for ppt_folder in tqdm(glob.glob("data/*/pptx/*"), desc="prepare ppt"):
                prepare_ppt_folder(ppt_folder)
        elif sys.argv[1] == "prepare_pdf":
            process_filetype("pdf", prepare_pdf_folder, int(sys.argv[2]))
        elif sys.argv[1] == "stat":
            data_stat()
            exit()
        print("process finished, waiting for next task")
