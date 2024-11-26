import glob
import json
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from time import sleep, time

import torch
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm

import llms
from induct import SlideInducter
from model_utils import (
    get_image_model,
    get_refined_doc,
    image_embedding,
    images_cosine_similarity,
    parse_pdf,
    prs_dedup,
)
from multimodal import ImageLabler
from presentation import Presentation
from utils import Config, older_than, pexists, pjoin, ppt_to_images

markdown_clean_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
device_count = torch.cuda.device_count()


def rm_folder(folder: str):
    try:
        shutil.rmtree(folder)
    except:
        for i in os.listdir(folder):
            try:
                rm_folder(pjoin(folder, i))
            except:
                pass


def process_filetype(file_type: str, func: callable, thread_num: int):
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
        list(executor.map(process_folder, folders, range(len(folders))))

    progress_bar.close()


def parse_pdfs(pdf_folders: list[str], idx: int):
    # require numpy==1.26.0, which is conflict with other packages
    from marker.models import load_all_models

    model = load_all_models(device=idx % device_count, dtype=torch.float16)
    for pdf_folder in pdf_folders:
        if not older_than(pdf_folder + "/original.pdf"):
            continue
        if not pexists(pjoin(pdf_folder, "source.md")):
            text_content = parse_pdf(
                pdf_folder + "/original.pdf",
                pdf_folder,
                model,
            )
            if len(text_content) < 512 or len(text_content) > 32768:
                rm_folder(pdf_folder)
                continue


def prepare_pdf_folder(pdf_folder: str, rank: int):
    image_model = get_image_model(f"cuda:{rank % device_count}")
    if not pexists(pjoin(pdf_folder, "source.md")):
        return
    if not pexists(pjoin(pdf_folder, "image_caption.json")):
        images_embeddings = image_embedding(pdf_folder, *image_model)
        images = [pjoin(pdf_folder, image) for image in images_embeddings]
        if len(images_embeddings) == 0:
            rm_folder(pdf_folder)
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
        caption_prompt = open("prompts/caption.txt").read()
        for image in images:
            image_stats[image] = llms.vision_model(caption_prompt, image)
            print(image_stats[image])
        with open(pjoin(pdf_folder, "image_caption.json"), mode="w") as f:
            json.dump(image_stats, f, indent=4, ensure_ascii=False)

    if not pexists(pjoin(pdf_folder, "refined_doc.json")):
        text_content = open(pjoin(pdf_folder, "source.md")).read()
        text_content = markdown_clean_pattern.sub("", text_content)
        doc_json = get_refined_doc(text_content)
        json.dump(
            doc_json,
            open(pjoin(pdf_folder, "refined_doc.json"), "w"),
            indent=4,
            ensure_ascii=False,
        )


def prepare_ppt_folder(ppt_folder: str, text_model):
    if os.path.exists(ppt_folder + "/image_stats.json") or not older_than(
        ppt_folder + "/original.pptx"
    ):
        return
    config = Config(rundir=ppt_folder, debug=False)
    presentation = Presentation.from_file(ppt_folder + "/original.pptx", config=config)
    ppt_image_folder = pjoin(config.RUN_DIR, "slide_images")
    ppt_to_images(presentation.source_file, ppt_image_folder)
    duplicates = prs_dedup(presentation, text_model)
    if len(duplicates) > len(presentation) / 2:
        rm_folder(ppt_folder)
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
    ImageLabler(presentation, config).caption_images()


def prepare_induction(induct_id: int):
    induct_llms = [
        (llms.qwen2_5, llms.qwen_vl),
        (llms.qwen2_5, llms.gpt4o),
        (llms.gpt4o, llms.qwen_vl),
        (llms.gpt4o, llms.gpt4o),
    ]

    def do_induct(llm: list[llms.LLM], ppt_folder: str, rank: int):
        llms.language_model = llm[0]
        llms.vision_model = llm[1]
        config = Config(rundir=ppt_folder)
        ppt_image_folder = pjoin(ppt_folder, "slide_images")
        template_image_folder = pjoin(ppt_folder, "template_images")
        image_model = get_image_model(f"cuda:{rank % device_count}")
        presentation = Presentation.from_file(
            pjoin(ppt_folder, "source_standard.pptx"), config
        )
        ImageLabler(presentation, config).caption_images()
        slide_inducter = SlideInducter(
            presentation, ppt_image_folder, template_image_folder, config, image_model
        )
        slide_inducter.content_induct()

    for idx, folder in enumerate(
        tqdm(sorted(glob.glob("data/*/pptx/*")), desc="prepare induction")
    ):
        do_induct(induct_llms[induct_id], folder, 0)


if __name__ == "__main__":
    if sys.argv[1] == "prepare_ppt":
        text_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=2)
        for ppt_folder in tqdm(glob.glob("data/*/pptx/*"), desc="prepare ppt"):
            prepare_ppt_folder(ppt_folder, text_model)
    elif sys.argv[1] == "prepare_induction":
        prepare_induction(int(sys.argv[2]))
    elif sys.argv[1] == "parse_pdf":
        multiprocessing.set_start_method("spawn", force=True)
        num_process = int(sys.argv[2])
        with ProcessPoolExecutor(max_workers=num_process) as executor:
            folders = glob.glob("data/*/pdf/*")
            subfolders = [[] for _ in range(num_process)]
            for idx, folder in enumerate(folders):
                subfolders[idx % num_process].append(folder)
            list(executor.map(parse_pdfs, subfolders, range(num_process)))
    elif sys.argv[1] == "prepare_pdf":
        prepare_pdf_folder = partial(prepare_pdf_folder)
        process_filetype("pdf", prepare_pdf_folder, int(sys.argv[2]))
