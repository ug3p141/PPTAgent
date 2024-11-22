import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tempfile import TemporaryDirectory
from time import sleep

import func_argparse
import pytorch_fid.fid_score as fid
import torch
from jinja2 import Template
from tqdm.auto import tqdm

import llms
from crawler import topics
from experiments import get_setting
from pptgen import get_slide_content
from presentation import Picture, Presentation
from utils import Config, older_than, pexists, pjoin, ppt_to_images

fid.tqdm = lambda x: x


def eval_general(presentations: list[Presentation]):
    # return (num_slides, num_chars, num_pictures)
    evals = defaultdict(list)
    for prs in presentations:
        evals["pages"].append(len(prs))
        evals["characters"].append(
            sum([len(slide.to_text(show_image=False)) for slide in prs.slides])
        )
        evals["figures"].append(
            sum([len(list(slide.shape_filter(Picture))) for slide in prs.slides])
        )
    return evals


def eval_fid(source_folders: list[str], setting: str, rank_id: int):
    device = f"cuda:{rank_id % torch.cuda.device_count()}"
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[64]]).to(device)
    fid_scores = []
    for ppt_folder in source_folders:
        source_folder = pjoin(ppt_folder, "slide_images")
        m1, s1 = fid.compute_statistics_of_path(source_folder, model, 128, 64, device)
        for result_folder in glob(pjoin(ppt_folder, f"final_images/{setting}/*")):
            if len(os.listdir(result_folder)) < 5:  # skip if less than 5 images
                continue
            m2, s2 = fid.compute_statistics_of_path(
                result_folder, model, 128, 64, device
            )
            fid_scores.append(fid.calculate_frechet_distance(m1, s1, m2, s2))
    return fid_scores


def eval_ppt(
    presentations: list[Presentation],
    slide_images_folders: list[str],
):
    evals = defaultdict(dict)

    # vision_scorer = open("prompts/ppteval_style.txt", "r").read()
    text_scorer = open("prompts/ppteval_content.txt", "r").read()
    for slide_image_folder in slide_images_folders:
        for slide_image in glob(pjoin(slide_image_folder, "*")):
            # evals["vision"][slide_image] = llms.vision_model(
            #     vision_scorer, slide_image, return_json=True
            # )
            evals["content"][slide_image] = llms.vision_model(
                text_scorer, slide_image, return_json=True
            )

    # 可以尝试改成使用多轮来得到一个最终分数
    # logic_scorer = Template(open("prompts/ppteval_logic.txt", "r").read())
    # for presentation in presentations:
    #     evals["logic"][presentation.source_file] = llms.language_model(
    #         logic_scorer.render(presentation=presentation.to_text()),
    #         return_json=True,
    #     )

    return evals


# ppt eval
def eval_experiment(
    agent_class: str,
    setting_id: int,
    thread_num: int = 10,
    general_eval: bool = False,
    ppt_eval: bool = False,
    fid_eval: bool = False,
):
    setting = get_setting(agent_class, setting_id)
    eval_stats = defaultdict(list)
    eval_file = pjoin("data", "eval", setting + ".json")
    if pexists(eval_file):
        eval_stats |= json.load(open(eval_file))
    source_folders = glob("data/*/pptx/*")
    source_splits = [source_folders[i::thread_num] for i in range(thread_num)]
    result_folders = glob("data/*/pptx/*/*")
    success_folders = [i for i in result_folders if pexists(pjoin(i, "final.pptx"))]
    config = Config("/tmp")
    presentations = [
        Presentation.from_file(pjoin(i, "final.pptx"), config) for i in success_folders
    ]

    if general_eval and "pages" not in eval_stats:
        eval_stats |= eval_general(presentations)

    if ppt_eval and "ppteval" not in eval_stats:
        slide_image_folders = glob(f"data/*/pptx/*/final_images/{setting}/*")
        slides_reference = [[] for _ in range(len(success_folders))]

        doc_jsons = {}
        for pdf_folder in glob("data/*/pdf/*"):
            doc_jsons[pdf_folder.split("/")[-1]] = json.load(
                open(pjoin(pdf_folder, "refined_doc.json"), "r")
            )
        for prs_idx, result_folder in enumerate(success_folders):
            doc_json = doc_jsons[result_folder.split("/")[-1]]
            outline = json.load(
                open(pjoin(result_folder, "presentation_outline.json"), "r")
            )
            for slide_title, slide in outline.items():
                slide_content = get_slide_content(doc_json, slide_title, slide)
                slides_reference[prs_idx].append(slide_content)

        eval_stats |= eval_ppt(presentations, slides_reference, slide_image_folders)

    with ThreadPoolExecutor(thread_num) as executor:
        if fid_eval and "fid" not in eval_stats:
            for fid_scores in executor.map(
                eval_fid, source_splits, [setting] * thread_num, range(thread_num)
            ):
                eval_stats["fid"].extend(fid_scores)
    for k, v in eval_stats.items():
        eval_stats[k] = sum(v) / len(v)
    print(eval_stats)
    json.dump(eval_stats, open(eval_file, "w"), indent=4)


def dataset_stat():
    pdf_stat = {}
    ppt_stat = {}
    tempdir = TemporaryDirectory()
    config = Config()
    config.set_rundir(tempdir.name)
    for topic in topics:
        markdown_contents = {
            f: len(open(f, "r").read()) for f in glob(f"data/{topic}/pdf/*/*.md")
        }
        pdf_stat |= markdown_contents
        avg_pdf_text_len = sum(markdown_contents.values()) / len(markdown_contents)
        num_images = 0
        for pdf_folder in glob(f"data/{topic}/pdf/*"):
            images = json.load(open(pjoin(pdf_folder, "image_caption.json")))
            num_images += len(images)
        avg_pdf_images = num_images / len(markdown_contents)
        ppt_text_len = 0
        ppt_pages = 0
        ppt_images = 0
        num_ppts = 10
        for ppt_folder in glob(f"data/{topic}/pptx/*"):
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "original.pptx"), config
            )
            ppt_stat[ppt_folder] = sum(
                [len(slide.to_text()) for slide in presentation.slides]
            )

            ppt_text_len += ppt_stat[ppt_folder]
            ppt_pages += len(presentation)
            ppt_images += len(os.listdir(pjoin(ppt_folder, "images")))

        avg_ppt_pages = ppt_pages / num_ppts
        avg_ppt_text_len = ppt_text_len / num_ppts
        avg_ppt_images = ppt_images / num_ppts
        print(
            f"{topic}: {avg_pdf_text_len:.2f}, {avg_pdf_images:.2f}, {avg_ppt_pages:.2f}, {avg_ppt_images:.2f}, {avg_ppt_text_len:.2f}"
        )

    json.dump({"pdf": pdf_stat, "ppt": ppt_stat}, open("data/stat.json", "w"), indent=4)


def pptx2images():
    while True:
        print("keep scanning for new pptx")
        for pptx in glob("data/*/pptx/*/*/*/final.pptx"):
            older_than(pptx)
            setting = pptx.split("/")[-3]
            pdf = pptx.split("/")[-2]
            ppt_folder = "/".join(pptx.split("/")[:-3])
            dst = pjoin(ppt_folder, "final_images", setting, pdf)
            if pexists(dst):
                continue
            try:
                ppt_to_images(pptx, dst)
            except:
                print("pptx to images failed")
        sleep(60)


if __name__ == "__main__":
    func_argparse.main(
        dataset_stat,
        eval_experiment,
        pptx2images,
    )
