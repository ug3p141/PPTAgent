import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from tempfile import TemporaryDirectory
from time import sleep

import func_argparse
import pytorch_fid.fid_score as fid
import torch
from jinja2 import Template
from rich import print

import llms
from crawler import topics
from experiments import get_setting
from presentation import Picture, Presentation
from utils import Config, older_than, pexists, pjoin, ppt_to_images

fid.tqdm = lambda x: x


def eval_general(presentations: list[Presentation], evals: dict[str, list[int]]):
    for prs in presentations:
        if prs.source_file not in evals:
            pages = len(prs)
            character = sum(
                [len(slide.to_text(show_image=False)) for slide in prs.slides]
            )
            figures = sum(
                [len(list(slide.shape_filter(Picture))) for slide in prs.slides]
            )
            evals[prs.source_file] = [pages, character, figures]


def eval_fid(source_folders: list[str], setting: str, rank_id: int):
    device = f"cuda:{rank_id % torch.cuda.device_count()}"
    model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[64]]).to(device)
    fid_scores = defaultdict(list)
    for ppt_folder in source_folders:
        source_folder = pjoin(ppt_folder, "source_slides")
        m1, s1 = fid.compute_statistics_of_path(source_folder, model, 128, 64, device)
        for result_folder in glob(pjoin(ppt_folder, f"final_images/{setting}/*")):
            if len(os.listdir(result_folder)) < 5:  # skip if less than 5 images
                continue
            m2, s2 = fid.compute_statistics_of_path(
                result_folder, model, 128, 64, device
            )
            fid_scores[ppt_folder].append(
                fid.calculate_frechet_distance(m1, s1, m2, s2)
            )
    return fid_scores


def eval_ppt(
    presentations: list[Presentation],
    slide_images_folders: list[str],
    evals: dict[str, dict[str, list[dict]]],
):
    vision_scorer = open("prompts/ppteval_style.txt", "r").read()
    text_scorer = open("prompts/ppteval_content.txt", "r").read()
    for slide_image_folder in slide_images_folders:
        for slide_image in glob(pjoin(slide_image_folder, "slide_*.jpg")):
            if slide_image not in evals["vision"]:
                evals["vision"][slide_image] = llms.vision_model(
                    vision_scorer, slide_image, return_json=True
                )
            if slide_image not in evals["content"]:
                evals["content"][slide_image] = llms.vision_model(
                    text_scorer, slide_image, return_json=True
                )

    # 可以尝试改成使用多轮来得到一个最终分数
    logic_scorer = Template(open("prompts/ppteval_logic.txt", "r").read())
    for presentation in presentations:
        if presentation.source_file not in evals["logic"]:
            evals["logic"][presentation.source_file] = llms.language_model(
                logic_scorer.render(presentation=presentation.to_text()),
                return_json=True,
            )

    return evals


# ppt eval
def eval_experiment(
    agent_class: str,
    setting_id: int,
    setting_name: str,
    thread_num: int = 8,
    general_eval: bool = True,
    ppt_eval: bool = True,
    fid_eval: bool = True,
):
    s = get_setting(agent_class, setting_id)
    setting = setting_name or s
    eval_file = pjoin("data", "eval", setting + ".json")
    if pexists(eval_file):
        eval_stats = json.load(open(eval_file))
    else:
        eval_stats = defaultdict(dict)
    source_folders = glob(f"data/*/pptx/*/*")
    result_folders = glob(f"data/*/pptx/*/{setting}/*")
    success_folders = [i for i in result_folders if pexists(pjoin(i, "final.pptx"))]
    config = Config("/tmp")
    presentations = [
        Presentation.from_file(pjoin(i, "final.pptx"), config) for i in success_folders
    ]

    if general_eval:
        eval_general(presentations, eval_stats["general"])

    if ppt_eval:
        if "ppteval" not in eval_stats:
            eval_stats["ppteval"] = defaultdict(dict)
        slide_image_folders = glob(f"data/*/pptx/*/final_images/{setting}/*")
        eval_ppt(presentations, slide_image_folders, eval_stats["ppteval"])

    with ThreadPoolExecutor(thread_num) as executor:
        if fid_eval:
            fid_folders = [i for i in source_folders if i not in eval_stats["fid"]]
            fid_splits = [
                fid_folders[i : i + len(fid_folders) // thread_num]
                for i in range(0, len(fid_folders), len(fid_folders) // thread_num)
            ]
            for fid_scores in executor.map(
                eval_fid,
                fid_splits,
                [setting] * thread_num,
                range(thread_num),
            ):
                eval_stats["fid"] |= fid_scores
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


def pptx2images(settings: str = "*"):
    while True:
        for folder in glob(f"data/*/pptx/*/{settings}/*/history"):
            folder = os.path.dirname(folder)
            pptx = pjoin(folder, "final.pptx")
            ppt_folder, setting, pdf = folder.rsplit("/", 2)
            dst = pjoin(ppt_folder, "final_images", setting, pdf)

            if not pexists(pptx):
                if pexists(dst):
                    print(f"remove {dst}")
                    shutil.rmtree(dst)
                continue

            older_than(pptx)
            if pexists(dst):
                continue
            try:
                ppt_to_images(pptx, dst)
            except:
                print("pptx to images failed")
        sleep(60)
        print("keep scanning for new pptx")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        func_argparse.main(
            dataset_stat,
            eval_experiment,
            pptx2images,
        )
    else:
        # setting dimension file: eval
        llms.language_model = llms.gpt4o
        llms.vision_model = llms.gpt4o

        config = Config("/tmp")
        eval_file = "human_eval/eval.json"
        if os.path.exists(eval_file):
            evals = json.load(open(eval_file))
        else:
            evals = {}
        for setting in ["baseline-gpt", "pptagent-real"]:  # os.listdir("human_eval"):
            if setting not in evals:
                evals[setting] = defaultdict(lambda: defaultdict(list))
            test_folders = sorted(glob(f"human_eval/{setting}/*"))[:3]
            presentations = [
                Presentation.from_file(pjoin(i, "final.pptx"), config)
                for i in test_folders
            ]
            eval_ppt(presentations, test_folders, evals[setting])
            print(setting, ":")
            for dimension, evals in evals[setting].items():
                scores = [i["score"] for i in evals.values() if isinstance(i, dict)]
                print(dimension, ":", sum(scores) / len(scores))
            json.dump(evals, open(eval_file, "w"), indent=4, ensure_ascii=False)
