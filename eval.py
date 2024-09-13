import json
import os
from functools import partial
from glob import glob
from itertools import product
from tempfile import TemporaryDirectory

import func_argparse
import pytorch_fid.fid_score as fid
from tqdm import tqdm

import llms
from agent import PPTAgent
from crawler import process_filetype, topics
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import app_config, pexists, pjoin, ppt_to_images

fid.tqdm = lambda x: x


def walk_data(file_type: str, topic: str):
    folders = glob.glob(f"data/topic/{topic}/{file_type}/*")
    for folder in folders:
        if folder.endswith(".DS_Store"):
            continue
        yield folder


def prepare_template():
    llm_models = [
        llms.qwen,
        llms.internvl_76,
        llms.internvl_40,
        llms.gpt4o,
        llms.gpt4omini,
    ]

    def gen_template(llm: llms.OPENAI, ppt_folder: str):
        llms.long_model = llm
        app_config.set_rundir(pjoin(ppt_folder, llm.model))
        ppt_image_folder = pjoin(ppt_folder, "slide_images")
        if not pexists(ppt_image_folder):
            raise Exception(f"ppt_image_folder not found: {ppt_image_folder}")
        presentation = Presentation.from_file(pjoin(ppt_folder, "source.pptx"))
        template_inducter = TemplateInducter(
            presentation, ppt_image_folder, pjoin(ppt_folder, "template_images")
        )
        if not pexists(template_inducter.slide_split_file):
            template_inducter.category_split()
        llms.caption_model = llms.internvl_multi
        # template_inducter.work()

    for llm in llm_models:
        print(f"Preparing templates using {llm.model}")
        process_filetype("pptx", partial(gen_template, llm))


def prepare_caption():
    def caption(ppt_folder: str):
        app_config.set_rundir(ppt_folder)
        presentation = Presentation.from_file(pjoin(ppt_folder, "source_standard.pptx"))
        labler = ImageLabler(presentation)
        labler.apply_stats(labler.caption_images())

    process_filetype("pptx", caption)


def generate_pres():
    llm_models = [
        llms.gpt4o,
        llms.gpt4o_mini,
        llms.internvl_76,
    ]
    # fix internvl problem
    for llm, topic in product(llm_models, topics):
        llms.agent_model = llm
        llms.caption_model = llm
        for ppt_folder, pdf_folder in tqdm(
            product(walk_data("pptx", topic), walk_data("pdf", topic))
        ):
            app_config.set_run_dir(ppt_folder)
            presentation = Presentation.from_file(pjoin(ppt_folder, "source.pptx"))
            ImageLabler(presentation)
            app_config.set_run_dir(pjoin(ppt_folder, llm.model))
            images = json.load(
                open(pjoin(pdf_folder, "image_caption.json"), "r"),
            )
            doc_json = json.load(
                open(pjoin(pdf_folder, "refined_doc.json"), "r"),
            )
            slide_cluster = json.load(open(pjoin(ppt_folder, "slide_cluster.json")))
            functional_keys, slide_cluster = (
                set(slide_cluster.pop("functional_keys")),
                slide_cluster,
            )
            PPTAgent(
                presentation, slide_cluster, images, 12, doc_json, functional_keys
            ).work()


def eval_fid(
    source_prs_file, generated_prs_file, batch_size=1, device="cuda", dims=2048
):
    with TemporaryDirectory() as temp_dir:
        source_folder = pjoin(temp_dir, "source")
        generated_folder = pjoin(temp_dir, "generated")
        os.makedirs(source_folder)
        os.makedirs(generated_folder)
        ppt_to_images(source_prs_file, source_folder)
        ppt_to_images(generated_prs_file, generated_folder)
        score = fid.calculate_fid_given_paths(
            [source_folder, generated_folder],
            batch_size=batch_size,
            device=device,
            dims=dims,
        )
        return score


def eval_blue():
    pass


def exp3_ablation_html(source_prs_file, generated_prs_file):
    pass


if __name__ == "__main__":
    # ? 4omini 等的split还没做
    app_config.DEBUG = False
    func_argparse.main(prepare_template, prepare_caption, generate_pres)
