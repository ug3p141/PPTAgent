import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from itertools import product
from tempfile import TemporaryDirectory

import func_argparse
import pytorch_fid.fid_score as fid
import torch
from tqdm.auto import tqdm

import llms
from agent import PPTAgent, get_slide_content
from apis import HistoryMark
from crawler import process_filetype, topics
from model_utils import get_text_embedding
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import Config, app_config, filename_normalize, pexists, pjoin, ppt_to_images

fid.tqdm = lambda x: x
eval_models = [
    llms.internvl_76,
    llms.qwen,
    llms.gpt4o,
]


# 每步都要deepcopy了生成一下
# 改batch api
def walk_data(file_type: str, topic: str):
    folders = glob(f"data/topic/{filename_normalize(topic)}/{file_type}/*")
    for folder in folders:
        if folder.endswith(".DS_Store"):
            continue
        yield folder


def prepare_template():
    def gen_template(llm: llms.OPENAI, ppt_folder: str):
        ppt_image_folder = pjoin(ppt_folder, "slide_images")
        with TemporaryDirectory() as temp_dir:
            app_config.set_rundir(temp_dir)
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx")
            )
        template_image_folder = pjoin(ppt_folder, "template_images")
        llms.long_model = llm
        llms.agent_model = llm
        if len(os.listdir(ppt_image_folder)) != len(presentation.slides):
            shutil.rmtree(ppt_image_folder)
            ppt_to_images(presentation.source_file, ppt_image_folder)
        if len(os.listdir(template_image_folder)) != len(presentation.slides):
            shutil.rmtree(template_image_folder)
            presentation.save(pjoin(ppt_folder, "template.pptx"), True)
            ppt_to_images(pjoin(ppt_folder, "template.pptx"), template_image_folder)
        app_config.set_rundir(pjoin(ppt_folder, llm.model))
        if not pexists(ppt_image_folder):
            raise Exception(f"ppt_image_folder not found: {ppt_image_folder}")
        template_inducter = TemplateInducter(
            presentation, ppt_image_folder, template_image_folder
        )
        if not pexists(template_inducter.slide_split_file):
            template_inducter.category_split()
        template_inducter.work()

    for llm in eval_models:
        print(f"Preparing templates using {llm.model}")
        process_filetype("pptx", partial(gen_template, llm))


def prepare_caption():
    def caption(ppt_folder: str):
        app_config.set_rundir(ppt_folder)
        presentation = Presentation.from_file(pjoin(ppt_folder, "source_standard.pptx"))
        labler = ImageLabler(presentation)
        labler.apply_stats(labler.caption_images())

    process_filetype("pptx", caption)


# 以intern vl 和 qwen 为锚点，把表现差的都删掉
def generate_pres(thread_num: int = 4):
    for llm, topic in product(eval_models, topics):
        llms.agent_model = llm
        ppt_folders = list(walk_data("pptx", topic))
        pdf_folders = list(walk_data("pdf", topic))
        progressbar = tqdm(
            ppt_folders,
            desc=f"Generating presentations using {llm.model} on {topic}",
            total=len(ppt_folders) * len(pdf_folders),
        )

        def process_ppt_folder(ppt_folder):
            app_config = Config()
            app_config.set_rundir(ppt_folder)
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx"),
                app_config,
            )
            ImageLabler(presentation, app_config)
            slide_cluster = json.load(
                open(
                    pjoin(
                        ppt_folder, llm.model, "template_induct", "slides_cluster.json"
                    )
                )
            )
            functional_keys, slide_cluster = (
                set(slide_cluster.pop("functional_keys")),
                slide_cluster,
            )
            layout_embeddings = torch.stack(
                get_text_embedding(list(slide_cluster.keys()))
            )
            for pdf_folder in pdf_folders:
                progressbar.update(1)
                app_config.set_rundir(
                    pjoin(ppt_folder, llm.model, os.path.basename(pdf_folder))
                )
                if pexists(
                    pjoin(
                        app_config.RUN_DIR,
                        "agent_steps.json",
                    )
                ):
                    continue
                images = json.load(
                    open(pjoin(pdf_folder, "image_caption.json"), "r"),
                )
                doc_json = json.load(
                    open(pjoin(pdf_folder, "refined_doc.json"), "r"),
                )

                PPTAgent(
                    presentation,
                    app_config,
                    slide_cluster,
                    images,
                    12,
                    doc_json,
                    functional_keys,
                    layout_embeddings,
                ).work()

        with ThreadPoolExecutor(thread_num) as executor:
            list(
                executor.map(process_ppt_folder, ppt_folders),
            )
        progressbar.close()


def folder_score(ppt_folder: str):
    score = 0
    for result_folder in glob(pjoin(ppt_folder, "*/*")):
        if pexists(pjoin(result_folder, "final.pptx")):
            score += 10
    steps = json.load(open(pjoin(ppt_folder, eval_models[0].model, "agent_steps.json")))
    step_sr = sum([step[0] == HistoryMark.API_CALL_CORRECT for step in steps]) / len(
        steps
    )
    return step_sr


def data_filter():
    for topic in topics:
        ppt_folders = glob(pjoin("data/topic", filename_normalize(topic), "pptx"))
        ppt_score = [
            (ppt_folder, folder_score(ppt_folder)) for ppt_folder in ppt_folders
        ]
        ppt_score.sort(key=lambda x: x[1], reverse=True)
        for ppt_folder, _ in ppt_score[:5]:
            print(ppt_folder)


def get_fid(source_folder, generated_folder, batch_size=1, device="cuda", dims=2048):
    score = fid.calculate_fid_given_paths(
        [source_folder, generated_folder],
        batch_size=batch_size,
        device=device,
        dims=dims,
    )
    return score


# success rate[step|all] coverage [source|blueprint] fid
# 跑完一个case后先把这个test跑起来防止却数据
def eval_experiment():
    stat = json.load(open("data/stat.json"))
    ppt_stat, pdf_stat = stat["ppt"], stat["pdf"]
    stat_succ_by_len = []
    for llm in eval_models:
        success_all = 0
        success_step = 0
        all_step = 0
        similarity_source = 0
        similarity_blueprint = 0
        fid = 0

        def calc_metrics(ppt_folder: str):
            source_presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx")
            )
            original_textembeds = {}
            work_dir = pjoin(ppt_folder, llm.model)
            for pdf_input in os.listdir(work_dir):
                if not os.path.isdir(pjoin(work_dir, pdf_input)):
                    continue
                steps, api_history, _ = json.load(
                    open(pjoin(work_dir, pdf_input, "agent_steps.json"))
                )
                all_step += len(api_history)
                success_step += sum(
                    [step[0] == HistoryMark.API_CALL_CORRECT for step in api_history]
                )
                stat_succ_by_len.append(0, ppt_stat[ppt_folder], pdf_stat[pdf_input])
                if pexists(pjoin(work_dir, pdf_input, "final.pptx")):
                    success_all += 1
                    final_presentation = Presentation.from_file(
                        pjoin(ppt_folder, pdf_input, "final.pptx")
                    )
                    ppt_to_images(
                        pjoin(work_dir, pdf_input, "final.pptx"),
                        pjoin(work_dir, pdf_input, "final_images"),
                    )
                    fid += get_fid(
                        pjoin(ppt_folder, pdf_input, "final_images"),
                        pjoin(ppt_folder, "slide_images"),
                    )
                    outline = json.load(
                        open(pjoin(work_dir, pdf_input, "outline.json"))
                    )
                    doc_json = json.load(
                        open(pjoin(ppt_folder, pdf_input, "refined_doc.json"))
                    )
                    for step, (slide_title, slide) in zip(steps, outline):
                        if step[0] not in original_textembeds:
                            original_textembeds[step[0]] = get_text_embedding(
                                source_presentation.slides[step[0]].to_text()
                            )
                        slide_content = get_slide_content(doc_json, slide_title, slide)
                        similarity_source += torch.cosine_similarity(
                            original_textembeds[step[0]],
                            get_text_embedding([slide_content]),
                            -1,
                        ).item()
                        similarity_blueprint += torch.cosine_similarity(
                            original_textembeds[step[0]],
                            get_text_embedding([slide_content]),
                            -1,
                        ).item()

        process_filetype("pptx", calc_metrics)
        print(
            "%s success_all: %d, success_step: %d, all_step: %d, similarity_source: %f, similarity_blueprint: %f, fid: %f"
            % (
                llm.model,
                success_all,
                success_step,
                all_step,
                similarity_source / success_all,
                similarity_blueprint / success_all,
                fid / success_all,
            )
        )


def dataset_stat():
    # save every ppt and pdf's length
    pdf_stat = {}
    ppt_stat = {}
    for topic in topics:
        markdown_contents = {
            f: len(open(f, "r").read())
            for f in glob(f"data/topic/{filename_normalize(topic)}/pdf/*.md")
        }
        pdf_stat |= markdown_contents
        topic_pdf_text_len = sum(markdown_contents.values()) / len(markdown_contents)
        num_images = 0
        for pdf_folder in walk_data("pdf", topic):
            images = json.load(open(pjoin(pdf_folder, "image_caption.json")))
            num_images += len(images)
        topic_avg_images = num_images / len(markdown_contents)
        ppt_text_len = 0
        ppt_pages = 0
        ppt_images = 0
        num_ppts = 0
        for ppt_folder in walk_data("pptx", topic):
            num_ppts += 1
            presentation = Presentation.from_file(pjoin(ppt_folder, "original.pptx"))
            ppt_stat[ppt_folder] = sum(
                [len(slide.to_text()) for slide in presentation.slides]
            )

            ppt_text_len += ppt_stat[ppt_folder]
            ppt_pages += len(presentation.slides)
            ppt_images += len(os.listdir(pjoin(ppt_folder, "images")))

        topic_avg_ppt_pages = ppt_pages / num_ppts
        topic_avg_ppt_text_len = ppt_text_len / num_ppts
        topic_avg_ppt_images = ppt_images / num_ppts
        print(
            f"{topic}: {topic_pdf_text_len}, {topic_avg_images}, {topic_avg_ppt_pages}, {topic_avg_ppt_images}, {topic_avg_ppt_text_len}"
        )
    # save to json
    json.dump({"pdf": pdf_stat, "ppt": ppt_stat}, open("data/stat.json", "w"), indent=4)


if __name__ == "__main__":
    app_config.DEBUG = True
    func_argparse.main(
        prepare_template, prepare_caption, generate_pres, dataset_stat, eval_experiment
    )
