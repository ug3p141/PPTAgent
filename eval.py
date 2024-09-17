from collections import defaultdict
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from itertools import product
from tempfile import TemporaryDirectory
import tempfile
import traceback

import func_argparse
from jinja2 import Template
import pytorch_fid.fid_score as fid
import torch
from tqdm.auto import tqdm

import llms
from agent import PPTAgent, get_slide_content
from crawler import process_filetype, topics
from model_utils import get_text_embedding
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import Config, app_config, filename_normalize, pexists, pjoin, ppt_to_images
from FlagEmbedding import BGEM3FlagModel

fid.tqdm = lambda x: x
eval_models = [
    (llms.qwen, llms.gpt4o),
    (llms.qwen, llms.internvl_76),
    (llms.internvl_76, llms.internvl_76),
    (llms.gpt4o, llms.gpt4o),
]


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

    for llm, layout_llm in eval_models:
        print(f"Preparing templates using {layout_llm.model}")
        process_filetype("pptx", partial(gen_template, layout_llm))


def prepare_caption():
    def caption(ppt_folder: str):
        app_config.set_rundir(ppt_folder)
        presentation = Presentation.from_file(pjoin(ppt_folder, "source_standard.pptx"))
        labler = ImageLabler(presentation)
        labler.apply_stats(labler.caption_images())

    process_filetype("pptx", caption)


def generate_pres(thread_num: int = 4):
    global eval_models
    text_models = [
        BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=i)
        for i in range(min(torch.cuda.device_count(), thread_num))
    ]
    if thread_num == 1:
        eval_models = [eval_models[-1]]
    for (llm, layout_llm), topic in product(eval_models, topics):
        llms.agent_model = llm
        if llm != layout_llm:
            model = f"{llm.model}_{layout_llm.model}"
        else:
            model = llm.model
        model = 'random'
        ppt_folders = list(walk_data("pptx", topic))
        pdf_folders = list(walk_data("pdf", topic))
        print(f"Generating presentations using {model} on {topic}")
        progressbar = tqdm(
            ppt_folders,
            total=len(ppt_folders) * len(pdf_folders),
        )

        def process_ppt_folder(arg):
            thread_id, ppt_folder = arg
            text_model = text_models[thread_id % len(text_models)]
            app_config = Config()
            app_config.set_rundir(ppt_folder)
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx"),
                app_config,
            )
            ImageLabler(presentation, app_config)
            cluster_file = pjoin(
                ppt_folder,
                layout_llm.model,
                "template_induct",
                "slides_cluster.json",
            )
            if not pexists(cluster_file):
                return
            slide_cluster = json.load(open(cluster_file))
            functional_keys, slide_cluster = (
                set(slide_cluster.pop("functional_keys")),
                slide_cluster,
            )
            layout_embeddings = torch.stack(
                get_text_embedding(list(slide_cluster.keys()), 32, text_model)
            )
            for pdf_folder in pdf_folders:
                app_config.set_rundir(
                    pjoin(ppt_folder, model, os.path.basename(pdf_folder))
                )
                if pexists(
                    pjoin(
                        app_config.RUN_DIR,
                        "agent_steps.json",
                    )
                ):
                    print("skip", ppt_folder, pdf_folder, "already generated")
                    progressbar.update(1)
                    continue
                try:
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
                        text_model,
                        doc_json,
                        functional_keys,
                        layout_embeddings,
                    ).work()
                except:
                    traceback.print_exc()
                    print(f"Error in {ppt_folder} {pdf_folder}")

                progressbar.update(1)

        if model.startswith("gpt-4o"):
            for ppt_folder in ppt_folders:
                process_ppt_folder((0, ppt_folder))  # async 不能多线程
                continue
        with ThreadPoolExecutor(thread_num) as executor:
            list(
                executor.map(process_ppt_folder, enumerate(ppt_folders)),
            )
        progressbar.close()

def get_fid():
    model = llms.gpt4o.model
    fid_scores = []
    for ppt_folder in glob(f"data/topic/*/pptx/*"):
        source_folder = pjoin(ppt_folder, "slide_images")
        with tempfile.TemporaryDirectory() as temp_dir:
            for result_folder in glob(pjoin(ppt_folder, f"final_images/{model}/*")):
                pdf = result_folder.split("/")[-1]
                for image in os.listdir(result_folder):
                    shutil.copy(
                        pjoin(result_folder, image),
                        pjoin(temp_dir, f"{pdf}_{image}"),
                    )
            if len(os.listdir(temp_dir)) == 0:
                continue
            fid_scores.append(fid.calculate_fid_given_paths(
                [source_folder, temp_dir], 128, 'cuda:0', 64
            ))
    print(model, sum(fid_scores) / len(fid_scores))

def get_gscore(slide_content, slide_ref):
    prompt = Template(open("prompts/llm_judge.txt").read())
    return int(llms.gpt4o(prompt.render(slide_content=slide_content, slide_ref=slide_ref)))

def eval_experiment():
    stat = json.load(open("data/stat.json"))
    pdf_stat = stat["pdf"]
    pdf_stat = {k.split("/")[-1].rsplit(".", 1)[0]: v for k, v in pdf_stat.items()}
    stat_succ_by_len = defaultdict(list)
    gscore = []
    for llm, layout_llm in eval_models[3:]:
        success_all = 0
        fidelity = 0
        if llm != layout_llm:
            model = f"{llm.model}_{layout_llm.model}"
        else:
            model = llm.model
        for ppt_folder in glob(f"data/topic/*/pptx/*"):
            source_presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx"), app_config
            )
            original_textembeds = {}
            topic = ppt_folder.split("/")[-3]
            work_dir = pjoin(ppt_folder, model)
            for pdf_input in os.listdir(work_dir):
                if pdf_input not in pdf_stat:
                    print('skip', pdf_input)
                    continue
                if pexists(pjoin(work_dir, pdf_input, "presentation_outline.json")):
                    outline = json.load(
                    open(pjoin(work_dir, pdf_input, "presentation_outline.json"))
                )
                elif pexists(pjoin('wastedata',work_dir, pdf_input, "presentation_outline.json")):
                    outline = json.load(
                    open(pjoin('wastedata',work_dir, pdf_input, "presentation_outline.json"))
                )
                else:
                    print('cannot find outline for', model, pdf_input)
                    continue

                if not pexists(pjoin(work_dir, pdf_input, "agent_steps.json")):
                    continue
                steps = json.load(open(pjoin(work_dir, pdf_input, "agent_steps.json")))
                if not pexists(pjoin(work_dir, pdf_input, "final.pptx")):
                    continue
                success_all += 1
                final_presentation = Presentation.from_file(
                    pjoin(work_dir, pdf_input, "final.pptx"), app_config
                )
                doc_json = json.load(
                    open(
                        pjoin(
                            "data/topic/",
                            topic,
                            "pdf",
                            pdf_input,
                            "refined_doc.json",
                        )
                    )
                )
                for slide_idx, (step, (slide_title, slide)) in enumerate(
                    zip(steps, outline.items())
                ):

                    if step[0] not in original_textembeds:
                        original_textembeds[step[0]] = get_text_embedding(
                            source_presentation.slides[step[0]].to_text()
                        )
                    slide_content = get_slide_content(doc_json, slide_title, slide)
                    if not len(gscore)>300:
                        for i in final_presentation.slides:
                            try:
                                gscore.append(get_gscore(i.to_text(), slide_content))
                            except:
                                pass
                    if model.startswith("gpt-4o") or model.startswith("Intern") or model.startswith("Qwen"):
                        continue
                    slide_embedding = get_text_embedding(
                        final_presentation.slides[slide_idx].to_text()
                    )
                    fidelity += torch.cosine_similarity(
                        get_text_embedding(slide_content),
                        slide_embedding,
                        -1,
                    ).item() / len(final_presentation.slides)
        print(
                "%s success_all: %d, similarity_source: %f, geval %f"
                % (
                    model,
                    success_all,
                    fidelity / success_all,
                    sum(gscore) / len(gscore),
                )
            )
        json.dump(stat_succ_by_len, open("data/stat_succ_by_len.json", "w"), indent=4)


def dataset_stat():
    pdf_stat = {}
    ppt_stat = {}
    tempdir = TemporaryDirectory()
    config = Config()
    config.set_rundir(tempdir.name)
    for topic in topics:
        markdown_contents = {
            f: len(open(f, "r").read())
            for f in glob(f"data/topic/{filename_normalize(topic)}/pdf/*/*.md")
        }
        pdf_stat |= markdown_contents
        avg_pdf_text_len = sum(markdown_contents.values()) / len(markdown_contents)
        num_images = 0
        for pdf_folder in walk_data("pdf", topic):
            images = json.load(open(pjoin(pdf_folder, "image_caption.json")))
            num_images += len(images)
        avg_pdf_images = num_images / len(markdown_contents)
        ppt_text_len = 0
        ppt_pages = 0
        ppt_images = 0
        num_ppts = 10
        for ppt_folder in walk_data("pptx", topic):
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "original.pptx"), config
            )
            ppt_stat[ppt_folder] = sum(
                [len(slide.to_text()) for slide in presentation.slides]
            )

            ppt_text_len += ppt_stat[ppt_folder]
            ppt_pages += len(presentation.slides)
            ppt_images += len(os.listdir(pjoin(ppt_folder, "images")))

        avg_ppt_pages = ppt_pages / num_ppts
        avg_ppt_text_len = ppt_text_len / num_ppts
        avg_ppt_images = ppt_images / num_ppts
        print(
            f"{topic}: {avg_pdf_text_len:.2f}, {avg_pdf_images:.2f}, {avg_ppt_pages:.2f}, {avg_ppt_images:.2f}, {avg_ppt_text_len:.2f}"
        )
    # save to json
    json.dump({"pdf": pdf_stat, "ppt": ppt_stat}, open("data/stat.json", "w"), indent=4)


def postprocess_final_pptx():
    def ppt2images(ppt_folder: str):
        for pptx in glob(pjoin(ppt_folder, "*/*", "final.pptx")):
            model =pptx.split('/')[-3]
            if model != 'pptc':
                continue
            pdf = pptx.split("/")[-2]
            dst = pjoin(ppt_folder, "final_images", model, pdf)
            if pexists(dst):
                continue
            ppt_to_images(pptx, dst)

    process_filetype("pptx", ppt2images)


if __name__ == "__main__":
    app_config.DEBUG = True
    func_argparse.main(
        prepare_template,
        prepare_caption,
        generate_pres,
        dataset_stat,
        eval_experiment,
        postprocess_final_pptx,
        get_fid
    )
