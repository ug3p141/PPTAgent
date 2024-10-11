from collections import defaultdict
import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from tempfile import TemporaryDirectory
import tempfile
import traceback

import func_argparse
from jinja2 import Template
import jsonlines
import pytorch_fid.fid_score as fid
import torch
from tqdm.auto import tqdm

import llms
from agent import PPTAgent, get_slide_content
from agent_random import PPTAgent as PPTAgentRandom
from agent_pptc import PPTAgent as PPTAgentPPTC
from crawler import process_filetype, topics
from model_utils import get_text_embedding
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import Config, filename_normalize, pexists, pjoin, ppt_to_images
from FlagEmbedding import BGEM3FlagModel

fid.tqdm = lambda x: x
eval_models = [
    (llms.qwen, llms.gpt4o),
    (llms.qwen, llms.internvl_76),
    (llms.internvl_76, llms.internvl_76),
    (llms.gpt4o, llms.gpt4o),
]


def get_setting(model_id: int, ablation_id=None):
    ablations = ["code", "layout", "feedback"]
    if isinstance(ablation_id, int) and ablation_id != -1:
        if model_id != 0:
            raise Exception("model_id must be 0 when ablation_id is int")
        return ablations[ablation_id]
    llm, layout_llm = eval_models[model_id]
    if llm != layout_llm:
        setting = f"{llm.model}_{layout_llm.model}"
    else:
        setting = llm.model
    return setting


def walk_data(file_type: str, topic: str):
    folders = glob(f"data/topic/{filename_normalize(topic)}/{file_type}/*")
    for folder in folders:
        if folder.endswith(".DS_Store"):
            continue
        yield folder


def prepare_template():
    def gen_template(llm: llms.OPENAI, ppt_folder: str):
        ppt_image_folder = pjoin(ppt_folder, "slide_images")
        config = Config()
        with TemporaryDirectory() as temp_dir:
            config.set_rundir(temp_dir)
            presentation = Presentation.from_file(
                pjoin(ppt_folder, "source_standard.pptx"), config
            )
        template_image_folder = pjoin(ppt_folder, "template_images")
        llms.long_model = llm
        llms.agent_model = llm
        if len(os.listdir(ppt_image_folder)) != len(presentation):
            shutil.rmtree(ppt_image_folder)
            ppt_to_images(presentation.source_file, ppt_image_folder)
        if len(os.listdir(template_image_folder)) != len(presentation):
            shutil.rmtree(template_image_folder)
            presentation.save(pjoin(ppt_folder, "template.pptx"), True)
            ppt_to_images(pjoin(ppt_folder, "template.pptx"), template_image_folder)
        config.set_rundir(pjoin(ppt_folder, llm.model))
        if not pexists(ppt_image_folder):
            raise Exception(f"ppt_image_folder not found: {ppt_image_folder}")
        template_inducter = TemplateInducter(
            presentation, ppt_image_folder, template_image_folder, config
        )
        if not pexists(template_inducter.slide_split_file):
            template_inducter.category_split()
        template_inducter.work()

    for llm, layout_llm in eval_models:

        print(f"Preparing templates using {layout_llm.model}")
        process_filetype("pptx", partial(gen_template, layout_llm))


def prepare_caption():
    def caption(ppt_folder: str):
        config = Config()
        config.set_rundir(ppt_folder)
        presentation = Presentation.from_file(
            pjoin(ppt_folder, "source_standard.pptx"), config
        )
        labler = ImageLabler(presentation, config)
        labler.caption_images()
        labler.apply_stats()

    process_filetype("pptx", caption)


def get_text_models(thread_num: int):
    text_models = [
        BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=i)
        for i in range(min(torch.cuda.device_count(), thread_num))
    ]
    return text_models


def do_generate(
    pptagent, progressbar, pdf_folders, text_models, layout, setting, retry_times, arg
):
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
        layout,
        "template_induct",
        "slides_cluster.json",
    )
    if not pexists(cluster_file):
        print("Error: cluster_file not found", cluster_file)
        exit(-1)
    slide_cluster = json.load(open(cluster_file))
    functional_keys, slide_cluster = (
        set(slide_cluster.pop("functional_keys")),
        slide_cluster,
    )
    layout_embeddings = torch.stack(
        get_text_embedding(list(slide_cluster.keys()), text_model)
    )
    for pdf_folder in pdf_folders:
        app_config.set_rundir(pjoin(ppt_folder, setting, os.path.basename(pdf_folder)))
        if pexists(
            pjoin(
                app_config.RUN_DIR,
                "agent_steps.jsonl",
            )
        ):
            print("skip", ppt_folder, pdf_folder, "already generated")
            progressbar.update(1)
            continue
        images = json.load(
            open(pjoin(pdf_folder, "image_caption.json"), "r"),
        )
        doc_json = json.load(
            open(pjoin(pdf_folder, "refined_doc.json"), "r"),
        )
        try:
            pptagent(
                presentation,
                app_config,
                slide_cluster,
                images,
                12,
                text_model,
                doc_json,
                functional_keys,
                layout_embeddings,
            ).work(retry_times)
        except:
            traceback.print_exc()
            print(f"Error in {ppt_folder} {pdf_folder}")

        progressbar.update(1)


# 先确保一下qwen+gpt4o 的完成率能和之前一样吧, 应该是exit(-1导致的)
# qwen 重测以分析错误原因
# qwen w/o code_render
# qwen w/o html
# qwen w/o layout
def generate_pres(model_id: int, thread_num: int = 16, ablation_id: int = None):
    retry_times = 1
    ablation_agents = [PPTAgentPPTC, PPTAgentRandom, PPTAgent]
    agentclass = PPTAgent
    text_models = get_text_models(thread_num)
    llm, layout_llm = eval_models[model_id]
    llms.agent_model = llm
    setting = get_setting(model_id, ablation_id)
    if isinstance(ablation_id, int):
        agentclass = ablation_agents[ablation_id]
        if ablation_id == 2:
            retry_times = 0
    for topic in topics:
        ppt_folders = list(walk_data("pptx", topic))
        pdf_folders = list(walk_data("pdf", topic))
        print(f"Generating presentations using {setting} on {topic}")
        progressbar = tqdm(
            ppt_folders,
            total=len(ppt_folders) * len(pdf_folders),
        )
        generate = partial(
            do_generate,
            agentclass,
            progressbar,
            pdf_folders,
            text_models,
            layout_llm.model,
            setting,
            retry_times,
        )
        with ThreadPoolExecutor(thread_num) as executor:
            list(
                executor.map(generate, enumerate(ppt_folders)),
            )
        progressbar.close()


# 把所以setting的都拿出来算一遍，还有dims
def get_fid(model_id: int, ablation_id: int = -1):
    setting = get_setting(model_id, ablation_id)
    device = f"cuda:{random.randint(0,torch.cuda.device_count()-1)}"
    print("calc fid for", setting, "on device:", device)
    pbar = tqdm(total=200, desc=f"calc fid for {setting}")
    results = defaultdict(dict)
    for dim in [64, 192, 768, 2048]:
        if os.path.exists(f"data/fid_{setting}_{dim}.json"):
            pbar.update(50)
            continue
        model = fid.InceptionV3([fid.InceptionV3.BLOCK_INDEX_BY_DIM[dim]]).to(device)
        fid_scores = []

        for ppt_folder in glob(f"data/topic/*/pptx/*"):
            pbar.update(1)
            source_folder = pjoin(ppt_folder, "slide_images")
            m1, s1 = fid.compute_statistics_of_path(
                source_folder, model, 128, dim, device
            )
            for result_folder in glob(pjoin(ppt_folder, f"final_images/{setting}/*")):
                if len(os.listdir(result_folder)) == 0:
                    continue
                m2, s2 = fid.compute_statistics_of_path(
                    result_folder, model, 128, dim, device
                )
                try:
                    fid_scores.append(fid.calculate_frechet_distance(m1, s1, m2, s2))
                except:
                    pass

        
        results[setting][dim] = sum(fid_scores) / len(fid_scores)
        json.dump(results, open(f"data/fid_{setting}_{dim}.json", "w"), indent=4)


def get_gscore(slide_content, slide_ref):
    prompt = Template(open("prompts/llm_judge.txt").read())
    return llms.gpt4o(
        prompt.render(slide_content=slide_content, slide_ref=slide_ref),
        delay_batch=True,
    )


# 成功率也做个分析吧
def eval_experiment(
    model_id: int = 0,
    ablation_id: int = None,
    setting: str = None,
    geval: bool = False,
    succ_stat: bool = False,
    thread_num: int = 10,
    refer_idx: int = 1,
):
    setting = setting or get_setting(model_id, ablation_id)
    stat = json.load(open("data/stat.json"))
    ppt_stat = stat["ppt"]
    pdf_stat = stat["pdf"]
    pdf_stat = {k.split("/")[-1].rsplit(".", 1)[0]: v for k, v in pdf_stat.items()}
    stat_succ_by_len = list()
    success_all = 0
    fidelity = 0
    source_config = Config()
    final_config = Config()

    if geval != -1:
        thread_num = 1
    text_models = get_text_models(thread_num)
    ppt_folders = glob(f"data/topic/*/pptx/*")
    pbar = tqdm(total=len(ppt_folders), desc="Evaluating")

    def eval_ppt(ppt_folder: str, rank_id: int):
        nonlocal success_all, fidelity
        text_model = text_models[rank_id % len(text_models)]
        source_config.set_rundir(ppt_folder)
        source_presentation = Presentation.from_file(
            pjoin(ppt_folder, "source_standard.pptx"), source_config
        )
        original_textembeds = {}
        sub_fidelity = []
        topic = ppt_folder.split("/")[-3]
        work_dir = pjoin(ppt_folder, setting)
        for pdf_input in os.listdir(work_dir):
            if pdf_input not in pdf_stat:
                print("skip", pdf_input)
                continue
            if pexists(pjoin(work_dir, pdf_input, "presentation_outline.json")):
                outline = json.load(
                    open(pjoin(work_dir, pdf_input, "presentation_outline.json"))
                )
            else:
                print("cannot find outline for", setting, pdf_input)
                continue

            output_file = glob(pjoin(work_dir, pdf_input, "agent_steps.json*"))
            if succ_stat:
                stat_succ_by_len.append([ppt_stat[ppt_folder], pdf_stat[pdf_input], 0])
            if len(output_file) != 1:
                continue
            if not pexists(pjoin(work_dir, pdf_input, "final.pptx")):
                continue
            success_all += 1
            if output_file[0].endswith("jsonl"):
                steps = jsonlines.Reader(open(output_file[0]))
            else:
                steps = json.load(open(output_file[0]))
            if succ_stat:
                stat_succ_by_len[-1][2] = 1
            final_config.set_rundir(pjoin(work_dir, pdf_input))
            final_presentation = Presentation.from_file(
                pjoin(work_dir, pdf_input, "final.pptx"), final_config
            )
            doc_json = json.load(
                open(
                    f"data/topic/{topic}/pdf/{pdf_input}/refined_doc.json",
                )
            )
            for slide_idx, (step, (slide_title, slide)) in enumerate(
                zip(steps, outline.items())
            ):
                if slide_idx == len(final_presentation.slides):
                    break
                assert isinstance(step[refer_idx], int)
                if step[refer_idx] not in original_textembeds:
                    original_textembeds[step[refer_idx]] = get_text_embedding(
                        source_presentation.slides[step[refer_idx] - 1].to_text(),
                        text_model,
                    )
                slide_content = get_slide_content(doc_json, slide_title, slide)
                slide_embedding = get_text_embedding(
                    final_presentation.slides[slide_idx].to_text(),
                    text_model,
                )
                similarity = torch.cosine_similarity(
                    get_text_embedding(slide_content, text_model),
                    slide_embedding,
                    -1,
                ).item()
                sub_fidelity.append(similarity)
                if not geval:
                    continue
                for i in final_presentation.slides:
                    get_gscore(i.to_text(), slide_content)
        if len(sub_fidelity) > 0:
            fidelity += sum(sub_fidelity) / len(sub_fidelity)
        pbar.update(1)

    with ThreadPoolExecutor(thread_num) as executor:
        list(executor.map(eval_ppt, ppt_folders, range(len(ppt_folders))))

    pbar.close()
    result = {"success": success_all, "fidelity": fidelity / len(ppt_folders)}
    if geval:
        gscore = []
        for result in llms.gpt4o.get_batch_result():
            try:
                gscore.append(int(result))
            except:
                pass
        result["gscore"] = sum(gscore) / len(gscore)
    print(json.dumps(result, indent=4))
    json.dump(result, open(f"data/result_{setting}.json", "w"), indent=4)
    if succ_stat:
        json.dump(
            stat_succ_by_len,
            open(f"data/stat_succ_by_len_{setting}.json", "w"),
            indent=4,
        )


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
            ppt_pages += len(presentation)
            ppt_images += len(os.listdir(pjoin(ppt_folder, "images")))

        avg_ppt_pages = ppt_pages / num_ppts
        avg_ppt_text_len = ppt_text_len / num_ppts
        avg_ppt_images = ppt_images / num_ppts
        print(
            f"{topic}: {avg_pdf_text_len:.2f}, {avg_pdf_images:.2f}, {avg_ppt_pages:.2f}, {avg_ppt_images:.2f}, {avg_ppt_text_len:.2f}"
        )

    json.dump({"pdf": pdf_stat, "ppt": ppt_stat}, open("data/stat.json", "w"), indent=4)


def postprocess_final_pptx():
    def ppt2images(ppt_folder: str):
        for pptx in glob(pjoin(ppt_folder, "*/*", "final.pptx")):
            model = pptx.split("/")[-3]
            pdf = pptx.split("/")[-2]
            dst = pjoin(ppt_folder, "final_images", model, pdf)
            if pexists(dst):
                continue
            try:
                ppt_to_images(pptx, dst)
            except:
                print(pptx, " failed")

    process_filetype("pptx", ppt2images)


def error_analysis(model_id: int, ablation_id: int = None):
    setting = get_setting(model_id, ablation_id)
    error_stats = defaultdict(list)
    undefined_errors = []
    element_grounding_errors = [
        "textframe ID",
        "The element is not a Picture",
        "Only the element_id",
        'element_id, text_id = textframe_id.split("_")',
        "for para in shape.text_frame.data",
        "shape = slide.shapes[element_id]",
        "shape = slide.shapes[int(element_id)]",
        "shape = slide.shapes[int(figure_id)]",
    ]
    instruction_following_errors = [
        "The function ",
        "has been",
        "invalid literal for int()",
        "SyntaxError",
    ]
    hallucination_errors = ["does not exist.", "got an unexpected keyword argument"]
    for step_file in glob(f"data/topic/*/pptx/*/{setting}/*/code_steps.jsonl"):
        steps = jsonlines.Reader(open(step_file))
        for code_step in steps:
            if code_step[-1] == None:
                continue
            error_reason = code_step[-1]

            if any(err in error_reason for err in element_grounding_errors):
                error_stats["element_grounding"].append(error_reason)
            elif any(err in error_reason for err in instruction_following_errors):
                error_stats["instruction_following"].append(error_reason)
            elif any(err in error_reason for err in hallucination_errors):
                error_stats["hallucination"].append(error_reason)
            elif "Incorrect shape: " in error_reason:
                breakpoint()
            else:
                undefined_errors.append(error_reason)

    for k, v in error_stats.items():
        print(k, len(v))
    print("undefined", len(undefined_errors))


if __name__ == "__main__":
    func_argparse.main(
        prepare_template,
        prepare_caption,
        generate_pres,
        dataset_stat,
        eval_experiment,
        postprocess_final_pptx,
        get_fid,
        error_analysis,
    )
