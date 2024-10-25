import json
from functools import partial
from glob import glob
from typing import Type

import func_argparse
import torch
from FlagEmbedding import BGEM3FlagModel

import llms
from crawler import topics
from eval import get_text_embedding
from model_utils import get_text_models
from multimodal import ImageLabler
from pptgen import PPTAgent, PPTCrew, PPTGen
from preprocess import process_filetype
from presentation import Presentation
from utils import Config, pbasename, pexists, pjoin

eval_models = {
    "PPTAgent": [
        (
            {"planner": llms.qwen2_5, "agent": llms.qwen2_5, "coder": llms.qwen2_5},
            llms.gpt4o,
        ),
        (
            {"planner": llms.qwen2_5, "agent": llms.qwen2_5, "coder": llms.qwen2_5},
            llms.qwen_vl,
        ),
        ({"planner": llms.gpt4o, "agent": llms.gpt4o, "coder": llms.gpt4o}, llms.gpt4o),
    ],
    "PPTCrew": [
        ({"planner": llms.gpt4o, "agent": llms.gpt4o, "coder": llms.gpt4o}, llms.gpt4o),
        (
            {"planner": llms.qwen2_5, "agent": llms.qwen2_5, "coder": llms.qwen2_5},
            llms.qwen_vl,
        ),
    ],
}
GENCLASS = {
    "PPTAgent": PPTAgent,
    "PPTCrew": PPTCrew,
}


def get_setting(class_name: str, roles: dict[str, llms.LLM]):
    role_string = "+".join(
        f"{role}_{llm.model.split('-')[0]}" for role, llm in roles.items()
    )
    return class_name + "-" + role_string


def do_generate(
    genclass: Type[PPTGen],
    roles: dict[str, llms.LLM],
    text_models: list[BGEM3FlagModel],
    layout_source: str,
    setting: str,
    ppt_folder: str,
    thread_id: int,
):
    text_model = text_models[thread_id % len(text_models)]
    app_config = Config(rundir=ppt_folder)
    presentation = Presentation.from_file(
        pjoin(ppt_folder, "source_standard.pptx"),
        app_config,
    )
    ImageLabler(presentation, app_config)
    cluster_file = pjoin(
        ppt_folder,
        layout_source,
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
        get_text_embedding(list(slide_cluster.keys()), text_model)
    )
    pptgen = genclass(roles, text_model).set_examplar(
        presentation,
        slide_cluster,
        functional_keys,
        layout_embeddings,
    )
    topic = ppt_folder.split("/")[1]
    for pdf_folder in glob(f"data/*/pdf/*"):
        if pdf_folder.split("/")[1] == topic:
            continue
        app_config.set_rundir(pjoin(ppt_folder, setting, pbasename(pdf_folder)))
        if pexists(pjoin(app_config.RUN_DIR, "history")):
            print("skip", ppt_folder, pdf_folder, "already generated")
            continue
        images = json.load(
            open(pjoin(pdf_folder, "image_caption.json"), "r"),
        )
        doc_json = json.load(
            open(pjoin(pdf_folder, "refined_doc.json"), "r"),
        )
        pptgen.generate_pres(app_config, images, 12, doc_json)
        pptgen.save_history()


def generate_pres(agent_class: str, setting_id: int, thread_num: int = 16):
    text_models = get_text_models(thread_num)
    roles, layout_llm = eval_models[agent_class][setting_id]
    setting = get_setting(agent_class, roles)
    generate = partial(
        do_generate,
        GENCLASS[agent_class],
        roles,
        text_models,
        layout_llm.model,
        setting,
    )
    process_filetype("pptx", generate, thread_num)


if __name__ == "__main__":
    func_argparse.main(
        generate_pres,
    )
