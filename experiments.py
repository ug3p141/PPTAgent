import json
import re
from functools import partial
from glob import glob
from typing import Literal, Type

import func_argparse
import torch

import llms
from model_utils import get_text_model
from multimodal import ImageLabler
from pptgen import PPTAgent, PPTCrew, PPTGen
from preprocess import process_filetype
from presentation import Presentation
from utils import Config, pbasename, pexists, pjoin

# language_model vision_model code_model
eval_models = {
    "PPTAgent": [
        (llms.qwen2_5, llms.qwen2_5, llms.gpt4o),
        (llms.qwen2_5, llms.qwen2_5, llms.qwen_vl),
    ],
    "PPTCrew": [
        (llms.qwen2_5, llms.qwen2_5, llms.qwen_vl),
        (llms.gpt4o, llms.gpt4o, llms.gpt4o),
    ],
}
GENCLASS = {
    "PPTAgent": PPTAgent,
    "PPTCrew": PPTCrew,
}


def get_setting(class_name: str, setting_id: int):
    language_model, code_model, vision_model = eval_models[class_name][setting_id]
    llms.language_model = language_model
    llms.code_model = code_model
    llms.vision_model = vision_model
    role_string = "+".join(
        re.search(r"^(.*?)-\d{2}", llm.model).group(1)
        for llm in [language_model, code_model, vision_model]
    )
    return class_name + "-" + role_string


# 所有template要重新prepare一遍，除了qwen2.5+qwen_vl
def do_generate(
    genclass: Type[PPTGen],
    setting: str,
    debug: bool,
    ppt_folder: str,
    thread_id: int,
):
    app_config = Config(rundir=ppt_folder, debug=debug)
    text_model = get_text_model(f"cuda:{thread_id % torch.cuda.device_count()}")
    presentation = Presentation.from_file(
        pjoin(ppt_folder, "source_standard.pptx"),
        app_config,
    )
    ImageLabler(presentation, app_config).caption_images()
    model_identifier = "+".join(
        (
            llms.language_model.model.split("-")[0],
            llms.vision_model.model.split("-")[0],
        )
    )
    induct_cache = pjoin(
        app_config.RUN_DIR, "template_induct", model_identifier, "induct_cache.json"
    )
    if not pexists(induct_cache):
        raise Exception(f"induct_cache not found: {induct_cache}")
    slide_induction = json.load(open(induct_cache))
    pptgen: PPTCrew = genclass(text_model).set_examplar(presentation, slide_induction)
    topic = ppt_folder.split("/")[1]
    for pdf_folder in glob(f"data/{topic}/pdf/*"):
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


def generate_pres(
    agent_class: str = "PPTCrew",
    setting_id: int = 0,
    setting_name: str = None,
    thread_num: int = 16,
    debug: bool = False,
):
    s = get_setting(agent_class, setting_id)
    print("generating slides using:", s)
    setting = setting_name or s
    generate = partial(
        do_generate,
        GENCLASS[agent_class],
        setting,
        debug,
    )
    process_filetype("pptx", generate, thread_num)


if __name__ == "__main__":
    func_argparse.main(
        generate_pres,
    )
