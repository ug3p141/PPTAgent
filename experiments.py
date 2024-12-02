import json
import re
from functools import partial
from glob import glob
from typing import Type

import func_argparse
import torch

import llms
from ablation import (
    PPTCrew_wo_Decoupling,
    PPTCrew_wo_HTML,
    PPTCrew_wo_LayoutInduction,
    PPTCrew_wo_SchemaInduction,
)
from model_utils import get_text_model
from multimodal import ImageLabler
from pptgen import PPTCrew
from preprocess import process_filetype
from presentation import Presentation
from utils import Config, older_than, pbasename, pexists, pjoin

# language_model vision_model code_model
EVAL_MODELS = [
    (llms.qwen2_5, llms.qwen2_5, llms.qwen_vl),
    (llms.gpt4o, llms.gpt4o, llms.gpt4o),
    (llms.qwen2_5, llms.qwen_coder, llms.qwen_vl),
]

# ablation
# 0: w/o layout induction: random layout
# 1: w/o schema induction: 只提供old data 的值，别的都不提供
# 2. w/o decoupling: pptagent
# 3: w/o html: use pptc
# 4: w/o typographer: pptagent
# 5. w/o comman generation? 给他新旧的值的对比

AGENT_CLASS = {
    -1: PPTCrew,
    0: PPTCrew_wo_LayoutInduction,
    1: PPTCrew_wo_SchemaInduction,
    2: PPTCrew_wo_Decoupling,
    3: PPTCrew_wo_HTML,
}


def get_setting(setting_id: int, ablation_id: int):
    language_model, code_model, vision_model = EVAL_MODELS[setting_id]
    agent_class = AGENT_CLASS.get(ablation_id)
    llms.language_model = language_model
    llms.code_model = code_model
    llms.vision_model = vision_model
    if ablation_id == -1:
        setting_name = "PPTCrew-" + "+".join(
            re.search(r"^(.*?)-\d{2}", llm.model).group(1)
            for llm in [language_model, code_model, vision_model]
        )
    else:
        setting_name = agent_class.__name__
    return agent_class, setting_name


# 所有template要重新prepare一遍，除了qwen2.5+qwen_vl
def do_generate(
    genclass: Type[PPTCrew],
    setting: str,
    debug: bool,
    ppt_folder: str,
    thread_id: int,
):
    app_config = Config(rundir=ppt_folder, debug=debug)
    text_model = get_text_model(f"cuda:{thread_id % torch.cuda.device_count()}")
    presentation = Presentation.from_file(
        pjoin(ppt_folder, "source.pptx"),
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
    if not older_than(induct_cache, wait=True):
        print(f"induct_cache not found: {induct_cache}")
        return
    slide_induction = json.load(open(induct_cache))
    pptgen: PPTCrew = genclass(text_model).set_examplar(presentation, slide_induction)
    topic = ppt_folder.split("/")[1]
    for pdf_folder in glob(f"data/{topic}/pdf/*"):
        app_config.set_rundir(pjoin(ppt_folder, setting, pbasename(pdf_folder)))
        if pexists(pjoin(app_config.RUN_DIR, "history")):
            continue
        images = json.load(
            open(pjoin(pdf_folder, "image_caption.json"), "r"),
        )
        doc_json = json.load(
            open(pjoin(pdf_folder, "refined_doc.json"), "r"),
        )
        pptgen.generate_pres(app_config, images, 12, doc_json)


def generate_pres(
    setting_id: int = 0,
    ablation_id: int = -1,
    setting_name: str = None,
    thread_num: int = 16,
    debug: bool = False,
):
    agent_class, s = get_setting(setting_id, ablation_id)
    print("generating slides using:", s)
    setting = setting_name or s
    generate = partial(
        do_generate,
        agent_class,
        setting,
        debug,
    )
    process_filetype("pptx", generate, thread_num)


if __name__ == "__main__":
    func_argparse.main(
        generate_pres,
    )
