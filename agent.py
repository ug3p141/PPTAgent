import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime

import json_repair
import PIL
import torch
from jinja2 import Template

import apis
import llms
from apis import API_TYPES, code_executor
from model_utils import get_text_embedding
from presentation import Presentation
from utils import Config, clear_slides, pexists, pjoin, print


class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        app_config: Config,
        template: dict,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
        functional_keys: set[str],
        layout_embeddings: torch.Tensor,
    ):
        self.presentation = presentation
        self.slide_templates = template
        self.doc_json = doc_json
        self.num_slides = num_slides
        self.image_information = ""
        self.app_config = app_config
        for k, v in images.items():
            if not pexists(k):
                raise FileNotFoundError(f"Image {k} not found")
            size = PIL.Image.open(k).size
            self.image_information += (
                f"Image path: {k}, size: {size[0]}*{size[1]} px\n caption: {v}\n"
            )
        self.functional_keys = functional_keys
        self.edit_template = Template(open("prompts/agent/edit.txt").read())
        self.layout_embeddings = layout_embeddings
        self.gen_prs = deepcopy(presentation)
        self.layout_names = list(self.slide_templates.keys())
        self.gen_prs.slides = []

        self.outline_file = pjoin(app_config.RUN_DIR, "presentation_outline.json")
        self.agent_steps = pjoin(app_config.RUN_DIR, "agent_steps.json")

    def work(self):
        if pexists(self.outline_file):
            self.outline = json.load(open(self.outline_file, "r"))
        else:
            self.outline = json_repair.loads(self.generate_outline())
            json.dump(
                self.outline, open(self.outline_file, "w"), ensure_ascii=False, indent=4
            )
        meta_data = "\n".join(
            [f"{k}: {v}" for k, v in self.doc_json.get("metadata", {}).items()]
        )
        self.metadata = f"\nMetadata of Presentation: \n{meta_data}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        self.simple_outline = "Outline of Presentation: \n" + "\n".join(
            [
                f"Slide {slide_idx+1}: {slide_title}"
                for slide_idx, slide_title in enumerate(self.outline)
            ]
        )
        self.generate_slides()

    def generate_slides(self):
        steps = []
        exit_flag = False
        code_executor.reset()
        self.gen_prs.slides = []
        # 由于GIL的存在，code executor的api_history和code_history在多线程中不会冲突
        for slide_data in enumerate(self.outline.items()):
            step, slide = self._generate_slide(slide_data)
            steps.append(step)
            if slide is not None:
                self.gen_prs.slides.append(slide)
            else:
                exit_flag = True
        json.dump(
            (steps, code_executor.api_history, code_executor.code_history),
            open(self.agent_steps, "w"),
        )
        if exit_flag:
            return
        # self.gen_prs.slides.sort(key=lambda x: x.slide_idx)
        self.gen_prs.save(pjoin(self.app_config.RUN_DIR, "final.pptx"))

    def _generate_slide(self, slide_data):
        slide_idx, (slide_title, slide) = slide_data
        images = "No Images"
        if slide["layout"] not in self.layout_names:
            layout_sim = torch.cosine_similarity(
                get_text_embedding(slide["layout"]), self.layout_embeddings
            )
            slide["layout"] = self.layout_names[layout_sim.argmax().item()]
        if any(
            [
                i in slide["layout"]
                for i in ["picture", "chart", "table", "diagram", "freeform"]
            ]
        ):
            images = self.image_information
        slide_content = f"Slide-{slide_idx+1} " + get_slide_content(
            self.doc_json, slide_title, slide
        )
        template_id = (
            max(
                self.slide_templates[slide["layout"]],
                key=lambda x: len(self.presentation.slides[x - 1].shapes),
            )
            - 1
        )
        edit_slide = deepcopy(self.presentation.slides[template_id])
        edit_slide.slide_idx = slide_idx
        edit_prompt = self.edit_template.render(
            api_documentation=code_executor.get_apis_docs(
                [API_TYPES.TEXT_EDITING, API_TYPES.IMAGE_EDITING]
            ),
            edit_target=edit_slide.to_html(),
            content=self.simple_outline + self.metadata + slide_content,
            images=images,
        )
        try:
            code_executor.execute_apis(
                edit_prompt,
                apis=llms.agent_model(
                    edit_prompt,
                ),
                edit_slide=edit_slide,
            )
            return (template_id, *code_executor.api_history[-1]), edit_slide
        except:
            return (template_id, *code_executor.api_history[-1]), None

    def generate_outline(self):
        template = Template(open("prompts/agent/outline.txt").read())
        prompt = template.render(
            num_slides=self.num_slides,
            layouts="\n".join(
                set(self.slide_templates.keys()).difference(self.functional_keys)
            ),
            functional_keys="\n".join(self.functional_keys),
            json_content=self.doc_json,
            images=self.image_information,
        )
        return llms.agent_model(prompt)


def get_slide_content(doc_json: dict, slide_title: str, slide: dict):
    slide_content = f"Title: {slide_title}\nSlide Description: {slide['description']}\n"
    if len(slide["subsection_keys"]) != 0:
        slide_content += "Slide Reference Text: "
        for key in slide["subsection_keys"]:
            for section in doc_json["sections"]:
                for subsection in section["subsections"]:
                    if key in subsection:
                        slide_content += f"SubSection {key}: {subsection[key]}\n"
    return slide_content
