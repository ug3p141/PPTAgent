import json
from copy import deepcopy
from datetime import datetime
import traceback

import json_repair
import PIL.Image
import jsonlines
import torch
from jinja2 import Template

import llms
from apis import CodeExecutor, get_code_executor, API_TYPES
from model_utils import get_text_embedding
from presentation import Presentation, SlidePage
from utils import Config, pexists, pjoin, print


class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        config: Config,
        template: dict,
        images: dict[str, str],
        num_slides: int,
        text_model,
        doc_json: dict[str, str],
        functional_keys: set[str],
        layout_embeddings: torch.Tensor,
    ):
        self.presentation = presentation
        self.slide_templates = template
        self.doc_json = doc_json
        self.num_slides = num_slides
        self.image_information = ""
        self.config = config
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
        self.text_model = text_model

        self.outline_file = pjoin(config.RUN_DIR, "presentation_outline.json")

    def work(self, retry_times: int = 1, force_pages: bool = True):
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
        self.generate_slides(retry_times, force_pages)

    def generate_slides(self, retry_times: int, force_pages: bool):
        succ_flag = True
        code_executor = get_code_executor(retry_times)
        self.gen_prs.slides = []
        for slide_data in enumerate(self.outline.items()):
            if force_pages and slide_data[0] == self.num_slides:
                break
            try:
                self.gen_prs.slides.append(
                    self._generate_slide(slide_data, code_executor)
                )
            except Exception as e:
                succ_flag = False
                if self.config.DEBUG:
                    traceback.print_exc()
                break
        with jsonlines.Writer(
            open(pjoin(self.config.RUN_DIR, "agent_steps.jsonl"), "w")
        ) as writer:
            writer.write_all(code_executor.api_history)
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "code_steps.jsonl"), "w"
        ) as writer:
            writer.write_all(code_executor.code_history)
        if succ_flag:
            self.gen_prs.save(pjoin(self.config.RUN_DIR, "final.pptx"))
        else:
            raise Exception("Failed to generate slide")

    def _generate_slide(self, slide_data, code_executor: CodeExecutor):
        slide_idx, (slide_title, slide) = slide_data
        images = "No Images"
        if slide["layout"] not in self.layout_names:
            layout_sim = torch.cosine_similarity(
                get_text_embedding(slide["layout"], model=self.text_model),
                self.layout_embeddings,
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
        edit_slide: SlidePage = deepcopy(self.presentation.slides[template_id])
        edit_prompt = self.edit_template.render(
            api_documentation=code_executor.get_apis_docs([API_TYPES.PPTAgent]),
            edit_target=edit_slide.to_html(),
            content=self.simple_outline + self.metadata + slide_content,
            images=images,
        )
        code_executor.execute_actions(
            edit_prompt,
            actions=llms.agent_model(
                edit_prompt,
            ),
            edit_slide=edit_slide,
        )
        edit_slide.slide_idx = slide_idx
        return edit_slide

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
    slide_desc = slide.get("description", "")
    slide_content = f"Title: {slide_title}\nSlide Description: {slide_desc}\n"
    if len(slide.get("subsection_keys", [])) != 0:
        slide_content += "Slide Reference Text: "
        for key in slide["subsection_keys"]:
            for section in doc_json["sections"]:
                for subsection in section.get("subsections", []):
                    if key in subsection:
                        slide_content += f"SubSection {key}: {subsection[key]}\n"
    return slide_content
