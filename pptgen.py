import json
import os
import traceback
from copy import deepcopy
from datetime import datetime

import jsonlines
import PIL.Image
import torch
from FlagEmbedding import BGEM3FlagModel

from apis import API_TYPES, CodeExecutor
from llms import LLM, Role
from model_utils import get_text_embedding
from presentation import Presentation, SlidePage
from utils import Config, get_slide_content, pexists, pjoin, tenacity


class PPTGen:
    def __init__(
        self,
        roles: dict[str, LLM],
        text_model: BGEM3FlagModel,
        retry_times: int = 3,
        force_pages: bool = False,
        error_exit: bool = True,
    ):
        self.text_model = text_model
        self.retry_times = retry_times
        self.force_pages = force_pages
        self.error_exit = error_exit
        self.staffs = {name: Role(name, llm) for name, llm in roles.items()}

    def set_examplar(
        self,
        presentation: Presentation,
        slide_templates: dict,
        functional_keys: set[str],
        layout_embeddings: torch.Tensor,
    ):
        self.presentation = presentation
        self.slide_templates = slide_templates
        self.functional_keys = functional_keys
        self.layout_embeddings = layout_embeddings
        self.layout_names = list(slide_templates.keys())
        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        return self

    def generate_pres(
        self,
        config: Config,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
    ):
        self.config = config
        os.makedirs(pjoin(config.RUN_DIR, "history"), exist_ok=True)
        self.doc_json = doc_json
        meta_data = "\n".join(
            [f"{k}: {v}" for k, v in self.doc_json.get("metadata", {}).items()]
        )
        self.metadata = f"\nMetadata of Presentation: \n{meta_data}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        self.image_information = ""
        for k, v in images.items():
            assert pexists(k), f"Image {k} not found"
            size = PIL.Image.open(k).size
            self.image_information += (
                f"Image path: {k}, size: {size[0]}*{size[1]} px\n caption: {v}\n"
            )
        succ_flag = True
        code_executor = CodeExecutor(self.staffs.get("coder", None), self.retry_times)
        self.gen_prs.slides = []
        self.outline = self._generate_outline(num_slides)
        self.simple_outline = "Outline of Presentation: \n" + "\n".join(
            [
                f"Slide {slide_idx+1}: {slide_title}"
                for slide_idx, slide_title in enumerate(self.outline)
            ]
        )
        for slide_data in enumerate(self.outline.items()):
            if self.force_pages and slide_data[0] == num_slides:
                break
            slide = self._generate_slide(slide_data, code_executor)
            if slide is not None:
                self.gen_prs.slides.append(slide)
                continue
            if self.config.DEBUG:
                traceback.print_exc()
            if self.error_exit:
                succ_flag = False
                break
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "code_steps.jsonl"), "w"
        ) as writer:
            writer.write_all(code_executor.code_history)
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "agent_steps.jsonl"), "w"
        ) as writer:
            writer.write_all(code_executor.api_history)
        if succ_flag:
            self.gen_prs.save(pjoin(self.config.RUN_DIR, "final.pptx"))
        else:
            raise Exception("Failed to generate slide")

    def save_history(self):
        for role in self.staffs.values():
            role.save_history(pjoin(self.config.RUN_DIR, "history"))

    def synergize(
        self,
        slide: SlidePage,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        pass

    def _generate_slide(self, slide_data, code_executor: CodeExecutor):
        slide_idx, (slide_title, slide) = slide_data
        image_info = "No Images"
        if slide["layout"] not in self.layout_names:
            layout_sim = torch.cosine_similarity(
                get_text_embedding(slide["layout"], self.text_model),
                self.layout_embeddings,
            )
            slide["layout"] = self.layout_names[layout_sim.argmax().item()]
        if any(
            [
                i in slide["layout"]
                for i in ["picture", "chart", "table", "diagram", "freeform"]
            ]
        ):
            image_info = self.image_information
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
        return self.synergize(
            deepcopy(self.presentation.slides[template_id]),
            slide_content,
            code_executor,
            image_info,
        )


class PPTAgent(PPTGen):
    def _generate_outline(self, num_slides: int):
        outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        if pexists(outline_file):
            outline = json.load(open(outline_file, "r"))
        else:
            outline = self.staffs["planner"](
                num_slides=num_slides,
                layouts="\n".join(
                    set(self.slide_templates.keys()).difference(self.functional_keys)
                ),
                functional_keys="\n".join(self.functional_keys),
                json_content=self.doc_json,
                image_information=self.image_information,
            )
            if isinstance(outline, dict) and all(
                set(slide.keys()) == {"layout", "subsection_keys", "description"}
                for slide in outline.values()
            ):
                json.dump(
                    outline, open(outline_file, "w"), ensure_ascii=False, indent=4
                )
            else:
                raise ValueError("Invalid outline structure")
        return outline

    def synergize(
        self,
        slide: SlidePage,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        actions = self.staffs["agent"](
            api_documentation=code_executor.get_apis_docs(API_TYPES.Agent.value),
            edit_target=slide.to_html(),
            content=self.simple_outline + self.metadata + slide_content,
            image_information=image_info,
        )
        code_executor.execute_actions(
            actions=actions,
            edit_slide=slide,
        )
        return slide


class PPTCrew(PPTGen):
    def synergize(
        self,
        slide: SlidePage,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        pass
