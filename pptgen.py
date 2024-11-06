import json
import os
import tempfile
import traceback
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

import jsonlines
import PIL.Image
import torch
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment, StrictUndefined

from apis import API_TYPES, CodeExecutor
from llms import Role
from model_utils import get_slide_image, get_text_embedding
from presentation import Presentation, SlidePage
from utils import Config, get_slide_content, pexists, pjoin, tenacity


@dataclass
class PPTGen(ABC):
    roles: list[str] = field(default_factory=list)

    def __init__(
        self,
        text_model: BGEM3FlagModel,
        retry_times: int = 3,
        force_pages: bool = False,
        error_exit: bool = True,
        **kwargs,
    ):
        self.text_model = text_model
        self.retry_times = retry_times
        self.force_pages = force_pages
        self.error_exit = error_exit
        self.staffs: dict[str, Role] = self._hire_staffs(**kwargs)

    def set_examplar(
        self,
        presentation: Presentation,
        slide_induction: dict,
    ):
        self.presentation = presentation
        self.slide_induction = slide_induction
        self.functional_keys = slide_induction.pop("functional_keys")
        self.layout_names = list(slide_induction.keys())
        self.layout_embeddings = torch.stack(
            get_text_embedding(self.layout_names, self.text_model)
        )
        self.empty_prs = deepcopy(presentation)
        self.empty_prs.slides = []
        return self

    def generate_pres(
        self,
        config: Config,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
    ):
        os.makedirs(pjoin(config.RUN_DIR, "history"), exist_ok=True)
        generated_slides = []
        self.config = config
        self.doc_json = doc_json
        meta_data = "\n".join(
            [f"{k}: {v}" for k, v in self.doc_json.get("metadata", {}).items()]
        )
        self.metadata = (
            f"{meta_data}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        )
        self.image_information = ""
        for k, v in images.items():
            assert pexists(k), f"Image {k} not found"
            size = PIL.Image.open(k).size
            self.image_information += (
                f"Image path: {k}, size: {size[0]}*{size[1]} px\n caption: {v}\n"
            )
        succ_flag = True
        code_executor = CodeExecutor(
            self.staffs.get("debugger", None), self.retry_times
        )
        try:  # ? 这里没有留下什么标记啊
            self.outline = self.generate_outline(num_slides)
        except:
            raise Exception("Failed to generate outline")
        self.simple_outline = "\n".join(
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
                generated_slides.append(slide)
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
            self.empty_prs.slides = generated_slides
            self.empty_prs.save(pjoin(self.config.RUN_DIR, "final.pptx"))
        else:
            raise Exception("Failed to generate slide")

    def save_history(self):
        for role in self.staffs.values():
            role.save_history(pjoin(self.config.RUN_DIR, "history"))

    def generate_outline(self, num_slides: int):
        outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        if pexists(outline_file):
            outline = json.load(open(outline_file, "r"))
        else:
            outline = self.staffs["planner"](
                num_slides=num_slides,
                layouts="\n".join(
                    set(self.slide_induction.keys()).difference(self.functional_keys)
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

    def _hire_staffs(self, **kwargs) -> dict[str, Role]:
        jinja_env = Environment(undefined=StrictUndefined)
        return {
            role: Role(role, env=jinja_env, **kwargs)
            for role in ["planner", "debugger"] + self.roles
        }

    @abstractmethod
    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        pass

    def _generate_slide(self, slide_data, code_executor: CodeExecutor):
        slide_idx, (slide_title, slide) = slide_data
        image_info = "No Images"
        layout_sim = torch.cosine_similarity(
            get_text_embedding(slide["layout"], self.text_model),
            self.layout_embeddings,
        )
        if layout_sim.max() < 0.7:
            raise ValueError(
                f"Layout {slide['layout']} not found, most similar is {slide['layout']}"
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
        template = self.slide_induction[slide["layout"]]
        return self.synergize(
            template,
            slide_content,
            code_executor,
            image_info,
        )


class PPTAgent(PPTGen):
    roles: list[str] = ["agent"]

    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        slide = deepcopy(self.presentation.slides[template["template_id"]])
        actions = self.staffs["agent"](
            api_documentation=code_executor.get_apis_docs(API_TYPES.Agent.value),
            edit_target=slide.html,
            content=self.simple_outline + self.metadata + slide_content,
            image_information=image_info,
        )
        code_executor.execute_actions(
            actions=actions,
            edit_slide=slide,
        )
        return slide


# 把debugger去掉
class PPTCrew(PPTGen):
    roles: list[str] = ["editor", "coder", "typographer", "reviewer"]

    def synergize(
        self,
        template: dict,
        slide_content: str,
        code_executor: CodeExecutor,
        images_info: str,
    ):
        slide = deepcopy(self.presentation.slides[template["template_id"]])
        content_schema = template["content_schema"]
        old_data = self._prepare_schema(content_schema)
        # ? 这个metadata也保存起来，免得到时候变了, 后面用来做评估
        # ? 看来还是得加个reviewer，因为还是有不符合规则的问题
        editor_output = self.staffs["editor"](
            schema=content_schema,
            outline=self.simple_outline,
            metadata=self.metadata,
            text=slide_content,
            images_info=images_info,
        )
        # remove elements, add new elements, change elements
        command_list = self._generate_commands(editor_output, content_schema, old_data)
        # 也许可以在这一步分配一个class name 方便后续css
        edit_actions = self.staffs["coder"](
            api_docs=code_executor.get_apis_docs(API_TYPES.Coder.value),
            edit_target=slide.to_html(textframe_id=True, image_id=True),
            command_list=command_list,
        )
        code_executor.execute_actions(
            actions=edit_actions,
            edit_slide=slide,
        )
        # 给所有role添加一个终止约束，例如reviewer，重试三次，或输出yes
        with tempfile.TemporaryFile(suffix=".jpg") as temp_file:
            get_slide_image(slide, self.empty_prs, temp_file.name)
            typography_actions = self.staffs["typographer"](
                api_docs=code_executor.get_apis_docs(API_TYPES.Typographer.value),
                current_css=slide.css,
                slide_image=temp_file.name,
            )
        code_executor.execute_actions(
            actions=typography_actions,
            edit_slide=slide,
        )

    def _prepare_schema(self, content_schema: dict):
        old_data = {}
        for el_name, el_info in content_schema.items():
            if el_info["type"] == "text":
                if isinstance(el_info["data"], list):
                    charater_counts = [len(i) for i in el_info["data"]]
                    content_schema[el_name]["suggestedCharacters"] = (
                        str(min(charater_counts)) + "-" + str(max(charater_counts))
                    )
                else:
                    content_schema[el_name]["suggestedCharacters"] = "<" + str(
                        len(el_info["data"])
                    )
            old_data[el_name] = el_info.pop("data")
            content_schema[el_name]["default_quantity"] = 1
            if isinstance(old_data[el_name], list):
                content_schema[el_name]["default_quantity"] = len(old_data[el_name])
        return old_data

    def _generate_commands(
        self, editor_output: dict, content_schema: dict, old_data: dict
    ):
        command_list = []
        for el_name, old_content in old_data.items():
            if el_name not in editor_output or editor_output[el_name]["data"] is None:
                command_list.append(
                    ("remove", content_schema[el_name]["type"], old_content)
                )
                continue
            new_content = editor_output[el_name]["data"]
            quantity_change = None

            if not isinstance(old_content, list):
                command_list.append(
                    (
                        "change",
                        content_schema[el_name]["type"],
                        old_content,
                        new_content,
                    )
                )
                continue

            assert isinstance(
                new_content, list
            ), "old_content is list, new_content must be list"
            old_len = len(old_content)
            new_len = len(new_content)
            if old_len != new_len:
                quantity_change = new_len - old_len

            command_list.append(
                (
                    "change",
                    content_schema[el_name]["type"],
                    f"quantity_change: {quantity_change}",
                    old_content,
                    new_content,
                )
            )

        return command_list
