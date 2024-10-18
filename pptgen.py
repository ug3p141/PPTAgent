import json
import traceback
from copy import deepcopy
from datetime import datetime

import jsonlines
import PIL.Image
import torch
from jinja2 import Template

import llms
from apis import API_TYPES, CodeExecutor, get_code_executor
from model_utils import get_text_embedding
from presentation import Presentation, SlidePage
from utils import Config, get_slide_content, pexists, pjoin, print


class Role:
    def __init__(self, name: str, llm: llms.OPENAI, config: dict = None):
        self.name = name
        self.llm = llm
        self.model = llm.model
        if config is None and os.path.exists(f"prompts/roles/{name}.json"):
            with open(f"prompts/roles/{name}.json", "r") as f:
                config = json.load(f)
        self.description = config["description"]
        self.template = Template(config["template"])
        self.prompt_args = config["jinja_args"]
        self.return_json = config["return_json"]

    def history_manager(self):
        pass

    def __call__(self, message: str, image_files: list[str] = None, **kwargs):
        assert self.prompt_args == kwargs.keys()
        return self.llm(
            self.template.render(message, **kwargs),
            image_files=image_files,
            return_json=self.return_json,
        )


class PPTGen:
    def __init__(
        self,
        text_model,
        retry_times: int = 3,
        force_pages: bool = False,
        error_exit: bool = True,
    ):
        self.staffs: dict[str, Role] = dict()
        self.rally()
        self.text_model = text_model
        self.retry_times = retry_times
        self.force_pages = force_pages
        self.error_exit = error_exit

    def hire(self, name: str, llm: llms.OPENAI, *args, **kwargs):
        self.staffs[name] = Role(name, llm, *args, **kwargs)

    def rally(self):
        pass

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
        meta_data = "\n".join(
            [f"{k}: {v}" for k, v in self.doc_json.get("metadata", {}).items()]
        )
        self.metadata = f"\nMetadata of Presentation: \n{meta_data}\nCurrent Time: {datetime.now().strftime('%Y-%m-%d')}\n"
        return self

    def generate_pres(
        self,
        config: Config,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
    ):
        self.config = config
        self.num_slides = num_slides
        self.doc_json = doc_json
        self.outline = self._generate_outline(self.staffs[0])
        self.simple_outline = "Outline of Presentation: \n" + "\n".join(
            [
                f"Slide {slide_idx+1}: {slide_title}"
                for slide_idx, slide_title in enumerate(self.outline)
            ]
        )
        self.image_information = ""
        for k, v in images.items():
            if not pexists(k):
                raise FileNotFoundError(f"Image {k} not found")
            size = PIL.Image.open(k).size
            self.image_information += (
                f"Image path: {k}, size: {size[0]}*{size[1]} px\n caption: {v}\n"
            )
        succ_flag = True
        code_executor = get_code_executor(self.retry_times)
        self.gen_prs.slides = []
        for slide_data in enumerate(self.outline.items()):
            if self.force_pages and slide_data[0] == self.num_slides:
                break
            try:
                self.gen_prs.slides.append(
                    self._generate_slide(slide_data, code_executor, self.staffs)
                )
            except Exception:
                if self.config.DEBUG:
                    traceback.print_exc()
                if self.error_exit:
                    succ_flag = False
                    break
        with jsonlines.open(
            pjoin(self.config.RUN_DIR, "agent_steps.jsonl"), "w"
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

    def _generate_slide(
        self, slide_data, code_executor: CodeExecutor, agent_model: Role
    ):
        slide_idx, (slide_title, slide) = slide_data
        images = "No Images"
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
        edit_slide = self._work(
            deepcopy(self.presentation.slides[template_id]), slide_content, images
        )
        return edit_slide

    def _work(
        self,
        slide: SlidePage,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        pass

    def _generate_outline(self, agent_model: Role):
        outline_file = pjoin(self.config.RUN_DIR, "presentation_outline.json")
        if pexists(outline_file):
            outline = json.load(open(outline_file, "r"))
        else:
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
            outline = self.staffs["planner"](prompt, return_json=True)
            json.dump(
                outline, open(self.outline_file, "w"), ensure_ascii=False, indent=4
            )
        return outline


class PPTAgent(PPTGen):
    def rally(self, agent_model: llms.OPENAI):
        self.hire("planner", agent_model)
        self.hire("agent", agent_model)

    def _work(
        self,
        slide: SlidePage,
        slide_content: str,
        code_executor: CodeExecutor,
        image_info: str,
    ):
        edit_prompt = self.edit_template.render(
            api_documentation=code_executor.get_apis_docs([API_TYPES.PPTAgent]),
            edit_target=slide.to_html(),
            content=self.simple_outline + self.metadata + slide_content,
            images=image_info,
        )
        code_executor.execute_actions(
            edit_prompt,
            actions=self.staffs[0].interact(
                edit_prompt,
            ),
            edit_slide=slide,
        )
        return slide


class PPTCrew(PPTGen):

    def _hire(self):
        self.staffs.append(Role("rephraser"))
        self.staffs.append(Role(""))

    def _generate_pres(
        self,
        agent_model: list[Role],
        config: Config,
        images: dict[str, str],
        num_slides: int,
        doc_json: dict[str, str],
    ):
        pass
