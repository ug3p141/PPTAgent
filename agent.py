import json
import pickle
from collections import defaultdict
from copy import deepcopy

from jinja2 import Template

import apis
from apis import API_TYPES, model_api
from llms import api_model, long_model
from presentation import Picture, Presentation, SlidePage
from utils import app_config, pexists, pjoin


# 设置合适的shape数量，看看需不需要增加或者删除,有无重点元素
# TODO 生成一个页面的描述符，描述一下字数，行数，页面类型，图片等节省token
# TODO prompt改成首行是system_prompt
# TODO 改成chathistory
# TODO MLLM 作为agent对生成内容进行collate chat session
class PPTAgent:
    def __init__(
        self,
        presentation: Presentation,
        template: dict,
        markdown_str: str,
        images: list[dict[str, str]],
        num_pages: int,
    ):
        self.presentation = presentation
        self.gen_prs = deepcopy(presentation)
        self.gen_prs.slides = []
        self.slide_templates = template
        self.markdown_str = markdown_str
        self.num_pages = num_pages
        self.images = images
        output_dir = pjoin(app_config.RUN_DIR, "agent")
        self.refined_doc = pjoin(output_dir, "refined_doc.json")
        self.outline_file = pjoin(output_dir, "slide_outline.json")
        self.step_result_file = pjoin(output_dir, "step_result.json")
        self.gen_prs_file = pjoin(output_dir, "generated_presentation.pptx")

    def work(self):
        if pexists(self.refined_doc):
            self.doc_json = json.load(open(self.refined_doc, "r"))
        else:
            self.doc_json = self.refine_doc()
            json.dump(
                self.doc_json, open(self.refined_doc, "w"), ensure_ascii=False, indent=4
            )

        if pexists(self.outline_file):
            self.outline = json.load(open(self.outline_file, "r"))
        else:
            self.outline = self.generate_outline()
            json.dump(
                self.outline, open(self.outline_file, "w"), ensure_ascii=False, indent=4
            )

        stylized_outline = ""
        for slide_idx, slide_title in enumerate(self.outline):
            slide = self.outline[slide_title]
            stylized_outline += f"Slide {slide_idx+1}: {slide_title}\nDescription: {slide['description']}\nKeyPoints: {slide['metadata_keys']+ slide['subsection_keys']}\n\n"
        self.stylized_outline = stylized_outline
        self.generate_slides()

    def generate_slides(self, use_cache=True):
        content_template = Template(open("prompts/agent/content_replacing.txt").read())
        layout_template = Template(open("prompts/agent/layout_adjust.txt").read())
        select_template = Template(open("prompts/agent/layout_select.txt").read())
        step_results = {
            "generated_slides": [],
            "layout_usage": defaultdict(int),
            "api": model_api.history,
        }
        if use_cache and pexists(self.step_result_file):
            step_results = pickle.load(open(self.step_result_file, "rb"))

        for slide_idx, slide_title in enumerate(
            list(self.outline.keys())[len(step_results["generated_slides"]) :]
        ):
            slide = self.outline[slide_title]
            slide_content = self.get_slide_content(slide_idx, slide_title, slide)

            # layout selection
            if slide_title in self.slide_templates["functional"]:
                template_ids = self.slide_templates["functional"][slide_title]["slides"]
            else:
                option_clusters = []
                for key in self.slide_templates["content"]:
                    if ("Picture" in key) == (len(slide["images"]) != 0):
                        option_clusters.extend(
                            [
                                (key, layouts)
                                for layouts in self.slide_templates["content"][key]
                            ]
                        )
                if len(option_clusters) == 1:
                    template_ids = option_clusters[0][1]
                elif len(option_clusters) == 0:
                    raise ValueError(
                        f"No suitable template found for slide {slide_title}"
                    )
                else:
                    select_prompt = select_template.render(
                        template_html_code="\n".join(
                            [
                                f"{cluster[0]}: Layout-{cluster[1][0]}\n----\n"
                                + self.presentation.slides[cluster[1][0]].to_html()
                                for cluster in option_clusters
                            ]
                        ),
                        slide_outline=self.stylized_outline,
                        slide_content=slide_content,
                    )
                    layout = api_model(select_prompt).split("-")[1]
                    for cluster in option_clusters:
                        if layout in cluster[1]:
                            template_ids = cluster[1]
                            break
            step_results["layout_usage"][template_ids[0]] += 1
            apis.template_slides = [
                self.presentation.slides[int(i)] for i in template_ids
            ]

            # layout adjustment
            layout_prompt = layout_template.render(
                api_documentation=model_api.get_apis_docs([API_TYPES.LAYOUT_ADJUST]),
                template_html_code="\n".join(
                    [
                        f"Template {idx}\n----\n" + i.to_html()
                        for idx, i in enumerate(apis.template_slides)
                    ]
                ),
                slide_outline=self.stylized_outline,
                slide_content=slide_content,
            )
            model_api.execute_apis(layout_prompt, api_model(layout_prompt))
            slide_template = apis.slide

            # content replacement
            shape_idxs = self.get_replace_ids(slide_template)
            content_prompt = content_template.render(
                api_documentation=model_api.get_apis_docs([API_TYPES.SET_CONTENT]),
                template_html_code=slide_template.to_html(),
                slide_outline=self.stylized_outline,
                slide_content=slide_content,
                element_ids=shape_idxs,
            )
            model_api.execute_apis(content_prompt, api_model(content_prompt))

            # result saving
            step_results["generated_slides"].append(apis.slide)
            pickle.dump(step_results, open(self.step_result_file, "wb"))
            self.gen_prs.slides = step_results["generated_slides"]
            self.gen_prs.save(self.gen_prs_file)

    def generate_outline(self):
        template = Template(open("prompts/agent/slide_outline.txt").read())
        prompt = template.render(
            num_pages=self.num_pages,
            functional_pages={
                k: v.get("metadata", None)
                for k, v in self.slide_templates["functional"].items()
            },
            json_content=self.doc_json,
            images=self.images,
        )
        return long_model(prompt)

    def get_replace_ids(self, slide: SlidePage):
        shape_idxs = []
        for shape in slide.shapes:
            if isinstance(shape, Picture) and not shape.is_background:
                shape_idxs.append(shape.shape_idx)
            elif shape.text_frame.is_textframe:
                for paragraph in shape.text_frame.data:
                    if paragraph["text"]:
                        shape_idxs.append(f"{shape.shape_idx}_{paragraph['idx']}")
        return shape_idxs

    def get_slide_content(self, slide_idx: int, slide_title: str, slide: dict):
        slide_content = f"Slide {slide_idx+1} {slide_title}\nSlide Description: {slide['description']}\nSlide Content: "
        for key in slide["metadata_keys"]:
            slide_content += f"{key}: {self.doc_json['metadata'][key]}\n"
        for key in slide["subsection_keys"]:
            for section in self.doc_json["sections"]:
                for subsection in section["subsections"]:
                    if key in subsection:
                        slide_content += f"SubSection {key}: {subsection[key]}\n"
        if slide["images"]:
            slide_content += "Images: "
            for image_path in slide["images"]:
                slide_content += f"{image_path}: {self.images[image_path]}\n"
        return slide_content

    def refine_doc(self):
        template = Template(open("prompts/agent/document_refine.txt").read())
        prompt = template.render(markdown_document=self.markdown_str)
        return long_model(prompt, use_json=False)


if __name__ == "__main__":
    prs = Presentation(app_config.TEST_PPT)
