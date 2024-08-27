import json
import os
import tempfile
import zipfile
from collections import defaultdict
from copy import deepcopy
from re import M

import requests
from jinja2 import Template
from pdf2image import convert_from_path

from llms import agent_model
from model_utils import get_cluster, image_embedding, images_cosine_similarity
from presentation import Presentation, SlidePage
from utils import app_config, filename_normalize, pexists, pjoin, tenacity


class TemplateInducter:
    def __init__(self, prs: Presentation):
        self.prs = prs
        self.layout_mapping = defaultdict(list)
        for slide in self.prs.slides:
            self.layout_mapping[slide.slide_layout_name].append(slide)
        self.output_dir = pjoin(app_config.RUN_DIR, "template_induct")
        os.makedirs(self.output_dir, exist_ok=True)
        self.slide_split_file = pjoin(self.output_dir, "slides_split.json")
        self.slide_cluster_file = pjoin(self.output_dir, "slides_cluster.json")
        self.template_image_folder = pjoin(self.output_dir, "template_images")
        self.ppt_image_folder = pjoin(self.output_dir, "ppt_images")

    def work(self):
        if pexists(self.slide_cluster_file):
            self.slide_cluster = json.load(open(self.slide_cluster_file))
            return self.slide_cluster
        self.ppt_to_images(self.prs.source_file, self.ppt_image_folder)
        self.slide_cluster = self.category_split()
        self.slide_cluster, content_slides_index = self.slide_cluster[
            "categories"
        ], set(self.slide_cluster["uncategorized"])
        functional_keys = list(self.slide_cluster.keys())
        if len(self.prs.slides) != len(content_slides_index) + sum(
            [len(i) for i in self.slide_cluster.values()]
        ):
            raise ValueError("slides number not match")
        self.layout_split(content_slides_index)
        for layout_name, cluster in self.slide_cluster.items():
            if app_config.DEBUG:
                os.makedirs(
                    pjoin(self.ppt_image_folder, filename_normalize(layout_name)),
                    exist_ok=True,
                )
                for slide_idx in cluster:
                    os.rename(
                        pjoin(
                            self.ppt_image_folder,
                            f"slide_{slide_idx:04d}.jpg",
                        ),
                        pjoin(
                            self.ppt_image_folder,
                            layout_name,
                            f"slide_{slide_idx:04d}.jpg",
                        ),
                    )
            if len(cluster) > 3:
                cluster = sorted(
                    cluster, key=lambda x: len(self.prs.slides[x].to_text())
                )
                self.slide_cluster[layout_name] = [
                    cluster[0],
                    cluster[len(cluster) // 2],
                    cluster[-1],
                ]
        self.slide_cluster["functional_keys"] = functional_keys
        json.dump(
            self.slides_cluster,
            open(self.slide_cluster_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slides_cluster

    def category_split(self):
        topic_split_template = Template(
            open("prompts/template_induct/category_split.txt").read()
        )
        return json.loads(
            agent_model(
                topic_split_template.render(
                    slides="\n----\n".join(
                        [
                            (
                                f"Slide {slide.slide_idx} of {len(self.prs.slides)}\n"
                                + (
                                    f"Title:{slide.slide_title}\n"
                                    if slide.slide_title
                                    else ""
                                )
                                + slide.to_text()
                            )
                            for slide in self.prs.slides
                        ]
                    ),
                )
            )
        )

    def layout_split(
        self,
        content_slides_index: set[int],
    ):
        deepcopy(self.prs).save(pjoin(self.output_dir, "template.pptx"))
        self.ppt_to_images(
            pjoin(self.output_dir, "template.pptx"),
            self.template_image_folder,
        )
        embeddings = image_embedding(self.template_image_folder)
        template = Template(open("prompts/template_induct/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx]
            content_types = slide.get_content_types()
            layout_name = slide.slide_layout_name
            if content_types:
                layout_name += f": ({', '.join(content_types)})"
            content_split[layout_name].append(slide)

        for layout_name, slides in content_split.items():
            if len(slides) < 3:
                continue
            sub_embeddings = [
                embeddings[f"slide_{slide.slide_idx:04d}.jpg"] for slide in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                cluster = [slides[i] for i in cluster]
                cluster_name = agent_model(
                    template.render(
                        existed_layoutnames=list(self.slide_cluster.keys())
                    ),
                    [f"slide_{slide.slide_idx:04d}.jpg" for slide in cluster],
                )
                self.slide_cluster[cluster_name] = [
                    slide.slide_idx for slide in cluster
                ]

    @tenacity
    def ppt_to_images(self, file: str, output_dir: str):
        if not file.endswith(".pptx"):
            raise ValueError("file must be a pptx")
        if pexists(file.replace(".pptx", ".pdf", 1)) and pexists(output_dir):
            return
        with open(file, "rb") as ppt_file:
            response = requests.post(
                app_config.PPT_TO_IMAGES_URL, files={"file": ppt_file}
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
            tmp_zip.write(response.content)
            tmp_zip_path = tmp_zip.name
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)


if __name__ == "__main__":
    prs = Presentation.from_file(app_config.TEST_PPT)
    mg = TemplateInducter(prs)
