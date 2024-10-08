import json
import os
import shutil
from collections import defaultdict

from jinja2 import Template

import llms
from model_utils import get_cluster, image_embedding, images_cosine_similarity
from presentation import Presentation
from utils import Config, get_json_from_response, pexists, pjoin, ppt_to_images


class TemplateInducter:
    def __init__(
        self, prs: Presentation, ppt_image_folder: str, template_image_folder: str, config:Config
    ):
        self.prs = prs
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        self.config = config
        self.output_dir = pjoin(config.RUN_DIR, "template_induct")
        self.slide_split_file = pjoin(self.output_dir, "slides_split.json")
        self.slide_cluster_file = pjoin(self.output_dir, "slides_cluster.json")
        os.makedirs(self.output_dir, exist_ok=True)

    def work(self, most_image = 3):
        if pexists(self.slide_cluster_file):
            self.slide_cluster = json.load(open(self.slide_cluster_file))
            return set(self.slide_cluster.pop("functional_keys")), self.slide_cluster
        if pexists(self.slide_split_file):
            category_cluster = json.load(open(self.slide_split_file))
        else:
            category_cluster = self.category_split()
        if "content" in category_cluster:
            content_slides_index, functional_cluster = (
                set(category_cluster.pop("content")),
                category_cluster,
            )
        elif "Uncategorized" in category_cluster:
            content_slides_index, functional_cluster = (
                set(
                    category_cluster.pop("Uncategorized"),
                ),
                category_cluster["categories"],
            )
        elif "Uncategorized" in category_cluster.get("categories",[]):
            content_slides_index, functional_cluster = (
                set(category_cluster["categories"].pop("Uncategorized")),
                category_cluster["categories"],
            )
        else:
            raise Exception(f"Unknown category cluster: {category_cluster}")
        self.slide_cluster = defaultdict(list)
        for layout_name, cluster in functional_cluster.items():
            for slide_idx in cluster:
                content_types = self.prs.slides[slide_idx - 1].get_content_types()
                content_type_name = (
                    f": ({', '.join(content_types)})"
                    if content_types
                    else ": plain text"
                )
                self.slide_cluster[layout_name + content_type_name].append(slide_idx)

        functional_keys = list(self.slide_cluster.keys())
        function_slides_index = set()
        for layout_name, cluster in self.slide_cluster.items():
            function_slides_index.update(cluster)
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index, most_image)
        if self.config.DEBUG:
            for layout_name, cluster in self.slide_cluster.items():
                os.makedirs(
                    pjoin(
                        self.output_dir,
                        "cluster_slides",
                        layout_name,
                    ),
                    exist_ok=True,
                )
                for slide_idx in cluster:
                    shutil.copy(
                        pjoin(
                            self.ppt_image_folder,
                            f"slide_{slide_idx:04d}.jpg",
                        ),
                        pjoin(
                            self.output_dir,
                            "cluster_slides",
                            layout_name,
                            f"slide_{slide_idx:04d}.jpg",
                        ),
                    )
        self.slide_cluster["functional_keys"] = functional_keys
        json.dump(
            self.slide_cluster,
            open(self.slide_cluster_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return set(self.slide_cluster.pop("functional_keys")), self.slide_cluster

    def category_split(self):
        category_split_template = Template(
            open("prompts/template_induct/category_split.txt").read()
        )
        response = llms.long_model(
            category_split_template.render(
                slides="\n----\n".join(
                    [
                        (
                            f"Slide {slide.slide_idx} of {len(self.prs.slides)}\n"
                            + (
                                f"Title:{slide.slide_title}\n"
                                if slide.slide_title
                                else ""
                            )
                            + f"Layout: {slide.slide_layout_name}\n"
                            + slide.to_text()
                        )
                        for slide in self.prs.slides
                    ]
                ),
            )
        )
        category_cluster = get_json_from_response(response)
        json.dump(
            category_cluster,
            open(self.slide_split_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return category_cluster

    def layout_split(
        self,
        content_slides_index: set[int],
        most_image:int
    ):
        embeddings = image_embedding(self.template_image_folder)
        template = Template(open("prompts/template_induct/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_types = slide.get_content_types()
            layout_name = slide.slide_layout_name
            content_type_name = (
                f": ({', '.join(content_types)})" if content_types else ": plain text"
            )
            content_split[(layout_name, content_type_name)].append(slide_idx)

        for (layout_name, content_type_name), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                cluster = [slides[i] for i in cluster]
                cluster_name = (
                    llms.caption_model(
                        template.render(
                            existed_layoutnames=list(self.slide_cluster.keys()),
                        ),
                        [
                            pjoin(self.ppt_image_folder, f"slide_{slide_idx:04d}.jpg")
                            for slide_idx in cluster[:most_image]
                        ],
                    )
                    + content_type_name
                )
                print(f"cluster_name: {cluster_name}")
                self.slide_cluster[cluster_name] = [slide_idx for slide_idx in cluster]