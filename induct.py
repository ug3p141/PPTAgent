import json
import os
import shutil
from collections import defaultdict

from jinja2 import Template

import llms
from model_utils import get_cluster, image_embedding, images_cosine_similarity
from presentation import Presentation
from utils import Config, pexists, pjoin, tenacity


class SlideInducter:
    def __init__(
        self,
        prs: Presentation,
        ppt_image_folder: str,
        template_image_folder: str,
        config: Config,
        image_models: list,
    ):
        self.prs = prs
        self.config = config
        self.slide_induction = defaultdict(lambda: defaultdict(list))
        self.ppt_image_folder = ppt_image_folder
        self.template_image_folder = template_image_folder
        self.image_models = image_models
        self.output_dir = pjoin(config.RUN_DIR)
        model_identifier = "+".join(
            (
                llms.language_model.model.split("-")[0],
                llms.vision_model.model.split("-")[0],
            )
        )
        self.induct_cache = pjoin(
            self.output_dir, f"induct_cache-{model_identifier}.json"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def layout_induct(
        self,
    ):
        if pexists(self.induct_cache):
            return json.load(open(self.induct_cache))
        content_slides_index, functional_cluster = self.category_split()
        for layout_name, cluster in functional_cluster.items():
            for slide_idx in cluster:
                content_types = self.prs.slides[slide_idx - 1].get_content_types()
                content_type_name = (
                    f":({'+'.join(content_types)})" if content_types else ":plain text"
                )
                self.slide_induction[layout_name + content_type_name]["slides"].append(
                    slide_idx
                )
        for layout_name, cluster in self.slide_induction.items():
            cluster["template_id"] = cluster["slides"][-1]

        functional_keys = list(self.slide_induction.keys())
        function_slides_index = set()
        for layout_name, cluster in self.slide_induction.items():
            function_slides_index.update(cluster["slides"])
        used_slides_index = function_slides_index.union(content_slides_index)
        for i in range(len(self.prs.slides)):
            if i + 1 not in used_slides_index:
                content_slides_index.add(i + 1)
        self.layout_split(content_slides_index)
        if self.config.DEBUG:
            for layout_name, cluster in self.slide_induction.items():
                os.makedirs(
                    pjoin(
                        self.output_dir,
                        "cluster_slides",
                        layout_name,
                    ),
                    exist_ok=True,
                )
                for slide_idx in cluster["slides"]:
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
        self.slide_induction["functional_keys"] = functional_keys
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction

    def category_split(self):
        category_split_template = Template(open("prompts/category_split.txt").read())
        category_cluster = llms.language_model(
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
            ),
            return_json=True,
        )
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
        elif "Uncategorized" in category_cluster.get("categories", []):
            content_slides_index, functional_cluster = (
                set(category_cluster["categories"].pop("Uncategorized")),
                category_cluster["categories"],
            )
        else:
            raise Exception(f"Unknown category cluster: {category_cluster}")
        return content_slides_index, functional_cluster

    def layout_split(self, content_slides_index: set[int]):
        embeddings = image_embedding(self.template_image_folder, *self.image_models)
        assert len(embeddings) == len(self.prs)
        template = Template(open("prompts/ask_category.txt").read())
        content_split = defaultdict(list)
        for slide_idx in content_slides_index:
            slide = self.prs.slides[slide_idx - 1]
            content_types = slide.get_content_types()
            layout_name = slide.slide_layout_name
            content_type_name = (
                f":({'+'.join(content_types)})" if content_types else ":plain text"
            )
            content_split[(layout_name, content_type_name)].append(slide_idx)

        for (layout_name, content_type_name), slides in content_split.items():
            sub_embeddings = [
                embeddings[f"slide_{slide_idx:04d}.jpg"] for slide_idx in slides
            ]
            similarity = images_cosine_similarity(sub_embeddings)
            for cluster in get_cluster(similarity):
                slide_indexs = [slides[i] for i in cluster]
                template_id = max(
                    slide_indexs,
                    key=lambda x: len(self.prs.slides[x - 1].shapes),
                )
                cluster_name = (
                    llms.vision_model(
                        template.render(
                            existed_layoutnames=list(self.slide_induction.keys()),
                        ),
                        pjoin(self.ppt_image_folder, f"slide_{template_id:04d}.jpg"),
                    )
                    + content_type_name
                )
                self.slide_induction[cluster_name]["template_id"] = template_id
                self.slide_induction[cluster_name]["slides"] = slide_indexs

    @tenacity
    def content_induct(self):
        self.slide_induction = self.layout_induct()
        content_induct_prompt = open("prompts/content_induct.txt").read()
        for layout_name, cluster in self.slide_induction.items():
            if "content_schema" not in cluster and "template_id" in cluster:
                self.slide_induction[layout_name]["content_schema"] = llms.vision_model(
                    content_induct_prompt,
                    images=pjoin(
                        self.ppt_image_folder,
                        f"slide_{cluster['template_id']:04d}.jpg",
                    ),
                    return_json=True,
                )
        json.dump(
            self.slide_induction,
            open(self.induct_cache, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slide_induction
