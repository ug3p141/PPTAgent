import json
import os
from collections import defaultdict

import numpy as np
import requests
from pdf2image import convert_from_path
from tenacity import retry, stop_after_attempt, wait_fixed

from llms import functional_split
from model_utils import fid_score, get_cluster
from presentation import Presentation, SlidePage
from utils import app_config, pexists, pjoin


class TemplateCluster:
    # 至多八个layout
    def __init__(self, prs: Presentation, output_dir: str):
        self.prs = prs
        self.layout_mapping = defaultdict(list)
        self.slide_split_file = pjoin(output_dir, "slides_split.json")
        self.slide_cluster_file = pjoin(output_dir, "slides_cluster.json")
        self.similarity_file = pjoin(output_dir, "similarity.json")
        self.template_pdf = pjoin(output_dir, "template.pdf")
        self.template_image_folder = pjoin(output_dir, "template_images")

    def work(self):
        if pexists(self.slide_cluster_file):
            return json.load(open(self.template_file))
        if pexists(self.slide_split_file):
            self.slides_split = json.load(open(self.slide_split_file))
        else:
            self.slides_split = functional_split(self.prs.to_html())
        functional_slides = self.slides_split["Auxiliary Slides"]
        content_slides_index = self.slides_split["Content Slides"]
        content_slides = [
            slide
            for slide in self.prs.slides
            if slide.slide_idx in content_slides_index
        ]
        similarity = self.calc_similarity(content_slides)
        content_cluster = self.layout_split(
            content_slides_index, content_slides, similarity
        )
        slides_cluster = {"content": content_cluster, "functional": functional_slides}
        json.dump(slides_cluster, open(self.slide_split_file, "w"), indent=4)
        return slides_cluster

    def layout_split(
        self,
        content_slides_index: list[int],
        content_slides: list[SlidePage],
        similarity: np.ndarray,
    ):
        content_split = defaultdict(list)
        for slide in content_slides:
            # TODO image和textframe 的数量 modality = json.dumps(slide.get_content_types())
            layout_name = (
                f"{slide.slide_layout_name}:({','.join(slide.get_content_types())})"
            )
            content_split[layout_name].append(slide.slide_idx)

        clusters = dict()
        for layout_name, slides_idx in content_split.items():
            sim_index = [content_slides_index.index(i) for i in slides_idx]
            sub_similarity = similarity[np.ix_(sim_index, sim_index)]
            sub_clusters = get_cluster(sub_similarity)
            # text frame 或者图片数量不同似乎就该不是同一个template
            # 不过现在这个版本也能将就用，先用着吧
            clusters[layout_name] = [
                sorted([slides_idx[i] for i in cluster]) for cluster in sub_clusters
            ]

        # text class的类型标注
        return clusters

        # layout_analyze_prompt = open("prompts/layout_analyze.txt").read()
        # layout_analyze_result = gemini(
        #     layout_analyze_prompt
        #     + prs.to_html(content_slides) + "Output:"
        # )
        # content_layouts = json.loads(layout_analyze_result.strip()) | functional_slides
        # assert sum(len(i) for i in content_layouts.values()) == len(prs.slides)
        # # 记得 -1
        # for slide_idx, slide in enumerate(prs.slides):
        #     for layout in prs.prs.slide_layouts:  # 按照modality进行划分
        #         if slide.slide_layout_name != layout.name:
        #             continue
        #         for k, v in content_layouts.items():
        #             if slide_idx in v:
        #                 layout_name = (
        #                     f"{k}-{layout.name}:({','.join(slide.get_content_types())})"
        #                 )

        #         self.layout_mapping[layout_name].append(slide)

    # 用来对具体元素进行识别
    # is background image的不figure在这一步不输出了吧，因为其实没有意义
    def calc_similarity(self, content_slides: list[SlidePage]):
        if pexists(self.similarity_file):
            return np.array(json.load(open(self.similarity_file)))
        template_pr = Presentation(
            content_slides,
            self.prs.slide_width,
            self.prs.slide_height,
            self.prs.source_file,
            len(content_slides),
        )
        template_pr.save(f"{app_config.RUN_DIR}/template.pptx")
        self.ppt_to_images(len(content_slides))
        similarity = fid_score(self.template_image_folder)
        json.dump(similarity.tolist(), open(self.similarity_file, "w"), indent=4)
        return similarity

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(6),
    )
    def ppt_to_images(self, num_pages: int):
        if not pexists(self.template_pdf):
            response = requests.request(
                "POST",
                "https://api.pspdfkit.com/build",
                headers={
                    "Authorization": "Bearer pdf_live_wejPGyz2D4vyYKtvK7JgEKQ0rHCfT8QrEf2qFFYzvBv"
                },
                files={
                    "file": open(
                        "./resource/中文信息联合党支部2022年述职报告.pptx", "rb"
                    )
                },
                data={"instructions": json.dumps({"parts": [{"file": "file"}]})},
                stream=True,
            )

            assert response.ok
            with open("result.pdf", "wb") as fd:
                for chunk in response.iter_content(chunk_size=8096):
                    fd.write(chunk)
        if not pexists(self.template_image_folder):
            os.mkdir(self.template_image_folder)
        pages = convert_from_path(self.template_pdf, 224)
        for i, page in enumerate(pages):
            page.save(f"{self.template_image_folder}/page_{(i+1):04d}.jpg", "JPEG")


if __name__ == "__main__":
    prs = Presentation.from_file(app_config.TEST_PPT)
    mg = TemplateCluster(prs)
    # slides_split = mg.functional_split()
    # functional_slides = slides_split["Auxiliary Slides"]
    # content_slides = slides_split["Content Slides"]
    # 将content slides按照布局划分，不要给出bg image
