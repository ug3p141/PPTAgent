import json
import os
from collections import defaultdict

import numpy as np
import requests
from pdf2image import convert_from_path
from tenacity import retry, stop_after_attempt, wait_fixed

from llms import agent_model
from model_utils import fid_score, get_cluster
from presentation import Presentation, SlidePage
from utils import app_config, pexists, pjoin


class TemplateInducter:
    # 至多八个layout
    def __init__(self, prs: Presentation):
        self.prs = prs
        self.layout_mapping = defaultdict(list)
        for slide in self.prs.slides:
            self.layout_mapping[slide.slide_layout_name].append(slide)
        output_dir = pjoin(app_config.RUN_DIR, "template_induct")
        self.slide_split_file = pjoin(output_dir, "slides_split.json")
        self.slide_cluster_file = pjoin(output_dir, "slides_cluster.json")
        self.similarity_file = pjoin(output_dir, "similarity.json")
        self.template_pre = pjoin(output_dir, "template.pptx")
        self.template_pdf = pjoin(output_dir, "template.pdf")
        self.template_image_folder = pjoin(output_dir, "template_images")

    # 每次发送一个layout 可能会更好 应该只有字数比较少的才会是 functional
    # functional/content->clusters
    def work(self):
        if pexists(self.slide_split_file):
            self.slides_cluster = json.load(open(self.slide_split_file))
        else:
            self.slides_cluster = self.functional_split()
            json.dump(
                self.slides_split,
                open(self.slide_split_file, "w"),
                indent=4,
                ensure_ascii=False,
            )
        content_slides_index = list(map(int, self.slides_split["content"]))
        content_slides = [
            slide
            for slide in self.prs.slides
            if slide.slide_idx in content_slides_index
        ]
        # TODO 每个cluster最多只接受3个slides和至多的cluster
        if pexists(self.similarity_file):
            similarity = np.array(json.load(open(self.similarity_file)))
        else:
            similarity = self.calc_similarity(content_slides)
            json.dump(
                similarity.tolist(),
                open(self.similarity_file, "w"),
                indent=4,
                ensure_ascii=False,
            )
        if pexists(self.slide_cluster_file):
            return json.load(open(self.slide_cluster_file))
        content_cluster = self.layout_split(
            content_slides_index, content_slides, similarity
        )
        self.slides_cluster = {
            "content": content_cluster,
            "functional": self.slides_split["functional"],
        }
        json.dump(
            self.slides_cluster,
            open(self.slide_cluster_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.slides_cluster

    # 200k
    def functional_split(self):
        self.slide_custer = {"functional": {}, "content": []}
        function_split_prompt = open(
            "prompts/template_induct/functional_split.txt"
        ).read()
        for slides in self.layout_mapping.values():
            split = json.loads(
                agent_model(
                    function_split_prompt
                    + "\n".join(
                        [
                            f"slide {slide.slide_idx}/{len(self.prs.slides)}: \n"
                            + slide.to_text()
                            + "----"
                            for slide in slides
                        ]
                    )
                    + "Output:"
                ).strip()
            )
            self.slide_custer["functional"] |= split["functional"]
            self.slide_custer["content"].extend(split["content"])

    def layout_split(
        self,
        content_slides_index: list[int],
        content_slides: list[SlidePage],
        similarity: np.ndarray,
    ):
        content_split = defaultdict(list)
        for slide in content_slides:
            content_types = slide.get_content_types()
            layout_name = slide.slide_layout_name
            if content_types:
                layout_name += f": ({', '.join(content_types)})"
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

        # textframe_tags = ["幻灯片标题", "小节标题", "标题", "固定文本"]
        # background_tags = ["固定文本"]
        # content layouts 的命名
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
        template_pr = Presentation(
            content_slides,
            self.prs.slide_width,
            self.prs.slide_height,
            self.prs.source_file,
            len(content_slides),
        )
        template_pr.save(self.template_pre)
        self.ppt_to_images(len(content_slides))
        similarity = fid_score(self.template_image_folder)

        return similarity

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(6),
    )
    def ppt_to_images(self):
        if not pexists(self.template_pdf):
            response = requests.request(
                "POST",
                "https://api.pspdfkit.com/build",
                headers={
                    "Authorization": "Bearer pdf_live_wejPGyz2D4vyYKtvK7JgEKQ0rHCfT8QrEf2qFFYzvBv"
                },
                files={"file": open(self.template_pre, "rb")},
                data={"instructions": json.dumps({"parts": [{"file": "file"}]})},
                stream=True,
            )

            assert response.ok
            with open("result.pdf", "wb") as fd:
                for chunk in response.iter_content(chunk_size=8096):
                    fd.write(chunk)
        os.makedirs(self.template_image_folder, exist_ok=True)
        pages = convert_from_path(self.template_pdf, 224)
        for i, page in enumerate(pages):
            page.save(f"{self.template_image_folder}/page_{(i+1):04d}.jpg", "JPEG")


if __name__ == "__main__":
    prs = Presentation.from_file(app_config.TEST_PPT)
    mg = TemplateInducter(prs)
