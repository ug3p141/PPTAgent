from collections import defaultdict
from presentation import Presentation
from llms import gemini
import json


class MasterGenerator:
    # 至多八个layout
    def __init__(self, prs: Presentation):
        self.prs = prs
        self.layout_mapping = defaultdict(list)
        # layout index = Function/Content Layout -> layout -> content_types
        #
        function_split_prompt = open("prompts/functional_split.txt").read()
        slides_split = json.loads(
            gemini(function_split_prompt + str(self.prs) + "Output:").strip()
        )
        functional_slides = slides_split["Auxiliary Slides"]
        content_slides = slides_split["Content Slides"]
        layout_analyze_prompt = open("prompts/layout_analyze.txt").read()
        layout_analyze_result = gemini(
            layout_analyze_prompt
            + "\n".join(
                [
                    str(slide)
                    for slide_idx, slide in enumerate(self.prs.slides)
                    if slide_idx in content_slides
                ]
            )
            + "Output:"
        )
        content_layouts = json.loads(layout_analyze_result.strip()) | functional_slides

        for slide_idx, slide in enumerate(prs.slides):
            for layout in prs.prs.slide_layouts:  # 按照modality进行划分
                if slide.slide_layout_name != layout.name:
                    continue
                for k, v in content_layouts.items():
                    if slide_idx in v:
                        layout_name = (
                            f"{k}-{layout.name}:({','.join(slide.get_content_types())})"
                        )

                self.layout_mapping[layout_name].append(slide)

    # 用来对具体元素进行识别
    # is background image的不figure在这一步不输出了吧，因为其实没有意义
    def template_induct(self):
        pass

    def build_mapping(self):
        pass


if __name__ == "__main__":
    prs = Presentation.from_file("resource/中文信息联合党支部2022年述职报告.pptx")
    mg = MasterGenerator(prs)
    slides_split = mg.functional_split()
    functional_slides = slides_split["Auxiliary Slides"]
    content_slides = slides_split["Content Slides"]
    # 将content slides按照布局划分，不要给出bg image
