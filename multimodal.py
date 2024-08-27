import json
import os

from jinja2 import Template
from tqdm.auto import tqdm

from llms import api_model, caption_model
from presentation import Picture, Presentation, SlidePage
from utils import app_config, pexists, pjoin, print


class ImageLabler:
    def __init__(self, presentation: Presentation):
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        self.stats_file = pjoin(app_config.RUN_DIR, "image_stats.json")
        self.collect_images()
        if pexists(self.stats_file):
            self.image_stats = json.load(open(self.stats_file, "r"))
        os.makedirs(pjoin(app_config.RUN_DIR, "images", "background"), exist_ok=True)
        os.makedirs(pjoin(app_config.RUN_DIR, "images", "content"), exist_ok=True)

    def apply_stats(self):
        json.dump(
            self.image_stats, open(self.stats_file, "w"), indent=4, ensure_ascii=False
        )
        for slide in self.presentation.slides:
            for shape in slide.shape_filter(Picture):
                stats = self.image_stats[shape.data[0]]
                if "caption" in stats:
                    shape.caption = stats["caption"]
                if "result" in stats:
                    shape.is_background = "background" == stats["result"]["label"]
                    shape.data[0] = pjoin(
                        app_config.RUN_DIR, stats["result"]["label"], shape.data[0]
                    )

    def caption_images(self):
        caption_prompt = open("prompts/image_label/caption.txt").read()
        for image, stats in tqdm(self.image_stats.items()):
            if "caption" not in stats:
                stats["caption"] = caption_model(caption_prompt, image)
                if app_config.DEBUG:
                    print(image, ": ", stats["caption"])
        self.apply_stats()

    def label_images(self, slide_cluster: dict[str, list[int]], images: dict[str, str]):
        template = Template(open("prompts/image_label/vision_cls.txt").read())
        image_labels = {
            "replace": [],
            "background": [],
        }
        for slide_idxs in slide_cluster.values():
            for slide_idx in slide_idxs:
                slide = self.presentation.slides[slide_idx]
                if not "Picture" in slide.get_content_types():
                    continue
                self.apply_labels(image_labels, slide)
                result = json.loads(
                    api_model(template.render(slide=slide, images=images))
                )
                for key, value in result.items():
                    image_labels[key].extend(value)
                self.apply_labels(result, slide)
        self.apply_stats()

    def apply_labels(self, image_labels: dict[str, str], slide: SlidePage):
        for shape in slide.shape_filter(Picture):
            if shape.img_path in image_labels["replace"]:
                shape.img_path = image_labels[shape.img_path]
                shape.is_background = True
            elif shape.img_path in image_labels["background"]:
                shape.is_background = True

    def collect_images(self):
        for slide_index, slide in enumerate(self.presentation.slides):
            for shape in slide.shape_filter(Picture):
                image_path = shape.data[0]
                image_path = shape.data[0]
                if image_path not in self.image_stats:
                    self.image_stats[image_path] = {
                        "appear_times": 0,
                        "slide_numbers": [],
                        "relative_area": shape.area / self.slide_area * 100,
                    }
                self.image_stats[image_path]["appear_times"] += 1
                self.image_stats[image_path]["slide_numbers"].append(slide_index + 1)
        for image_path, stats in self.image_stats.items():
            ranges = self._find_ranges(stats["slide_numbers"])
            top_ranges = sorted(ranges, key=lambda x: x[1] - x[0], reverse=True)[:3]
            top_ranges_str = ", ".join(
                [f"{r[0]}-{r[1]}" if r[0] != r[1] else f"{r[0]}" for r in top_ranges]
            )
            stats["top_ranges_str"] = top_ranges_str

    def _find_ranges(self, numbers):
        ranges = []
        start = numbers[0]
        end = numbers[0]
        for num in numbers[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append((start, end))
                start = num
                end = num
        ranges.append((start, end))
        return ranges
