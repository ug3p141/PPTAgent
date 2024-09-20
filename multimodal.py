import json
from collections import defaultdict

import json_repair
import PIL
from jinja2 import Template

import llms
import utils
from presentation import Picture, Presentation, SlidePage
from utils import Config, pbasename, pexists, pjoin, print


class ImageLabler:
    def __init__(self, presentation: Presentation, app_config: Config = None):
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        self.stats_file = pjoin(app_config.RUN_DIR, "image_stats.json")
        if app_config is None:
            app_config = utils.app_config
        self.app_config = app_config
        self.collect_images()
        if pexists(self.stats_file):
            self.image_stats = json.load(open(self.stats_file, "r"))
            self.apply_stats()
        self.image_stats["resource/pic_placeholder.png"] = {
            "label": "content",
            "caption": "it's a placeholder",
        }

    def apply_stats(self, image_stats: dict = None):
        if image_stats is None:
            image_stats = self.image_stats
        for slide in self.presentation.slides:
            for shape in slide.shape_filter(Picture):
                stats = image_stats[shape.img_path]
                shape.caption = stats["caption"]

    def caption_images(self):
        caption_prompt = open("prompts/image_label/caption.txt").read()
        for image, stats in self.image_stats.items():
            if "caption" not in stats:
                stats["caption"] = llms.caption_model(caption_prompt, image)
                if self.app_config.DEBUG:
                    print(image, ": ", stats["caption"])
        json.dump(
            self.image_stats,
            open(self.stats_file, "w"),
            indent=4,
            ensure_ascii=False,
        )
        return self.image_stats

    def label_images(
        self,
        functional_keys: list[str],
        slide_cluster: dict[str, list[int]],
        images: dict[str, str],
    ):
        template = Template(open("prompts/image_label/vision_cls.txt").read())

        image_labels = defaultdict(lambda: defaultdict(int))
        for layout_name, cluster in slide_cluster.items():
            for slide_idx in cluster:
                slide_images = ""
                if all(
                    [
                        "label" in self.image_stats[shape.img_path]
                        for shape in self.presentation.slides[
                            slide_idx - 1
                        ].shape_filter(Picture)
                    ]
                ):
                    continue
                for shape in self.presentation.slides[slide_idx - 1].shape_filter(
                    Picture
                ):
                    if shape.img_path == "resource/pic_placeholder.png":
                        continue
                    slide_images += f"{pbasename(shape.img_path)}: {self.image_stats[shape.img_path]['caption']}\n"
                    slide_images += f"Appeared {self.image_stats[shape.img_path]['appear_times']} times including slides: {self.image_stats[shape.img_path]['top_ranges_str']}\n"
                    slide_images += f"Coverage Area: {self.image_stats[shape.img_path]['relative_area']:.2f}% of the slide\n"
                    slide_images += "----\n"
                if not "Picture" in layout_name or all(
                    [
                        shape.img_path == "resource/pic_placeholder.png"
                        for shape in self.presentation.slides[
                            slide_idx - 1
                        ].shape_filter(Picture)
                    ]
                ):
                    continue
                slide_type = (
                    "Structural" if layout_name in functional_keys else "Non-Structural"
                )
                results = json_repair.loads(
                    llms.caption_model(
                        template.render(
                            slide_type=slide_type,
                            slide_images=slide_images,
                            provided_images="----\n".join(
                                [f"{pbasename(k)}: {v}" for k, v in images.items()]
                            )
                            + "----",
                        ),
                        pjoin(
                            self.app_config.RUN_DIR,
                            "slide_images",
                            f"slide_{slide_idx:04d}.jpg",
                        ),
                    )
                )
                if not isinstance(results, list):
                    results = [results]
                if "results" in results:
                    results = results["results"]
                for result in results:
                    if result["label"] != "background":
                        continue
                    image_labels[result["image"]][result["replace"]] += 1
        image_labels = {
            pjoin(self.app_config.IMAGE_DIR, k): pjoin(
                self.app_config.IMAGE_DIR,
                max(v.items(), key=lambda x: x[1])[0],
            )
            for k, v in image_labels.items()
        }
        self.apply_labels(image_labels)

    def apply_labels(self, image_labels: dict[str, str]):
        for slide in self.presentation.slides:
            for shape in slide.shape_filter(Picture):
                if shape.img_path in image_labels:
                    shape.is_background = True
                    self.image_stats[shape.img_path]["label"] = "background"
                    self.image_stats[shape.img_path]["replace"] = image_labels[
                        shape.img_path
                    ]
                else:
                    self.image_stats[shape.img_path]["label"] = "content"
        json.dump(
            self.image_stats, open(self.stats_file, "w"), indent=4, ensure_ascii=False
        )
        self.apply_stats()

    def collect_images(self):
        for slide_index, slide in enumerate(self.presentation.slides):
            for shape in slide.shape_filter(Picture):
                image_path = shape.data[0]
                self.image_stats[image_path] = {
                    "appear_times": 0,
                    "slide_numbers": set(),
                    "relative_area": shape.area / self.slide_area * 100,
                    "size": PIL.Image.open(image_path).size,
                }
                self.image_stats[image_path]["appear_times"] += 1
                self.image_stats[image_path]["slide_numbers"].add(slide_index + 1)
        for image_path, stats in self.image_stats.items():
            stats["slide_numbers"] = sorted(list(stats["slide_numbers"]))
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
