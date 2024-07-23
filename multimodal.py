import json
from presentation import Presentation, Picture
from utils import base_config, pjoin, print
from tqdm.auto import tqdm
from llms import (
    get_prs_outline,
    label_image_withcap,
    label_image_withcap_outline,
    label_image_withgemini_image,
    label_image_withoutcap,
    label_image_withqwen,
    label_image_withslide,
)


# TODO: layout中的背景图片需要识别吗，感觉不需要了
class ImageLabler:
    def __init__(self, presentation: Presentation):
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        # self.llm = InternVL()
        self.collect_images()
        self.image_stats = json.load(open("resource/image_stats.json", "r"))
        self.caption_images()
        self.outline = get_prs_outline(self.presentation)
        # self.label_images()
        # print(self.outline, self.image_stats)

    def caption_images(self, batch_size=1):
        # for image, stats in tqdm(self.image_stats.items()):
        # stats["caption"] = self.llm.caption_image(image)
        for slide in self.presentation.slides:
            for shape in slide.shapes:
                if not isinstance(shape, Picture):
                    continue
                image_path = shape.data[0]
                stats = self.image_stats[image_path]
                shape.caption = stats["caption"]

    def label_images(self, batch_size=1):
        for image, stats in tqdm(self.image_stats.items()):
            stats["result"] = self.llm.label_image(image, self.outline, **stats)
        for slide in self.presentation.slides:
            for shape in slide.shapes:
                if not isinstance(shape, Picture):
                    continue
                image_path = shape.data[0]
                stats = self.image_stats[image_path]
                shape.is_background = "background" in stats["result"]["label"]

    def collect_images(self):
        for slide_index, slide in enumerate(self.presentation.slides):
            for shape in slide.shapes:
                if not isinstance(shape, Picture):
                    continue
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


# 三种方法的acc对比
# 1. 直接使用intern-vl，无caption仅few-shot，约75左右
# 2. 先用intern-vl做caption，再用qwen做分类，约
# 3. 用intern-vl做caption，再用intern-vl做分类，约
# 4. 不传入image

if __name__ == "__main__":
    prs = Presentation.from_file(
        pjoin(
            base_config.PPT_DIR,
            "中文信息联合党支部2022年述职报告.pptx",
        )
    )
    LABEL = ["background", "content"]
    ground_truth = {
        "./output/images/图片 26.jpg": 0,
        "./output/images/图片 33.png": 0,
        "./output/images/图片 30.png": 0,
        "./output/images/图片 23.png": 0,
        "./output/images/图片 22.png": 0,
        "./output/images/图片 7.png": 0,
        "./output/images/图片 6.jpg": 1,
        "./output/images/图片 7.jpg": 1,
        "./output/images/图片 8.jpg": 1,
        "./output/images/图片 9.jpg": 1,
        "./output/images/Picture 2.jpg": 1,
        "./output/images/图片 9.png": 1,
        "./output/images/图片 15.png": 1,
        "./output/images/图片 21.png": 1,
        "./output/images/图片 2.png": 1,
        "./output/images/图片 2.jpg": 0,
    }
    image_stats = json.load(open("resource/image_stats.json", "r"))
    funcs = [
        label_image_withgemini_image,
        label_image_withcap,
        label_image_withoutcap,
        label_image_withqwen,
        label_image_withcap_outline,
    ]
    outline = get_prs_outline(prs)
    for func in funcs:
        false_samples = []
        for k, v in image_stats.items():
            if func(image_file=k, outline=outline, **v) != LABEL[ground_truth[k]]:
                false_samples.append([k, v])
        print(
            f"Accuracy of {func.__name__}: {100*(1-len(false_samples)/len(image_stats))}%"
        )
