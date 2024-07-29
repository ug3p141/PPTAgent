import json
import os

from tqdm.auto import tqdm

from llms import caption_image, label_image
from presentation import Picture, Presentation
from utils import app_config, pjoin, print


class ImageLabler:
    def __init__(self, presentation: Presentation, output_dir: str, batchsize: int = 8):
        self.presentation = presentation
        self.slide_area = presentation.slide_width.pt * presentation.slide_height.pt
        self.image_stats = {}
        self.out_file = pjoin(output_dir, "image_stats.json")

    def work(self):
        if os.path.exists(self.out_file):
            self.image_stats = json.load(open(self.out_file, "r"))
        else:
            self.collect_images()
            self.caption_images()
            self.label_images()
            json.dump(self.image_stats, open(self.out_file, "w"), indent=4)
        self.apply_stats()

    def apply_stats(self):
        for slide in self.presentation.slides:
            for shape in slide.shapes:
                if not isinstance(shape, Picture):
                    continue
                stats = self.image_stats[shape.data[0]]
                shape.caption = stats["caption"]
                shape.is_background = "background" == stats["result"]["label"]

    def caption_images(self):
        for image, stats in tqdm(self.image_stats.items()):
            self.image_stats[image]["caption"] = caption_image(image)

    def label_images(self):
        for image, stats in tqdm(self.image_stats.items()):
            self.image_stats[image]["result"] = label_image(image, **stats)

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


if __name__ == "__main__":
    prs = Presentation.from_file(app_config.TEST_PPT)
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
    image_stats = json.load(open("image_stats.json", "r"))
    false_samples = []
    for k, v in tqdm(image_stats.items()):
        output = label_image(image_file=k, outline=None, **v)
        image_stats[k]["result"] = output
        if output["label"] != LABEL[ground_truth[k]]:
            false_samples.append([k, v, output])
        json.dump(image_stats, open("image_stats.json", "w"))

    print(
        f"Accuracy of {label_image.__name__}: {100*(1-len(false_samples)/len(image_stats))}%"
    )
