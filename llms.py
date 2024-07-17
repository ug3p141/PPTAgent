import torch
import torchvision.transforms as T
import logging
import json
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


# class LLM(metaclass=SingletonMeta):
#     @classmethod
#     def from_api(self, text, device_map: dict = None):
#         raise NotImplementedError
#     @classmethod
#     def from_local(self, text, device_map: dict = None):
#         raise NotImplementedError


class InternVL(metaclass=SingletonMeta):
    def __init__(self, model_id="OpenGVLab/InternVL2-8B", device_map: dict = None):
        if device_map is None:
            device_map = {"": 0}
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        ).eval()
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.prompt_head = open("resource/prompt_image.txt").read()

    def label_image(
        self,
        image_file: str,
        appear_times: int,
        top_ranges_str: str,
        relative_area: float,
        **kwargs,
    ):
        # 这里需要设置一下可能
        aspect_ratio, pixel_values = load_image(image_file)
        # 加上图片的aspect ratio
        prompt = (
            self.prompt_head
            + f"""Input:
- <image>
- Appear times: {appear_times}
- Range of slide numbers: {top_ranges_str}
- Aspect ratio: {aspect_ratio} 
- The area of the image relative to the total slide area: {relative_area:.2f}%
Output:"""
        )
        response = self.model.chat(
            self.tokenizer,
            pixel_values.to(torch.bfloat16).cuda(),
            prompt,
            self.generation_config,
        )
        logging.info(f"InvernVLImageLabeler: {response}")
        return json.loads(response)


class Gemini:
    pass


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return target_aspect_ratio, processed_images


def load_image(image_file, input_size=448, max_num=16):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    target_aspect_ratio, images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return target_aspect_ratio, pixel_values


if __name__ == "__main__":
    model = InternVL()
    model.label_image("output/images/图片 2.jpg", 2, "1,3", 0.5)
