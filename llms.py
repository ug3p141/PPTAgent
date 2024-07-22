from copy import deepcopy
import os
import requests
import torch
import logging
import json
from transformers import AutoModel, AutoTokenizer
import google.generativeai as genai
from model_utils import load_image
from utils import print


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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

    def caption_image(self, image_file: str):
        _, pixel_values = load_image(image_file)
        prompt = "Please describe the image shortly\n<image>"
        response = self.model.chat(
            self.tokenizer,
            pixel_values.to(torch.bfloat16).cuda(),
            prompt,
            self.generation_config,
        )
        logging.info(f"InvernVLImageCaptioner: {response}")
        return response

    def label_image(
        self,
        image_file: str,
        outline: str,
        appear_times: int,
        top_ranges_str: str,
        relative_area: float,
        caption: str,
        **kwargs,
    ):
        # 这里需要设置一下可能
        aspect_ratio, pixel_values = load_image(image_file)
        # 加上图片的aspect ratio
        prompt = (
            self.prompt_head
            + "Input:\n"
            + {
                "image": "<image>",
                "caption": caption,
                "outline": outline,
                "appear_times": appear_times,
                "slide_range": top_ranges_str,
                "aspect_ratio": aspect_ratio,
                "relative_area": relative_area,
            }
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
    def __init__(self) -> None:
        proxy = "http://124.16.138.148:7890"
        os.environ["https_proxy"] = proxy
        os.environ["http_proxy"] = proxy
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.safety_settings = [
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        self.generation_config = genai.GenerationConfig(
            #response_mime_type="application/json"  , response_schema=list[*DATACLASSES]
        )

    def chat(self, content: str) -> str:
        response = self.model.generate_content(
            content,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config,
        )
        assert response.status_code == 200, response.text
        return response.text


class QWEN2:
    def __init__(self) -> None:
        self.api = "http://124.16.138.147:7819/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        self.template_data = {
            "model": "Qwen2-72B-Instruct-GPTQ-Int4",
            "temperature": 0.0,
            "max_tokens": 100,
            "stream": False,
        }

    def chat(self, content: str) -> str:
        data = deepcopy(self.template_data) | {
            "messages": [{"role": "user", "content": content}]
        }
        response = requests.post(self.api, headers=self.headers, data=json.dumps(data))
        assert response.status_code == 200, response.text


if __name__ == "__main__":
    # internvl = InternVL()
    # internvl.label_image("output/images/图片 2.jpg", 2, "1,3", 0.5)
    qwen = QWEN2()
    print(qwen.chat("你是谁"))
    # gemini = Gemini()
    # print(gemini.chat("你是谁"))
