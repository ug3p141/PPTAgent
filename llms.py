import base64
import json
import random
from time import sleep, time

import google.generativeai as genai
import PIL
import torch
from jinja2 import Template
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed
from transformers import AutoModel, AutoTokenizer

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

    def __init__(self, model_id="models/InternVL2-8B", device_map: dict = None):
        self._initialized = False
        self._model_id = model_id
        self._device_map = device_map if device_map is not None else "auto"

    def _initialize(self):
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self._model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self._device_map,
        ).eval()
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=10240,
            do_sample=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            trust_remote_code=True,
        )
        self._initialized = True

    def __call__(self, pixel_values: torch.Tensor, prompt: str):
        if not self._initialized:
            self._initialize()
        return self.model.chat(
            self.tokenizer,
            pixel_values.to(torch.bfloat16).cuda(),
            prompt,
            self.generation_config,
        )


class Gemini:
    def __init__(self, time_limit: int = 60) -> None:
        self.time_limit = time_limit
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        self.api_config = (
            [
                "AIzaSyBks5-bQ96-uyhvIOCtoPToVqKIdl4szcQ",
                "AIzaSyAQSqXqih0qx0E0U5eUWz1WI_-pt4LjpYk",
                "AIzaSyBA7Bd1jyVePHOKVFmhM7zA1RSxvvIoCS0",
                "AIzaSyD8-BuQYK5njOKBBkeIKhjEaoYmGkjoA84",
            ],
            [0, 0, 0, 0],
        )
        self._call_idx = random.randint(0, len(self.api_config[0]) - 1)
        self.generation_config = genai.GenerationConfig()

    def start_chart(self):
        self._chat = self.model.start_chat(history=[])

    def prepare(self):
        self._call_idx = (self._call_idx + 1) % len(self.api_config[0])
        self.use_apikey = self.api_config[0][self._call_idx]
        genai.configure(api_key=self.use_apikey)
        call_time = time()
        if call_time - self.api_config[1][self._call_idx] < self.time_limit:
            sleep(self.time_limit - (call_time - self.api_config[1][self._call_idx]))
        self.api_config[1][self._call_idx] = call_time

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(6),
    )
    def __call__(
        self, content: str, image_file: str = None, use_json: bool = True
    ) -> dict:
        if use_json:
            self.generation_config.response_mime_type = "application/json"
        else:
            self.generation_config.response_mime_type = "text/plain"
        self.prepare()
        if image_file is not None:
            content = [content, PIL.Image.open(image_file)]
        response = self.model.generate_content(
            content,
            safety_settings=self.safety_settings,
            generation_config=self.generation_config,
        )
        if use_json:
            return json.loads(response.text.strip())
        else:
            return response.text.strip()

    def chat(self, content: str, image_file: str = None) -> str:
        self.prepare()
        if image_file is not None:
            content = [content, PIL.Image.open(image_file)]
        response = self._chat.send_message(content)
        return json.loads(response.text.strip())


class OPENAI:
    def __init__(self, model: str = "gpt-4o-2024-08-06") -> None:
        self.client = OpenAI()
        self.model = model

    @retry(
        wait=wait_exponential(min=1, max=8),
        stop=stop_after_attempt(3),
    )
    def __call__(
        self,
        content: str,
        image_file: str = None,
        system_message: str = None,
        chat_history: list = None,
    ) -> dict:
        messages = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        if chat_history is not None:
            messages = chat_history + messages
        elif system_message is not None:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            )
        if image_file is not None:
            with open(image_file, "rb") as image:
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(image.read()).decode('utf-8')}"
                        },
                    }
                )
        completion = self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return completion["choices"][0]["message"]


gemini = Gemini()
intern = InternVL()
gpt4o = OPENAI()
vl_model = intern
long_model = gemini
api_model = gpt4o


def caption_image(image_file: str):
    _, pixel_values = load_image(image_file)
    prompt = open("prompts/caption.txt").read()
    return vl_model(pixel_values.to(torch.bfloat16).cuda(), prompt)


def label_image(
    image_file: str,
    appear_times: int,
    top_ranges_str: str,
    relative_area: float,
    caption: str,
    **kwargs,
):
    prompt_head = open("prompts/image_cls_withcap.txt").read()
    aspect_ratio, _ = load_image(image_file)
    prompt = (
        prompt_head
        + "Input:\n"
        + str(
            {
                "caption": caption,
                "appear_times": appear_times,
                "slide_range": top_ranges_str,
                "aspect_ratio": aspect_ratio,
                "relative_area": relative_area,
            }
        )
    )
    return json.loads(vl_model(prompt, image_file).strip())


def get_outline(content: str):
    template = Template(open("prompts/get_outline.txt").read())
    prompt = template(paper_md=content).render()
    return long_model(prompt)


if __name__ == "__main__":
    # internvl = InternVL()
    # internvl("who r u")
    # print(qwen("你是谁"))
    gemini = Gemini()
    gemini.chat("小红是小明的爸爸")
    print(gemini.chat("小红和小明的关系是什么"))
