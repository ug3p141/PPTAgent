import base64
import json
import random
from time import sleep, time

import google.generativeai as genai
import PIL
import torch
from jinja2 import Template
from openai import OpenAI

from model_utils import internvl_load_image
from utils import print, tenacity


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


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

    @tenacity
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
    def __init__(self, model: str = "gpt-4o-2024-08-06", api_base: str = None) -> None:
        self.client = OpenAI(base_url=api_base)
        self.model = model

    @tenacity
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
        return completion.choices[0].message.content


gemini = Gemini()
intern = OPENAI(
    model="OminiPreGen/models/InternVL2-8B", api_base="http://124.16.138.143:8000/v1"
)
gpt4o = OPENAI()
gpt4omini = OPENAI(model="gpt-4o-mini")
caption_model = gpt4omini
agent_model = gpt4o


def label_image(
    image_file: str,
    appear_times: int,
    top_ranges_str: str,
    relative_area: float,
    caption: str,
    **kwargs,
):
    prompt_head = open("prompts/image_label/image_cls_withcap.txt").read()
    aspect_ratio, _ = internvl_load_image(image_file)
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
    return agent_model(prompt, image_file)


def get_outline(content: str):
    template = Template(open("prompts/get_outline.txt").read())
    prompt = template(paper_md=content).render()
    return agent_model(prompt)


if __name__ == "__main__":
    # internvl = InternVL()
    # internvl("who r u")
    # print(qwen("你是谁"))
    gemini = Gemini()
    gemini.chat("小红是小明的爸爸")
    print(gemini.chat("小红和小明的关系是什么"))
