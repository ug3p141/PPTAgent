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


class Gemini:
    def __init__(self, time_limit: int = 60) -> None:
        self.time_limit = time_limit
        self.model = "gemini-1.5-pro-latest"
        self.client = genai.GenerativeModel(self.model)
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
        self._chat = self.client.start_chat(history=[])

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
        response = self.client.generate_content(
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
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        history_limit: int = 8,
    ) -> None:
        self.client = OpenAI(base_url=api_base)
        self.model = model
        self.system_message = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant"}],
            }
        ]
        self.history = []
        self.history_limit = history_limit
        assert history_limit % 2 == 0, "history_limit must be even"

    @tenacity
    def __call__(
        self,
        content: str,
        image_files: list[str] = None,
        save_history: bool = False,
    ) -> str:
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit :]
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
            self.system_message[0]["content"][0]["text"] = system_message
        messages = self.history + [
            {"role": "user", "content": [{"type": "text", "text": content}]}
        ]
        if image_files is not None:
            if not isinstance(image_files, list):
                image_files = [image_files]
            for image_file in image_files:
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
            model=self.model, messages=self.system_message + messages
        )
        response = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": response})
        if save_history:
            self.history = messages
        return response

    def clear_history(self):
        self.history = []


class Agent:
    def __init__(self, model: OPENAI):
        self.model = model

    def __call__(self, *args, **kwargs) -> str:
        return self.model(*args, **kwargs)

    def clear_history(self):
        self.model.clear_history()

    def set_model(self, model):
        self.model = model


gpt4o = OPENAI()
gpt4omini = OPENAI(model="gpt-4o-mini")

gemini = Gemini()

internvl = OPENAI(
    model="InternVL2-Llama3-76B", api_base="http://124.16.138.150:8000/v1"
)
qwen = OPENAI(model="Qwen2-72B-Instruct", api_base="http://124.16.138.150:7999/v1")

caption_model = internvl
long_model = qwen
agent_model = Agent(gpt4o)


def get_refined_doc(text_content: str):
    template = Template(open("prompts/document_refine.txt").read())
    prompt = template.render(markdown_document=text_content)
    return json.loads(long_model(prompt))


if __name__ == "__main__":
    print(internvl("who r u"))
    print(qwen("你是谁"))
    # gemini = Gemini()
    # gemini.chat("小红是小明的爸爸")
    # print(gemini.chat("小红和小明的关系是什么"))
