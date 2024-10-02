import asyncio
import base64
import random
from time import sleep, time

import google.generativeai as genai
import json_repair
import PIL.Image
import requests
from jinja2 import Template
from oaib import Auto
from openai import OpenAI

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
            return json_repair.loads(response.text.strip())
        else:
            return response.text.strip()

    def chat(self, content: str, image_file: str = None) -> str:
        self.prepare()
        if image_file is not None:
            content = [content, PIL.Image.open(image_file)]
        response = self._chat.send_message(content)
        return json_repair.loads(response.text.strip())


class OPENAI:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        use_batch: bool = False,
        history_limit: int = 8,
    ) -> None:
        self.client = OpenAI(base_url=api_base)
        if use_batch:
            self.oai_batch = Auto(loglevel=0)
        self._model = model
        self.model = model
        self._use_batch = use_batch
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
        if not self._use_batch:
            completion = self.client.chat.completions.create(
                model=self._model, messages=self.system_message + messages
            )
            response = completion.choices[0].message.content
        else:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                job = loop.create_task(self._run_batch(messages))
                asyncio.get_event_loop().run_until_complete(job)
            else:
                job = loop.run_until_complete(self._run_batch(messages))
            response = job.to_dict()["result"][0]["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": response})
        if save_history:
            self.history = messages
        return response

    async def _run_batch(self, message: list):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self._model,
            messages=self.system_message + message,
        )
        return await self.oai_batch.run()

    def clear_history(self):
        self.history = []

    def __repr__(self):
        return f"OPENAI Server (model={self.model}, use_batch={self._use_batch})"


class APIModel:
    def __init__(self, model: str, api_base: str):
        self.api_base = api_base
        self.model = model

    def __call__(self, content: str, image_files: list[str] = None):
        system_message = "You are a helpful assistant"
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
        data = {"prompt": content, "system": system_message, "image": []}

        if image_files:
            for image_file in image_files:
                with open(image_file, "rb") as image:
                    base64_image = base64.b64encode(image.read()).decode("utf-8")
                    data["image"].append(f"data:image/jpeg;base64,{base64_image}")

        response = requests.post(self.api_base, json=data)
        response.raise_for_status()
        return response.text


gpt4o = OPENAI(use_batch=True)
gpt4omini = OPENAI(model="gpt-4o-mini")
gemini = Gemini()
qwen = OPENAI(model="Qwen2-72B-Instruct", api_base="http://localhost:7999/v1")
internvl_multi = APIModel(
    model="InternVL2-Llama3-76B",
    api_base="http://124.16.138.150:5000/generate",
)
internvl_76 = OPENAI(model="InternVL2-Llama3-76B", api_base="http://127.0.0.1:8000/v1")
caption_model = internvl_76
long_model = qwen
agent_model = gpt4o


def get_refined_doc(text_content: str):
    template = Template(open("prompts/document_refine.txt").read())
    prompt = template.render(markdown_document=text_content)
    return json_repair.loads(long_model(prompt))


if __name__ == "__main__":
    qwen("who r u")
