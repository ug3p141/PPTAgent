import asyncio
import base64

import requests
from oaib import Auto
from openai import OpenAI

from utils import get_json_from_response, print, tenacity


def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    if loop.is_running():
        job = loop.create_task(coroutine)
        loop.run_until_complete(job)
    else:
        job = loop.run_until_complete(coroutine)
    return job


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
        delay_batch: bool = False,
        return_json: bool = False,
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
            try:
                completion = self.client.chat.completions.create(
                    model=self._model, messages=self.system_message + messages
                )
            except Exception as e:
                print(e)
                raise e
            response = completion.choices[0].message.content
        else:
            result = run_async(self._run_batch(messages, delay_batch))
            if delay_batch:
                return
            response = result.to_dict()["result"][0]["choices"][0]["message"]["content"]
        messages.append({"role": "assistant", "content": response})
        if save_history:
            self.history = messages
        return response if not return_json else get_json_from_response(response)

    async def _run_batch(self, message: list, delay_batch: bool = False):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self._model,
            messages=self.system_message + message,
        )
        if delay_batch:
            return
        return await self.oai_batch.run()

    def get_batch_result(self):
        results = run_async(self.oai_batch.run())
        return [
            r["choices"][0]["message"]["content"]
            for r in results.to_dict()["result"].values()
        ]

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
qwen = OPENAI(model="Qwen2-72B-Instruct", api_base="http://localhost:7999/v1")
qwen2_5 = OPENAI(
    model="Qwen2.5-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.143:7812/v1"
)
qwen_vl = OPENAI(
    model="Qwen2-VL-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.144:7813/v1"
)
internvl_multi = APIModel(
    model="InternVL2-Llama3-76B",
    api_base="http://124.16.138.150:5000/generate",
)
internvl_76 = OPENAI(model="InternVL2-Llama3-76B", api_base="http://127.0.0.1:8000/v1")
caption_model = qwen_vl
long_model = qwen2_5
agent_model = qwen2_5

if __name__ == "__main__":
    gpt4o("who r u")
