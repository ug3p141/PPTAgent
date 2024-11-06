import asyncio
import base64

import jsonlines
import requests
import yaml
from FlagEmbedding import BGEM3FlagModel
from jinja2 import Environment
from oaib import Auto
from openai import OpenAI

from model_utils import get_text_embedding
from utils import get_json_from_response, pexists, pjoin, print, tenacity


def run_async(coroutine):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    job = loop.run_until_complete(coroutine)
    return job


class LLM:
    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_base: str = None,
        use_openai: bool = True,
        use_batch: bool = False,
    ) -> None:
        if use_openai:
            self.client = OpenAI(base_url=api_base)
        if use_batch:
            assert use_openai, "use_batch must be used with use_openai"
            self.oai_batch = Auto(loglevel=0)
        self.model = model
        self.api_base = api_base
        self._use_openai = use_openai
        self._use_batch = use_batch

    def __call__(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
        history: list = None,
        delay_batch: bool = False,
        return_json: bool = False,
        return_message: bool = False,
    ) -> str:
        if content.startswith("You are"):
            system_message, content = content.split("\n", 1)
        if history is None:
            history = []
        system, message = self.format_message(content, images, system_message)
        if self._use_batch:
            result = run_async(self._run_batch(system + history + message, delay_batch))
            if delay_batch:
                return
            response = result.to_dict()["result"][0]["choices"][0]["message"]["content"]
        elif self._use_openai:
            completion = self.client.chat.completions.create(
                model=self.model, messages=system + history + message
            )
            response = completion.choices[0].message.content
        else:
            response = requests.post(
                self.api_base,
                json={
                    "system": system_message,
                    "prompt": content,
                    "image": [
                        i["image_url"]["url"]
                        for i in message[-1]["content"]
                        if i["type"] == "image_url"
                    ],
                },
            )
            response.raise_for_status()
            response = response.text
        message.append({"role": "assistant", "content": response})
        if return_json:
            response = get_json_from_response(response)
        if return_message:
            return response, message
        return response

    def __repr__(self) -> str:
        return f"LLM(model={self.model}, api_base={self.api_base})"

    async def _run_batch(self, messages: list, delay_batch: bool = False):
        await self.oai_batch.add(
            "chat.completions.create",
            model=self.model,
            messages=messages,
        )
        if delay_batch:
            return
        return await self.oai_batch.run()

    def format_message(
        self,
        content: str,
        images: list[str] = None,
        system_message: str = None,
    ):
        if system_message is None:
            system_message = "You are a helpful assistant"
        system = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            }
        ]
        message = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        if images is not None:
            if not isinstance(images, list):
                images = [images]
            for image in images:
                with open(image, "rb") as f:
                    message[0]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode('utf-8')}"
                            },
                        }
                    )
        return system, message

    def get_batch_result(self):
        results = run_async(self.oai_batch.run())
        return [
            r["choices"][0]["message"]["content"]
            for r in results.to_dict()["result"].values()
        ]

    def clear_history(self):
        self.history = []


# by type, by relevance, by time
class Role:
    def __init__(
        self,
        name: str,
        env: Environment,
        text_model: BGEM3FlagModel = None,
        llm: LLM = None,
        config: dict = None,
    ):
        self.name = name
        if config is None:
            with open(f"roles/{name}.yaml", "r") as f:
                config = yaml.safe_load(f)
        if llm is None:
            llm = globals()[config["use_model"] + "_model"]
        self.llm = llm
        self.model = llm.model
        self.text_model = text_model
        self.prompt_args = set(config["jinja_args"])
        self.system_message = config["system_prompt"]
        self.return_json = config["return_json"]
        self.template = env.from_string(config["template"])
        self._history = []

    def append_history(self, message: list):
        # calc similarity of request
        if self.text_model is not None:
            request = message[0]["content"][0]["text"]
            embedding = get_text_embedding(request, self.text_model)
            self._history.append((embedding, message))
        else:
            self._history.append(message)

    def add_validator(self, validator):
        self.validator = validator

    def get_history(self, add_history: int = 0):
        if add_history > 0:
            return self._history[-add_history:]

    def save_history(self, output_dir: str):
        history_file = pjoin(output_dir, f"{self.name}.jsonl")
        if pexists(history_file) and len(self._history) == 0:
            return
        with jsonlines.open(history_file, "w") as writer:
            writer.write_all(self._history)

    def __repr__(self) -> str:
        return f"Role(name={self.name}, model={self.model})"

    def __call__(
        self,
        images: list[str] = None,
        add_history: int = 0,
        **jinja_args,
    ):
        assert self.prompt_args == set(jinja_args.keys()), "Invalid arguments"
        response, message = self.llm(
            self.template.render(**jinja_args),
            system_message=self.system_message,
            history=self.get_history(add_history),
            images=images,
            return_json=self.return_json,
            return_message=True,
        )
        self.append_history(message)
        return response


gpt4o = LLM(use_batch=True)
gpt4omini = LLM(model="gpt-4o-mini")
qwen2_5 = LLM(model="Qwen2.5-72B-Instruct", api_base="http://127.0.0.1:7999/v1")
qwen_vl = LLM(
    model="Qwen2-VL-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.144:7813/v1"
)
internvl_76 = LLM(model="InternVL2-Llama3-76B", api_base="http://127.0.0.1:8000/v1")
internvl_multi = LLM(
    model="InternVL2-Llama3-76B",
    api_base="http://127.0.0.1:5000/generate",
    use_openai=False,
)
language_model = qwen2_5
code_model = qwen2_5
vision_model = internvl_76


if __name__ == "__main__":
    print(
        qwen2_5(
            "who r u",
        )
    )
