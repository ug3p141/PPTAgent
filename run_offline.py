from typing import List, NamedTuple, Optional

from flask import Flask, jsonify, request
from PIL.Image import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image

app = Flask(__name__)

model_name = "/141nfs/zhenghao2022/Llama-3.2-11B-Vision-Instruct"
llm = LLM(
    model=model_name,
    tensor_parallel_size=4,
    max_model_len=4096,
    limit_mm_per_prompt={"image": 3},
    gpu_memory_utilization=0.5,
)


class ModelRequestData(NamedTuple):
    text: str
    image_data: List[Image]


def preprocess_qwen2vl(messages: List[dict]) -> ModelRequestData:
    processor = AutoProcessor.from_pretrained(model_name)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_data, _ = process_vision_info(messages)
    return ModelRequestData(
        prompt=text,
        image_data=image_data,
    )


def preprocess_mllama(messages: List[dict]) -> ModelRequestData:
    image_data = []
    for turn in messages:
        if len(turn["content"]) > 1:
            prompt: str = turn["content"][0]["text"]
            assert len(turn["content"]) - 1 == prompt.count("<|image|>")
            image_data.extend(
                [fetch_image(url["image_url"]["url"]) for url in turn["content"][1:]]
            )
        return ModelRequestData(
            prompt=prompt,
            image_data=image_data,
        )


def run_generate(messages: List[dict]):
    if "Qwen" in model_name:
        req_data = preprocess_qwen2vl(messages)
    else:
        req_data = preprocess_mllama(messages)
    outputs = req_data.llm.generate(
        {
            "prompt": req_data.text,
            "multi_modal_data": {"image": req_data.image_data},
        },
        use_tqdm=False,
    )

    return outputs[0].outputs[0].text


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    model = data.get("model")
    if model not in model_name:
        return jsonify({"error": "Invalid model"}), 400
    messages = data.get("messages")

    try:
        response = run_generate(model, messages)
        return jsonify({"generated_text": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
