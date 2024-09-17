"""
This example shows how to use vLLM for running offline inference with
multi-image input on vision language models, using the chat template defined
by the model.
"""

import base64
import sys
from typing import List

from transformers import AutoTokenizer

from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image


app = Flask(__name__)
llm = LLM(
    model="/mnt/shared_home/zhenghao2022/InternVL2-Llama3-76B",
    trust_remote_code=True,
    max_num_seqs=1,
    max_model_len=int(sys.argv[2]),
    limit_mm_per_prompt={"image": 3},
    tensor_parallel_size=8,
)


tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/shared_home/zhenghao2022/InternVL2-Llama3-76B", trust_remote_code=True
)


def load_internvl(system: str, prompt, image_urls: List[str]):
    placeholders = "\n".join(
        f"Image-{i}: <image>\n" for i, _ in enumerate(image_urls, start=1)
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{placeholders}\n{prompt}"},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Stop tokens for InternVL
    # models variants may have different stop tokens
    # please refer to the model card for the correct "stop words":
    # https://huggingface.co/OpenGVLab/InternVL2-2B#service
    stop_token_ids = [128001, 128002, 128003]
    return llm, prompt, stop_token_ids


def run_generate(system: str, question: str, image_urls: List[str]):
    llm, prompt, stop_token_ids = load_internvl(system, question, image_urls)

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": [fetch_image(url) for url in image_urls]},
        },
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    return outputs[0].outputs[0].text


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    system = data.get("system", "You are an helpful assistant.")
    prompt = data.get("prompt")
    image_base64 = data.get("image", [])
    response = run_generate(system, prompt, image_base64)
    return response


if __name__ == "__main__":
    with open("resource/doc2ppt_images/fig.1.png", "rb") as image:
        base64_image = base64.b64encode(image.read()).decode("utf-8")
    run_generate(
        "You are an helpful assistant.",
        "What is in the picture?",
        [
            f"data:image/jpeg;base64,{base64_image}",
            f"data:image/jpeg;base64,{base64_image}",
            f"data:image/jpeg;base64,{base64_image}",
        ],
    )
    app.run(host=sys.argv[1], port=5000)
