import os
import datetime
import json
import logging
import sys
from pathlib import Path
import google.generativeai as genai
from datamodel import json_to_dataclass, DATACLASSES
from pptx import Presentation
from tenacity import retry, wait_fixed, stop_after_attempt, after_log
from flask import Flask, send_file

app = Flask(__name__)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)
os.makedirs("output/requests/", exist_ok=True)
os.makedirs("output/ppts/", exist_ok=True)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
GOOGLE_API_KEY = "AIzaSyAQSqXqih0qx0E0U5eUWz1WI_-pt4LjpYk"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")
safety_settings = [
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
generation_config = genai.GenerationConfig(
    response_mime_type="application/json"  # , response_schema=list[*DATACLASSES]
)
prompt_head = open("prompt_head_en.txt", "r").read()


@app.route("/")
def hello():
    return "ppt generation service running successfully"

@app.route("/generate_ppt", methods=["POST"])
@retry(wait=wait_fixed(30), stop=stop_after_attempt(5), after=after_log(logger, logging.DEBUG))
def ppt_gen(filename:str, pdf_md:str, number_of_slides:str):
    model_input = prompt_head.replace(
        r"{{number_of_slides}}", str(number_of_slides)
    ).replace(r"{{{paper}}}", pdf_md)
    response = model.generate_content(
        model_input,
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    with open(
        f"output/requests/{Path(filename).stem}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json",
        mode="w",
    ) as f:
        json.dump(
            {"input": model_input, **response.to_dict()}, f, ensure_ascii=False, indent=4
        )
    json_data = json.loads(
        response.text[response.text.find("[") : response.text.rfind("]") + 1]
    )
    ppt = Presentation("./MasterSlide.pptx")
    for page in json_to_dataclass(json_data):
        page.to_slide(ppt)
    ppt_path = f"output/ppts/{Path(filename).stem}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pptx"
    ppt.save(ppt_path)
    send_file(ppt_path, as_attachment=True)
    return "ppt generated successfully"


if __name__=="__main__":
    import socket
    s = socket.socket()
    s.bind(("", 0))
    random_port = s.getsockname()[1]
    s.close()
    app.run(debug=True, host="0.0.0.0", port=random_port)
    # paper_md = open("./DOC2PPT.md").read()
    # number_of_slides = 20
    # filename = "DOC2PPT.md"
    # ppt_gen(filename, paper_md, number_of_slides)
