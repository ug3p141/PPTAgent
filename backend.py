# TODO 改成多页同时生成，生成成功的页面都发回去，不成功的就不要了
import asyncio
import hashlib
import io
import json
import os
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from glob import glob
from typing import Dict, List

import PIL.Image
import PyPDF2
import torch
from fastapi import (FastAPI, File, Form, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from FlagEmbedding import BGEM3FlagModel

import llms
from agent import PPTAgent
from model_utils import get_text_embedding, prs_dedup
from multimodal import ImageLabler
from presentation import Presentation
from template_induct import TemplateInducter
from utils import (IMAGE_EXTENSIONS, Config, parse_pdf, pjoin, ppt_to_images,
                   print)

RUNS_DIR = "runs"
STAGES = [
    "PPT Parsing",
    "PDF Parsing (1min/page)",
    "Document Refinement",
    "PPT Image Captioning",
    "PPT to Images",
    "Slides Deduplication",
    "PDF Image Captioning",
    "Template Induction",
    "PPT Generation",
    "PPT Saving",
]
qwen = llms.OPENAI(
    model="Qwen2.5-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.143:7812/v1"
)
qwen_vl = llms.OPENAI(
    model="Qwen2-VL-72B-Instruct-GPTQ-Int4", api_base="http://124.16.138.144:7813/v1"
)
llms.long_model = qwen
llms.agent_model = qwen
llms.caption_model = qwen_vl

# Allow CORS for frontend requests
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for progress and files
text_models = [
    BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=i)
    for i in range(torch.cuda.device_count())
]
executor = ThreadPoolExecutor(max_workers=30)
progress_store: Dict[str, Dict] = {}
result_store: Dict[str, str] = {}
active_connections: Dict[str, WebSocket] = {}


class ProgressManager:
    def __init__(self, task_id: str, stages: List[str], debug: bool = True):
        self.task_id = task_id
        self.stages = stages
        self.debug = debug
        self.failed = False
        self.current_stage = 0
        self.total_stages = len(stages)

    def run_stage(self, func, *args, **kwargs):
        if self.task_id not in active_connections:
            self.failed = True
        if self.failed:
            return
        try:
            self.report_progress()
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            self.fail_stage(str(e))

    def report_progress(self):
        self.current_stage += 1
        progress = int((self.current_stage / self.total_stages) * 100)
        asyncio.run(
            send_progress(
                active_connections[self.task_id],
                f"Stage: {self.stages[self.current_stage - 1]}",
                progress,
            )
        )

    def fail_stage(self, error_message: str):
        asyncio.run(
            send_progress(
                active_connections[self.task_id],
                f"{self.stages[self.current_stage]} Error: {error_message}",
                100,
            )
        )
        self.failed = True
        active_connections.pop(self.task_id, None)
        if self.debug:
            logger.error(
                f"{self.task_id}: {self.stages[self.current_stage]} Error: {error_message}"
            )


@app.post("/api/upload")
async def create_task(
    pptxFile: UploadFile = File(...),
    pdfFile: UploadFile = File(...),
    selectedModel: str = Form(...),
    numberOfPages: int = Form(...),
):
    # create time 20xx-xx-xx
    task_id = datetime.now().strftime("20%y-%m-%d") + "/" + str(uuid.uuid4())
    os.makedirs(pjoin(RUNS_DIR, task_id))
    pptx_path = pjoin(RUNS_DIR, task_id, "source.pptx")
    pdf_path = pjoin(RUNS_DIR, task_id, "source.pdf")
    with open(pptx_path, "wb") as f:
        f.write(await pptxFile.read())
    with open(pdf_path, "wb") as f:
        f.write(await pdfFile.read())
    progress_store[task_id.replace("/", "|")] = {
        "selectedModel": selectedModel,
        "numberOfPages": numberOfPages,
    }
    executor.submit(ppt_gen, task_id.replace("/", "|"))
    return {"task_id": task_id.replace("/", "|")}


async def send_progress(websocket: WebSocket, status: str, progress: int):
    await websocket.send_json({"progress": progress, "status": status})


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    if task_id in progress_store:
        await websocket.accept()
        await send_progress(websocket, "websocket connected successfully", 3)
    else:
        raise HTTPException(status_code=404, detail="Task not found")
    active_connections[task_id] = websocket
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("websocket disconnected", task_id)
        active_connections.pop(task_id, None)


@app.get("/api/download")
def download(task_id: str):
    if task_id not in result_store:
        raise HTTPException(status_code=404, detail="Task not found")
    file_path = result_store[task_id]
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pptx",
            headers={"Content-Disposition": f"attachment; filename=pptagent.pptx"},
        )
    return {"error": "File not found"}


@app.get("/")
def hello():
    if len(active_connections) < 30:
        return {"message": "Hello, World!"}
    else:
        raise HTTPException(
            status_code=429, detail="Too many running connections, limit is 30"
        )


def ppt_gen(task_id: str):
    generation_config = Config(task_id.replace("|", "/"))
    ppt_image_folder = pjoin(generation_config.RUN_DIR, "slide_images")
    text_model = text_models[ord(task_id[-1]) % len(text_models)]
    # wait for websocket connection
    for i in range(50):
        if task_id in active_connections:
            break
        time.sleep(0.1)
        if i == 49:
            progress_store.pop(task_id)
            return
    task = progress_store.pop(task_id)
    json.dump(task, open(pjoin(generation_config.RUN_DIR, "task.json"), "w"))
    num_pages = task["numberOfPages"]
    # model = task["selectedModel"]
    asyncio.run(
        send_progress(active_connections[task_id], "task initialized successfully", 5)
    )

    progress = ProgressManager(task_id, STAGES)

    with open(pjoin(generation_config.RUN_DIR, "source.pdf"), "rb") as f:
        pdf_content = f.read()
    with open(pjoin(generation_config.RUN_DIR, "source.pptx"), "rb") as f:
        pptx_content = f.read()
    pdf_md5 = hashlib.md5(pdf_content).hexdigest()
    pptx_md5 = hashlib.md5(pptx_content).hexdigest()
    parsedpdf_dir = pjoin(RUNS_DIR, "cache", "PDF", pdf_md5)
    pptx_config = Config(pjoin("cache", "PPTX", pptx_md5))
    try:
        presentation = progress.run_stage(
            Presentation.from_file,
            pjoin(generation_config.RUN_DIR, "source.pptx"),
            pptx_config,
        )
        if len(presentation) < 5:
            asyncio.run(
                send_progress(
                    active_connections[task_id],
                    "PPT Parsing Error: too short, you should upload a normal presentation(>5)",
                    100,
                )
            )
        if len(glob(parsedpdf_dir + "/source/*.md")) == 0:
            if len(PyPDF2.PdfReader(io.BytesIO(pdf_content)).pages) > 20:
                raise Exception(
                    "PDF Parsing Error: too long, you should upload a short PDF"
                )
            progress.run_stage(
                parse_pdf,
                pjoin(generation_config.RUN_DIR, "source.pdf"),
                parsedpdf_dir,
                "http://192.168.14.17:11223/convert",
            )
        else:
            progress.report_progress()
        text_content = open(glob(parsedpdf_dir + "/source/*.md")[0]).read()
        if len(text_content) > 10_000:
            raise Exception(
                "PDF Parsing Error: too long, you should upload a short PDF"
            )
        doc_json = progress.run_stage(llms.get_refined_doc, text_content)
        json.dump(
            doc_json, open(pjoin(generation_config.RUN_DIR, "refined_doc.json"), "w")
        )

        labler = ImageLabler(presentation, pptx_config)
        progress.run_stage(labler.caption_images)
        labler.apply_stats()

        progress.run_stage(ppt_to_images, presentation.source_file, ppt_image_folder)

        duplicates = progress.run_stage(
            prs_dedup, presentation, ppt_image_folder, model=text_model
        )
        for slide in duplicates:
            os.remove(pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"))
        for err_idx, _ in presentation.error_history:
            os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
        assert len(presentation) == len(
            [i for i in os.listdir(ppt_image_folder) if i.endswith(".jpg")]
        )
        for i, slide in enumerate(presentation.slides, 1):
            slide.slide_idx = i
            os.rename(
                pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
                pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
            )

        caption_prompt = open("prompts/image_label/caption.txt").read()
        if not os.path.exists(pjoin(parsedpdf_dir, "caption.json")):
            images = progress.run_stage(
                lambda: {
                    pjoin(parsedpdf_dir, k): [
                        llms.caption_model(caption_prompt, [pjoin(parsedpdf_dir, k)]),
                        PIL.Image.open(pjoin(parsedpdf_dir, k)).size,
                    ]
                    for k in os.listdir(parsedpdf_dir)
                    if k.split(".")[-1] in IMAGE_EXTENSIONS
                }
            )
            json.dump(images, open(pjoin(parsedpdf_dir, "caption.json"), "w"))
        else:
            images = json.load(open(pjoin(parsedpdf_dir, "caption.json")))
            progress.report_progress()
        deepcopy(presentation).normalize().save(
            pjoin(generation_config.RUN_DIR, "template.pptx"), layout_only=True
        )
        ppt_to_images(
            pjoin(generation_config.RUN_DIR, "template.pptx"),
            pjoin(generation_config.RUN_DIR, "template_images"),
        )
        template_inducter = TemplateInducter(
            presentation,
            ppt_image_folder,
            pjoin(generation_config.RUN_DIR, "template_images"),
            pptx_config,
        )
        functional_keys, slide_cluster = progress.run_stage(
            template_inducter.work, most_image=1
        )
        presentation = presentation.normalize()

        progress.run_stage(
            PPTAgent(
                presentation,
                generation_config,
                slide_cluster,
                images,
                num_pages,
                text_model,
                doc_json,
                functional_keys,
                torch.stack(
                    get_text_embedding(list(slide_cluster.keys()), model=text_model)
                ),
            ).work,
            3,
        )
        if task_id in active_connections:
            result_store[task_id] = pjoin(generation_config.RUN_DIR, "final.pptx")
        progress.report_progress()
    except Exception as e:
        progress.fail_stage(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="192.168.14.17", port=9297)
