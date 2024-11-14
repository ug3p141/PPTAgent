import asyncio
import hashlib
import importlib
import itertools
import json
import os
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from typing import Dict, List

import PIL.Image
import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from FlagEmbedding import BGEM3FlagModel
from marker.models import load_all_models

import llms
import pptgen
from induct import SlideInducter
from model_utils import get_image_model, get_refined_doc, parse_pdf
from multimodal import ImageLabler
from presentation import Presentation
from utils import IMAGE_EXTENSIONS, Config, pjoin, ppt_to_images

# constants
RUNS_DIR = "runs"
STAGES = [
    "PPT Parsing",
    "PDF Parsing",
    "Document Refinement",
    "Slide Induction",
    "PPT Generation",
    "Success!",
]
NUM_MODELS = 1 if len(sys.argv) == 1 else int(sys.argv[1])
NUM_INSTANCES_PER_MODEL = 4
DEVICE_COUNT = torch.cuda.device_count()

# models
text_models = [
    BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=i % DEVICE_COUNT)
    for i in range(NUM_MODELS)
]
image_models = [get_image_model(device=i % DEVICE_COUNT) for i in range(NUM_MODELS)]
marker_models = [
    load_all_models(device=i % DEVICE_COUNT, dtype=torch.float16)
    for i in range(NUM_MODELS)
]

# server
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
progress_store: Dict[str, Dict] = {}
active_connections: Dict[str, WebSocket] = {}
counter = itertools.cycle(range(NUM_MODELS))
executor = ThreadPoolExecutor(max_workers=NUM_MODELS * NUM_INSTANCES_PER_MODEL)


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
    numberOfPages: int = Form(...),
):
    importlib.reload(llms)
    importlib.reload(pptgen)
    task_id = datetime.now().strftime("20%y-%m-%d") + "/" + str(uuid.uuid4())
    os.makedirs(pjoin(RUNS_DIR, task_id))
    pptx_blob = await pptxFile.read()
    pdf_blob = await pdfFile.read()
    task = {"numberOfPages": numberOfPages, "model_idx": next(counter)}
    for file_type, blob in [("pptx", pptx_blob), ("pdf", pdf_blob)]:
        file_md5 = hashlib.md5(blob).hexdigest()
        task[file_type] = file_md5
        file_dir = pjoin(RUNS_DIR, file_type, file_md5)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, exist_ok=True)
            with open(pjoin(file_dir, "source." + file_type), "wb") as f:
                f.write(blob)
    progress_store[task_id] = task
    executor.submit(ppt_gen, task_id)
    return {"task_id": task_id.replace("/", "|")}


async def send_progress(websocket: WebSocket, status: str, progress: int):
    await websocket.send_json({"progress": progress, "status": status})


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    task_id = task_id.replace("|", "/")
    if task_id in progress_store:
        await websocket.accept()
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
    if not os.path.exists(pjoin(RUNS_DIR, task_id)):
        raise HTTPException(status_code=404, detail="Task not created yet")
    file_path = pjoin(RUNS_DIR, task_id, "final.pptx")
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="application/pptx",
            headers={"Content-Disposition": f"attachment; filename=pptagent.pptx"},
        )
    raise HTTPException(status_code=404, detail="Task not finished yet")


@app.get("/")
def hello():
    if len(active_connections) < NUM_MODELS:
        return {"message": "Hello, World!"}
    else:
        raise HTTPException(
            status_code=429,
            detail=f"Too many running connections, limit is {NUM_MODELS}",
        )


def ppt_gen(task_id: str):
    for _ in range(100):
        if task_id in active_connections:
            break
        time.sleep(0.02)
    else:
        progress_store.pop(task_id)
        return
    task = progress_store.pop(task_id)
    pptx_md5 = task["pptx"]
    pdf_md5 = task["pdf"]
    generation_config = Config(pjoin(RUNS_DIR, task_id))
    pptx_config = Config(pjoin(RUNS_DIR, "pptx", pptx_md5))
    json.dump(task, open(pjoin(generation_config.RUN_DIR, "task.json"), "w"))

    model_idx = task["model_idx"]
    text_model, image_model, marker_model = (
        text_models[model_idx],
        image_models[model_idx],
        marker_models[model_idx],
    )

    progress = ProgressManager(task_id, STAGES)
    parsedpdf_dir = pjoin(RUNS_DIR, "pdf", pdf_md5)
    ppt_image_folder = pjoin(pptx_config.RUN_DIR, "slide_images")

    asyncio.run(
        send_progress(active_connections[task_id], "task initialized successfully", 10)
    )

    try:
        # ppt parsing
        presentation = Presentation.from_file(
            pjoin(pptx_config.RUN_DIR, "source.pptx"), pptx_config
        )
        ppt_to_images(pjoin(pptx_config.RUN_DIR, "source.pptx"), ppt_image_folder)
        assert len(os.listdir(ppt_image_folder)) == len(
            presentation
        ), "Number of parsed slides and images do not match"

        for err_idx, _ in presentation.error_history:
            os.remove(pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg"))
        for i, slide in enumerate(presentation.slides, 1):
            slide.slide_idx = i
            os.rename(
                pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg"),
                pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg"),
            )

        labler = ImageLabler(presentation, pptx_config)
        progress.run_stage(labler.caption_images)

        # pdf parsing
        if not os.path.exists(pjoin(parsedpdf_dir, "source.md")):
            text_content = progress.run_stage(
                parse_pdf,
                pjoin(RUNS_DIR, "pdf", pdf_md5, "source.pdf"),
                parsedpdf_dir,
                marker_model,
                8,
            )
        else:
            text_content = open(pjoin(parsedpdf_dir, "source.md")).read()
            progress.report_progress()

        # doc refine and caption
        if not os.path.exists(pjoin(parsedpdf_dir, "caption.json")):
            caption_prompt = open("prompts/caption.txt").read()
            images = {}
            for k in os.listdir(parsedpdf_dir):
                if k.split(".")[-1] in IMAGE_EXTENSIONS:
                    try:
                        images[k] = [
                            llms.vision_model(
                                caption_prompt, [pjoin(parsedpdf_dir, k)]
                            ),
                            PIL.Image.open(pjoin(parsedpdf_dir, k)).size,
                        ]
                    except Exception as e:
                        logger.error(f"Error captioning image {k}: {e}")
            json.dump(images, open(pjoin(parsedpdf_dir, "caption.json"), "w"))
        else:
            images = json.load(open(pjoin(parsedpdf_dir, "caption.json")))
        if not os.path.exists(pjoin(parsedpdf_dir, "refined_doc.json")):
            doc_json = progress.run_stage(get_refined_doc, text_content)
            json.dump(doc_json, open(pjoin(parsedpdf_dir, "refined_doc.json"), "w"))
        else:
            doc_json = json.load(open(pjoin(parsedpdf_dir, "refined_doc.json")))
            progress.report_progress()

        # Slide Induction
        deepcopy(presentation).normalize().save(
            pjoin(generation_config.RUN_DIR, "template.pptx"), layout_only=True
        )
        ppt_to_images(
            pjoin(generation_config.RUN_DIR, "template.pptx"),
            pjoin(generation_config.RUN_DIR, "template_images"),
        )
        slide_inducter = SlideInducter(
            presentation,
            ppt_image_folder,
            pjoin(generation_config.RUN_DIR, "template_images"),
            pptx_config,
            image_model,
        )
        slide_induction = progress.run_stage(slide_inducter.content_induct)
        presentation = presentation.normalize()

        # PPT Generation
        progress.run_stage(
            pptgen.PPTCrew(text_model, error_exit=False)
            .set_examplar(presentation, slide_induction)
            .generate_pres,
            generation_config,
            images,
            task["numberOfPages"],
            doc_json,
        )
        progress.report_progress()
    except Exception as e:
        progress.fail_stage(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    import subprocess

    import uvicorn

    ip = (
        subprocess.check_output(
            "hostname -I | tr ' ' '\n' | grep '^124\\.'", shell=True
        )
        .decode()
        .strip()
    )
    print(f"backend running on {ip}:9297")
    uvicorn.run(app, host=ip, port=9297)
