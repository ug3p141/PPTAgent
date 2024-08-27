import os
import tempfile
import zipfile

import win32com.client
from flask import Flask, request, send_file

from utils import pjoin

app = Flask(__name__)


@app.route("/convert", methods=["POST"])
def convert_ppt_to_images():
    ppt_file = request.files["file"]
    with tempfile.TemporaryDirectory() as temp_dir:
        ppt_path = pjoin(temp_dir, ppt_file.filename)
        ppt_file.save(ppt_path)
        Application = win32com.client.Dispatch("PowerPoint.Application")
        Presentation = Application.Presentations.Open(ppt_path)
        zip_path = pjoin(temp_dir, "slides.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for i, slide in enumerate(Presentation.Slides):
                image_path = pjoin(temp_dir, f"slide_{(i+1):04d}.jpg")
                slide.Export(image_path, "JPG")
                zipf.write(image_path, arcname=f"slide_{(i+1):04d}.jpg")
        Presentation.Close()
        Application.Quit()
        return send_file(zip_path, as_attachment=True, download_name="slides.zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
