import glob
import json
import os
import re
import subprocess
import tempfile

from tqdm import tqdm

import llms
from utils import pexists, ppt_to_images

slides = """
Slides should include a title page. Following slides should contain an informative slide title
and short, concise bullet points. Longer slides should be broken up into multiple slides.
"""

convert_to_latex = (
    "Summarize the following input in a {} style."
    "Style parameters: {}"
    "Format the output document as a latex file:\n"
    "Input: {}\n\n"
    "Output:"
)

sure_prompt = (
    f"Given the input text, extract the document title and authors."
    "For each section in the given input text, extract the most important sentences."
    "Format the output using the following json template:\n"
    "{}\n\n"
    "Input: {}\n"
    "Output:"
)

internal_representation_wo_image = """{
    "Document Title": "TITLE",
    "Document Authors": ["AUTHOR 1", "AUTHOR 2", "...", "AUTHOR N"],
    "SECTION TITLE 1": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    },
    "SECTION TITLE 2": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    },
    "...": {},
    "SECTION TITLE N": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    }
}"""


internal_representation = """{
    "Document Title": "TITLE",
    "Document Authors": ["AUTHOR 1", "AUTHOR 2", "...", "AUTHOR N"],
    "SECTION TITLE 1": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ],
        "Figures": [
            {
                "Name": "Figure K",
                "Caption": "CAPTION"
            },
            {
                "Name": "Figure K",
                "Caption": "CAPTION"
            }
        ]
    },
    "SECTION TITLE 2": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ]
    },
    "...": {},
    "SECTION TITLE N": {
        "Content": [
            "SENTENCE 1",
            "SENTENCE 2",
            "...",
            "SENTENCE N"
        ],
        "Figures": [
            {
                "Path": "path/to/figure/1.png",
                "Caption": "CAPTION"
            },
            {
                "Path": "path/to/figure/2.png",
                "Caption": "CAPTION"
            }
        ]
    }
}"""


def replace_mentions_of_figures(latex, figure_dir):
    latex = latex.split("\n")
    for i in range(len(latex)):
        paragraph = latex[i]
        matches = re.findall(r"\\includegraphics.*?{([^}]+)}", paragraph)
        for match in matches:
            if pexists(match):
                continue
            if match == os.path.basename(match):
                if pexists(os.path.join(figure_dir, match)):
                    latex[i] = paragraph.replace(match, f"{figure_dir}/{match}")
                    continue
            raise ValueError(f"Figure {match} not found")
    return "\n".join(latex)


def kctv_gen_ppt(doc_dir):
    # Take input doc
    input_json = json.load(open(doc_dir + "/refined_doc.json"))
    images = json.load(open(doc_dir + "/image_caption.json"))
    input_json["Figures"] = images
    model_name = llms.get_simple_modelname(llms.language_model)
    output_base = os.path.join(doc_dir, "kctv_" + model_name)

    if os.path.exists(output_base + "_slide_images"):
        return

    prompt = sure_prompt.format(internal_representation, input_json)
    gpt_response = llms.language_model(prompt, return_json=True)

    with open(
        output_base + ".json",
        "w",
        encoding="utf-8",
    ) as fout:
        json.dump(gpt_response, fout, indent=4)

    latex_prompt = convert_to_latex.format("slide", slides, gpt_response)
    gpt_latex = llms.language_model(
        latex_prompt,
    )
    gpt_latex = gpt_latex.replace("```latex", "").replace("```", "")
    gpt_latex = replace_mentions_of_figures(gpt_latex, doc_dir)
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as f:
        with open(f.name, "w") as fout:
            fout.write(gpt_latex.replace("\\ ", " "))
        subprocess.run(
            ["pdflatex", f.name],
            timeout=30,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        pdf_file = f.name.replace(".tex", ".pdf")
        if not pexists(pdf_file):
            raise ValueError(f"PDF not compiled successfully")
        os.rename(pdf_file, output_base + ".pdf")
    ppt_to_images(output_base + ".pdf", output_base + "_slide_images")


if __name__ == "__main__":
    for pdf_folder in tqdm(glob.glob("data/*/pdf/*")):
        try:
            kctv_gen_ppt(pdf_folder)
        except Exception as e:
            continue
        print("success generated ppt for ", pdf_folder)
