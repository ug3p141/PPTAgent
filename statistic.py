import json
import os
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 读取 Excel 文件
file_path = "./human_eval/final_ppt评分标注结果_2024-12-13.xlsx"
human_eval = pd.read_excel(file_path).to_dict("records")
llm_eval = json.load(open("./ppt_eval_12-16_qwen+intern.json"))


def plot_correlation(data, x, y, title):

    corr_data = defaultdict(dict)
    eval_file = "data/evals/PPTCrew-gpt-4o+gpt-4o+gpt-4o.json"
    setting = os.path.basename(eval_file).removesuffix(".json")
    eval_stats = json.load(open(eval_file))
    for dimension in ["ppl", "fid", "content", "vision", "logic"]:
        pairs = eval_stats[dimension]
        for pptx, score in pairs.items():
            if isinstance(score, list):
                score = score[0]
                print(eval_file, score)
            if isinstance(score, dict):
                score = score["score"]
            if isinstance(score, str):
                continue
            if score > 5000:
                continue
            if dimension == "logic":
                dimension = "coherence"
            if dimension == "vision":
                dimension = "design"
            print(pptx, dimension, score)
            corr_data[pptx][dimension] = score
    data = list(corr_data.values())
    new_data = []
    for i in data:
        if len(i) != 4:
            continue
        new_data.append(i)
    numeric_df = pd.DataFrame(new_data)

    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    # Plot the heatmap with axis limits set from -1 to 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,
        vmax=1,
        annot_kws={"size": 15},
    )
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("correlation.pdf", bbox_inches="tight")
    plt.show()


def organize():
    data = []
    for dimension, files in llm_eval.items():
        scores_llm = []
        scores_human = []
        setting_scores = defaultdict(
            lambda: {"llm": defaultdict(list), "human": defaultdict(list)}
        )
        outlierset = set()
        for filename, values in files.items():
            try:
                setting, basename = filename.split("/", 2)[1:]
                basename = basename.split("/")[0]
                if isinstance(values["score"], int):
                    setting_scores[setting]["llm"][basename].append(values["score"])
            except:
                continue

        for record in human_eval:
            setting = record.get("模型")
            score = record[dimension]
            basename = record["PPT"]
            try:
                score = int(score)
            except:
                continue
            if basename in setting_scores[setting]["llm"]:
                setting_scores[setting]["human"][basename].append(score)

        for setting, scores_data in sorted(setting_scores.items(), key=lambda x: x[0]):
            sorted_keys = sorted(scores_data["llm"].keys())
            scores_llm = [
                sum(scores_data["llm"][k]) / len(scores_data["llm"][k])
                for k in sorted_keys
            ]
            scores_human = [
                sum(scores_data["human"][k]) / len(scores_data["human"][k])
                for k in sorted_keys
            ]
            for k, v1 in zip(sorted_keys, scores_llm):
                data.append(
                    {
                        "setting": setting,
                        "sample": k,
                        "dimension": dimension,
                        "score": v1,
                        "source": "llm",
                    }
                )
            for k, v2 in zip(sorted_keys, scores_human):
                data.append(
                    {
                        "setting": setting,
                        "sample": k,
                        "dimension": dimension,
                        "score": v2,
                        "source": "human",
                    }
                )
        print(setting, dimension, sum(scores_llm) / len(scores_llm))
        pearson_corr, _ = pearsonr(scores_llm, scores_human)
        spearman_corr, _ = spearmanr(scores_llm, scores_human)
        print(f"{dimension}, at a length of {len(scores_llm)}")
        print(f"pearson: {pearson_corr}, spearman: {spearman_corr}")
    df = pd.DataFrame(data)


def setting_corr():
    data_human = df[df["source"] == "human"][df["dimension"] == "logic"]
    data_llm = df[df["source"] == "llm"][df["dimension"] == "logic"]

    # 按 setting, sample, dimension 对齐
    merged = pd.merge(
        data_human,
        data_llm,
        on=["setting", "sample", "dimension"],
        suffixes=("_human", "_llm"),
    )

    # 提取 score 列
    scores_human = merged["score_human"]
    scores_llm = merged["score_llm"]

    # 计算 Pearson 相关系数
    if len(scores_human) > 1:  # 确保样本数足够
        pearson_correlation = pearsonr(scores_human, scores_llm)
        spearman_correlation = spearmanr(scores_human, scores_llm)

    # 输出结果
    print(f"Pearson correlation between human and llm: {pearson_correlation}")
    print(f"Spearman correlation between human and llm: {spearman_correlation}")


@contextmanager
def science_plot(font_size=16):
    import scienceplots

    with plt.style.context(["ieee", "grid", "no-latex", "light"]):
        plt.rcParams.update({"font.size": font_size})
        yield
