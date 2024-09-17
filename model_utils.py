import os
from copy import deepcopy

import numpy as np
import torch
import torchvision.transforms as T
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoFeatureExtractor, AutoModel

from presentation import Presentation
from utils import IMAGE_EXTENSIONS, pjoin

DEFAULT_DEVICE = "cuda:3"

text_embed_model = None
image_embed_model = None
extractor = None


def get_text_embedding(text: list[str], batchsize: int = 32, model = None):
    global text_embed_model
    if model is None and text_embed_model is None:
        text_embed_model = BGEM3FlagModel(
            "BAAI/bge-m3", use_fp16=True, device=DEFAULT_DEVICE
        )
    if model is None:
        model = text_embed_model
    if isinstance(text, str):
        return torch.tensor(model.encode(text)["dense_vecs"]).to(model.device)
    result = []
    for i in range(0, len(text), batchsize):
        result.extend(
            torch.tensor(
                model.encode(text[i : i + batchsize])["dense_vecs"]
            ).to(model.device)
        )
    return result


def prs_dedup(presentation: Presentation, ppt_image_folder: str, batchsize: int = 32, model = None):
    text_embeddings = get_text_embedding(
        [i.to_text() for i in presentation.slides], batchsize, model
    )
    pre_embedding = text_embeddings[0]
    slide_idx = 1
    duplicates = []
    while slide_idx < len(presentation.slides):
        cur_embedding = text_embeddings[slide_idx]
        if torch.cosine_similarity(pre_embedding, cur_embedding, -1) > 0.8:
            duplicates.append(slide_idx - 1)
        slide_idx += 1
        pre_embedding = cur_embedding
    return [presentation.slides.pop(i) for i in reversed(duplicates)]


def image_embedding(image_dir: str, batchsize: int = 16, model = None):
    global extractor
    global image_embed_model
    if model is None:
        model = image_embed_model
    if image_embed_model is None:
        image_embed_model = AutoModel.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            torch_dtype=torch.float16,
            device_map=DEFAULT_DEVICE,
        ).eval()
    if extractor is None:
        extractor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
    transform = T.Compose(
        [
            T.Resize(int((256 / 224) * extractor.size["height"])),
            T.CenterCrop(extractor.size["height"]),
            T.ToTensor(),
            T.Normalize(mean=extractor.image_mean, std=extractor.image_std),
        ]
    )

    inputs = []
    embeddings = []
    images = [
        i for i in sorted(os.listdir(image_dir)) if i.split(".")[-1] in IMAGE_EXTENSIONS
    ]
    for file in images:
        image = Image.open(pjoin(image_dir, file)).convert("RGB")
        inputs.append(transform(image))
        if len(inputs) % batchsize == 0 or file == images[-1]:
            batch = {"pixel_values": torch.stack(inputs).to(model.device)}
            embeddings.extend(model(**batch).last_hidden_state.detach())
            inputs.clear()
    return {image: embedding.flatten() for image, embedding in zip(images, embeddings)}


def images_cosine_similarity(embeddings: list[torch.Tensor]):
    embeddings = [embedding for embedding in embeddings]
    sim_matrix = torch.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim_matrix[i, j] = sim_matrix[j, i] = torch.cosine_similarity(
                embeddings[i], embeddings[j], -1
            )
    return sim_matrix


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def average_distance(similarity, idx, cluster_idx):
    """
    Calculate the average distance between a point (idx) and a cluster (cluster_idx).
    """
    if idx in cluster_idx:
        return 0
    total_similarity = 0
    for idx_in_cluster in cluster_idx:
        total_similarity += similarity[idx, idx_in_cluster]
    return total_similarity / len(cluster_idx)


def get_cluster(similarity: np.ndarray, sim_bound: float = 0.65):
    num_points = similarity.shape[0]
    clusters = []
    sim_copy = deepcopy(similarity)
    added = [False] * num_points
    while True:
        max_avg_dist = sim_bound
        best_cluster = None
        best_point = None

        for c in clusters:
            for point_idx in range(num_points):
                if added[point_idx]:
                    continue
                avg_dist = average_distance(sim_copy, point_idx, c)
                if avg_dist > max_avg_dist:
                    max_avg_dist = avg_dist
                    best_cluster = c
                    best_point = point_idx

        if best_point is not None:
            best_cluster.append(best_point)
            added[best_point] = True
            similarity[best_point, :] = 0
            similarity[:, best_point] = 0
        else:
            if similarity.max() < sim_bound:
                break
            i, j = np.unravel_index(np.argmax(similarity), similarity.shape)
            clusters.append([int(i), int(j)])
            added[i] = True
            added[j] = True
            similarity[i, :] = 0
            similarity[:, i] = 0
            similarity[j, :] = 0
            similarity[:, j] = 0
    return clusters


def internvl_build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(
                (input_size, input_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def internvl_find_closest_aspect_ratio(
    aspect_ratio, target_ratios, width, height, image_size
):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def internvl_dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return target_aspect_ratio, processed_images


def internvl_load_image(image_file, input_size=448, max_num=16):
    image = Image.open(image_file).convert("RGB")
    transform = internvl_build_transform(input_size=input_size)
    target_aspect_ratio, images = internvl_dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return target_aspect_ratio, pixel_values
