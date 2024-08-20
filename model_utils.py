import os
from copy import deepcopy

import numpy as np
import pytorch_fid.fid_score as fid
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from utils import pjoin

fid.tqdm = lambda x: x

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def fid_score(img_dir: str):
    img_files = sorted([f for f in os.listdir(img_dir)])
    num_images = len(img_files)
    fid_scores = np.zeros((num_images, num_images))
    os.system("rm -rf .temp_1")
    os.system("rm -rf .temp_2")
    os.makedirs(".temp_1")
    os.makedirs(".temp_2")
    for i in range(num_images):
        for j in range(i + 1, num_images):
            img_i_path = pjoin(img_dir, img_files[i])
            img_j_path = pjoin(img_dir, img_files[j])
            os.rename(img_i_path, pjoin(".temp_1", img_files[i]))
            os.rename(img_j_path, pjoin(".temp_2", img_files[j]))
            fid_value = fid.calculate_fid_given_paths(
                [".temp_1", ".temp_2"],
                batch_size=1,
                device="cuda",
                dims=2048,
            )
            fid_scores[i, j] = fid_value
            fid_scores[j, i] = fid_value

            if fid_value < 150:
                print(f"{img_files[i]} - {img_files[j]}: {fid_value}")
            os.rename(pjoin(".temp_1", img_files[i]), img_i_path)
            os.rename(pjoin(".temp_2", img_files[j]), img_j_path)

    print(fid_scores)


MAX_SIM = 999
SIM_BOUND = 50


def average_distance(similarity, idx, cluster_idx):
    """
    Calculate the average distance between a point (idx) and a cluster (cluster_idx).
    """
    total_distance = 0
    for idx_in_cluster in cluster_idx:
        total_distance += similarity[idx, idx_in_cluster]
    return total_distance / len(cluster_idx)


def get_cluster(similarity: np.ndarray):
    similarity[similarity == 0] = MAX_SIM
    num_points = similarity.shape[0]
    cluster = []
    sim_copy = deepcopy(similarity)

    while True:
        min_avg_dist = SIM_BOUND
        best_cluster = None
        best_point = None

        for c in cluster:
            for point_idx in range(num_points):
                avg_dist = average_distance(sim_copy, point_idx, c)
                if avg_dist < min_avg_dist:
                    min_avg_dist = avg_dist
                    best_cluster = c
                    best_point = point_idx

        if best_point is not None:  # or 考虑方差
            best_cluster.append(best_point)
            similarity[best_point, :] = MAX_SIM
            similarity[:, best_point] = MAX_SIM
        else:
            miss_flag = False
            min_edge_val = similarity.min()
            if min_edge_val > SIM_BOUND:
                miss_flag = True
                break
            i, j = np.unravel_index(np.argmin(similarity), similarity.shape)
            # if added[i] or added[j]: # 凡是added过了，已经MAX_SIM了
            cluster.append([i, j])
            similarity[i, :] = MAX_SIM
            similarity[:, i] = MAX_SIM
            similarity[j, :] = MAX_SIM
            similarity[:, j] = MAX_SIM
            if miss_flag:
                break
    cluster.extend(
        [[idx] for idx in range(num_points) if all([idx not in c for c in cluster])]
    )
    return cluster


def build_transform(input_size):
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


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return target_aspect_ratio, processed_images


def load_image(image_file, input_size=448, max_num=16):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    target_aspect_ratio, images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return target_aspect_ratio, pixel_values
