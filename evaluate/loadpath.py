import os
from pathlib import Path

import cv2

SUPPORTED_DATASETS = ("SD360", "SVGC_AVA", "AVS-ODV")

def load_saliency_map_from_png(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image could not be loaded: {path}")
    return image.T

load_ground_map_from_png = load_saliency_map_from_png
load_ground_fix_from_png = load_saliency_map_from_png

def _ground_path(saliency_folder, frame, root, dataset, kind):
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    video_id = os.path.basename(os.path.normpath(saliency_folder))
    if dataset == "AVS-ODV":
        frame = frame[:-2] if frame.endswith("_e") else frame
        filename = f"{frame}_e.jpg" if kind == "maps" else f"{frame}_efix.png"
    else:
        frame = f"{int(frame):04d}" if dataset == "SD360" else str(int(frame))
        filename = frame + ".png"
    return os.path.join(root, kind, video_id, filename)

def get_ground_map_path(saliency_folder, saliency_file, ground_truth_root, DatasetName):
    return _ground_path(saliency_folder, saliency_file, ground_truth_root, DatasetName, "maps")

def get_ground_fix_path(saliency_folder, saliency_namefile, ground_truth_root, Dataname):
    return _ground_path(saliency_folder, saliency_namefile, ground_truth_root, Dataname, "fixation")

def get_all_ground_fix(ground_fix_root, Dataname):
    if Dataname not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {Dataname}")
    return [load_ground_fix_from_png(path) for path in sorted((Path(ground_fix_root)/"fixation").glob("*/*.png"))]
