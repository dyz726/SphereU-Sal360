import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from trimesh_utils import IcoSphereRef, asSpherical

def _build_sphere_grid(sphere_node_type, sphere_rank):
    icosphere_ref = IcoSphereRef(sphere_node_type)
    normals = icosphere_ref.get_normals(rank=sphere_rank)
    normals_rphitheta = asSpherical(normals)
    normals_wh = np.stack(
        (
            normals_rphitheta[:, 2] / 180,
            normals_rphitheta[:, 1] / 180 * 2 - 1,
        ),
        axis=1,
    ).astype(np.float32)
    return torch.from_numpy(normals_wh).reshape(1, -1, 1, 2)

def _sample_erp_periodic(tensor, sphere_grid_tensor, mode="bilinear"):
    width = tensor.shape[-1]
    padded = F.pad(tensor, (1, 1, 0, 0), mode="circular")
    periodic_grid = sphere_grid_tensor.clone()
    periodic_grid[..., 0] *= width / (width + 2.0)
    return F.grid_sample(
        padded,
        periodic_grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )

def _sample_to_sphere(sphere_grid_tensor, rgb, saliency, fixation):
    rgb_tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().unsqueeze(0)
    rgb_sampled = _sample_erp_periodic(
        rgb_tensor, sphere_grid_tensor
    ).squeeze(0).squeeze(-1).transpose(0, 1).numpy()

    def sample_grayscale(image):
        tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).unsqueeze(0).float()
        return _sample_erp_periodic(
            tensor, sphere_grid_tensor
        ).reshape(-1).numpy().astype(np.float32)

    fixation_tensor = (
        torch.from_numpy(np.ascontiguousarray(fixation))
        .unsqueeze(0)
        .unsqueeze(0)
        .float()
    )
    sphere_fixation = _sample_erp_periodic(
        fixation_tensor,
        sphere_grid_tensor,
        mode="nearest",
    ).reshape(-1).numpy().astype(np.float32)
    return rgb_sampled, sample_grayscale(saliency), sphere_fixation

def _sample_clip_augmentation(augmentation_kwargs):
    yaw_ratio = 0.0
    if augmentation_kwargs.get("yaw_rotation_augmentation", False) and random.random() < 0.5:
        yaw_ratio = random.random()
    lr_flip = (
        augmentation_kwargs.get("lr_flip_augmentation", False)
        and random.random() < 0.5
    )
    color_factors = None
    if augmentation_kwargs.get("color_augmentation", False) and random.random() < 0.5:
        color_factors = tuple(random.uniform(0.9, 1.1) for _ in range(3))
    return yaw_ratio, lr_flip, color_factors

def _transform_sphere_grid(sphere_grid_tensor, yaw_ratio, lr_flip):
    if not yaw_ratio and not lr_flip:
        return sphere_grid_tensor
    grid = sphere_grid_tensor.clone()
    longitude = -grid[..., 0] if lr_flip else grid[..., 0]
    grid[..., 0] = torch.remainder(longitude - 2.0 * yaw_ratio + 1.0, 2.0) - 1.0
    return grid

def _transform_erp_saliency(saliency, yaw_ratio, lr_flip):
    if yaw_ratio:
        shift = yaw_ratio * saliency.shape[1]
        integer_shift = int(shift)
        fraction = shift - integer_shift
        saliency = saliency.astype(np.float32)
        shifted = np.roll(saliency, integer_shift, axis=1)
        if fraction:
            shifted = (
                (1.0 - fraction) * shifted
                + fraction * np.roll(saliency, integer_shift + 1, axis=1)
            )
        saliency = shifted
    if lr_flip:
        saliency = np.fliplr(saliency)
    return saliency

def _apply_color_augmentation(sphere_rgb, color_factors):
    if color_factors is None:
        return sphere_rgb
    brightness, contrast, saturation = color_factors
    gray = np.sum(
        sphere_rgb * np.array([0.2989, 0.5870, 0.1140], dtype=np.float32),
        axis=-1,
        keepdims=True,
    )
    rgb = gray + saturation * (sphere_rgb - gray)
    mean_gray = gray.mean()
    rgb = mean_gray + contrast * (rgb - mean_gray)
    return np.clip(rgb * brightness, 0.0, 255.0).astype(np.float32)

class SaliencyDataset(Dataset):

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

    def __init__(
        self,
        dataname,
        root_dir,
        video_id,
        dataset_kwargs,
        augmentation_kwargs=None,
        data_type="train",
        include_partial=False,
    ):
        self.dataname = dataname
        self.root_dir = Path(root_dir)
        self.video_ids = [str(item) for item in video_id]
        self.seq_length = dataset_kwargs["seq_length"]
        self.sphere_rank = dataset_kwargs["sphere_rank"]
        self.sphere_node_type = dataset_kwargs["sphere_node_type"]
        self.data_type = data_type
        self.include_partial = include_partial

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")
        if data_type not in {"train", "test"}:
            raise ValueError(f"Unsupported data_type: {data_type}")

        self.sphere_grid_tensor = _build_sphere_grid(self.sphere_node_type, self.sphere_rank)

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])
        self.augmentation_kwargs = augmentation_kwargs or {}

        if self.dataname not in {"SD360", "SVGC_AVA"}:
            raise ValueError(f"Unsupported video dataset: {self.dataname}")
        self.video_base_dir = self.root_dir / "videos"
        self.annotation_base_dir = self.root_dir
        self.data_list = self._build_video_data_list()

        if not self.data_list:
            raise RuntimeError(
                f"No valid samples found for {self.dataname} under {self.root_dir} "
                f"(split={self.data_type})"
            )
        print(
            f"Loaded {len(self.data_list)} {self.data_type} clips from "
            f"{len(self.video_ids)} videos (sphere={self.sphere_node_type}:{self.sphere_rank})"
        )

    def _find_video_file(self, video_id):
        exact_candidate = self.video_base_dir / video_id
        if (
            exact_candidate.is_file()
            and exact_candidate.suffix.lower() in self.VIDEO_EXTENSIONS
        ):
            return exact_candidate
        for extension in self.VIDEO_EXTENSIONS:
            candidate = self.video_base_dir / f"{video_id}{extension}"
            if candidate.is_file():
                return candidate
            candidate = self.video_base_dir / f"{video_id}{extension.upper()}"
            if candidate.is_file():
                return candidate
        return None

    @staticmethod
    def _indexed_annotations(directory):
        files = {}
        for path in Path(directory).iterdir():
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            try:
                frame_number = int(path.stem)
            except ValueError:
                continue
            files[frame_number] = path
        return files

    def _build_video_data_list(self):
        data_list = []
        for video_id in self.video_ids:
            video_path = self._find_video_file(video_id)
            annotation_dir = self.annotation_base_dir
            maps_dir = annotation_dir / "maps" / video_id
            fixation_dir = annotation_dir / "fixation" / video_id
            if video_path is None:
                print(f"Warning: video not found for {video_id} in {self.video_base_dir}")
                continue
            if not maps_dir.is_dir() or not fixation_dir.is_dir():
                print(f"Warning: annotations not found for {video_id} in {annotation_dir}")
                continue

            saliency_files = self._indexed_annotations(maps_dir)
            fixation_files = self._indexed_annotations(fixation_dir)
            paired_frame_numbers = set(saliency_files) & set(fixation_files)
            if not paired_frame_numbers:
                print(
                    f"Warning: skipping {video_id}; no paired map/fixation annotations found "
                    f"(maps={len(saliency_files)}, fixation={len(fixation_files)})"
                )
                continue

            capture = cv2.VideoCapture(str(video_path))
            total_video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if 0 in paired_frame_numbers:
                label_start_index = 0
            elif total_video_frames in paired_frame_numbers:
                label_start_index = 1
            else:
                print(
                    f"Warning: cannot confirm annotation index base for {video_id}; "
                    "assuming 1-based labels"
                )
                label_start_index = 1

            missing_clips = 0
            for start in range(0, total_video_frames, self.seq_length):
                valid_length = min(self.seq_length, total_video_frames - start)
                if valid_length < self.seq_length and not self.include_partial:
                    break
                frame_indices = list(range(start, start + valid_length))
                if valid_length < self.seq_length:
                    frame_indices.extend([frame_indices[-1]] * (self.seq_length - valid_length))
                label_numbers = [index + label_start_index for index in frame_indices]
                if not all(
                    number in saliency_files and number in fixation_files
                    for number in label_numbers
                ):
                    missing_clips += 1
                    continue
                data_list.append(
                    {
                        "video_path": str(video_path),
                        "frame_indices": frame_indices,
                        "sal_seq": [str(saliency_files[number]) for number in label_numbers],
                        "fix_seq": [str(fixation_files[number]) for number in label_numbers],
                        "videoID": video_id,
                        "valid_length": valid_length,
                    }
                )
            if missing_clips:
                print(
                    f"Warning: skipped {missing_clips} clips from {video_id} "
                    "because map or fixation frames are missing"
                )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def _iter_video_frames(self, video_path, frame_indices):
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        previous_index = frame_indices[0] - 1
        previous_frame = None
        try:
            for frame_index in frame_indices:
                if frame_index == previous_index:
                    yield previous_frame
                    continue
                if frame_index != previous_index + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Failed to decode frame {frame_index} from {video_path}")
                previous_index = frame_index
                previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield previous_frame
        finally:
            capture.release()

    def __getitem__(self, idx):
        item = self.data_list[idx]
        rgb_source = self._iter_video_frames(item["video_path"], item["frame_indices"])
        yaw_ratio, lr_flip, color_factors = _sample_clip_augmentation(self.augmentation_kwargs)
        sphere_grid = _transform_sphere_grid(self.sphere_grid_tensor, yaw_ratio, lr_flip)

        sphere_rgbs = []
        sphere_sals = []
        sphere_fixations = []
        sals = []
        for rgb, sal_path, fix_path in zip(rgb_source, item["sal_seq"], item["fix_seq"]):
            saliency = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            fixation = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
            if saliency is None or fixation is None:
                raise RuntimeError(f"Failed to read annotations: {sal_path}, {fix_path}")
            sphere_rgb, sphere_sal, sphere_fix = self._convert_to_sphere(
                rgb, saliency, fixation, sphere_grid_tensor=sphere_grid
            )
            sphere_rgbs.append(_apply_color_augmentation(sphere_rgb, color_factors))
            sphere_sals.append(sphere_sal)
            sphere_fixations.append(sphere_fix)
            sals.append(_transform_erp_saliency(saliency, yaw_ratio, lr_flip))

        if len(sphere_rgbs) != self.seq_length:
            raise RuntimeError(
                f"Expected {self.seq_length} frames but decoded {len(sphere_rgbs)} "
                f"from {item['video_path']}"
            )

        sphere_rgb = torch.from_numpy(np.stack(sphere_rgbs)).float().div_(255.0)
        sphere_sal = torch.from_numpy(np.stack(sphere_sals)).float()
        sphere_fix = torch.from_numpy(np.stack(sphere_fixations)).float()
        return {
            "sphere_rgb": sphere_rgb,
            "sphere_sal": sphere_sal,
            "sphere_fix": sphere_fix,
            "erp_sal": torch.from_numpy(np.stack(sals)).float().div_(255.0),
            "normalized_sphere_rgb": (sphere_rgb - self.norm_mean) / self.norm_std,
            "normalized_sphere_sal": sphere_sal / 255.0,
            "normalized_sphere_fix": sphere_fix / 255.0,
            "seq_path": item["sal_seq"],
            "videoID": item["videoID"],
            "valid_length": item["valid_length"],
        }

    def _convert_to_sphere(self, rgb, saliency, fixation, sphere_grid_tensor=None):
        if sphere_grid_tensor is None:
            sphere_grid_tensor = self.sphere_grid_tensor
        return _sample_to_sphere(sphere_grid_tensor, rgb, saliency, fixation)

class AVSSaliencyDataset(Dataset):

    IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, dataname, root_dir, video_id, dataset_kwargs, augmentation_kwargs=None):
        self.dataname = dataname
        self.root_dir = Path(root_dir)
        self.video_ids = [str(item) for item in video_id]
        self.seq_length = dataset_kwargs["seq_length"]
        self.sphere_rank = dataset_kwargs["sphere_rank"]
        self.sphere_node_type = dataset_kwargs["sphere_node_type"]

        if self.seq_length <= 0:
            raise ValueError("seq_length must be positive")

        self.sphere_grid_tensor = _build_sphere_grid(self.sphere_node_type, self.sphere_rank)

        self.norm_mean = torch.tensor([0.485, 0.456, 0.406])
        self.norm_std = torch.tensor([0.229, 0.224, 0.225])

        self.augmentation_kwargs = augmentation_kwargs or {}

        self.data_list = self._build_data_list()
        if not self.data_list:
            raise RuntimeError(
                f"No valid samples found for {self.dataname} under {self.root_dir}"
            )
        print(
            f"Loaded {len(self.data_list)} clips from {len(self.video_ids)} videos "
            f"(sphere={self.sphere_node_type}:{self.sphere_rank})"
        )

    def _build_data_list(self):
        data_list = []
        for video_id in self.video_ids:
            frames_dir = self.root_dir / "frames" / video_id
            maps_dir = self.root_dir / "maps" / video_id
            fixation_dir = self.root_dir / "fixation" / video_id
            if not (frames_dir.is_dir() and maps_dir.is_dir() and fixation_dir.is_dir()):
                print(f"Warning: skipping {video_id}; frames/maps/fixation not found under {self.root_dir}")
                continue

            frame_files = sorted(
                p for p in frames_dir.iterdir()
                if p.suffix.lower() in self.IMAGE_EXTENSIONS
            )

            for start in range(0, len(frame_files) - self.seq_length + 1, self.seq_length):
                rgb_seq = []
                sal_seq = []
                fix_seq = []
                for frame_path in frame_files[start:start + self.seq_length]:
                    base_name = frame_path.stem
                    sal_path = maps_dir / f"{base_name}_e.jpg"
                    fix_path = fixation_dir / f"{base_name}_efix.png"
                    if sal_path.is_file() and fix_path.is_file():
                        rgb_seq.append(str(frame_path))
                        sal_seq.append(str(sal_path))
                        fix_seq.append(str(fix_path))
                    else:
                        print(f"Warning: annotations missing for {video_id}/{base_name}, drop this clip")
                if len(rgb_seq) == self.seq_length:
                    data_list.append(
                        {
                            "rgb_seq": rgb_seq,
                            "sal_seq": sal_seq,
                            "fix_seq": fix_seq,
                            "videoID": video_id,
                        }
                    )
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        yaw_ratio, lr_flip, color_factors = _sample_clip_augmentation(self.augmentation_kwargs)
        sphere_grid = _transform_sphere_grid(self.sphere_grid_tensor, yaw_ratio, lr_flip)

        sals = []
        sphere_rgbs = []
        sphere_sals = []
        sphere_fixs = []
        for rgb_path, sal_path, fix_path in zip(item["rgb_seq"], item["sal_seq"], item["fix_seq"]):
            rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
            sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
            fix = cv2.imread(fix_path, cv2.IMREAD_GRAYSCALE)
            if rgb is None or sal is None or fix is None:
                raise RuntimeError(f"Corrupted data: {rgb_path}, {sal_path}, {fix_path}")

            sphere_rgb, sphere_sal, sphere_fix = _sample_to_sphere(
                sphere_grid,
                rgb,
                sal,
                fix,
            )
            sphere_rgbs.append(_apply_color_augmentation(sphere_rgb, color_factors))
            sphere_sals.append(sphere_sal)
            sphere_fixs.append(sphere_fix)
            sals.append(_transform_erp_saliency(sal, yaw_ratio, lr_flip))

        sphere_rgb = torch.from_numpy(np.stack(sphere_rgbs)).float().div_(255.0)
        sphere_sal = torch.from_numpy(np.stack(sphere_sals)).float()
        sphere_fix = torch.from_numpy(np.stack(sphere_fixs)).float()
        return {
            "sphere_rgb": sphere_rgb,
            "sphere_sal": sphere_sal,
            "sphere_fix": sphere_fix,
            "erp_sal": torch.from_numpy(np.stack(sals)).float().div_(255.0),
            "normalized_sphere_rgb": (sphere_rgb - self.norm_mean) / self.norm_std,
            "normalized_sphere_sal": sphere_sal / 255.0,
            "normalized_sphere_fix": sphere_fix / 255.0,
            "seq_path": item["sal_seq"],
            "videoID": item["videoID"],
            "valid_length": self.seq_length,
        }
