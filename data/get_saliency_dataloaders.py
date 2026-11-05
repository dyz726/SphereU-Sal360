import os
import random
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .DataLoader360Video import AVSSaliencyDataset, SaliencyDataset

SUPPORTED_DATASETS = ("SD360", "AVS-ODV", "SVGC_AVA")
NO_AUGMENTATION = {
    "color_augmentation": False,
    "lr_flip_augmentation": False,
    "yaw_rotation_augmentation": False,
}

def resolve_dataset_split(dataset_name, dataset_split):
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    supported = (1, 2, 3) if dataset_name == "AVS-ODV" else (1,)
    if dataset_split not in supported:
        raise ValueError(f"{dataset_name} supports dataset_split={supported}; got {dataset_split}")
    return dataset_split

def _split_video_ids(dataset_root_dir, data_type, dataset_split, dataset_name):
    list_file = os.path.join(dataset_root_dir, f"{data_type}_list_{dataset_split}.txt")
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"{dataset_name} split file not found: {list_file}")
    with open(list_file, encoding="utf-8") as stream:
        video_ids = [line.split()[0] for line in stream if line.strip()]
    if not video_ids:
        raise ValueError(f"Empty split file: {list_file}")
    if len(video_ids) != len(set(video_ids)):
        raise ValueError(f"Duplicate video IDs in split file: {list_file}")
    return video_ids

def get_dataloaders(
    is_test: bool,
    dataset_name: str,
    dataset_root_dir: Optional[str],
    dataset_kwargs: Dict[str, Any],
    augmentation_kwargs: Dict[str, Any],
    train_batch_size: int,
    val_batch_size: int,
    num_workers: int,
    pin_memory: bool,
    dataset_split: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    if dataset_root_dir is None:
        raise ValueError("dataset_root_dir is required")
    dataset_split = resolve_dataset_split(dataset_name, dataset_split)
    train_videos = _split_video_ids(dataset_root_dir, "train", dataset_split, dataset_name)
    test_videos = _split_video_ids(dataset_root_dir, "test", dataset_split, dataset_name)
    overlap = sorted(set(train_videos) & set(test_videos))
    if overlap:
        raise ValueError(f"{dataset_name} train and test lists overlap: {overlap}")

    dataset_class = AVSSaliencyDataset if dataset_name == "AVS-ODV" else SaliencyDataset

    def make_dataset(video_ids, augmentation, data_type):
        kwargs = dict(
            dataname=dataset_name,
            root_dir=dataset_root_dir,
            video_id=video_ids,
            dataset_kwargs=dataset_kwargs,
            augmentation_kwargs=augmentation,
        )
        if dataset_class is SaliencyDataset:
            kwargs.update(data_type=data_type, include_partial=False)
        return dataset_class(**kwargs)

    if is_test:
        print(f"{dataset_name} (split {dataset_split}) test videos: {len(test_videos)}")
        dataset_train = dataset_val = make_dataset(test_videos, NO_AUGMENTATION, "test")
    else:
        if len(train_videos) < 2:
            raise ValueError(f"At least two {dataset_name} training videos are required")

        rng = random.Random(33)
        rng.shuffle(train_videos)
        split_idx = int(0.8 * len(train_videos))
        val_videos = train_videos[split_idx:]
        train_videos = train_videos[:split_idx]
        print(
            f"{dataset_name} (split {dataset_split}): {len(train_videos)} train, "
            f"{len(val_videos)} validation; {len(test_videos)} test videos held out"
        )
        dataset_train = make_dataset(train_videos, augmentation_kwargs, "train")
        dataset_val = make_dataset(val_videos, NO_AUGMENTATION, "train")

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    loader_train = DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=not is_test, **loader_kwargs
    )
    loader_val = DataLoader(dataset_val, batch_size=val_batch_size, shuffle=False, **loader_kwargs)
    return loader_train, loader_val
