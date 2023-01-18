from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclasses.dataclass
class DistanceMetrics:
    """Class to store the distance metrics between frames."""

    count_matches: np.ndarray
    offset: np.ndarray
    descriptor_distance: np.ndarray
    deep_feats_distance: np.ndarray


@dataclasses.dataclass
class Video:
    name: str
    path: str
    fps: float
    resolution: Tuple[int, int]
    num_frames: int
    fourcc: str
    duration: float = 0.0
    frames: List[Frame] = dataclasses.field(default_factory=list)
    distance_metrics: DistanceMetrics = dataclasses.field(init=False)

    def __post_init__(self):
        self.duration = self.num_frames / self.fps
        self.distance_metrics = DistanceMetrics(
            count_matches=np.zeros((self.num_frames, self.num_frames)),
            offset=np.zeros((self.num_frames, self.num_frames)),
            descriptor_distance=np.zeros((self.num_frames, self.num_frames)),
            deep_feats_distance=np.zeros((self.num_frames, self.num_frames)),
        )

    def add_frame(self, frame: Frame) -> None:
        self.frames.append(frame)

    def remove_frame(self, indices: List[int]) -> None:
        # remove the frames
        self.frames = [frame for i, frame in enumerate(self.frames) if i not in indices]

        # change the index of the frames
        for i in range(len(self.frames)):
            self.frames[i].index = i
        # remove the rows and columns of the distance metrics
        self.distance_metrics.count_matches = np.delete(
            self.distance_metrics.count_matches, indices, axis=0
        )
        self.distance_metrics.count_matches = np.delete(
            self.distance_metrics.count_matches, indices, axis=1
        )
        self.distance_metrics.offset = np.delete(
            self.distance_metrics.offset, indices, axis=0
        )
        self.distance_metrics.offset = np.delete(
            self.distance_metrics.offset, indices, axis=1
        )
        self.distance_metrics.descriptor_distance = np.delete(
            self.distance_metrics.descriptor_distance, indices, axis=0
        )
        self.distance_metrics.descriptor_distance = np.delete(
            self.distance_metrics.descriptor_distance, indices, axis=1
        )
        self.distance_metrics.deep_feats_distance = np.delete(
            self.distance_metrics.deep_feats_distance, indices, axis=0
        )
        self.distance_metrics.deep_feats_distance = np.delete(
            self.distance_metrics.deep_feats_distance, indices, axis=1
        )
        # update the number of frames
        self.num_frames = len(self.frames)
        self.duration = self.num_frames / self.fps

    def __str__(self):
        return f"Video: {self.name}, {self.num_frames} frames, {self.duration} seconds"

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]

    def set_metrics(self, i: int, j: int, metrics: Dict[str, float]) -> None:
        # set the value of the metrics between the frames i and j
        self.distance_metrics.count_matches[i, j] = metrics["count_matches"]
        self.distance_metrics.offset[i, j] = metrics["offset"]
        self.distance_metrics.descriptor_distance[i, j] = metrics["descriptor_distance"]
        self.distance_metrics.deep_feats_distance[i, j] = metrics["deep_feats_distance"]
        # set the value of the metrics between the frames j and i
        self.distance_metrics.count_matches[j, i] = metrics["count_matches"]
        self.distance_metrics.offset[j, i] = metrics["offset"]
        self.distance_metrics.descriptor_distance[j, i] = metrics["descriptor_distance"]
        self.distance_metrics.deep_feats_distance[j, i] = metrics["deep_feats_distance"]


@dataclasses.dataclass
class Frame:
    name: str
    path: str
    index: int = 0
    key_points: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    descriptors: np.ndarray = None
    deep_features: np.ndarray = None

    def set_deep_features(self, deep_features: np.ndarray) -> None:
        self.deep_features = deep_features

    def set_key_points(self, key_points: List[Dict[str, Any]]) -> None:
        self.key_points = key_points

    def set_descriptors(self, descriptors: np.ndarray) -> None:
        self.descriptors = descriptors

    def __str__(self):
        return f"Frame: {self.name}"

    def __repr__(self):
        return f"Frame: {self.name}"
