from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import cv2
import numpy as np

from video_recover.data_models.video import Frame


class InterFrameMetric(ABC):
    """Base class for all inter-frames metrics."""

    @abstractmethod
    def __call__(self, query_frame: Frame, train_frame: Frame) -> Dict[str, float]:
        """Compute the metrics between two frames."""
        pass


class MatchingMetrics(InterFrameMetric):
    """Compute the matching metrics between two frames.

    - Count matches: the number of matches between the two frames.
    - Offset: the `average` or `median` distance between the matched key points.
    - Distance: the `average` or `median` distance between the matched descriptors.
    """

    def __init__(
        self, matcher_name: str = "FLANN", aggregator_name: str = "median"
    ) -> None:

        self.matcher_name = matcher_name
        aggregators = {"median": np.median, "mean": np.mean}
        # check if the aggregator is valid
        if aggregator_name not in aggregators:
            raise ValueError(
                f"Invalid aggregator: {aggregators}. Valid values are:",
                f" {list(aggregators.keys())}",
            )
        self.aggregator = aggregators[aggregator_name]

    def __call__(self, query_frame: Frame, train_frame: Frame) -> Dict[str, float]:
        """Compute the metrics between two frames."""
        # create the matcher object based on the matcher name
        # the matcher is created here and not in the __init__ method to avoid the
        # problem of the multiprocessing module when using the cv2 module
        # ( the multiprocessing module cannot pickle the cv2 module)

        if self.matcher_name == "FLANN":
            index_params = dict(
                algorithm=6, table_number=6, key_size=12, multi_probe_level=2
            )
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif self.matcher_name == "BF":
            matcher = cv2.BFMatcher()

        # match the features
        matches = matcher.knnMatch(query_frame.descriptors, train_frame.descriptors, k=2)
        # filter the matches
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # compute "Offset" metric & "Distance" metric
        offsets = []
        distances = []
        for match in matches:
            # get the query and train key points
            query_key_point = query_frame.key_points[match.queryIdx]
            train_key_point = train_frame.key_points[match.trainIdx]
            # get the coordinates
            assert "pt" in query_key_point.keys()
            assert "pt" in train_key_point.keys()
            x1, y1 = query_key_point["pt"]
            x2, y2 = train_key_point["pt"]
            # compute the offset
            offset = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            offsets.append(offset)
            # append the distance
            distances.append(match.distance)

        # compute the "Offset" metric
        offset = self.aggregator(offsets)
        # compute the "Distance" metric
        distance = self.aggregator(distances)
        return {
            "count_matches": len(matches),
            "offset": offset,
            "descriptor_distance": distance,
        }


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def cosine_distance(x, y):
    return (1 - cosine_similarity(x, y)) / 2


class DistanceDeepFeatures(InterFrameMetric):
    """Compute the distance between the deep features of two frames."""

    def __init__(self, distance_type: str = "cosine") -> None:

        distance = {
            "cosine": cosine_distance,
            "euclidean": euclidean_distance,
        }
        # check if the distance is valid
        if distance_type not in distance:
            raise ValueError(
                f"Invalid distance: {distance_type}. Valid values are:",
                f" {list(distance.keys())}",
            )
        self.distance = distance[distance_type]

    def __call__(self, query_frame: Frame, train_frame: Frame) -> Dict[str, float]:
        # compute the distance between the deep features
        distance = self.distance(query_frame.deep_features, train_frame.deep_features)
        return {"deep_feats_distance": distance}


class BatchMetrics:
    """Compute the metrics between two frames."""

    def __init__(self, metrics: List[InterFrameMetric]) -> None:
        self._metrics = metrics

    @property
    def metrics(self) -> List[InterFrameMetric]:
        return self._metrics

    @metrics.setter
    def metrics(self, metrics: List[InterFrameMetric]) -> None:
        self._metrics = metrics

    def __call__(self, args: Tuple[Frame, Frame]) -> Dict[str, float]:
        query_frame, train_frame = args
        print(
            f"Computing metrics between frame {query_frame.index} and frame ",
            f"{train_frame.index}",
        )
        metrics = {}
        for metric in self.metrics:
            metrics.update(metric(query_frame, train_frame))
        return metrics
