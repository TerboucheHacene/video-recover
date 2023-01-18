from typing import List

import numpy as np
from sklearn.ensemble import IsolationForest

from video_recover.data_models.video import Video


class OutliersDetection:
    def __init__(self):
        self.anomaly_detection_model = IsolationForest(random_state=0, n_estimators=120)

    def __call__(self, video: Video) -> List[int]:
        # get the features
        features = self.get_features(video)
        # fit the model
        self.anomaly_detection_model.fit(features)
        # get the indices of the noisy frames
        noisy_frames_indices = np.where(
            self.anomaly_detection_model.predict(features) == -1
        )[0]
        return noisy_frames_indices.tolist()

    def get_features(self, video: Video) -> np.ndarray:
        features = []
        for frame in video.frames:
            features.append(frame.deep_features)
        features = np.array(features)
        return features


class SimpleOutliersDetection:
    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def __call__(self, video: Video) -> List[int]:
        # get the count matches matrix
        count_matches_matrix = video.distance_metrics.count_matches
        # get the number of frames
        n_frames = len(count_matches_matrix)
        # get the number of matches for each frame
        n_matches = np.sum(count_matches_matrix, axis=1)
        # get the mean number of matches
        mean = np.mean(n_matches)
        # get the standard deviation of the number of matches
        std = np.std(n_matches)
        # get the outliers
        outliers = []
        for i in range(n_frames):
            if n_matches[i] < mean - self.threshold * std:
                outliers.append(i)
        return outliers
