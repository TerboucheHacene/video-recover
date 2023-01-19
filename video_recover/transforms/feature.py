from typing import Any, Callable, Dict, List

import cv2
import torch
from PIL import Image

from video_recover.data_models.video import Video


class DeepFeatureExtractor:
    """Class to extract the deep features of the frames in a video."""

    def __init__(
        self,
        model: torch.nn.Module,
        transforms: Callable,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.model = model
        self.transforms = transforms
        self.batch_size = batch_size
        self.device = device
        # set the model to eval mode
        self.model.eval()
        # move the model to the device
        self.model.to(self.device)

    def __call__(self, video: Video) -> Video:
        # load the frames per batch and extract the deep features
        for i in range(0, len(video), self.batch_size):
            # get the frames
            frames_objects = video[i : i + self.batch_size]
            # get the frames paths
            frames_paths = [frame.path for frame in frames_objects]
            # load the frames
            frames = [Image.open(path) for path in frames_paths]
            # convert to RGB
            frames = [frame.convert("RGB") for frame in frames]
            # apply the transforms
            frames = [self.transforms(frame) for frame in frames]
            # stack the frames
            frames_tensor: torch.Tensor = torch.stack(frames)

            # forward pass
            with torch.no_grad():
                deep_features = self.model(frames_tensor.to(self.device)).squeeze()
            # convert to numpy
            deep_features = deep_features.cpu().numpy()
            # set the deep features
            for frame, deep_feature in zip(frames_objects, deep_features):
                frame.set_deep_features(deep_feature)
        return video


class FeatureDescriptor:
    """Class to extract the features and descriptors of the frames in a video."""

    def __init__(self, feature_extractor: str = "ORB"):
        descriptor = {
            "ORB": cv2.ORB_create,
            "SIFT": cv2.SIFT_create,
            "SURF": cv2.xfeatures2d.SURF_create,
        }
        # check if the feature extractor is valid
        if feature_extractor not in descriptor:
            raise ValueError(
                f"Invalid feature extractor: {feature_extractor}. Valid values are: ",
                f"{list(descriptor.keys())}",
            )
        self.feature_extractor = descriptor[feature_extractor]()

    def __call__(self, video: Video) -> Video:
        # extract the features and descriptors
        for frame_object in iter(video):
            # load the frame
            frame = cv2.imread(frame_object.path)
            # convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # extract the features
            key_points, descriptors = self.feature_extractor.detectAndCompute(
                frame, None
            )
            # convert the key points to dict
            key_points = self.convert_cv_key_points_to_dict(key_points)
            # set the features and descriptors
            frame_object.set_key_points(key_points)
            frame_object.set_descriptors(descriptors)
        return video

    @staticmethod
    def convert_cv_key_points_to_dict(key_points) -> List[Dict[str, Any]]:
        """Convert the key points to a list of dicts.

        This is needed to be able to serialize the key points when using the
        multiprocessing module.
        """
        return [
            {
                "pt": key_point.pt,
                "size": key_point.size,
                "angle": key_point.angle,
                "response": key_point.response,
                "octave": key_point.octave,
                "class_id": key_point.class_id,
            }
            for key_point in key_points
        ]
