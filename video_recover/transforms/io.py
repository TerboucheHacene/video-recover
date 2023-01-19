import os
from pathlib import Path
from typing import List

import cv2

from video_recover.data_models.video import Frame, Video


class ExtractFrames:
    """Class to extract frames from a video.

    It extracts every n frames. By default, n=1. It creates a folder with the same name
    as the video and saves the frames in that folder. It also creates a video object
    and adds the frames to it.
    """

    def __init__(self, every_n_frames=1):
        self.every_n_frames = every_n_frames

    def __call__(self, root_path: Path, video_name: str, extract_path: Path) -> Video:
        video_path = os.path.join(root_path, video_name)
        frames_path = os.path.join(extract_path, video_name.split(".")[0])
        os.makedirs(frames_path, exist_ok=True)
        # read the video cap
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        # get the frame rate
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        # get the total number of frames
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # get the fourcc
        fourcc = vidcap.get(cv2.CAP_PROP_FOURCC)
        # get the frame width and height
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # intialize a video object
        video = Video(
            name=video_name,
            path=video_path,
            fps=fps,
            resolution=(int(width), int(height)),
            num_frames=int(total_frames),
            fourcc=fourcc,
        )
        count = 0
        while success:
            if count % self.every_n_frames == 0:
                # save frame as JPEG file 0000000000.jpg
                frame_path = os.path.join(frames_path, f"{count:07d}.jpg")
                cv2.imwrite(frame_path, image)
            success, image = vidcap.read()
            count += 1
            # create a frame object
            frame = Frame(name=f"{count:07d}.jpg", path=frame_path, index=count)
            # add the frame to the video
            video.add_frame(frame)
        vidcap.release()

        return video


class VideoFromFrames:
    """
    Class to create a video from a list of frames.

    It creates a video object and adds the frames to it.
    """

    def __call__(
        self, video: Video, output_path: Path, frames_indices: List[int]
    ) -> Video:
        # get the frames paths
        frames_paths = [video.frames[index].path for index in frames_indices]
        output_video_path = os.path.join(output_path, video.name)
        # initialize a video object
        output_video = Video(
            name=video.name,
            path=output_video_path,
            fps=video.fps,
            resolution=video.resolution,
            num_frames=len(frames_indices),
            # fourcc=video.fourcc,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        )
        # initialize a video writer
        out_cap = cv2.VideoWriter(
            output_video_path,
            int(output_video.fourcc),
            output_video.fps,
            (int(output_video.resolution[0]), int(output_video.resolution[1])),
        )
        for i, path in enumerate(frames_paths):
            frame = cv2.imread(path)
            out_cap.write(frame)
            # create a frame object
            frame = Frame(name=os.path.basename(path), path=path, index=i)
            # add the frame to the video
            output_video.add_frame(frame)

        out_cap.release()
        return output_video
