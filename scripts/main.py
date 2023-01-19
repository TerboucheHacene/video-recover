from argparse import ArgumentParser, Namespace
from multiprocessing import Pool, cpu_count
from pathlib import Path

from video_recover.filtering.outliers import OutliersDetection
from video_recover.sorting.sort import Sort
from video_recover.transforms.feature import DeepFeatureExtractor, FeatureDescriptor
from video_recover.transforms.io import ExtractFrames, VideoFromFrames
from video_recover.transforms.match import (
    BatchMetrics,
    DistanceDeepFeatures,
    MatchingMetrics,
)
from video_recover.vision.cnn import get_resnet50_model, get_transforms


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--root_path", type=str, default="data/videos/corrupted/")
    parser.add_argument("--output_path", type=str, default="data/videos/recovered/")
    parser.add_argument("--extract_path", type=str, default="data/extracted/")
    parser.add_argument("--video_name", type=str, default="video_1.mp4")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--ordering_matrix", type=str, default="offset")
    parser.add_argument("--combine_matrices", type=bool, default=False)
    parser.add_argument("--initial_frame", type=int, default=-1)

    return parser.parse_args()


def run(args: Namespace) -> None:
    root_path = Path(args.root_path)
    output_path = Path(args.output_path)
    extract_path = Path(args.extract_path)
    video_name = args.video_name

    print(f"Start the video recovery for {video_name}...")
    # extract the frames
    extract_frames = ExtractFrames()
    print("Extracting frames...")
    video = extract_frames(
        root_path=root_path, video_name=video_name, extract_path=extract_path
    )
    print(f"Number of frames: {len(video)}")

    # get deep features
    print("Getting deep features...")
    deep_feature_extractor = DeepFeatureExtractor(
        model=get_resnet50_model(),
        transforms=get_transforms(),
        batch_size=args.batch_size,
        device=args.device,
    )
    video = deep_feature_extractor(video)
    print("Getting features...")
    feature_descriptor = FeatureDescriptor()
    video = feature_descriptor(video)

    # get the outliers
    print("Getting outliers...")
    outliers_detection = OutliersDetection()
    noisy_frames_indices = outliers_detection(video)

    # get the path of noisy frames
    noisy_frames_paths = [video[index].path for index in noisy_frames_indices]
    print(f"Number of noisy frames: {len(noisy_frames_paths)}")
    print(f"Noisy frames paths: {noisy_frames_paths}")
    print("noisy frames indices: ", noisy_frames_indices)

    # remove the outliers
    print("Removing outliers...")
    video.remove_frame(noisy_frames_indices)

    # get the distance metrics
    distance_deep_features = DistanceDeepFeatures()

    matching_metrics = MatchingMetrics()
    batch_metrics = BatchMetrics(metrics=[distance_deep_features, matching_metrics])

    # compute the metrics between the frames in parrallel (using multiprocessing)
    pool = Pool(processes=cpu_count())
    print("Number of processes: ", cpu_count())
    pool_args = []
    # generate all the calculations
    for i in range(len(video)):
        for j in range(i + 1, len(video)):
            if i != j:
                pool_args.append((video[j], video[i]))
    # compute the metrics
    metrics = pool.map(batch_metrics, pool_args)
    pool.close()
    pool.join()

    # fill the metrics in the video
    for i in range(len(video)):
        for j in range(i + 1, len(video)):
            if i == j:
                continue
            pair_metrics = metrics.pop(0)
            video.set_metrics(i, j, pair_metrics)

    # sort the frames
    print("Sorting frames...")
    sort = Sort(
        ordering_matrix=args.ordering_matrix, combine_matrices=args.combine_matrices
    )
    # initial_frame = 28, '../data/frames/0000030.jpg'
    # the algorithm will start from this frame and will try to sort the frames
    # if you don't know which frame to choose, you can set it to -1
    sorted_indices = sort(video, initial_frame=args.initial_frame)
    print("Sorted indices: ", sorted_indices)

    # create a new video from the sorted frames
    video_from_frames = VideoFromFrames()
    sorted_video = video_from_frames(video, output_path, sorted_indices)
    print("Sorted video: ", sorted_video.name)


if __name__ == "__main__":
    args = parse_args()
    run(args)
