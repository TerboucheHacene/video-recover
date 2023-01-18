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


def main():
    root_path = Path("data/videos/corrupted/")
    output_path = Path("data/videos/recovered/")
    extract_path = Path("data/extracted/")

    video_name = "video_1.mp4"

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
        batch_size=100,
        device="cpu",
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
    args = []
    # generate all the calculations
    for i in range(len(video)):
        for j in range(i + 1, len(video)):
            if i != j:
                args.append((video[j], video[i]))
    # compute the metrics
    metrics = pool.map(batch_metrics, args)

    # fill the metrics in the video
    for i in range(len(video)):
        for j in range(i + 1, len(video)):
            if i == j:
                continue
            pair_metrics = metrics.pop(0)
            video.set_metrics(i, j, pair_metrics)

    # sort the frames
    print("Sorting frames...")
    sort = Sort(ordering_matrix="offset", combine_matrices=False)
    # initial_frame = 28, '../data/frames/0000030.jpg'
    sorted_indices = sort(video, initial_frame=None)
    print("Sorted indices: ", sorted_indices)

    # create a new video from the sorted frames
    video_from_frames = VideoFromFrames()
    sorted_video = video_from_frames(video, output_path, sorted_indices)
    print("Sorted video: ", sorted_video.name)


if __name__ == "__main__":
    main()
