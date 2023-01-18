from typing import List, Tuple

import numpy as np

from video_recover.data_models.video import Video


class Sort:
    """Class to sort the frames in a video."""

    def __init__(self, ordering_matrix: str = "offset", combine_matrices: bool = False):
        # check if ordering_matrix is valid
        if ordering_matrix not in [
            "offset",
            "count_matches",
            "descriptor_distance",
            "deep_feats_distance",
        ]:
            raise ValueError("Invalid ordering matrix")

        self.ordering_matrix = ordering_matrix
        self.combine = combine_matrices

    def __call__(
        self,
        video: Video,
        initial_frame: int = -1,
    ) -> List[int]:
        """Sort the frames in the video using the distance metrics."""
        distance_metrics = video.distance_metrics
        # Get the ordering matrix
        if not self.combine:
            ordering_matrix = getattr(distance_metrics, self.ordering_matrix)
        else:
            offset = getattr(distance_metrics, "offset")
            descriptor_distance = getattr(distance_metrics, "descriptor_distance")
            deep_feats_distance = getattr(distance_metrics, "deep_feats_distance")
            # normalize the matrices
            offset = offset / np.max(offset)
            descriptor_distance = descriptor_distance / np.max(descriptor_distance)
            deep_feats_distance = deep_feats_distance / np.max(deep_feats_distance)
            # combine the matrices
            ordering_matrix = offset + descriptor_distance + deep_feats_distance
        # get the count matches matrix
        count_matches = getattr(distance_metrics, "count_matches")

        # sort the frames
        sorted_indices = self.sort(count_matches, ordering_matrix, initial_frame)
        return sorted_indices

    def sort(
        self, count_matches_matrix, ordering_matrix, initial_frame: int = -1
    ) -> List[int]:
        """Sort the frames in the video using the distance metrics."""
        # create a frame groups array
        frame_groups = [[i] for i in range(len(count_matches_matrix))]

        # while not all frames are combined
        while len(frame_groups) > 1:
            # find the most powerful frame match
            if initial_frame != -1:
                sub_ordering_matrix = ordering_matrix.copy()[initial_frame].reshape(
                    1, -1
                )
                sub_count_matches_matrix = count_matches_matrix.copy()[
                    initial_frame
                ].reshape(1, -1)
                # get the best match
                best_match = self.get_best_match(
                    sub_count_matches_matrix, sub_ordering_matrix
                )
                best_match = (initial_frame, best_match[1])
                print(count_matches_matrix.shape)
            else:
                sub_ordering_matrix = ordering_matrix.copy()
                sub_count_matches_matrix = count_matches_matrix.copy()
                # get the best match
                best_match = self.get_best_match(
                    sub_count_matches_matrix, sub_ordering_matrix
                )
            im, jm = best_match
            if any([(im in group) and (jm in group) for group in frame_groups]):
                count_matches_matrix[im, jm] = -1
            else:
                # combine the frames
                new_group = self.update_frame_groups(frame_groups, best_match)
                # update the count matches matrix
                initial_frame = self.update_count_matches_matrix(
                    count_matches_matrix, new_group, best_match, initial_frame
                )
        return frame_groups[0][::-1]

    @staticmethod
    def get_best_match(
        match_matrix: np.ndarray, distance_matrix: np.ndarray
    ) -> Tuple[int, int]:
        """Get the best match from the match matrix and distance matrix."""
        match_matrix = np.triu(match_matrix)
        np.fill_diagonal(match_matrix, -1)
        max_matches = np.max(match_matrix)
        # get all the pairs with the maximum number of matches
        matched_frame_pairs = np.where(match_matrix == max_matches)
        # get the distances of the matched frame pairs
        distance_pairs = distance_matrix[matched_frame_pairs]
        # sort the pairs based on the distance
        pairs = [*zip(matched_frame_pairs[0], matched_frame_pairs[1], distance_pairs)]
        pairs = sorted(pairs, key=lambda x: x[2])
        return pairs[0][:2]

    @staticmethod
    def update_frame_groups(
        frame_groups: List[List[int]], best_match: Tuple[int, int]
    ) -> List[int]:
        """Update the frame groups array."""
        im, jm = best_match
        group1 = [group for group in frame_groups if im in group][0]
        group2 = [group for group in frame_groups if jm in group][0]
        # get the index if im, jm in the group
        im_index = group1.index(im)
        jm_index = group2.index(jm)
        # create a new group based on the indices of im, jm
        # im is in the end of group1 and jm is in the beginning of group2
        if im_index == len(group1) - 1 and jm_index == 0:
            new_group = group1 + group2
        # im is in the beginning of group1 and jm is in the end of group2
        elif im_index == 0 and jm_index == len(group2) - 1:
            new_group = group2 + group1
        # im is in the beginning of group1 and jm is in the beginning of group2
        elif im_index == 0 and jm_index == 0:
            new_group = group2[::-1] + group1
        # im is in the end of group1 and jm is in the end of group2
        elif im_index == len(group1) - 1 and jm_index == len(group2) - 1:
            new_group = group1 + group2[::-1]
        # remove the groups from the frame groups array and add the new group

        frame_groups.remove(group1)
        frame_groups.remove(group2)
        frame_groups.append(new_group)
        return new_group

    @staticmethod
    def update_count_matches_matrix(
        count_matches_matrix, new_group, best_match, initial_frame
    ) -> int:
        """Update the count matches matrix."""
        im, jm = best_match
        if initial_frame is not None:
            initial_frame = -1
            count_matches_matrix[initial_frame, :] = -1
            count_matches_matrix[:, initial_frame] = -1

        if len(new_group) > 2:
            # If im, jm are in the middle of a group, then set their row and column to -1
            # so that they are not matched again with any other frame
            im_index = new_group.index(im)
            jm_index = new_group.index(jm)
            if im_index != 0 and im_index != len(new_group) - 1:
                count_matches_matrix[im, :] = -1
                count_matches_matrix[:, im] = -1
            if jm_index != 0 and jm_index != len(new_group) - 1:
                count_matches_matrix[jm, :] = -1
                count_matches_matrix[:, jm] = -1
        return initial_frame
