"""
This module provides an extension to learn the features of the Blobinator calibration sheet.
"""

import cv2 as cv
import numpy as np
import pdf2image
import torch


MAX_ROTATION = 90
MIN_SCALE = 0.2
MAX_SCALE = 0.8
MAX_WIGGLE = 0.2

IMAGE_SIZE = (1024, 1024)


class BlobinatorDataset(torch.utils.data.IterableDataset):
    """
    This is the Blobinator dataset.

    It warps the Blobinator sheet into a number of background images via randomly generated homographies.
    """

    def __init__(
        self,
        blob_sheet_path: str,
    ):
        self.blob_sheet = np.array(pdf2image.convert_from_path(blob_sheet_path)[0])
        self.keypoints = list(
            map(
                lambda k: (
                    (
                        k[0][0] * self.blob_sheet.shape[0],
                        k[0][1] * self.blob_sheet.shape[1],
                    ),
                    k[1],
                    k[2],
                ),
                [
                    ((0.2, 0.2), 100, 0),
                    ((0.8, 0.2), 100, 0),
                    ((0.8, 0.8), 100, 0),
                    ((0.2, 0.8), 100, 0),
                ],
            )
        )
        self.backgrounds = np.zeros(shape=(256, *IMAGE_SIZE, 3))

    def __iter__(self):
        for background in self.backgrounds:
            transform = self.generate_homography()
            mapped_image = self.map_blobs(background, transform)
            mapped_keypoints = self.map_keypoints(transform)
            filtered_keypoints = []
            filtered_mapped_keypoints = []
            for k, k_mapped in zip(self.keypoints, mapped_keypoints):
                (x_mapped, y_mapped), _, _ = k_mapped
                if (
                    x_mapped >= 0
                    and x_mapped < self.backgrounds.shape[1]
                    and y_mapped >= 0
                    and y_mapped < self.backgrounds.shape[2]
                ):
                    filtered_keypoints.append(k)
                    filtered_mapped_keypoints.append(k_mapped)
            yield (
                self.blob_sheet,
                mapped_image,
                filtered_keypoints,
                filtered_mapped_keypoints,
            )

    def generate_homography(self) -> np.ndarray:
        """
        Generate a random homography according to the configuration values.

        The configurable values are:
        - `MAX_ROTATION`: The maximum rotation either clock-wise or counter clockwise in degrees. (Note: the range of
            rotations is [`-MAX_ROTATION`°, `MAX_ROTATION`°])
        - `MIN_SCALE`: The smallest possible scaling factor.
        - `MAX_SCALE`: The largest possible scaling factor.
        - `MAX_WIGGLE`: The amount points can be wiggled in the x and y direction. This value is a fraction of the
            shorter sidelength of the blobsheet.
        """
        scale = np.random.uniform(MIN_SCALE, MAX_SCALE)
        rotation = np.random.uniform(-MAX_ROTATION, MAX_ROTATION) * np.pi / 180
        translation_x = np.random.uniform(
            0, self.backgrounds.shape[1] - self.blob_sheet.shape[0]
        )
        translation_y = np.random.uniform(
            0, self.backgrounds.shape[2] - self.blob_sheet.shape[1]
        )
        translation = np.array([[translation_x, translation_y]] * 4, np.float32)
        # First scale, then rotate, then translate
        unit_points = np.array(
            [
                [0, 0],
                [self.blob_sheet.shape[0], 0],
                [0, self.blob_sheet.shape[1]],
                [self.blob_sheet.shape[0], self.blob_sheet.shape[1]],
            ],
            np.float32,
        )
        transformed_points = unit_points * scale
        for i in range(transformed_points.shape[0]):
            transformed_points[i][0] = transformed_points[i][0] * np.cos(
                rotation
            ) - transformed_points[i][0] * np.sin(rotation)
            transformed_points[i][1] = transformed_points[i][0] * np.sin(
                rotation
            ) + transformed_points[i][1] * np.cos(rotation)
        transformed_points += translation
        # Wiggle points to get perspective transformation
        transformed_points += np.random.uniform(
            -MAX_WIGGLE
            * scale
            * min(self.blob_sheet.shape[0], self.blob_sheet.shape[1]),
            MAX_WIGGLE
            * scale
            * min(self.blob_sheet.shape[0], self.blob_sheet.shape[1]),
            (4, 2),
        )
        homography = cv.getPerspectiveTransform(unit_points, transformed_points)
        return homography

    def map_blobs(self, background, homography):
        """
        Warps the blobsheet into the background according to a homography.
        """
        warped_blobs = cv.warpPerspective(self.blob_sheet, homography, IMAGE_SIZE)
        return warped_blobs

    def map_keypoints(self, homography):
        """
        Maps the list of keypoints according to the homography.
        """
        def map_keypoint(k):
            (x, y), scale, rotation = k
            mapped_location = homography @ np.array([x, y, 1])
            mapped_location = mapped_location[:2] / mapped_location[2]
            second_location = homography @ np.array(
                [x - np.sin(rotation) * scale, y + np.cos(rotation) * scale, 1]
            )
            second_location = second_location[:2] / second_location[2]
            new_rotation = np.arccos(
                np.dot(second_location - mapped_location, np.array([0, 1]))
                / np.linalg.norm(second_location - mapped_location)
            )
            if second_location[0] > mapped_location[0]:
                new_rotation = 2 * np.pi - new_rotation
            new_scale = np.linalg.norm(second_location - mapped_location)
            return mapped_location, new_scale, new_rotation

        return list(map(map_keypoint, self.keypoints))
