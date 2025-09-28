"""
This module provides an extension to learn the features of the Blobinator calibration sheet.
"""

from typing import Generator, Sequence, TypeAlias, TypedDict, Union

import cv2 as cv
import numpy as np
import pdf2image
import torch


class ImageWrapper(TypedDict):
    img: np.ndarray
    padUp: int
    padLeft: int


Keypoint: TypeAlias = tuple[np.ndarray, float, float]


class BlobinatorDataset(torch.utils.data.IterableDataset):
    """
    This is the Blobinator dataset.

    It warps the Blobinator sheet into a number of background images via randomly generated homographies.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        blobs = np.array(pdf2image.convert_from_path(cfg.BLOBINATOR.BLOB_SHEET, grayscale=True)[0])
        self.blob_sheet: np.ndarray = np.zeros((1, self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))
        self.blob_sheet_pad_left: int = (cfg.TRAINING.PAD_TO - blobs.shape[0]) // 2
        self.blob_sheet_pad_up: int = (cfg.TRAINING.PAD_TO - blobs.shape[1]) // 2
        self.blob_sheet[
            :,
            self.blob_sheet_pad_left:self.blob_sheet_pad_left+blobs.shape[0],
            self.blob_sheet_pad_up: self.blob_sheet_pad_up + blobs.shape[1]
        ] = blobs
        self.keypoints: Sequence[Keypoint] = list(
            map(
                lambda k: (
                    np.array([
                        k[0][0] * blobs.shape[0] + self.blob_sheet_pad_left,
                        k[0][1] * blobs.shape[1] + self.blob_sheet_pad_up,
                    ]),
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
        self.backgrounds: np.ndarray = np.zeros(shape=(256, self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))

    def __iter__(self) -> Generator[tuple[
        ImageWrapper,
        ImageWrapper,
        Keypoint,
        Keypoint,
        str,
        str,
        float,
        float
    ]]:
        for i, background in enumerate(self.backgrounds):
            transform = self.generate_homography()
            mapped_image = self.map_blobs(background, transform)
            mapped_keypoints = self.map_keypoints(transform)
            for k, k_mapped in zip(self.keypoints, mapped_keypoints):
                yield (
                    {
                        "img": self.blob_sheet,
                        "padLeft": self.blob_sheet_pad_left,
                        "padUp": self.blob_sheet_pad_up
                    },
                    {
                        "img": mapped_image,
                        "padLeft": 0,
                        "padUp": 0
                    },
                    (k[0], k[1], k[2]),
                    (k_mapped[0], k_mapped[1], k_mapped[2]),
                    "img0000",
                    f"img{i+1:04}",
                    1,
                    0
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
        scale = np.random.uniform(self.cfg.BLOBINATOR.MIN_SCALE, self.cfg.BLOBINATOR.MAX_SCALE)
        rotation = np.random.uniform(-self.cfg.BLOBINATOR.MAX_ROTATION, self.cfg.BLOBINATOR.MAX_ROTATION) * np.pi / 180
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
            -self.cfg.BLOBINATOR.MAX_WIGGLE
            * scale
            * min(self.blob_sheet.shape[0], self.blob_sheet.shape[1]),
            self.cfg.BLOBINATOR.MAX_WIGGLE
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
        warped_blobs = np.zeros(shape=(1, self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO))
        warped_blobs[0, :, :] = cv.warpPerspective(
            self.blob_sheet[0, :, :],
            homography,
            (self.cfg.TRAINING.PAD_TO, self.cfg.TRAINING.PAD_TO)
        )
        return warped_blobs

    def map_keypoints(self, homography: np.ndarray) -> Sequence[Keypoint]:
        """
        Maps the list of keypoints according to the homography.
        """
        def map_keypoint(k: Keypoint) -> Keypoint:
            location, scale, rotation = k
            x = location[0]
            y = location[1]
            mapped_location = homography @ np.array([x, y, 1])
            mapped_location = mapped_location[:2] / mapped_location[2]
            second_location = homography @ np.array([x - np.sin(rotation) * scale, y + np.cos(rotation) * scale, 1])
            second_location = second_location[:2] / second_location[2]
            new_rotation = np.arccos(
                np.dot(second_location - mapped_location, np.array([0, 1]))
                / np.linalg.norm(second_location - mapped_location)
            )
            if second_location[0] > mapped_location[0]:
                new_rotation = (2 * np.pi - new_rotation) % (2 * np.pi)
            new_scale = float(np.linalg.norm(second_location - mapped_location))
            return mapped_location, new_scale, new_rotation.item()

        return list(map(map_keypoint, self.keypoints))
