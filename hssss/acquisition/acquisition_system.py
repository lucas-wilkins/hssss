""" Basic data structures """


from abc import ABC
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops

from acquisition.textures import GreyscaleImage
from hssss.camera import Camera



@dataclass
class AcquisitionResult:
    """ Result of a single step of acquisition """
    projection_image: np.ndarray
    camera_1: np.ndarray
    camera_2: np.ndarray


def write_acquisition_results(data: list["AcquisitionResult"], directory: str):
    """ Create a directory a fill it with the results from an acquisition """

    os.makedirs(directory, exist_ok=True)

    for index, result in enumerate(data):

        for prefix, data in [("camera_1", result.camera_1),
                             ("camera_2", result.camera_2),
                             ("projected", result.projection_image)]:

            filename = os.path.join(directory, f"{prefix}-{index:03}.png")
            cv2.imwrite(filename, data)

@dataclass
class SegmentationParameters:
    """ Parameters used to clean up the segmentation """
    minimal_contrast: float = 10
    minimal_quality: float = 0.5


@dataclass
class AssignmentEntry:
    """ Results from the assignments of regions to images """
    assignment: np.ndarray  # Image with the regions numbered, but no accounting for quality/contrast
    quality: np.ndarray     # Number, in ideality in [0, 1], but might be bigger than 1 in some cases
    contrast: np.ndarray    # Absolute difference between white and black

    def show_raw(self):

        plt.subplot(2, 2, 1)
        plt.imshow(self.assignment)

        plt.subplot(2, 2, 2)
        plt.imshow(self.quality, vmin=0.0, vmax=0.0)

        plt.subplot(2, 2, 3)
        plt.imshow(self.contrast, vmin=0.0)

    def segment(self, parameters: SegmentationParameters):
        """ Exclude dubious regions (low contrast / low quality metric) """
        good_regions = np.logical_and(
            self.quality >= parameters.minimal_quality,
            self.contrast >= parameters.minimal_contrast )

        output = -np.ones_like(self.assignment)

        output[good_regions] = self.assignment[good_regions]

        return output

    def colourised(self, segmentation_parameters: SegmentationParameters):
        """ Colour the image by region """
        # Unique colours by incomensurate periods
        segmented = self.segment(segmentation_parameters)
        hues = np.array(((segmented * (np.pi-2)) % 3) * 60, dtype=np.uint8) # Value in 0-180
        sats = 156 + np.array( ((segmented * (np.exp(1)-2.5)) % 2) * 50, dtype=np.uint8)  # Value in 156-256
        values = np.zeros_like(segmented, dtype=np.uint8)   # = 0
        values[segmented != -1] = 255 # in {0, 255}
        sats[segmented == -1] = 0

        merged = cv2.merge((hues, sats, values))

        return cv2.cvtColor(merged, cv2.COLOR_HSV2RGB)

@dataclass
class SegmentedImages:
    """ Data containing the segmented images and data needed to do triangulation """
    camera_1: np.ndarray
    camera_2: np.ndarray

    def image_points(self):
        # Get the centre of mass for each unique region in each of the images
        props = regionprops(self.camera_1)

        centres_1 = {}
        for prop in props:
            label = int(prop.label)
            centres_1[label] = prop.centroid

        props = regionprops(self.camera_2)

        centres_2 = {}
        for prop in props:
            label = int(prop.label)
            centres_2[label] = prop.centroid

        labels = set(centres_1.keys()).intersection(set(centres_2.keys()))
        if -1 in labels:
            labels.remove(-1)

        # Combine, and only for the labels in both
        paired_centres_1 = np.array([centres_1[int(label)] for label in labels])
        paired_centres_2 = np.array([centres_2[int(label)] for label in labels])

        return paired_centres_1, paired_centres_2

@dataclass
class Assignment:
    """ Assignment of pixels to regions """
    camera_1: AssignmentEntry
    camera_2: AssignmentEntry

    def segment(self, segmentation_parameters: SegmentationParameters):
        """ Returns 'images' containing the assigned numbers for each camera, and a list of numbers present in both """
        camera_1_seg = self.camera_1.segment(segmentation_parameters)
        camera_2_seg = self.camera_2.segment(segmentation_parameters)

        return SegmentedImages(camera_1_seg, camera_2_seg)

class Encoder:
    """ Creates images to encode/decode image regions """

    def __init__(self, image_x: int, image_y: int, resolution_x: int, resolution_y: int):
        self.image_x = image_x
        self.image_y = image_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y

    def image_count(self) -> int:
        """ Number of images """

    def get_image(self, index: int) -> np.ndarray:
        """ Encoding image"""

    def assign_locations(self, images: list[AcquisitionResult]) -> Assignment:
        """ Convert a list of images to a matching array with x and y positions """

    def image_reel(self):

        while True:
            for i in range(self.image_count()):
                im = GreyscaleImage(self.get_image(i))
                im.show(f"Index {i}")


class AcquisitionSystem(ABC):
    """ Base class for image acquisition """

    def __init__(self, camera_1: Camera, camera_2: Camera):
        self.camera_1 = camera_1
        self.camera_2 = camera_2

    def run_scan(self, encoder: Encoder, *args) -> list[AcquisitionResult]:
        """ Run a scan """

