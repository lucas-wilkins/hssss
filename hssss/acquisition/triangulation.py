from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from hssss.acquisition.acquisition_system import SegmentedImages
from hssss.camera import Camera

@dataclass
class TriangulationResult:
    points: np.ndarray
    distances: np.ndarray

    def plot_points(self, threshold: float | None = None):
        """ Do a 3D plot of the triangulared points, they will be coloured by how well the lines  """

        ax = plt.gcf().add_subplot(projection='3d')

        norm = Normalize(vmin=0.0, vmax=np.max(self.distances) if threshold is None else threshold)
        cmap = cm.get_cmap("coolwarm")
        colors = cmap(norm(self.distances))

        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2],
                   s=10, c=colors, marker='o', depthshade=False)

        ax.axis("equal")

        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    def threshold(self, distance: float) -> np.ndarray:
        """ Get points where the distance between the rays that should intersect is less than `distance` """

        return self.points[self.distances < distance, :]



def triangulate(segmented_images: SegmentedImages, camera_1: Camera, camera_2: Camera):
    """ Convert segmented images into a point cloud """

    # Get the centroids of the images
    centres_1, centres_2 = segmented_images.image_points()

    # Convert to points in an arbitrary image plane
    image_points_1 = camera_1.pixels_to_point(centres_1)
    image_points_2 = camera_2.pixels_to_point(centres_2)

    # Now we want to get rays passing through the position parameter of the camera and each image point,
    # and then find the points on each line where they are closest. From this we then want to
    # 1) find the midpoint to add to the point cloud
    # 2) find the distance between them to measure quality
    #
    # Set up the problem as follows...
    # Two lines using provided points given by:
    #   L1 = X1 + t1 V1
    #   L2 = X2 + t2 V2

    n_points = image_points_1.shape[0]

    x1 = np.repeat(np.array((camera_1.position,), dtype=float), n_points, axis=0)
    x2 = np.repeat(np.array((camera_2.position,), dtype=float), n_points, axis=0)

    v1 = image_points_1 - x1
    v2 = image_points_2 - x2

    # Normalise to keep things nice (they should never be zero)
    # v1 /= np.sqrt(np.sum(v1**2, axis=1)).reshape(-1, 1)
    # v2 /= np.sqrt(np.sum(v2**2, axis=1)).reshape(-1, 1)

    # Next, define a line between the two closest points, L3
    #   L3 = X1 + t2 V1 + t3 V3
    #
    # At the points which are closest, L1 and L2 should be perpendicular to L3, leading to
    #   V3 = V2 x V1

    v3 = np.cross(v2, v1) # Checked this, it broadcasts correctly

    # Then we want to find where L3 = L2
    #   X1 + t1 V1 + t3 V3 = X2 + t2 V2
    #
    # Rearranging, we get
    #   (t1, t2, t3) . (V1, -V2, V3) = X2 - X1
    #
    # Which we can easily solve to get t1, t2, t3
    #

    v = np.array((v1, -v2, v3)).transpose(1, 2, 0)
    x = (x2 - x1).reshape(n_points, 3, 1)

    t = np.linalg.solve(v, x)

    #
    # Now the 3D points can be obtained by the line equations
    #

    p1 = t[:,0,:]*v1 + x1
    p2 = t[:,1,:]*v2 + x2

    # Get the midpoints and distances

    points = 0.5*(p1 + p2)
    distances = np.sqrt(np.sum((p2 - p1)**2, axis=1))

    return TriangulationResult(points, distances)
