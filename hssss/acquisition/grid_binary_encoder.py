import numpy as np

from acquisition.acquisition_system import Encoder, AcquisitionResult, Assignment, AssignmentEntry


class GridBinaryEncoder(Encoder):
    def __init__(self, image_x: int, image_y: int, resolution_x: int, resolution_y: int, grid_thickness: int):
        """

        Image sequence is black, white, x value series, y value series

        :param image_x:
        :param image_y:
        :param resolution_x:
        :param resolution_y:
        """

        super().__init__(image_x, image_y, resolution_x, resolution_y)

        self.grid_thickness = grid_thickness

        self.x_bit_depth = int(np.ceil(np.log2(resolution_x)))
        self.y_bit_depth = int(np.ceil(np.log2(resolution_y)))

        self.x_values = resolution_x * np.arange(image_x) / image_x
        self.y_values = resolution_y * np.arange(image_y) / image_y

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)




    def image_count(self):
        return self.x_bit_depth + self.y_bit_depth + 3

    def get_image(self, index: int):
        """ Get the images for each index, begins with black then white, then x, then y"""
        if index == 0:
            return np.zeros_like(self.x_mesh, dtype=float)

        elif index == 1:
            return np.ones_like(self.x_mesh, dtype=float)

        elif index == 2:
            # Grid

            # [0-1) over each region
            x_region = self.x_mesh % 1.0
            y_region = self.y_mesh % 1.0

            # Convert back to pixels
            x_pixels_per_region = self.image_x / self.resolution_x
            y_pixels_per_region = self.image_y / self.resolution_y
            x_pixels = x_region * x_pixels_per_region
            y_pixels = y_region * y_pixels_per_region

            border = 0.5 * self.grid_thickness

            return np.array(
                np.logical_and(
                    np.logical_and(
                        border <= x_pixels,
                        x_pixels < x_pixels_per_region - border),
                    np.logical_and(
                        border <= y_pixels,
                        y_pixels < y_pixels_per_region - border)),
                dtype=float)


        elif 2 < index < 3 + self.x_bit_depth:
            this_bit_depth = self.x_bit_depth + 2 - index
            scale = 2**this_bit_depth
            return np.array(self.x_mesh % (2*scale) > scale, dtype=float)

        else:
            this_bit_depth = self.x_bit_depth + self.y_bit_depth + 2 - index
            scale = 2**this_bit_depth
            return np.array(self.y_mesh % (2 * scale) > scale, dtype=float)

    def assign_locations(self, results: list[AcquisitionResult]) -> Assignment:

        out = []

        # Iterate over image sets for each camera
        for images in [[np.array(result.camera_1, dtype=float) for result in results],
                       [np.array(result.camera_2, dtype=float) for result in results]]:

            images = [np.sum(image, axis=2) for image in images]

            black = images[0]
            white = images[1]
            grid = images[2]

            contrast = white - black

            # initial quality is just the grid
            quality =  (grid - black) / contrast

            bits = []
            for i in range(3, self.image_count()):
                im = images[i]

                values = (im - black) / contrast

                # Store the bit type
                bits.append(values > 0.5)

                # Get the distance from 0.5 for measuring quality
                distance = 2*np.abs(values - 0.5)

                # Overall quality metric is the arraywise minimum over all
                quality = np.minimum(quality, distance)

            # Convert to numbers

            x_bits = reversed(bits[:self.x_bit_depth])
            y_bits = reversed(bits[self.x_bit_depth:])

            x_number = np.zeros_like(black, dtype=np.long)
            y_number = np.zeros_like(black, dtype=np.long)

            for power, bitmap in enumerate(x_bits):
                x_number += (2**power) * bitmap.astype(np.long)

            for power, bitmap in enumerate(y_bits):
                y_number += (2**power) * bitmap.astype(np.long)

            assignment = self.resolution_x * y_number + x_number

            out.append(AssignmentEntry(
                assignment=assignment,
                quality=quality,
                contrast=contrast))

        return Assignment(out[0], out[1])

if __name__ == "__main__":

    encoder = GridBinaryEncoder(800, 600, 10, 10, 5)

    # encoder.image_reel()

    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(encoder.get_image(2))

    plt.figure()
    plt.imshow(encoder.get_image(encoder.image_count()-1))

    plt.show()

    # GridBinaryEncoder(800, 600, 64, 64, 2).image_reel()