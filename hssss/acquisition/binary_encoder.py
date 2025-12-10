import numpy as np

from acquisition.acquisition_system import Encoder, AcquisitionResult, Assignment, AssignmentEntry


class BinaryEncoder(Encoder):
    def __init__(self, image_x: int, image_y: int, resolution_x: int, resolution_y: int):
        """

        Image sequence is black, white, x value series, y value series

        :param image_x:
        :param image_y:
        :param resolution_x:
        :param resolution_y:
        """

        super().__init__(image_x, image_y, resolution_x, resolution_y)

        self.x_bit_depth = int(np.ceil(np.log2(resolution_x)))
        self.y_bit_depth = int(np.ceil(np.log2(resolution_y)))


        self.x_values = resolution_x * np.arange(image_x) / image_x
        self.y_values = resolution_y * np.arange(image_y) / image_y

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_values, self.y_values)


    def image_count(self):
        return self.x_bit_depth + self.y_bit_depth + 2

    def get_image(self, index: int):
        """ Get the images for each index, begins with black then white, then x, then y"""
        if index == 0:
            return np.zeros_like(self.x_mesh, dtype=float)

        elif index == 1:
            return np.ones_like(self.x_mesh, dtype=float)

        elif 1 < index < 2 + self.x_bit_depth:
            this_bit_depth = self.x_bit_depth + 1 - index
            scale = 2**this_bit_depth
            return np.array(self.x_mesh % (2*scale) > scale, dtype=float)

        else:
            this_bit_depth = self.x_bit_depth + self.y_bit_depth + 1 - index
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

            contrast = white - black

            quality = np.ones_like(contrast, dtype=float)

            bits = []
            for i in range(2, self.image_count()):
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
    # BinaryEncoder(800, 600, 5, 11).image_reel()
    BinaryEncoder(800, 600, 64, 64).image_reel()