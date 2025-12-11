
import numpy as np


class Camera:
    """ Data defining a camera """

    def __init__(self,
        position: tuple[float, float, float],
        look_at: tuple[float, float, float],
        up: tuple[float, float, float],
        fov_deg: float,
        horizontal_pixels: int,
        vertical_pixels: int):

        self.position = position
        self.look_at = look_at
        self.up = up
        self.fov_deg = fov_deg
        self.horizontal_pixels = horizontal_pixels
        self.vertical_pixels = vertical_pixels

    @property
    def aspect_ratio(self) -> float:
        return self.horizontal_pixels / self.vertical_pixels

    def pixels_to_point(self, pixel_coords: np.ndarray, distance=1):
        """ Convert a pixel to a point in 3D space """

        # Should just be able to invert the view-projection matrix
        n_points = pixel_coords.shape[0]

        # points should be in [-1, -1], not [0, n_px] (convert to normalised device coordinates)
        x_device = ((pixel_coords[:, 1] / self.horizontal_pixels) * 2 - 1) # TODO: Negate?
        y_device = -((pixel_coords[:, 0] / self.vertical_pixels) * 2 - 1)
        z_device = np.ones_like(y_device) #-np.ones_like(y_device) # -1 is near plane, 1 is far plane
        w_device = np.ones_like(y_device)

        vec4ified = np.vstack((
              x_device,
              y_device,
              z_device,
              w_device))


        view_projection = self.perspective_matrix(0.01*distance, distance*100) @ self.view_matrix()
        inverted = np.linalg.inv(view_projection)

        world_unnormalised = (inverted @ vec4ified).T
        normalised = world_unnormalised[:, :3] / world_unnormalised[:, 3].reshape(-1, 1)

        return normalised

    def perspective_matrix(self, near: float = 0.01, far: float = 100):
        """ Get the perspective/projection matrix for GL

        :param near: near clipping plane distance
        :param far: far clipping plane distance

        :return: 4x4 float32 frustum matrix
        """

        f = 1.0 / np.tan(np.radians(self.fov_deg) / 2)
        m = np.zeros((4, 4), dtype=np.float32)

        m[0, 0] = f / self.aspect_ratio
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1

        return m

    def view_matrix(self):
        """
        Transform from world position to camera relative
        """

        eye = np.array(self.position, dtype=np.float32)
        target = np.array(self.look_at, dtype=np.float32)
        up = np.array(self.up, dtype=np.float32)

        # Look at matrix
        f = target - eye
        f = f / np.linalg.norm(f)

        s = np.cross(f, up)
        s = s / np.linalg.norm(s)

        u = np.cross(s, f)

        view = np.identity(4, dtype=np.float32)
        view[0, :3] = s
        view[1, :3] = u
        view[2, :3] = -f
        view[0, 3] = -np.dot(s, eye)
        view[1, 3] = -np.dot(u, eye)
        view[2, 3] = np.dot(f, eye)

        return view