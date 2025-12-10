import numpy as np
import glfw

from camera import Camera


class WindowCamera(Camera):
    """ Object to hold the state of a GL camera and generate an appropriate view matrix """

    def __init__(self,
                 fov: float = 45,
                 horizontal_pixels: int = 800,
                 vertical_pixels: int = 600,
                 yaw: float = 0.0,
                 pitch: float = 0.0,
                 camera_distance: float = 5.0,
                 mouse_sensitivity: float = 0.2,
                 zoom_sensitivity: float = 0.5):

        self.yaw = yaw
        self.pitch = pitch
        self.last_x = 0
        self.last_y = 0
        self.mouse_down = False
        self.first_mouse = True
        self.mouse_sensitivity = mouse_sensitivity
        self.zoom_sensitivity = zoom_sensitivity
        self.camera_distance = camera_distance  # how far camera is from origin

        super().__init__(
            position=(0,0,0),
            look_at=(0,0,0),
            up=(0,0,1),
            fov_deg=fov,
            horizontal_pixels=horizontal_pixels,
            vertical_pixels=vertical_pixels)

        self.update()

    def update(self):

        # Convert to radians
        ry = np.radians(self.yaw)
        rp = np.radians(self.pitch)

        x = -self.camera_distance * np.cos(rp) * np.cos(ry)
        y = self.camera_distance * np.cos(rp) * np.sin(ry)
        z = self.camera_distance * np.sin(rp)

        self.position = (x, y, z)

    def mouse_move_callback(self):
        """ Creates a glfw callback to deal the mouse movement """
        ref = self

        def fun(window, xpos, ypos):

            if not ref.mouse_down:
                return

            if ref.first_mouse:
                ref.last_x = xpos
                ref.last_y = ypos
                ref.first_mouse = False
                return

            dx = xpos - ref.last_x
            dy = ref.last_y - ypos  # reversed so +dy means look up

            ref.last_x = xpos
            ref.last_y = ypos

            ref.yaw += dx * ref.mouse_sensitivity
            ref.pitch -= dy * ref.mouse_sensitivity

            # clamp pitch to avoid flipping
            ref.pitch = max(-89.0, min(89.0, ref.pitch))

            # print("Window state:", self.yaw, self.pitch)
            ref.update()

        return fun

    def mouse_button_callback(self):
        """ Creates a glfw callback to deal with mouse buttons """
        ref = self

        def fun(window, button, action, mods):

            global mouse_down, first_mouse
            if button == glfw.MOUSE_BUTTON_LEFT:
                if action == glfw.PRESS:
                    ref.mouse_down = True
                    ref.first_mouse = True  # reset to avoid jump
                else:
                    ref.mouse_down = False

        return fun

    def scroll_callback(self):
        """ Creates a glfw callback function to deal with scroll requests """
        ref = self

        def fun(window, xoffset, yoffset):

            ref.camera_distance -= yoffset * ref.zoom_sensitivity
            ref.camera_distance = max(0.01, min(100.0, ref.camera_distance))  # clamp distance

            ref.update()

        return fun
    #
    # def view_matrix(self):
    #     """
    #     Returns a 4x4 view matrix based on yaw/pitch camera angles.
    #     Camera orbits around (0,0,0) at camera_distance.
    #     """
    #
    #     eye = np.array(self.position, dtype=np.float32)
    #     target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    #     up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    #
    #     # Look at matrix
    #     f = target - eye
    #     f = f / np.linalg.norm(f)
    #
    #     s = np.cross(f, up)
    #     s = s / np.linalg.norm(s)
    #
    #     u = np.cross(s, f)
    #
    #     view = np.identity(4, dtype=np.float32)
    #     view[0, :3] = s
    #     view[1, :3] = u
    #     view[2, :3] = -f
    #     view[0, 3] = -np.dot(s, eye)
    #     view[1, 3] = -np.dot(u, eye)
    #     view[2, 3] = np.dot(f, eye)
    #
    #     return view
