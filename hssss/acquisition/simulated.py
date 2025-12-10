import time

from glfw.library import glfw

import numpy as np

import glfw
from OpenGL.GL import *

from acquisition.binary_encoder import BinaryEncoder
from acquisition.acquisition_system import Encoder
from acquisition.textures import GreyscaleImage
from hssss.window_camera import WindowCamera


from hssss.acquisition.acquisition_system import AcquisitionSystem, AcquisitionResult

from hssss.acquisition.shaders import load_shaders
from hssss.acquisition.stl_util import load_stl_into_vao_vbo
from hssss.camera import Camera

class CameraSelector:
    def __init__(self, camera_1: Camera, camera_2: Camera, projector: Camera, window_camera: WindowCamera):
        self.camera_1 = camera_1
        self.camera_2 = camera_2
        self.projector = projector
        self.window_camera = window_camera

        self.selected_camera = window_camera

    def key_callback(self):
        this = self

        def fun(window, key, scancode, action, mods):
            if action == glfw.PRESS:

                match key:
                    case glfw.KEY_1:
                        this.selected_camera = self.camera_1
                    case glfw.KEY_2:
                        this.selected_camera = self.camera_2
                    case glfw.KEY_3:
                        this.selected_camera = self.projector
                    case glfw.KEY_4:
                        this.selected_camera = self.window_camera


            # if action == glfw.RELEASE:
            #     print(f"Key {key} released")

        return fun


class SimulatedAcquisitionSystem(AcquisitionSystem):
    def __init__(self,
                 camera_1: Camera,
                 camera_2: Camera,
                 projector: Camera,
                 model_file: str):

        super().__init__(camera_1, camera_2)

        self.projector = projector
        self.model_filename = model_file

    def show_with_projection_shader(self, encoder: Encoder, index: int, shader: str | None = None):

        if index < 0:
            index = encoder.image_count() - index

        projection_data = encoder.get_image(index)
        projection_texture = GreyscaleImage(projection_data)


        # Display the input model

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        movable_camera = WindowCamera()

        camera_selector = CameraSelector(
            self.camera_1,
            self.camera_2,
            self.projector,
            movable_camera)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(
            movable_camera.horizontal_pixels,
            movable_camera.vertical_pixels,
            "STL Shader Viewer",
            None, None)

        if not window:
            raise RuntimeError("Window failed")
        glfw.make_context_current(window)
        glEnable(GL_DEPTH_TEST)

        vao, vbo, n_verts = load_stl_into_vao_vbo(self.model_filename)

        # Load in the shader
        prog = load_shaders(fragment_filename="projection_simple" if shader is None else shader)
        glUseProgram(prog)

        # Textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, projection_texture.gl_texture())

        # Uniform locations
        loc_model = glGetUniformLocation(prog, "uModel")
        loc_mvp = glGetUniformLocation(prog, "uMVP")

        loc_projector_mvp = glGetUniformLocation(prog, "uProjectorMVP")
        loc_projector_position = glGetUniformLocation(prog, "uProjectorPos")

        loc_light = glGetUniformLocation(prog, "uLightDir")
        loc_color = glGetUniformLocation(prog, "uColor")

        glUniform1i(glGetUniformLocation(prog, "uProjectorTexture"), 0)

        # Example values:
        light_dir = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        glUniform3fv(loc_light, 1, light_dir)
        glUniform3fv(loc_color, 1, color)
        glUniform3fv(loc_projector_position, 1, np.array(self.projector.position, np.float32))

        # model transformation
        model = np.identity(4, np.float32)


        # projector projection matrix
        projector_model_view = self.projector.view_matrix() @ model
        projector_mvp = self.projector.perspective_matrix() @ projector_model_view

        # callbacks for window interaction
        glfw.set_cursor_pos_callback(window, movable_camera.mouse_move_callback())
        glfw.set_mouse_button_callback(window, movable_camera.mouse_button_callback())
        glfw.set_scroll_callback(window, movable_camera.scroll_callback())

        glfw.set_key_callback(window, camera_selector.key_callback())

        # main loop
        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0.1, 0.1, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            selected_camera = camera_selector.selected_camera
            mvp = selected_camera.perspective_matrix() @ selected_camera.view_matrix() @ model


            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glUniformMatrix4fv(loc_model, 1, GL_TRUE, model)
            glUniformMatrix4fv(loc_projector_mvp, 1, GL_TRUE, projector_mvp)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

            glfw.swap_buffers(window)

        glfw.terminate()


    def show_model(self, shader: str | None = None):
        # Display the input model

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        camera = WindowCamera()

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(camera.horizontal_pixels, camera.vertical_pixels, "STL Shader Viewer", None, None)
        if not window:
            raise RuntimeError("Window failed")
        glfw.make_context_current(window)
        glEnable(GL_DEPTH_TEST)

        vao, vbo, n_verts = load_stl_into_vao_vbo(self.model_filename)

        # Load in the diffuse shader
        prog = load_shaders(fragment_filename="diffuse" if shader is None else shader)
        glUseProgram(prog)

        # Uniform locations
        loc_model = glGetUniformLocation(prog, "uModel")
        loc_mvp = glGetUniformLocation(prog, "uMVP")
        loc_light = glGetUniformLocation(prog, "uLightDir")
        loc_color = glGetUniformLocation(prog, "uColor")


        # Example values:
        light_dir = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        glUniform3fv(loc_light, 1, light_dir)
        glUniform3fv(loc_color, 1, color)

        # camera projection matrix
        proj = camera.perspective_matrix()

        # callbacks for window interaction
        glfw.set_cursor_pos_callback(window, camera.mouse_move_callback())
        glfw.set_mouse_button_callback(window, camera.mouse_button_callback())

        # main loop
        while not glfw.window_should_close(window):
            glfw.poll_events()
            glClearColor(0.1, 0.1, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            model = np.identity(4, np.float32)
            mvp = proj @ camera.view_matrix() @ model

            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glUniformMatrix4fv(loc_model, 1, GL_TRUE, model)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

            glfw.swap_buffers(window)

        glfw.terminate()

    def run_scan(self,
                 encoder: Encoder,
                 shader: str | None = None,
                 show_process: bool = False) -> list[AcquisitionResult]:

        first_projection = encoder.get_image(0)
        projection_texture = GreyscaleImage(first_projection)


        # Display the input model

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        movable_camera = WindowCamera()

        camera_selector = CameraSelector(
            self.camera_1,
            self.camera_2,
            self.projector,
            movable_camera)

        # Only show the window if required
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if show_process else glfw.FALSE)

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        window = glfw.create_window(
            movable_camera.horizontal_pixels,
            movable_camera.vertical_pixels,
            "STL Shader Viewer",
            None, None)

        if not window:
            raise RuntimeError("Window failed")
        glfw.make_context_current(window)
        glEnable(GL_DEPTH_TEST)

        vao, vbo, n_verts = load_stl_into_vao_vbo(self.model_filename)

        # Load in the shader
        prog = load_shaders(fragment_filename="projection_simple" if shader is None else shader)
        glUseProgram(prog)

        # Textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, projection_texture.gl_texture())

        # Uniform locations
        loc_model = glGetUniformLocation(prog, "uModel")
        loc_mvp = glGetUniformLocation(prog, "uMVP")

        loc_projector_mvp = glGetUniformLocation(prog, "uProjectorMVP")
        loc_projector_position = glGetUniformLocation(prog, "uProjectorPos")

        loc_light = glGetUniformLocation(prog, "uLightDir")
        loc_color = glGetUniformLocation(prog, "uColor")

        glUniform1i(glGetUniformLocation(prog, "uProjectorTexture"), 0)

        # Example values:
        light_dir = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        glUniform3fv(loc_light, 1, light_dir)
        glUniform3fv(loc_color, 1, color)
        glUniform3fv(loc_projector_position, 1, np.array(self.projector.position, np.float32))

        # model transformation
        model = np.identity(4, np.float32)


        # projector projection matrix
        projector_model_view = self.projector.view_matrix() @ model
        projector_mvp = self.projector.perspective_matrix() @ projector_model_view

        # callbacks for window interaction
        glfw.set_cursor_pos_callback(window, movable_camera.mouse_move_callback())
        glfw.set_mouse_button_callback(window, movable_camera.mouse_button_callback())
        glfw.set_scroll_callback(window, movable_camera.scroll_callback())

        glfw.set_key_callback(window, camera_selector.key_callback())

        width, height = glfw.get_framebuffer_size(window)

        output = []

        for index in range(encoder.image_count()):


            # Bind the texture for this part
            texture = GreyscaleImage(encoder.get_image(index))

            # Replace (reallocate)
            texture.gl_texture()

            # texture.show()

            # Main rendering - Camera 1
            glfw.poll_events()
            glClearColor(0.1, 0.1, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            mvp = self.camera_1.perspective_matrix() @ self.camera_1.view_matrix() @ model

            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glUniformMatrix4fv(loc_model, 1, GL_TRUE, model)
            glUniformMatrix4fv(loc_projector_mvp, 1, GL_TRUE, projector_mvp)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)


            # Get image data for camera 1
            glPixelStorei(GL_PACK_ALIGNMENT, 1)  # Important for tightly packed rows
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

            # Convert to a NumPy array
            image_1 = np.frombuffer(pixels, dtype=np.uint8)
            image_1 = image_1.reshape(height, width, 3)
            image_1 = np.flipud(image_1)

            # Remember to swap buffers after saving, not before
            glfw.swap_buffers(window)

            if show_process:
                time.sleep(0.5)



            # Main rendering - Camera 2
            glfw.poll_events()
            glClearColor(0.1, 0.1, 0.1, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            mvp = self.camera_2.perspective_matrix() @ self.camera_2.view_matrix() @ model

            glUseProgram(prog)
            glUniformMatrix4fv(loc_mvp, 1, GL_TRUE, mvp)
            glUniformMatrix4fv(loc_model, 1, GL_TRUE, model)
            glUniformMatrix4fv(loc_projector_mvp, 1, GL_TRUE, projector_mvp)

            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)

            # Get image data for camera 1
            glPixelStorei(GL_PACK_ALIGNMENT, 1)  # Important for tightly packed rows
            pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

            # Convert to a NumPy array
            image_2 = np.frombuffer(pixels, dtype=np.uint8)
            image_2 = image_2.reshape(height, width, 3)
            image_2 = np.flipud(image_2)

            # Remember to swap buffers after saving, not before
            glfw.swap_buffers(window)

            # collect images

            output.append(AcquisitionResult(
                projection_image=texture.data.astype(float)/255,
                camera_1 = image_1,
                camera_2 = image_2 ))

            if show_process:
                time.sleep(0.5)



        glfw.terminate()

        return output


if __name__ == "__main__":
    camera_1 = Camera(
        position=(-5, 2, 0),
        look_at=(0,0,0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=60)

    camera_2 = Camera(
        position=(-5, -2, 0),
        look_at=(0, 0, 0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=60)

    projector = Camera(
        position=(-10, 0, 0),
        look_at=(0,0,0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=5)

    system = SimulatedAcquisitionSystem(camera_1, camera_2, projector,"../monkey.stl")
    # system = SimulatedAcquisitionSystem(camera, camera, projector,"../cube.stl")


    # encoder = BinaryEncoder(800, 600, 64, 64)
    encoder = BinaryEncoder(800, 600, 16, 16)



    # Display stuff

    # encoder.image_reel()
    # system.show_model()
    # system.show_with_projection_shader(encoder, -1)
    system.run_scan(encoder, show_process=True)

