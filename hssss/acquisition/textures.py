import numpy as np
from OpenGL.GL import (glGenTextures, glBindTexture, glTexImage2D, glGenerateMipmap, glTexParameteri, glTexParameterfv,
                       GL_TEXTURE_2D, GL_RED, GL_R8, GL_UNSIGNED_BYTE, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
                       GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR,
                       GL_CLAMP_TO_BORDER, GL_TEXTURE_BORDER_COLOR, GL_ONE,
                       GL_TEXTURE_SWIZZLE_R, GL_TEXTURE_SWIZZLE_G, GL_TEXTURE_SWIZZLE_B, GL_TEXTURE_SWIZZLE_A)


class GreyscaleImage:

    def __init__(self, data: np.ndarray):
        if len(data.shape) != 2:
            raise ValueError("Expected data to be two dimensional ")

        if np.isdtype(data.dtype, 'integral'):
            data = np.array(data, dtype=np.uint8)

        elif np.isdtype(data.dtype, 'real floating'):
            data = np.array(255*data, dtype=np.uint8)

        else:
            raise ValueError("Expected float or int array")

        self.data = data

    def show(self, title: str | None = None):
        import matplotlib.pyplot as plt

        if title is not None:
            plt.figure(title)

        plt.imshow(self.data, "gray", vmin=0, vmax=255)

        plt.show()

    def gl_texture(self):
        """ Get a texture for using in open GL stuff

         Binds it too
         """
        width, height = self.data.shape

        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, height, width, 0, GL_RED, GL_UNSIGNED_BYTE,
                     self.data.reshape(-1))
        glGenerateMipmap(GL_TEXTURE_2D)

        # Set wrap mode to clamp to border
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)

        # Set border color to black
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [0.0, 0.0, 0.0, 1.0])

        # Mipmap
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Swizzles
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ONE)

        return tex