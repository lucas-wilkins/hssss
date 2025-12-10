import numpy as np
import ctypes

from OpenGL.GL import (glGenVertexArrays, glGenBuffers, glBindVertexArray, glBindBuffer,
                       glBufferData, glEnableVertexAttribArray, glVertexAttribPointer,
                       GL_ARRAY_BUFFER, GL_STATIC_DRAW, GL_FLOAT, GL_FALSE)


from stl import mesh


def _load_stl_as_interleaved_vertices(filename):
    """
    Returns flattened array [pos.x, pos.y, pos.z, norm.x, norm.y, norm.z, ...]
    and number of vertices.
    """
    m = mesh.Mesh.from_file(filename)

    vertices = m.vectors.reshape(-1, 3)
    normals = np.repeat(m.normals, 3, axis=0)  # One normal per vertex (per triangle)

    interleaved = np.hstack((vertices, normals)).astype(np.float32)

    return interleaved, len(vertices)

def _create_vao_vbo_from_interleaved(interleaved_vertices):
    """ Create the Vertex Array/Buffer Objects """

    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, interleaved_vertices.nbytes,
                 interleaved_vertices, GL_STATIC_DRAW)

    stride = interleaved_vertices.strides[0]

    # position attribute for shaders (location = 0)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

    offset = ctypes.c_void_p(12)  # 3x4 bytes

    # normal attribute for shaders (location = 1)
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, offset)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vbo

def load_stl_into_vao_vbo(filename):
    """ Create Vertex Array/Buffer Objects for an STL """
    interleaved, n_verts = _load_stl_as_interleaved_vertices(filename)
    vao, vbo = _create_vao_vbo_from_interleaved(interleaved)

    return vao, vbo, n_verts