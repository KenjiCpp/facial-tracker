import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
from PIL import Image
import cv2
from ctypes import pointer

class OpenGLGraphics:
    VERTEX_SHADER = """
        #version 330

        in vec4 position;
        in vec2 InTexCoords;

        out vec2 OutTexCoords;
              
        uniform mat4 transform; 

        void main() {
            vec4 pos = transform * vec4(position.xyz, 1.0);
            gl_Position = vec4(pos.xyz, 1.0);
            OutTexCoords = vec2(InTexCoords.x, 1.0 - InTexCoords.y);
        }
    """
    FRAGMENT_SHADER = """
        #version 330

        in vec2 OutTexCoords;

        out vec4 outColor;
        uniform sampler2D samplerTex;

        void main() {
            outColor = texture(samplerTex, OutTexCoords);
        }
    """

    def __init__(self):
        if not glfw.init():
            return
        window = glfw.create_window(600, 600, "Pyopengl", None, None)
        if not window:
            glfw.terminate()
            return
        glfw.make_context_current(window)

        self.shader_vertex   = OpenGL.GL.shaders.compileShader(OpenGLGraphics.VERTEX_SHADER, GL_VERTEX_SHADER)
        self.shader_fragment = OpenGL.GL.shaders.compileShader(OpenGLGraphics.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.shader = OpenGL.GL.shaders.compileProgram(self.shader_vertex, self.shader_fragment)
        glUseProgram(self.shader)
        self.transform_loc = glGetUniformLocation(self.shader, "transform")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        
        glDepthFunc(GL_LESS)
        glDepthRange(-100.0, 100.0)

        glClearColor(0.0, 0.0, 0.0, 1.0)

    def set_transform(self, mat):
        glUniformMatrix4fv(self.transform_loc, 1, GL_FALSE, mat)

    def clear(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def draw(self, mesh, transform, texture):
        self.set_transform(transform)
        texture.bind()
        mesh.bind()
        position = glGetAttribLocation(self.shader, 'position')
        glVertexAttribPointer(position, 4, GL_FLOAT, GL_FALSE, 4 * 6, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position) 
        texCoords = glGetAttribLocation(self.shader, "InTexCoords")
        glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE,  4 * 6, ctypes.c_void_p(16))
        glEnableVertexAttribArray(texCoords)
        glUseProgram(self.shader)
        glDrawElements(GL_TRIANGLES, len(mesh.triangles), GL_UNSIGNED_INT, None)

    def release(self):
        pass

class OpenGLMesh:
    def __init__(self, vertices, tex_coords, triangles):
        self.n_vertices = len(vertices)
        self.vertices   = np.hstack([np.array(vertices), np.array(tex_coords)]).astype(np.float32).flatten()
        self.triangles  = np.array(triangles).astype(np.int32).flatten()

        self.buffers = glGenBuffers(2)
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBufferData(GL_ARRAY_BUFFER, 4 * len(self.vertices), self.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4 * len(self.triangles), self.triangles, GL_STATIC_DRAW)

    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.buffers[0])
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffers[1])

    def release(self):
        glDeleteBuffers(2, self.buffers)
        pass

class OpenGLTexture:
    def __init__(self, file=None, data=None, width=None, height=None):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        img_data = data
        if file is not None:
            image = Image.open(file)
            img_data = np.array(list(image.getdata()), np.uint8)
            width = image.width
            height = image.height
        assert(img_data is not None)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    def bind(self):
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def release(self):
        glDeleteTextures(1, [self.texture])
        pass

class OpenGLCanvas:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.frame_buffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        self.color_buffer = glGenRenderbuffers(1)

        glBindRenderbuffer(GL_RENDERBUFFER, self.color_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.color_buffer)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.frame_buffer)
        glViewport(0, 0, self.width, self.height)

    def get_color_buffer(self):
        self.bind()
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGBA, GL_FLOAT)
        data = np.frombuffer(data, np.float32)
        data = np.array(data).reshape((self.height, self.width, 4))
        data = cv2.flip(data, 0)
        return data

    def release(self):
        glDeleteRenderbuffers(1, [self.color_buffer])
        glDeleteFramebuffers(1, [self.frame_buffer])
        pass