import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import torch
from pinn import make_forward_fn, LinearNN
from logistic_equation import make_loss_fn
import torchopt
import ctypes

# Vertex and Fragment shader source
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
void main() {
    gl_Position = vec4(aPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
"""

def main():
    # Initialize GLFW
    if not glfw.init():
        return

    # Set GLFW window hints
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(800, 600, "PINNs Visualization", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Compile and link shaders
    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    # Setup your PINN model
    torch.manual_seed(42)
    num_hidden = 5
    dim_hidden = 5
    batch_size = 30
    num_iter = 100
    learning_rate = 1e-1
    domain = (-5.0, 5.0)

    model = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=1)
    funcs = make_forward_fn(model, derivative_order=1)
    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))
    params = tuple(model.parameters())

    # Train the model
    for i in range(num_iter):
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])
        loss = loss_fn(params, x)
        params = optimizer.step(loss, params)
        print(f"Iteration {i} with loss {float(loss)}")

    # Prepare data for OpenGL
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    f_eval = f(x_eval, params)
    vertices = np.array([
        [float(x), float(y), 0.0] for x, y in zip(x_eval, f_eval)
    ], dtype=np.float32)

    # Create a Vertex Array Object (VAO)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    # Create a Vertex Buffer Object (VBO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Configure the vertex attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Unbind the VBO and VAO
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # Render loop
    while not glfw.window_should_close(window):
        # Clear the screen
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw the points
        glBindVertexArray(VAO)
        glDrawArrays(GL_LINE_STRIP, 0, len(vertices))
        glBindVertexArray(0)

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()

    # Clean up
    glDeleteBuffers(1, [VBO])
    glDeleteVertexArrays(1, [VAO])
    glfw.terminate()

if __name__ == "__main__":
    main()
