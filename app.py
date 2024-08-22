import argparse
import torch
from torch import nn
import numpy as np
import torchopt
import OpenGL.GL as gl
import OpenGL.GLUT as glut
from pinn import make_forward_fn, LinearNN, make_loss_fn

# Global variables
params = None
points = None
camera_angle_x = 0
camera_angle_y = 0

# opengl display callback

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # apply camera rotation
    gl.glLoadIdentity()
    gl.glRotatef(camera_angle_x, 1, 0, 0)
    gl.glRotatef(camera_angle_y, 0, 1, 0)

    gl.glBegin(gl.GL_POINTS)
    for point in points:
        # using the f_eval value to vary the color
        gl.glColor3f(point[3], 0.0, 1.0 - point[3])  # Simple color map
        gl.glVertex3f(point[0], point[1], point[2])
    gl.glEnd()

    glut.glutSwapBuffers()


# OpenGL reshape callback
def reshape(width, height):
    gl.glViewport(0, 0, width, height)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    gl.glOrtho(-1.5, 1.5, -1.5, 1.5, -10.0, 10.0)  # Simple orthographic view
    gl.glMatrixMode(gl.GL_MODELVIEW)

# OpenGL keyboard callback
def keyboard(key, x, y):
    global camera_angle_x, camera_angle_y
    if key == b'q':
        camera_angle_x += 5
    elif key == b'e':
        camera_angle_x -= 5
    elif key == b'a':
        camera_angle_y += 5
    elif key == b'd':
        camera_angle_y -= 5
    glut.glutPostRedisplay()

# Initialize OpenGL window

def init_glut_window():
    glut.glutInit()  # Initialize the GLUT library
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB | glut.GLUT_DEPTH)
    glut.glutInitWindowSize(800, 600)
    glut.glutCreateWindow(b"3D PINN Visualization")
    glut.glutDisplayFunc(display)
    glut.glutReshapeFunc(reshape)
    glut.glutKeyboardFunc(keyboard)
    
    # Enable depth testing and lighting
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LIGHTING)
    gl.glEnable(gl.GL_LIGHT0)
    
    # Light properties
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (0, 0, 10, 1))
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (1, 1, 1, 1))
    gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, (1, 1, 1, 1))
    
    gl.glClearColor(0.1, 0.1, 0.1, 1.0)
    gl.glPointSize(3.0)


# OpenGL update function for animation (if needed)
def update(value):
    global points
    # Modify points or parameters to animate over time (optional)
    glut.glutPostRedisplay()
    glut.glutTimerFunc(16, update, 0)  # ~60 FPS

    




def train_pinn(args):
    global params, points

    # Configuration
    num_hidden = args.num_hidden
    dim_hidden = args.dim_hidden
    batch_size = args.batch_size
    num_iter = args.num_epochs
    learning_rate = args.learning_rate
    domain = (-1.0, 1.0)

    # Function versions of model forward, gradient and loss
    model = LinearNN(num_layers=num_hidden, num_neurons=dim_hidden, num_inputs=3)
    funcs = make_forward_fn(model, derivative_order=1)

    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    # Choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    # Initial parameters randomly initialized
    params = tuple(model.parameters())

    # Train the model
    loss_evolution = []
    for i in range(num_iter):

        # Sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size, 1).uniform_(domain[0], domain[1])
        y = torch.FloatTensor(batch_size, 1).uniform_(domain[0], domain[1])
        z = torch.FloatTensor(batch_size, 1).uniform_(domain[0], domain[1])

        # Concatenate x, y, z to form the input tensor of shape (batch_size, 3)
        inputs = torch.cat([x, y, z], dim=1)

        # Compute the loss with the current parameters
        loss = loss_fn(params, inputs)  # Pass `inputs` directly to `loss_fn`

        # Update the parameters with functional optimizer
        params = optimizer.step(loss, params)

        print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))

    return loss_evolution

def main():
    # Parse input from user
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num-hidden", type=int, default=5)
    parser.add_argument("-d", "--dim-hidden", type=int, default=5)
    parser.add_argument("-b", "--batch-size", type=int, default=100)
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-e", "--num-epochs", type=int, default=1000)

    args = parser.parse_args()

    # Train the PINN
    loss_evolution = train_pinn(args)

    # Initialize OpenGL and run the visualization
    init_glut_window()
    glut.glutMainLoop()

if __name__ == "__main__":
    main()

