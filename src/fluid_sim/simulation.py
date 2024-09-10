# src/fluid_sim/simulation.py

import numpy as np
import torch
from pinn.model import NavierStokesPINN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class FluidSimulation:
    def __init__(self, nx, ny, dx, dy, dt, rho, nu, alpha):
        self.nx, self.ny = nx, ny
        self.dx, self.dy, self.dt = dx, dy, dt
        self.rho, self.nu, self.alpha = rho, nu, alpha
        
        # Initialize fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.T = np.zeros((ny, nx))  # Temperature field
        
        # Obstacle mask (1 for fluid, 0 for obstacle)
        self.mask = np.ones((ny, nx))
        
        # Adaptive mesh refinement
        self.refinement_level = np.ones((ny, nx), dtype=int)
        
        # Load PINN model
        self.pinn_model = self.load_pinn_model()

    def load_pinn_model(self):
        model = NavierStokesPINN(input_dim=4, output_dim=4, hidden_layers=[50, 50, 50, 50])
        model.load_state_dict(torch.load('models/navier_stokes_pinn.pth', weights_only=True))
        model.eval()
        return model

    def set_initial_conditions(self, condition='uniform'):
        if condition == 'uniform':
            self.u = np.ones((self.ny, self.nx)) * 0.1
            self.v = np.zeros((self.ny, self.nx))
        elif condition == 'vortex':
            y, x = np.meshgrid(np.linspace(0, 1, self.ny), np.linspace(0, 1, self.nx))
            self.u = -np.sin(np.pi * x) * np.cos(np.pi * y)
            self.v = np.cos(np.pi * x) * np.sin(np.pi * y)
        self.p = np.zeros((self.ny, self.nx))
        self.T = np.zeros((self.ny, self.nx))

    def set_boundary_conditions(self, condition='no_slip'):
        if condition == 'no_slip':
            self.u[0, :] = self.u[-1, :] = self.u[:, 0] = self.u[:, -1] = 0
            self.v[0, :] = self.v[-1, :] = self.v[:, 0] = self.v[:, -1] = 0
        elif condition == 'periodic':
            self.u[0, :] = self.u[-1, :]
            self.u[:, 0] = self.u[:, -1]
            self.v[0, :] = self.v[-1, :]
            self.v[:, 0] = self.v[:, -1]

    def add_obstacle(self, x, y, radius):
        y_grid, x_grid = np.ogrid[:self.ny, :self.nx]
        dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
        self.mask[dist_from_center <= radius] = 0

    def refine_mesh(self):
        # Simple refinement based on velocity gradient
        grad_u = np.gradient(self.u)
        grad_v = np.gradient(self.v)
        grad_mag = np.sqrt(grad_u[0]**2 + grad_u[1]**2 + grad_v[0]**2 + grad_v[1]**2)
        self.refinement_level = np.minimum(np.maximum(grad_mag * 10, 1), 3).astype(int)

    def pinn_prediction(self, t, x, y, T):
        input_tensor = torch.tensor([[t, x, y, T]], dtype=torch.float32)
        with torch.no_grad():
            prediction = self.pinn_model(input_tensor)
        return prediction.numpy()[0]

    def update_fields(self):
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                if self.mask[i, j]:
                    x, y = j * self.dx, i * self.dy
                    t = self.dt
                    T = self.T[i, j]
                    u, v, p, T_new = self.pinn_prediction(t, x, y, T)
                    
                    # Apply refinement
                    ref_level = self.refinement_level[i, j]
                    weight = ref_level / 3.0
                    
                    self.u[i, j] = (1 - weight) * self.u[i, j] + weight * u
                    self.v[i, j] = (1 - weight) * self.v[i, j] + weight * v
                    self.p[i, j] = (1 - weight) * self.p[i, j] + weight * p
                    self.T[i, j] = (1 - weight) * self.T[i, j] + weight * T_new

        self.set_boundary_conditions()

    def simulate_step(self):
        self.refine_mesh()
        self.update_fields()

    def get_velocity_field(self):
        return self.u, self.v

    def get_pressure_field(self):
        return self.p

    def get_temperature_field(self):
        return self.T

    def visualize_simulation(self, num_frames=200, interval=50):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        u, v = self.get_velocity_field()
        p = self.get_pressure_field()
        
        # Reduce resolution for visualization
        downsample = 4
        u_vis = u[::downsample, ::downsample]
        v_vis = v[::downsample, ::downsample]
        p_vis = p[::downsample, ::downsample]
        
        vel_mag = np.sqrt(u_vis**2 + v_vis**2)
        
        im1 = ax1.imshow(vel_mag, cmap='viridis', animated=True, aspect='equal')
        ax1.set_title('Velocity Magnitude')
        fig.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(p_vis, cmap='coolwarm', animated=True, aspect='equal')
        ax2.set_title('Pressure')
        fig.colorbar(im2, ax=ax2)
        
        # Add interactivity
        ax_obstacle = plt.axes([0.1, 0.01, 0.2, 0.03])
        button_obstacle = Button(ax_obstacle, 'Add Obstacle')
        
        ax_reset = plt.axes([0.35, 0.01, 0.2, 0.03])
        button_reset = Button(ax_reset, 'Reset Simulation')
        
        def on_click(event):
            if event.inaxes == ax1:
                x, y = int(event.xdata * downsample), int(event.ydata * downsample)
                self.add_obstacle(x, y, 3)
        
        def reset_simulation(event):
            self.set_initial_conditions()
            self.mask = np.ones((self.ny, self.nx))
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        button_obstacle.on_clicked(lambda x: None)
        button_reset.on_clicked(reset_simulation)
        
        def update(frame):
            for _ in range(5):  # Perform multiple simulation steps per frame
                self.simulate_step()
            
            u, v = self.get_velocity_field()
            p = self.get_pressure_field()
            
            u_vis = u[::downsample, ::downsample]
            v_vis = v[::downsample, ::downsample]
            p_vis = p[::downsample, ::downsample]
            
            vel_mag = np.sqrt(u_vis**2 + v_vis**2)
            im1.set_array(vel_mag)
            im2.set_array(p_vis)
            
            return im1, im2
        
        anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
        
        return anim
