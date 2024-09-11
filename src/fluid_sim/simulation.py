import numpy as np
import torch
import torch.nn as nn
from pinn.model import NavierStokesPINN
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from numba import jit, prange

class FluidSimulation:
    def __init__(self, nx, ny, dx, dy, dt, rho, nu, alpha):
        self.nx, self.ny = nx, ny
        self.dx, self.dy, self.dt = dx, dy, dt
        self.rho, self.nu, self.alpha = rho, nu, alpha
        
        # Initialize fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.T = np.zeros((ny, nx))
        
        self.mask = np.ones((ny, nx))
        self.refinement_level = np.ones((ny, nx), dtype=int)
        
        self.obstacle_size = 5
        self.obstacle_type = 'circle'

        # Load and optimize PINN model
        self.pinn_model = self.load_pinn_model()

    def load_pinn_model(self):
        model = NavierStokesPINN(input_dim=4, output_dim=4, hidden_layers=[50, 50, 50, 50])
        model.load_state_dict(torch.load('models/navier_stokes_pinn.pth', map_location=torch.device('cpu')))
        model.eval()
        return torch.jit.script(model)  # JIT compile the model

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
        grad_u = np.gradient(self.u)
        grad_v = np.gradient(self.v)
        grad_mag = np.sqrt(grad_u[0]**2 + grad_u[1]**2 + grad_v[0]**2 + grad_v[1]**2)
        self.refinement_level = np.minimum(np.maximum(grad_mag * 10, 1), 3).astype(int)

    def pinn_prediction_batch(self, t, x, y, T):
        input_tensor = torch.tensor(np.column_stack([t, x, y, T]), dtype=torch.float32)
        with torch.no_grad():
            prediction = self.pinn_model(input_tensor)
        return prediction.numpy()

    def update_fields(self):
        t = np.full((self.ny, self.nx), self.dt)
        x = np.arange(0, self.nx * self.dx, self.dx)
        y = np.arange(0, self.ny * self.dy, self.dy)
        X, Y = np.meshgrid(x, y)
        
        inputs = np.column_stack([t.flatten(), X.flatten(), Y.flatten(), self.T.flatten()])
        predictions = self.pinn_prediction_batch(inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3])
        
        u_pred = predictions[:, 0].reshape(self.ny, self.nx)
        v_pred = predictions[:, 1].reshape(self.ny, self.nx)
        p_pred = predictions[:, 2].reshape(self.ny, self.nx)
        T_pred = predictions[:, 3].reshape(self.ny, self.nx)
        
        weight = self.refinement_level / 3.0
        
        self.u = np.where(self.mask, (1 - weight) * self.u + weight * u_pred, 0)
        self.v = np.where(self.mask, (1 - weight) * self.v + weight * v_pred, 0)
        self.p = np.where(self.mask, (1 - weight) * self.p + weight * p_pred, self.p)
        self.T = np.where(self.mask, (1 - weight) * self.T + weight * T_pred, self.T)
        
        self.set_boundary_conditions()

    @staticmethod
    @jit(nopython=True, parallel=True)
    def update_fields_numba(u, v, p, T, mask, refinement_level, dx, dy, dt, pinn_prediction):
        ny, nx = u.shape
        for i in prange(1, ny - 1):
            for j in prange(1, nx - 1):
                if mask[i, j]:
                    x, y = j * dx, i * dy
                    t = dt
                    T_val = T[i, j]
                    u_new, v_new, p_new, T_new = pinn_prediction(t, x, y, T_val)
                    
                    ref_level = refinement_level[i, j]
                    weight = ref_level / 3.0
                    
                    u[i, j] = (1 - weight) * u[i, j] + weight * u_new
                    v[i, j] = (1 - weight) * v[i, j] + weight * v_new
                    p[i, j] = (1 - weight) * p[i, j] + weight * p_new
                    T[i, j] = (1 - weight) * T[i, j] + weight * T_new
        return u, v, p, T

    def simulate_step(self):
        self.refine_mesh()
        self.update_fields()

    def get_velocity_field(self):
        return self.u, self.v

    def get_pressure_field(self):
        return self.p

    def get_temperature_field(self):
        return self.T

    def add_obstacle(self, x, y, size):
        y_grid, x_grid = np.ogrid[:self.ny, :self.nx]
        if self.obstacle_type == 'circle':
            dist_from_center = np.sqrt((x_grid - x)**2 + (y_grid - y)**2)
            self.mask[dist_from_center <= size] = 0
        elif self.obstacle_type == 'square':
            self.mask[max(0, y-size):min(self.ny, y+size), max(0, x-size):min(self.nx, x+size)] = 0

    def visualize_simulation(self, num_frames=200, interval=50):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        u, v = self.get_velocity_field()
        p = self.get_pressure_field()
        
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
        ax_obstacle = plt.axes([0.1, 0.01, 0.1, 0.03])
        button_obstacle = Button(ax_obstacle, 'Add Obstacle')
        
        ax_reset = plt.axes([0.25, 0.01, 0.1, 0.03])
        button_reset = Button(ax_reset, 'Reset Simulation')
        
        ax_viscosity = plt.axes([0.4, 0.01, 0.3, 0.03])
        slider_viscosity = Slider(ax_viscosity, 'Viscosity', 0.001, 0.1, valinit=self.nu)
        
        ax_obstacle_type = plt.axes([0.75, 0.01, 0.1, 0.03])
        button_obstacle_type = Button(ax_obstacle_type, 'Toggle Shape')
        
        # Add text annotations
        text_info = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, va='top', ha='left')
        
        def on_click(event):
            if event.inaxes == ax1:
                x, y = int(event.xdata * downsample), int(event.ydata * downsample)
                self.add_obstacle(x, y, self.obstacle_size)
                # Visual indicator
                circle = plt.Circle((event.xdata, event.ydata), self.obstacle_size/downsample, 
                                    fill=False, color='r')
                ax1.add_artist(circle)
                fig.canvas.draw_idle()
        
        def reset_simulation(event):
            self.set_initial_conditions()
            self.mask = np.ones((self.ny, self.nx))
        
        def update_viscosity(val):
            self.nu = slider_viscosity.val
        
        def toggle_obstacle_type(event):
            self.obstacle_type = 'square' if self.obstacle_type == 'circle' else 'circle'
            button_obstacle_type.label.set_text(f'Shape: {self.obstacle_type}')
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        button_obstacle.on_clicked(lambda x: None)
        button_reset.on_clicked(reset_simulation)
        slider_viscosity.on_changed(update_viscosity)
        button_obstacle_type.on_clicked(toggle_obstacle_type)
        
        def update(frame):
            for _ in range(5):
                self.simulate_step()
            
            u, v = self.get_velocity_field()
            p = self.get_pressure_field()
            
            u_vis = u[::downsample, ::downsample]
            v_vis = v[::downsample, ::downsample]
            p_vis = p[::downsample, ::downsample]
            
            vel_mag = np.sqrt(u_vis**2 + v_vis**2)
            im1.set_array(vel_mag)
            im2.set_array(p_vis)
            
            # Update text annotations
            avg_vel = np.mean(vel_mag)
            max_pressure = np.max(np.abs(p))
            info_text = f'Avg Velocity: {avg_vel:.4f}\nMax Pressure: {max_pressure:.4f}\nViscosity: {self.nu:.4f}'
            text_info.set_text(info_text)
            
            return im1, im2, text_info
        
        anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=True)
        plt.tight_layout()
        plt.show()
        
        return anim

# Helper function for running the simulation
def run_simulation(nx=200, ny=200, num_frames=500, interval=20):
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    dt = 0.001
    rho, nu, alpha = 1.0, 0.01, 0.005
    
    sim = FluidSimulation(nx, ny, dx, dy, dt, rho, nu, alpha)
    sim.set_initial_conditions('uniform')
    sim.set_boundary_conditions('no_slip')
    sim.add_obstacle(nx//2, ny//2, 10)
    
    anim = sim.visualize_simulation(num_frames=num_frames, interval=interval)
    return sim, anim

if __name__ == "__main__":
    sim, anim = run_simulation()
    plt.show()
