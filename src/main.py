# main.py

import numpy as np
from pinn.model import NavierStokesPINN
from pinn.train import generate_fluid_data, train_navier_stokes_pinn, save_model, load_model
from fluid_sim.simulation import FluidSimulation

def train_pinn():
    # Generate data
    n_samples = 10000
    x_train, y_train = generate_fluid_data(n_samples, t_range=(0, 1), x_range=(0, 1), y_range=(0, 1), include_temp=True)
    x_val, y_val = generate_fluid_data(n_samples // 10, t_range=(0, 1), x_range=(0, 1), y_range=(0, 1), include_temp=True)
    
    # Initialize model
    input_dim = 4  # t, x, y, T
    output_dim = 4  # u, v, p, T
    hidden_layers = [50, 50, 50, 50]
    model = NavierStokesPINN(input_dim, output_dim, hidden_layers)
    
    # Set fluid parameters
    rho = 1.0  # fluid density
    nu = 0.01  # kinematic viscosity
    alpha = 0.005  # thermal diffusivity
    
    # Train model
    trained_model, train_losses, val_losses = train_navier_stokes_pinn(
        model, x_train, y_train, x_val, y_val, rho, nu, alpha,
        epochs=1000, batch_size=64, lr=0.001, patience=50
    )
    
    # Save the trained model
    save_model(trained_model)
    
    print("Training completed and model saved.")
    return trained_model

def run_simulation():
    nx, ny = 200, 200  # Increased resolution for better accuracy
    dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)
    dt = 0.001
    rho, nu, alpha = 1.0, 0.01, 0.005
    
    simulation = FluidSimulation(nx, ny, dx, dy, dt, rho, nu, alpha)
    
    # Set initial conditions (you can change to 'vortex' for different initial conditions)
    simulation.set_initial_conditions('uniform')
    simulation.set_boundary_conditions('no_slip')
    
    # Add an initial obstacle
    simulation.add_obstacle(nx//2, ny//2, 10)
    
    return simulation

def main():
    # Check if a trained model exists, if not, train a new one
    try:
        model = load_model(NavierStokesPINN(4, 4, [50, 50, 50, 50]))
        print("Loaded existing model.")
    except (FileNotFoundError, RuntimeError):
        print("No existing model found or model incompatible. Training a new model.")
        model = train_pinn()

    # Run fluid simulation
    simulation = run_simulation()
    
    # Visualize simulation
    print("Starting visualization. Close the plot window to end the simulation.")
    print("Click on the velocity plot to add obstacles.")
    print("Use the 'Add Obstacle' button to enable/disable obstacle addition.")
    print("Use the 'Reset Simulation' button to clear all obstacles and reset the flow.")
    
    anim = simulation.visualize_simulation(num_frames=1000, interval=1)

if __name__ == "__main__":
    main()
