import torch
import torch.nn as nn

class NavierStokesPINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super(NavierStokesPINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.Tanh())
        
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def navier_stokes_loss(model, x, y_true, rho, nu, alpha):
    x = x.detach().requires_grad_(True) 
    y_pred = model(x)
    u, v, p, T = y_pred[:, 0:1], y_pred[:, 1:2], y_pred[:, 2:3], y_pred[:, 3:4]
    
    du_dt = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0:1]
    du_dx = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 1:2]
    du_dy = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 2:3]
    
    dv_dt = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 0:1]
    dv_dx = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1:2]
    dv_dy = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 2:3]
    
    dp_dx = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 1:2]
    dp_dy = torch.autograd.grad(p.sum(), x, create_graph=True)[0][:, 2:3]
    
    d2u_dx2 = torch.autograd.grad(du_dx.sum(), x, create_graph=True)[0][:, 1:2]
    d2u_dy2 = torch.autograd.grad(du_dy.sum(), x, create_graph=True)[0][:, 2:3]
    
    d2v_dx2 = torch.autograd.grad(dv_dx.sum(), x, create_graph=True)[0][:, 1:2]
    d2v_dy2 = torch.autograd.grad(dv_dy.sum(), x, create_graph=True)[0][:, 2:3]
    
    dT_dt = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 0:1]
    dT_dx = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 1:2]
    dT_dy = torch.autograd.grad(T.sum(), x, create_graph=True)[0][:, 2:3]
    d2T_dx2 = torch.autograd.grad(dT_dx.sum(), x, create_graph=True)[0][:, 1:2]
    d2T_dy2 = torch.autograd.grad(dT_dy.sum(), x, create_graph=True)[0][:, 2:3]
    
    ns_u = du_dt + u * du_dx + v * du_dy + (1/rho) * dp_dx - nu * (d2u_dx2 + d2u_dy2)
    ns_v = dv_dt + u * dv_dx + v * dv_dy + (1/rho) * dp_dy - nu * (d2v_dx2 + d2v_dy2)
    continuity = du_dx + dv_dy
    energy = dT_dt + u * dT_dx + v * dT_dy - alpha * (d2T_dx2 + d2T_dy2)
    
    mse_loss = nn.MSELoss()(y_pred, y_true)
    physics_loss = torch.mean(ns_u**2 + ns_v**2 + continuity**2 + energy**2)
    
    total_loss = mse_loss + physics_loss
    
    return total_loss
