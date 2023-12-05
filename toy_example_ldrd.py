"""
cd $WORK/output_torchBragg
libtbx.python $MODULES/torchBragg/toy_example_ldrd.py
"""
import torch
import numpy as np

def model(x, coeff_0, coeff_1, operations):
    x_0, x_1 = x
    y_0 = operations[0](x_0, coeff_0)
    y_1 = operations[1](x_1, coeff_1)
    z_0 = y_0
    z_1 = operations[2](y_0, y_1)
    return(torch.stack([z_0, z_1], axis=0))

def data_fit(x, coeff_0, coeff_1, operations, observations):
    z = model(x, coeff_0, coeff_1,operations)
    return(torch.mean((z-observations)**2))

def construct_ground_truth_x(x_min, x_max, num_points):
    x_0_vec = torch.linspace(x_min, x_max, num_points)
    x_1_vec = torch.linspace(x_min, x_max, num_points)
    x0_m, x1_m = torch.meshgrid(x_0_vec, x_1_vec, indexing='ij')
    ground_truth_x = torch.stack([x0_m, x1_m], axis=2)
    ground_truth_x = ground_truth_x.reshape(2, -1)
    return(ground_truth_x)

def find_x(x_min, x_max, num_points, 
           ground_truth_coeff_0, ground_truth_coeff_1,
           assumed_coeff_0_np, assumed_coeff_1_np,
           ground_truth_operations,
           assumed_operations,
           num_iter, let_coeff_vary, lr=0.01, init_x=5):
    
    ground_truth_x = construct_ground_truth_x(x_min, x_max, num_points)
    observed_z = model(ground_truth_x, ground_truth_coeff_0, ground_truth_coeff_1, ground_truth_operations)
    
    x_0 = torch.tensor(np.ones([2, observed_z.shape[1]])*init_x, requires_grad=True)

    if let_coeff_vary:
        assumed_coeff_0 = torch.tensor(assumed_coeff_0_np, requires_grad=True)
        assumed_coeff_1 = torch.tensor(assumed_coeff_1_np, requires_grad=True)

        # optimizer = torch.optim.Adam([x_0, assumed_coeff_0, assumed_coeff_1], lr=lr)
        optimizer = torch.optim.LBFGS([x_0, assumed_coeff_0, assumed_coeff_1], lr=lr)
    else:
        assumed_coeff_0 = torch.tensor(assumed_coeff_0_np, requires_grad=False)
        assumed_coeff_1 = torch.tensor(assumed_coeff_1_np, requires_grad=False)

        # optimizer = torch.optim.Adam([x_0], lr=lr)
        optimizer = torch.optim.LBFGS([x_0], lr=lr)

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            loss = data_fit(x_0, assumed_coeff_0, assumed_coeff_1, assumed_operations, observed_z)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    actual_MSE = torch.mean((x_0-ground_truth_x)**2).detach().numpy()
    optimization_loss = closure().detach().numpy()
    print('Optimization loss is: ', optimization_loss)
    print('Actual coeff are: ', ground_truth_coeff_0.detach().numpy(), ground_truth_coeff_1.detach().numpy())
    print('Final coeff are: ', assumed_coeff_0.detach().numpy(), assumed_coeff_1.detach().numpy())
    print('Actual MSE loss is: ', actual_MSE)
    return(optimization_loss, actual_MSE, ground_truth_x.detach().numpy(), x_0.detach().numpy())
    

if __name__ == '__main__':

    ground_truth_operations = [torch.multiply, torch.multiply, torch.add]
    ground_truth_coeff_0 = torch.Tensor([5])
    ground_truth_coeff_1 = torch.Tensor([4])
    assumed_operations = [torch.multiply, torch.multiply, torch.add]
    x_min = -10 
    x_max = 10
    num_points = 201
    assumed_coeff_0_np = np.array(5.)
    assumed_coeff_1_np = np.array(4.)
    num_iter = 1000
    let_coeff_vary = True
    init_x = 0

    optimization_loss, actual_MSE, true_x, final_x = find_x(x_min, x_max, num_points, 
                                           ground_truth_coeff_0, ground_truth_coeff_1,
                                           assumed_coeff_0_np, assumed_coeff_1_np,
                                           ground_truth_operations,
                                           assumed_operations,
                                           num_iter, let_coeff_vary, lr=0.01, init_x=init_x)
    breakpoint()
    

