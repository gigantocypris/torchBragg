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

if __name__ == '__main__':
    let_coeff_vary = True

    x_0_vec = torch.arange(0, 10, 1)
    x_1_vec = torch.arange(0, 10, 1)

    num_iter = 1000

    x0_m, x1_m = torch.meshgrid(x_0_vec, x_1_vec, indexing='ij')
    ground_truth_x = torch.stack([x0_m, x1_m], axis=2)
    ground_truth_x = ground_truth_x.reshape(2, -1)
    ground_truth_operations = [torch.multiply, torch.multiply, torch.add]
    ground_truth_coeff_0 = torch.Tensor([5])
    ground_truth_coeff_1 = torch.Tensor([4])
    observed_z = model(ground_truth_x, ground_truth_coeff_0, ground_truth_coeff_1, ground_truth_operations)

    assumed_operations = [torch.multiply, torch.multiply, torch.add]

    # find the x that fits the observed z

    if let_coeff_vary:
        # allow the coefficients to vary
        x_0 = torch.tensor(np.ones([2, observed_z.shape[1]])*5, requires_grad=True)
        assumed_coeff_0 = torch.tensor(4., requires_grad=True)
        assumed_coeff_1 = torch.tensor(3., requires_grad=True)

        optimizer = torch.optim.Adam([x_0, assumed_coeff_0, assumed_coeff_1], lr=0.01)
        # optimizer = torch.optim.LBFGS([x_0, assumed_coeff_0, assumed_coeff_1], lr=0.0001)
    else:
        # do not allow the coefficients to vary
        x_0 = torch.tensor(np.ones([2, observed_z.shape[1]])*5, requires_grad=True)
        assumed_coeff_0 = torch.tensor(4., requires_grad=False)
        assumed_coeff_1 = torch.tensor(3., requires_grad=False)

        optimizer = torch.optim.Adam([x_0], lr=0.01)
        # optimizer = torch.optim.LBFGS([x_0], lr=0.0001)

    for i in range(num_iter):
        def closure():
            optimizer.zero_grad()
            loss = data_fit(x_0, assumed_coeff_0, assumed_coeff_1, assumed_operations, observed_z)
            loss.backward()
            return loss
        optimizer.step(closure)
    
    print(closure())
    print(assumed_coeff_0.detach().numpy(), assumed_coeff_1.detach().numpy())



    
    # do not allow the coefficients to vary

    # try all sets of operations