import torch

def model(x, coeff_0, coeff_1, operations):
    x_0, x_1 = x
    y_0 = operations[0](x_0, coeff_0)
    y_1 = operations[1](x_1, coeff_1)
    z_0 = y_0
    z_1 = operations[2](y_0, y_1)
    return(torch.stack([z_0, z_1], axis=0))

def data_fit(x, operations, observations):
    z = model(x, operations)
    return(torch.sum((z-observations)**2))

if __name__ == '__main__':
    x_0_vec = torch.arange(0, 10, 1)
    x_1_vec = torch.arange(0, 10, 1)

    x0_m, x1_m = torch.meshgrid(x_0_vec, x_1_vec)
    ground_truth_x = torch.stack([x0_m, x1_m], axis=2)
    ground_truth_x = ground_truth_x.reshape(2, -1)
    ground_truth_operations = [torch.multiply, torch.multiply, torch.multiply]
    ground_truth_coeff_0 = torch.Tensor([4])
    ground_truth_coeff_1 = torch.Tensor([3])
    observed_z = model(ground_truth_x, ground_truth_coeff_0, ground_truth_coeff_1, ground_truth_operations)

    assumed_coeff_vec = torch.repeat_interleave([[4],[3]], ground_truth_x.shape[1], axis=1)
    assumed_operations = [torch.multiply, torch.multiply, torch.add]

    # find the x that fits the observed z, given assumed coeff and operations

    # initial conditions
    x_0 = torch.ones([2, observed_z.shape[1]])*5

    # allow the coefficients to vary

    # do not allow the coefficients to vary

    # try all sets of operations

    breakpoint()