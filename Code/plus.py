import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
def cos_plus(lambda_max=None, lambda_min=None, E_loops=11,highpass=True):
    if highpass:
        def calculate_lambda_n1(E_n):
            return F.relu(lambda_max + 0.5 * lambda_min * (1 + torch.cos((1 + torch.min(E_n, torch.tensor(E_loops, dtype=torch.float32)) / E_loops) * torch.pi)))

        epochs = torch.arange(0, E_loops + 1, dtype=torch.float32)
        lambda_values = torch.stack([calculate_lambda_n1(E_n) for E_n in epochs])
    else:
        def calculate_lambda_n2(E_n):
            return F.relu(lambda_max - 0.5 * lambda_min * (1 + torch.cos(
                (1 + torch.min(E_n, torch.tensor(E_loops, dtype=torch.float32)) / E_loops) * torch.pi)))

        epochs = torch.arange(0, E_loops + 1, dtype=torch.float32)
        lambda_values = torch.stack([calculate_lambda_n2(E_n) for E_n in epochs])
    return lambda_values
