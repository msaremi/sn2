import torch
from semnan_cuda import SEMNAN

cuda_device = torch.device("cuda")
dtype = torch.double
learning_rate = 0.001
num_iterations = 10000

struct = torch.tensor([  # Upper-triangular matrix with latent space
        [1, 1, 1, 0],  # latent space
        [0, 1, 0, 1],  # latent space
        [1, 0, 0, 0],  # latent space
        [0, 1, 0, 0],  # latent space
        [0, 0, 1, 0],  # latent space
        [0, 0, 0, 1],  # latent space
        [0, 1, 1, 1],  # visible space
        [0, 0, 1, 0],  # visible space
        [0, 0, 0, 1],  # visible space
        [0, 0, 0, 0],  # visible space
], device=cuda_device, dtype=torch.bool)

sample_covar_inv = torch.linalg.pinv(torch.tensor([
        [2,  3,  6,  8],
        [3,  7, 12, 16],
        [6, 12, 23, 30],
        [8, 16, 30, 41],
], dtype=dtype))

params = torch.ones_like(struct, dtype=dtype)
params += torch.randn_like(struct, dtype=dtype)  # add noise to the ground-truth parameters

semnan = SEMNAN(struct, params, dtype)
semnan.set_sample_covariance_inv(sample_covar_inv)
optim = torch.optim.Adamax([semnan.get_weights()], lr=learning_rate)
semnan.forward()
print(semnan.get_visible_covariance())

for i in range(num_iterations):
    semnan.forward()
    semnan.backward()
    optim.step()

    if i % (num_iterations / 100) == 0:
        print(f"iter={i}  \terror={semnan.kullback_leibler_loss_forward().item()}")

print(semnan.get_visible_covariance())
