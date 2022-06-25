import torch
from semnan_cuda import SEMNANSolver, loss

device = torch.device("cuda")  # always CUDA
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
], device=device, dtype=torch.bool)

sample_covar = torch.tensor([
        [2,  3,  6,  8],
        [3,  7, 12, 16],
        [6, 12, 23, 30],
        [8, 16, 30, 41],
], dtype=dtype)

params = torch.ones_like(struct, dtype=dtype)
params += torch.empty_like(params).normal_(0, 1)  # add noise to the ground-truth parameters

semnan = SEMNANSolver(struct,                               # AMASEM structure
                      weights=params,                       # initial parameters
                      dtype=dtype,                          # torch.double or torch.float
                      loss=loss.KullbackLeibler(),          # or .Bhattacharyya or a subclass of .LossBase
                      )
semnan.sample_covariance = sample_covar
optim = torch.optim.Adamax([semnan.weights], lr=learning_rate)

semnan.forward()                    # create the visible covariance matrix
print(semnan.visible_covariance_)   # ... and display it

for i in range(num_iterations):
    semnan.forward()
    semnan.backward()
    optim.step()

    if i % (num_iterations / 100) == 0:
        print(f"iter={i:<10}"
              f"kullback_leibler_loss={semnan.loss().item():<15.5}"
              f"weights_error={torch.dist(semnan.weights, struct.to(dtype)).item():<15.5}"
              )

print(semnan.visible_covariance_)
