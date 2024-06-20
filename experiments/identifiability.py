import json
import torch
import argparse
from pathlib import Path
from sn2_cuda import SN2Solver


def get_results(data, num_expr=10, max_iterations=12000):
    def get_average_effective_distance(pmDAG_true, pmDAG, struct, identification_mask):
        num_vars = struct.shape[1]
        return torch.sum(torch.abs(pmDAG_true.weights[-num_vars:, :] - pmDAG.weights[-num_vars:, :]) * identification_mask) / \
               torch.sum(identification_mask)

    def get_intervened_covariance(pmDAG_true, struct, x_dims, y_dims):
        x_dims_fixed = x_dims + struct.shape[0] - struct.shape[1]
        pmDAG_true_intervened = SN2Solver(struct)
        pmDAG_true_intervened.weights = torch.clone(pmDAG_true.weights)
        pmDAG_true_intervened.weights[x_dims_fixed, :] = 0.0
        pmDAG_true_intervened.forward()
        return pmDAG_true_intervened.visible_covariance_[y_dims, :][:, y_dims]

    def get_kldiv(covariance1, covariance2):
        kldiv = torch.trace(torch.mm(torch.inverse(covariance1), covariance2))
        return (kldiv + torch.logdet(covariance1) - torch.logdet(covariance2) - covariance2.shape[0]) / 2.0


    device = torch.device("cuda")
    results = {}

    for name, pmDAG in data.items():
        print(f"Working on '{name}'")
        results[name] = {}
        struct = torch.tensor(pmDAG['struct'], dtype=torch.bool, device=device)
        mask = torch.tensor(pmDAG['xy identification mask'], dtype=torch.bool, device=device)
        x_dims = torch.tensor(pmDAG['x dims'], device=device)
        y_dims = torch.tensor(pmDAG['y dims'], device=device)

        loss = float('inf')
        expr = 0

        while expr < num_expr:
            print(f"Experiment {expr + 1}")
            results[name][expr] = {}
            pmDAG = SN2Solver(struct, method=SN2Solver.METHODS.ACCUM)
            pmDAG_true = SN2Solver(struct)

            pmDAG_true.forward()
            pmDAG.sample_covariance = pmDAG_true.visible_covariance_
            intervened_covariance_true = get_intervened_covariance(pmDAG_true, struct, x_dims, y_dims)
            optim = torch.optim.Adamax([pmDAG.weights], lr=0.001)
            prev_loss = float('inf')

            for i in range(max_iterations + 1):
                pmDAG.forward()
                pmDAG.backward()
                optim.step()

                if i % (max_iterations / 50) == 0:
                    loss = pmDAG.loss().item()
                    intervened_covariance = get_intervened_covariance(pmDAG, struct, x_dims, y_dims)
                    kldiv = get_kldiv(intervened_covariance_true, intervened_covariance).item()
                    average_effective_distance = get_average_effective_distance(pmDAG_true, pmDAG, struct, identification_mask=mask).item()
                    print(f"iteration={i:<10} loss={loss:<15.5} average effective distance={average_effective_distance:<15.5} KL div={kldiv:<15.5}")
                    results[name][expr][i] = {'loss': loss, 'average effective distance': average_effective_distance, 'KL div': kldiv}

                    if loss > 1e-4 and abs(prev_loss - loss) <= 1e-6:
                        break

                    prev_loss = loss

            if loss <= 1e-5:
                expr += 1
            else:
                print("Did not converge. Re-doing the experiment.")

    return results


def _generate_results(num_expr=10, max_iterations=12000, dir="."):
    with open(Path(dir, 'pmDAGs.json'), 'r') as fp:
        data = json.load(fp)

    results = get_results(data, num_expr, max_iterations)

    with open(Path(dir, 'results.json'), 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default=".")
    parser.add_argument("-num_expr", default=10, type=int)
    parser.add_argument("-max_iterations", default=12000, type=int)
    args = parser.parse_args()

    _generate_results(args.num_expr, args.max_iterations, args.dir)
