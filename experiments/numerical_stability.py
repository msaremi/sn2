import json
import torch
import argparse
from pathlib import Path
from sn2_cuda import SN2Solver
from sn2_cuda.loss import KullbackLeibler, Bhattacharyya, SquaredHellinger
device = torch.device("cuda")

def get_results(data, num_expr=500, dtype=torch.float64, lrs=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10), max_iterations=100):
    device = torch.device("cuda")
    results = {}

    for name, pmDAG in data.items():
        print(f"Working on '{name}'")
        results[name] = {}
        struct = torch.tensor(pmDAG['struct'], dtype=torch.bool, device=device)

        for lr in lrs:
            print(f"\tLearning rate set to {lr}")
            results[name][lr] = {}

            for loss in (KullbackLeibler, Bhattacharyya, SquaredHellinger):
                results[name][lr][loss.__name__] = {}

                for method in (SN2Solver.METHODS.ACCUM, SN2Solver.METHODS.COVAR):
                    results[name][lr][loss.__name__][method.name] = 0

            for expr in range(num_expr):
                print(f"\t\tExperiment {expr + 1:3} of {num_expr:3}", end='')
                pmDAG_true = SN2Solver(struct, dtype=dtype)
                pmDAG_true.forward()
                weights = torch.randn_like(struct, dtype=dtype, device=device)

                for loss in (KullbackLeibler, Bhattacharyya, SquaredHellinger):
                    for method in (SN2Solver.METHODS.ACCUM, SN2Solver.METHODS.COVAR):
                        print(f"\t{loss.__name__}+{method.name}", end=' ')
                        pmDAG = SN2Solver(struct, weights=weights.clone(), method=method, loss=loss(), dtype=dtype)
                        pmDAG.sample_covariance = pmDAG_true.visible_covariance_
                        optim = torch.optim.SGD([pmDAG.weights], lr=lr)
                        failure = False

                        for i in range(max_iterations):
                            pmDAG.forward()
                            error = pmDAG.loss().item()

                            if error < 1e-5:
                                break

                            pmDAG.backward()
                            optim.step()

                            if not torch.all(torch.isfinite(pmDAG.weights.grad)):
                                failure = True
                                break

                        if failure:
                            print("fail", end='')
                        else:
                            results[name][lr][loss.__name__][method.name] += 1
                            print("pass", end='')

                print()

    return results


def _generate_results(num_expr=500, dtype=torch.float64, lrs=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10), max_iterations=100, dir="."):
    with open(Path(dir, 'pmDAGs.json'), 'r') as fp:
        data = json.load(fp)

    results = get_results(data, num_expr, dtype, lrs, max_iterations)

    with open(Path(dir, f"numerical_stability_results_{dtype.__reduce__()}_n{num_expr}.json"), 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default=".")
    parser.add_argument("-num_expr", default=220, type=int)
    parser.add_argument("-dtype", choices=('float32', 'float64'), default='float64', type=str)
    parser.add_argument("-lrs", default=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10), nargs='+', type=float)
    parser.add_argument("-max_iterations", default=100, type=int)
    args = parser.parse_args()

    _generate_results(
        args.num_expr, torch.float32 if args.dtype == 'float32' else torch.float64,
        args.lrs, args.max_iterations, args.dir
    )