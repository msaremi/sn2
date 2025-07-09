import json
import torch
import argparse
from pathlib import Path
from sn2_cuda import SN2Solver
device = torch.device("cuda")

def get_results(data, num_expr=500, dtype=torch.float32, lrs=(1e-4, 1e-3, 1e-2, 1e-1, 1), max_iterations=100):
    device = torch.device("cuda")
    results = {}

    for name, pmDAG in data.items():
        print(f"Working on '{name}'")
        results[name] = {}
        struct = torch.tensor(pmDAG['struct'], dtype=torch.bool, device=device)

        for method in (SN2Solver.METHODS.ACCUM, SN2Solver.METHODS.COVAR):
            print(f"\tMethod set to '{method}'")
            results[name][str(method)] = {}

            for lr in lrs:
                print(f"\t\tLearning rate set to {lr}")
                num_success = 0

                for expr in range(num_expr):
                    print(f"\t\t\tExperiment {expr + 1} of {num_expr}", end='')
                    pmDAG = SN2Solver(struct, method=method, dtype=dtype)
                    pmDAG_true = SN2Solver(struct, dtype=dtype)
                    pmDAG_true.forward()
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
                        print(" failed.")
                    else:
                        num_success += 1
                        print(" succeeded.")

                results[name][str(method)][lr] = num_success / num_expr

    return results


def _generate_results(num_expr=500, dtype=torch.float32, lrs=(1e-4, 1e-3, 1e-2, 1e-1, 1), max_iterations=100, dir="."):
    with open(Path(dir, 'pmDAGs.json'), 'r') as fp:
        data = json.load(fp)

    results = get_results(data, num_expr, dtype, lrs, max_iterations)

    with open(Path(dir, f"numerical_stability_results_{str(dtype).rsplit('.')[-1]}_n{num_expr}.json"), 'w') as fp:
        json.dump(results, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", default=".")
    parser.add_argument("-num_expr", default=500, type=int)
    parser.add_argument("-dtype", choices=('float32', 'float64'), default='float64', type=str)
    parser.add_argument("-lrs", default=(1e-4, 1e-3, 1e-2, 1e-1, 1), nargs='+', type=float)
    parser.add_argument("-max_iterations", default=100, type=int)
    args = parser.parse_args()

    _generate_results(
        args.num_expr, torch.float32 if args.dtype == 'float32' else torch.float64,
        args.lrs, args.max_iterations, args.dir
    )