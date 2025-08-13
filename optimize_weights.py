import datetime
import logging
import os

import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import analysis
import common
import synthesis


class WeightOptimizer(ElementwiseProblem):
    def __init__(
            self, texton_library: analysis.result_objects.PrimaryTextonResult, synthetic_size
    ):
        self.texton_library = texton_library
        self.synthetic_size = synthetic_size
        self.target_global_coverage = self.texton_library.global_coverage

        parameter_count = len(synthesis.Weights().to_array())
        super().__init__(
            n_var=parameter_count,
            xu=np.full(parameter_count, 1),
            xl=np.full(parameter_count, -1),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        weights = synthesis.Weights.from_array(x)

        primary_textons = synthesis.primary_texton_distro(
            self.texton_library.primary_textons, self.synthetic_size, weights=weights,
        )

        actual_distro = synthesis.primary_textons.get_coverage(primary_textons)
        out["F"] = (actual_distro - self.target_global_coverage) ** 2


class WeightOptimizerCallback(Callback):
    def __init__(self, callback=None, patience: int = 0, termination=None, min_error: float = 0) -> None:
        super().__init__()
        self.printed_header = False
        self.start = datetime.datetime.now()
        self.last_score = None
        self.no_improvement = 0
        self.callback = callback
        self.patience = patience
        self.termination = termination
        self.min_error = min_error

    def notify(self, algorithm: GA):
        best_score = algorithm.pop.get("F")[0][0]

        timestamp = str(datetime.datetime.now() - self.start)
        timestamp = timestamp[:timestamp.index(".")]

        if not self.printed_header:
            print(" ELAPSED  GEN       SCORE     CHANGE  PARAMETERS")
            self.printed_header = True

        print("{:>8} {:>4} {:>11.8f}  {:>9.6f}  [{}]".format(
            timestamp, algorithm.n_gen, best_score,
            0 if self.last_score is None else best_score - self.last_score,
            ", ".join("{:>9.6f}".format(i) for i in algorithm.pop.get("X")[0])
        ))

        if self.last_score == best_score:
            self.no_improvement += 1
            if self.no_improvement > self.patience:
                print("No improvement in {} generations. Terminating...".format(self.patience))
                self.termination.terminate()
                self.termination.perc = 1

        else:
            self.no_improvement = 0

        if best_score <= self.min_error:
            print(
                "Found solution with score {:.8f} (below {}). Terminating...".format(best_score, self.min_error)
            )
            self.termination.terminate()
            self.termination.perc = 1

        self.last_score = best_score


def optimize_weights(config: synthesis.OptimizeConfig) -> synthesis.Weights:
    logging.info("Reading analyzed files...")
    extracted_primary_textons = analysis.result_objects.PrimaryTextonResult.load(os.path.join(config.intermediate_directory, "primary_textons.dat"))

    problem = WeightOptimizer(extracted_primary_textons, (config.result_size, config.result_size))

    algorithm = GA(
        pop_size=config.population_size,
        sampling=synthesis.Weights().to_array(),
        eliminate_duplicates=True
    )

    termination = get_termination("n_gen", config.generations)

    start = datetime.datetime.now()
    res = minimize(
        problem,
        algorithm,
        termination,
        save_history=False,
        callback=WeightOptimizerCallback(
            patience=config.patience, termination=termination, min_error=config.min_error
        ),
        copy_termination=False
    )
    end = datetime.datetime.now()

    best_parameters = res.X
    best_score = res.F[0]

    weights = synthesis.Weights.from_array(best_parameters)
    weights.to_json(os.path.join(config.intermediate_directory, "placement_weights.json"), best_score)

    return weights


if __name__ == '__main__':
    common.logger.configure_logger(level=logging.ERROR)
    cfg = synthesis.OptimizeConfig.from_argv()
    optimize_weights(cfg)

