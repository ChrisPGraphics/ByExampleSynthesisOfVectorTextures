import synthesis.synthesis_config as synthesis_config


class OptimizeConfig(synthesis_config.SynthesisConfig):
    _description = ("Fine tunes weights to improve synthesized results. "
                    "It is recommended to use --skip_density_correction flag on results synthesized with optimized weights")

    generations: int = 50
    result_size: int = 400
    population_size: int = 10
    patience: int = 5
    min_error: float = 1e-6

    @staticmethod
    def _add_args(parser):
        parser.add_argument(
            "--generations",
            help="The maximum number of generations to run with the genetic algorithm. "
                 "More generations will yield better parameters with diminishing returns "
                 "but will significantly increase computation time",
            type=int,
            default=50
        )

        parser.add_argument(
            "--result_size",
            help="The size of the exemplar to synthesize during the optimization process. "
                 "Larger values will be less prone to the stochastic nature of synthesis "
                 "but will increase computation time",
            type=int,
            default=400
        )

        parser.add_argument(
            "--population_size",
            help="The number of children to create per iteration. Larger values will be less prone to outliers but will "
                 "significantly increase computation time",
            type=int,
            default=10
        )

        parser.add_argument(
            "--patience",
            help="If the optimizer goes 'patience' iterations without improvement, it will terminate early. "
                 "This is useful to quickly end and restart the process if the optimizer gets stuck in a local minima",
            type=int,
            default=5
        )

        parser.add_argument(
            "--min_error",
            help="Terminate if the score drops below this threshold",
            type=float,
            default=1e-6
        )

    @staticmethod
    def _set_properties(config, args):
        config.generations = args.generations
        config.result_size = args.result_size
        config.population_size = args.population_size
        config.patience = args.patience
        config.min_error = args.min_error
