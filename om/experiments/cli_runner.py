import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from om.experiments.experiment_utils import (
    experiment1_ising1d,
    experiment2_ising1d__high_dim,
    experiment3_varselect_mice_data,
)
from om.experiments.multichain_bias_experiments_iter import (
    load_and_show_plots as load_iter_bias_plots,
    multi_chain_expected_func_compute_and_plot_itr,
)
from om.experiments.multichain_bias_experiments_time import (
    load_and_show_plots_TIME,
    multi_chain_expected_func_compute_and_plot_time,
)
from om.experiments.multichain_kl_experiments import (
    load_and_show_plots as load_kl_plots,
    multi_chain_kl_compute_and_plot,
)

DEFAULT_BVS_DATA_PATH = PROJECT_ROOT / "om" / "models" / "lifespan-merged.csv"


@dataclass(frozen=True)
class ExperimentPreset:
    family: str
    model: str
    description: str
    generator: Callable
    generator_args: dict
    num_records_per_chain: int
    num_independent_chains: int
    num_mcmc_samples_per_chain: int | None = None
    sampling_time_per_chain_seconds: float | None = None
    ground_truth_expected_func_value_already_known: float | None = None
    particle_arg_name: str | None = None
    default_run_mcmc_opad: bool = True
    default_run_dvpi: bool = False
    default_run_greedy_bfs: bool = False
    default_run_greedy_nwss: bool = False
    notes: str = ""


class ExperimentRunner:
    def __init__(self):
        self.presets = self._build_presets()

    @staticmethod
    def parse_y_limit(value: str):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError("Expected y-limits in the form min,max")
        return float(parts[0]), float(parts[1])

    def _build_presets(self) -> dict[tuple[str, str], ExperimentPreset]:
        return {
            ("kl", "ising15"): ExperimentPreset(
                family="kl",
                model="ising15",
                description="Ising15",
                generator=experiment1_ising1d,
                generator_args={"num_particles": 10, "latice_size": 15, "J": 1, "h": 0.1, "beta": 0.5},
                num_mcmc_samples_per_chain=20000,
                num_records_per_chain=100,
                num_independent_chains=20,
                particle_arg_name="num_particles",
                default_run_dvpi=True,
                default_run_greedy_nwss=True,
                notes="Paper KL setting: MCMC family, DVPI, DVPI OPAD+, and NWSS.",
            ),
            ("kl", "bvs"): ExperimentPreset(
                family="kl",
                model="bvs",
                description="BVS",
                generator=experiment3_varselect_mice_data,
                generator_args={"data_path": str(DEFAULT_BVS_DATA_PATH)},
                num_mcmc_samples_per_chain=10000,
                num_records_per_chain=100,
                num_independent_chains=20,
                default_run_dvpi=True,
                default_run_greedy_nwss=True,
                notes="Paper KL setting: MCMC family, DVPI, DVPI OPAD+, and NWSS.",
            ),
            ("bias_iter", "ising15"): ExperimentPreset(
                family="bias_iter",
                model="ising15",
                description="Ising15",
                generator=experiment1_ising1d,
                generator_args={"num_particles": 100, "latice_size": 15, "J": 1, "h": 0.1, "beta": 0.5},
                num_mcmc_samples_per_chain=100000,
                num_records_per_chain=100,
                num_independent_chains=30,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Iter-based consistency setting; not a paper experiment.",
            ),
            ("bias_iter", "ising30"): ExperimentPreset(
                family="bias_iter",
                model="ising30",
                description="Ising30",
                generator=experiment2_ising1d__high_dim,
                generator_args={"num_particles": 100, "latice_size": 30, "J": 1, "beta": 0.5},
                num_mcmc_samples_per_chain=10000,
                num_records_per_chain=100,
                num_independent_chains=30,
                ground_truth_expected_func_value_already_known=0,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Iter-based consistency setting; not a paper experiment.",
            ),
            ("bias_iter", "ising40"): ExperimentPreset(
                family="bias_iter",
                model="ising40",
                description="Ising40",
                generator=experiment2_ising1d__high_dim,
                generator_args={"num_particles": 100, "latice_size": 40, "J": 1, "beta": 0.5},
                num_mcmc_samples_per_chain=200000,
                num_records_per_chain=200,
                num_independent_chains=30,
                ground_truth_expected_func_value_already_known=0,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Iter-based consistency setting; not a paper experiment.",
            ),
            ("bias_iter", "bvs"): ExperimentPreset(
                family="bias_iter",
                model="bvs",
                description="MiceVarSelect",
                generator=experiment3_varselect_mice_data,
                generator_args={"data_path": str(DEFAULT_BVS_DATA_PATH), "num_variational_particles": 100},
                num_mcmc_samples_per_chain=100000,
                num_records_per_chain=100,
                num_independent_chains=30,
                particle_arg_name="num_variational_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Iter-based consistency setting; not a paper experiment.",
            ),
            ("bias_time", "ising15"): ExperimentPreset(
                family="bias_time",
                model="ising15",
                description="Ising15",
                generator=experiment1_ising1d,
                generator_args={"num_particles": 100, "latice_size": 15, "J": 1, "h": 0.1, "beta": 0.5},
                sampling_time_per_chain_seconds=12,
                num_records_per_chain=100,
                num_independent_chains=30,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Paper time-based experiment.",
            ),
            ("bias_time", "ising30"): ExperimentPreset(
                family="bias_time",
                model="ising30",
                description="Ising30",
                generator=experiment2_ising1d__high_dim,
                generator_args={"num_particles": 100, "latice_size": 30, "J": 1, "beta": 0.5},
                sampling_time_per_chain_seconds=12,
                num_records_per_chain=100,
                num_independent_chains=30,
                ground_truth_expected_func_value_already_known=0,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Paper time-based experiment.",
            ),
            ("bias_time", "ising40"): ExperimentPreset(
                family="bias_time",
                model="ising40",
                description="Ising40",
                generator=experiment2_ising1d__high_dim,
                generator_args={"num_particles": 100, "latice_size": 40, "J": 1, "beta": 0.5},
                sampling_time_per_chain_seconds=12,
                num_records_per_chain=100,
                num_independent_chains=30,
                ground_truth_expected_func_value_already_known=0,
                particle_arg_name="num_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Paper time-based experiment.",
            ),
            ("bias_time", "bvs"): ExperimentPreset(
                family="bias_time",
                model="bvs",
                description="BVS",
                generator=experiment3_varselect_mice_data,
                generator_args={"data_path": str(DEFAULT_BVS_DATA_PATH), "num_variational_particles": 100},
                sampling_time_per_chain_seconds=12,
                num_records_per_chain=100,
                num_independent_chains=30,
                particle_arg_name="num_variational_particles",
                default_run_greedy_bfs=True,
                default_run_greedy_nwss=True,
                notes="Paper time-based experiment.",
            ),
        }

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run OPAD paper experiments from the command line."
        )
        parser.add_argument("--list-presets", action="store_true", help="List available (family, model) presets and exit.")
        parser.add_argument("--family", choices=("kl", "bias_iter", "bias_time"))
        parser.add_argument("--model", choices=("ising15", "ising30", "ising40", "bvs"))
        parser.add_argument("--plot-results-dir", type=Path, default=None, help="Instead of running an experiment, reload a saved result folder and regenerate its plots.")
        parser.add_argument("--output-dir", type=Path, default=None, help="Directory under which timestamped result folders will be created.")
        parser.add_argument("--description", default=None, help="Override the default timestamp prefix for the run.")
        parser.add_argument("--num-particles", type=int, default=None, help="Override the preset particle count argument when supported.")
        parser.add_argument("--num-records", type=int, default=None, help="Override records per chain.")
        parser.add_argument("--num-chains", type=int, default=None, help="Override number of independent chains.")
        parser.add_argument("--num-mcmc-samples", type=int, default=None, help="Override MCMC samples per chain for KL/bias_iter experiments.")
        parser.add_argument("--sampling-time-seconds", type=float, default=None, help="Override total sampling time per chain for bias_time experiments.")
        parser.add_argument("--dataset-path", type=Path, default=None, help="Override BVS dataset path.")
        parser.add_argument("--observable", choices=("sum", "first_spin", "min_spin"), default="sum", help="Observable used for bias experiments.")
        parser.add_argument("--rhat-y-lim", type=self.parse_y_limit, default=None, help="For bias plot regeneration, set R-hat y-limits as min,max.")
        parser.add_argument("--variance-y-lim", type=self.parse_y_limit, default=None, help="For bias plot regeneration, set variance-plot y-limits as min,max.")
        parser.add_argument("--kl-y-lim", type=self.parse_y_limit, default=None, help="For KL plot regeneration, set KL y-limits as min,max.")
        parser.add_argument("--run-mcmc-opad", action=argparse.BooleanOptionalAction, default=None, help="Run the MCMC family (MCMC, OPAD, OPAD+, RB/WR).")
        parser.add_argument("--run-dvpi", action=argparse.BooleanOptionalAction, default=None, help="Run DVPI in KL experiments.")
        parser.add_argument("--run-greedy-bfs", action=argparse.BooleanOptionalAction, default=None, help="Run BFS greedy search when supported.")
        parser.add_argument("--run-greedy-nwss", action=argparse.BooleanOptionalAction, default=None, help="Run NWSS greedy search when supported.")
        return parser

    def list_presets(self) -> str:
        lines = []
        for key in sorted(self.presets):
            preset = self.presets[key]
            lines.append(
                f"{preset.family:10s} {preset.model:8s} "
                f"description={preset.description} notes={preset.notes}"
            )
        return "\n".join(lines)

    def get_preset(self, family: str, model: str) -> ExperimentPreset:
        key = (family, model)
        if key not in self.presets:
            raise ValueError(f"No preset registered for family={family!r}, model={model!r}.")
        return self.presets[key]

    def default_output_dir(self, family: str) -> Path:
        return {
            "kl": PROJECT_ROOT / "results" / "kl",
            "bias_iter": PROJECT_ROOT / "results" / "mse_vs_itr",
            "bias_time": PROJECT_ROOT / "results" / "mse_vs_time",
        }[family]

    def observable_from_name(self, observable_name: str) -> Callable:
        if observable_name == "sum":
            return lambda x: sum(x)
        if observable_name == "first_spin":
            return lambda x: x[0]
        if observable_name == "min_spin":
            return lambda x: min(x)
        raise ValueError(f"Unknown observable: {observable_name}")

    def run(self, args: argparse.Namespace):
        if args.list_presets:
            print(self.list_presets())
            return None

        if args.family is None:
            raise ValueError("--family is required unless --list-presets is used.")

        if args.plot_results_dir is not None:
            return self.regenerate_plots(args)

        if args.model is None:
            raise ValueError("--model is required when running a new experiment.")

        preset = self.get_preset(args.family, args.model)
        generator_args = dict(preset.generator_args)

        if args.num_particles is not None:
            if preset.particle_arg_name is None:
                raise ValueError(f"Preset ({args.family}, {args.model}) does not support --num-particles.")
            generator_args[preset.particle_arg_name] = args.num_particles

        if args.dataset_path is not None:
            if "data_path" not in generator_args:
                raise ValueError(f"Preset ({args.family}, {args.model}) does not use a dataset path.")
            generator_args["data_path"] = str(args.dataset_path.resolve())

        description = args.description or preset.description
        output_dir = args.output_dir.resolve() if args.output_dir else self.default_output_dir(args.family)
        run_mcmc_opad = preset.default_run_mcmc_opad if args.run_mcmc_opad is None else args.run_mcmc_opad
        run_dvpi = preset.default_run_dvpi if args.run_dvpi is None else args.run_dvpi
        run_greedy_bfs = preset.default_run_greedy_bfs if args.run_greedy_bfs is None else args.run_greedy_bfs
        run_greedy_nwss = preset.default_run_greedy_nwss if args.run_greedy_nwss is None else args.run_greedy_nwss

        common_kwargs = {
            "path": str(output_dir),
            "model_sampler_init_generator": preset.generator,
            "generator_args": generator_args,
            "num_records_per_chain": args.num_records or preset.num_records_per_chain,
            "num_independent_chains": args.num_chains or preset.num_independent_chains,
            "description": description,
        }

        if args.family == "kl":
            return multi_chain_kl_compute_and_plot(
                **common_kwargs,
                num_mcmc_samples_per_chain=args.num_mcmc_samples or preset.num_mcmc_samples_per_chain,
                do_run_mcmc_opad=run_mcmc_opad,
                do_run_dvpi=run_dvpi,
                do_run_greedy_bfs=run_greedy_bfs,
                do_run_greedy_nwss=run_greedy_nwss,
            )

        observable = self.observable_from_name(args.observable)

        if args.family == "bias_iter":
            return multi_chain_expected_func_compute_and_plot_itr(
                **common_kwargs,
                num_mcmc_samples_per_chain=args.num_mcmc_samples or preset.num_mcmc_samples_per_chain,
                func_state=observable,
                ground_truth_expected_func_value_already_known=preset.ground_truth_expected_func_value_already_known,
                do_run_mcmc_opad=run_mcmc_opad,
                do_run_greedy_bfs=run_greedy_bfs,
                do_run_greedy_nwss=run_greedy_nwss,
            )

        if args.family == "bias_time":
            return multi_chain_expected_func_compute_and_plot_time(
                **common_kwargs,
                sampling_time_per_chain_seconds=args.sampling_time_seconds or preset.sampling_time_per_chain_seconds,
                func_state=observable,
                ground_truth_expected_func_value_already_known=preset.ground_truth_expected_func_value_already_known,
                do_run_mcmc_opad=run_mcmc_opad,
                do_run_greedy_bfs=run_greedy_bfs,
                do_run_greedy_nwss=run_greedy_nwss,
            )

        raise ValueError(f"Unsupported family: {args.family}")

    def regenerate_plots(self, args: argparse.Namespace):
        result_path = args.plot_results_dir.resolve()
        result_dir = str(result_path)

        if args.family == "kl":
            has_mcmc = (result_path / "MCMC_KL_results.csv").exists()
            has_dvpi = (result_path / "DPVI_KL_results.csv").exists()
            has_greedy_bfs = (result_path / "back_KL_results.csv").exists()
            has_greedy_nwss = (result_path / "back_KL_results3.csv").exists()
        else:
            has_mcmc = (result_path / "MCMC_KL_results.csv").exists()
            has_dvpi = False
            has_greedy_bfs = (
                (result_path / "greedy_bfs_KL_results.csv").exists()
                and (result_path / "greedy_bfs_Recorded_Iters_for_all_algs.csv").exists()
            ) or (
                (result_path / "greedy_bfs_KL_results.csv").exists()
                and (result_path / "greedy_bfs_Recorded_Times_for_all_algs.csv").exists()
            )
            has_greedy_nwss = (
                (result_path / "greedy_nwss_KL_results.csv").exists()
                and (result_path / "greedy_nwss_Recorded_Iters_for_all_algs.csv").exists()
            ) or (
                (result_path / "greedy_nwss_KL_results.csv").exists()
                and (result_path / "greedy_nwss_Recorded_Times_for_all_algs.csv").exists()
            )

        run_mcmc_opad = has_mcmc if args.run_mcmc_opad is None else args.run_mcmc_opad
        run_dvpi = has_dvpi if args.run_dvpi is None else args.run_dvpi
        run_greedy_bfs = has_greedy_bfs if args.run_greedy_bfs is None else args.run_greedy_bfs
        run_greedy_nwss = has_greedy_nwss if args.run_greedy_nwss is None else args.run_greedy_nwss

        if args.family == "kl":
            return load_kl_plots(
                RESPATH=result_dir,
                do_plot_mcmc=run_mcmc_opad,
                do_plot_mcmc_opad=run_mcmc_opad,
                do_plot_mcmc_opad_plus=run_mcmc_opad,
                do_plot_mcmc_rb_wr=run_mcmc_opad,
                do_plot_dpvi=run_dvpi,
                do_plot_dvpi_plus=run_dvpi,
                do_plot_greedy_bfs=run_greedy_bfs,
                do_plot_greedy_nwss=run_greedy_nwss,
                y_limits=args.kl_y_lim,
            )

        if args.family == "bias_iter":
            return load_iter_bias_plots(
                RESPATH=result_dir,
                do_plot_greedy_bfs=run_greedy_bfs,
                do_plot_greedy_nwss=run_greedy_nwss,
                rhat_plotting_y_lim=args.rhat_y_lim,
                variance_plotting_y_lim=args.variance_y_lim,
            )

        if args.family == "bias_time":
            return load_and_show_plots_TIME(
                RESPATH=result_dir,
                do_plot_greedy_bfs=run_greedy_bfs,
                do_plot_greedy_nwss=run_greedy_nwss,
                rhat_plotting_y_lim=args.rhat_y_lim,
                variance_plotting_y_lim=args.variance_y_lim,
            )

        raise ValueError(f"Unsupported family for plot regeneration: {args.family}")


def main(argv=None):
    runner = ExperimentRunner()
    parser = runner.build_parser()
    args = parser.parse_args(argv)
    runner.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
