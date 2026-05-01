# Experiments

This directory contains the main experiment entrypoint for reproducing the experiments of the paper under review:
*Improving Discrete MCMC via Optimal Particle Weighting*

## Command-Line Interface
- `om/experiments/cli_runner.py`

The CLI replaces the old pattern of editing hardcoded `main()` blocks by hand.

## Quick Start

From the project root:

```bash
python om/experiments/cli_runner.py --list-presets
```

Run a paper-style KL experiment:

```bash
python om/experiments/cli_runner.py --family kl --model ising15
python om/experiments/cli_runner.py --family kl --model bvs
```

Run a paper-style bias/variance-vs-time experiment:

```bash
python om/experiments/cli_runner.py --family bias_time --model ising15
python om/experiments/cli_runner.py --family bias_time --model ising30
python om/experiments/cli_runner.py --family bias_time --model ising40
python om/experiments/cli_runner.py --family bias_time --model bvs
```

Iter-based bias/variance experiments 
(i.e. experiments that plot MSE, bias, variance and R-hat versus the number of drawn samples) 
are also supported for consistency, but they are not the paper experiments:

```bash
python om/experiments/cli_runner.py --family bias_iter --model ising15
```

Regenerate plots from an existing saved results folder:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --plot-results-dir om/experiments/results/mse_vs_time/Ising30_2026-04-30_18-37-21 \
  --rhat-y-lim 0.999,1.02 \
  --no-run-greedy-bfs \
  --no-run-greedy-nwss
```

When using `--plot-results-dir`, `--model` is not required. The CLI auto-detects which saved algorithm files exist in that folder unless you explicitly override the `--run-*` flags.

## Paper Presets

The current presets are:

- `kl ising15`
- `kl bvs`
- `bias_time ising15`
- `bias_time ising30`
- `bias_time ising40`
- `bias_time bvs`
- `bias_iter ising15`
- `bias_iter ising30`
- `bias_iter ising40`
- `bias_iter bvs`

Notes:

- The paper bias/variance experiments are the `bias_time` presets with `12` seconds per chain.
- The paper KL experiments are `kl ising15` and `kl bvs`.
- `bias_iter` presets are kept for consistency.

## Default Algorithms

For KL presets:

- `MCMC`
- `OPAD`
- `OPAD+`
- `RB/WR`
- `DVPI`
- `DVPI OPAD+`
- `NWSS`

For bias/variance presets:

- `MCMC`
- `OPAD`
- `OPAD+`
- `RB/WR`
- `BFS`
- `NWSS`

`DVPI` is not part of the bias/variance defaults.

## Main Arguments

Show all arguments with:

```bash
python om/experiments/cli_runner.py --help
```

Most useful arguments:

- `--family {kl,bias_iter,bias_time}`
- `--model {ising15,ising30,ising40,bvs}`
- `--output-dir PATH`
- `--description NAME`
- `--num-particles N`
- `--num-records N`
- `--num-chains N`
- `--num-mcmc-samples N`
- `--sampling-time-seconds T`
- `--dataset-path PATH`
- `--observable {sum,first_spin,min_spin}`
- `--plot-results-dir PATH`
- `--rhat-y-lim min,max`
- `--variance-y-lim min,max`
- `--kl-y-lim min,max`
- `--run-mcmc-opad` / `--no-run-mcmc-opad`
- `--run-dvpi` / `--no-run-dvpi`
- `--run-greedy-bfs` / `--no-run-greedy-bfs`
- `--run-greedy-nwss` / `--no-run-greedy-nwss`

## Examples

Override the output directory:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --model ising15 \
  --output-dir results/paper_runs
```

Override the run description:

```bash
python om/experiments/cli_runner.py \
  --family kl \
  --model ising15 \
  --description ising15_rb_added
```

Run only the MCMC family for a bias/time experiment:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --model ising15 \
  --no-run-greedy-bfs \
  --no-run-greedy-nwss
```

Run a KL experiment without DVPI:

```bash
python om/experiments/cli_runner.py \
  --family kl \
  --model ising15 \
  --no-run-dvpi
```

Regenerate saved bias/time plots with custom limits:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --plot-results-dir om/experiments/results/mse_vs_time/Ising15_2026-04-30_17-51-46 \
  --rhat-y-lim 0.999,1.02 \
  --variance-y-lim 0.0,0.2 \
  --run-greedy-bfs \
  --run-greedy-nwss
```

Regenerate saved KL plots:

```bash
python om/experiments/cli_runner.py \
  --family kl \
  --plot-results-dir results/kl/Ising15_2026-05-01_12-34-56 \
  --run-mcmc-opad \
  --run-dvpi \
  --run-greedy-nwss \
  --kl-y-lim 0.001,10
```

Regenerate saved iter-based bias/variance plots:

```bash
python om/experiments/cli_runner.py \
  --family bias_iter \
  --plot-results-dir om/experiments/results/mse_vs_itr/Ising15_2026-05-01_02-06-40 \
  --rhat-y-lim 0.999,1.02
```

Use a different observable in a bias experiment:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --model ising15 \
  --observable first_spin
```

Override the BVS dataset location:

```bash
python om/experiments/cli_runner.py \
  --family bias_time \
  --model bvs \
  --dataset-path om/models/lifespan-merged.csv
```

## Output Layout

Each run creates a timestamped result folder under the selected output directory, for example:

```text
results/mse_vs_time/Ising15_2026-05-01_12-34-56
results/kl/BVS_2026-05-01_12-40-10
```

These folders contain:

- `metadata.csv`
- saved CSV result tables
- saved plots

## Recommended Workflow

Common usage is:

1. Inspect presets with `--list-presets`.
2. Run one preset directly.
3. Override only the few parameters you want to change.
4. Keep paper runs and exploratory runs in different `--output-dir` locations.
