import csv
import datetime
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def _write_dataframe(path, data):
    pd.DataFrame(data).to_csv(path, index=True)


def run_multi_chain_kl_experiment(
        model_sampler_init_generator,
        generator_args,
        num_mcmc_samples_per_chain,
        num_records_per_chain,
        num_independent_chains,
        description,
        kl_compute,
        results_dir_name='om_results',
):
    print("===========\n", description, "\n===========\n")
    chains = [f'C_{chain_index}' for chain_index in range(num_independent_chains)]

    experiments_dir = Path(__file__).resolve().parent
    main_results_path = experiments_dir / results_dir_name
    os.makedirs(main_results_path, exist_ok=True)

    result_path = main_results_path / datetime.datetime.now().strftime(description + '_%Y-%m-%d_%H-%M-%S')
    os.makedirs(result_path, exist_ok=True)

    metadata_dict = {
        "Num_experiments": num_independent_chains,
        "Description": description,
        "num_records_per_chain": num_records_per_chain,
        "num_mcmc_samples_per_chain": num_mcmc_samples_per_chain,
    }
    for key, value in generator_args.items():
        metadata_dict[f'gen.arg.{key}'] = value

    with open(result_path / 'metadata.csv', 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, metadata_dict.keys())
        writer.writeheader()
        writer.writerow(metadata_dict)

    recorded_iters_for_all_chains = None
    chain_to_kls_mcmc = {}
    chain_to_kls_opad = {}
    chain_to_kls_opad_plus = {}

    for chain_count, chain_name in enumerate(chains):
        print("{i}: Chain name: {c}".format(i=chain_count, c=chain_name))

        generator_args['seed'] = chain_count
        model_sampler_init = model_sampler_init_generator(**generator_args)

        results = kl_compute(
            num_mcmc_samples=num_mcmc_samples_per_chain,
            num_records=num_records_per_chain,
            sampler=model_sampler_init.sampler,
            init_state=model_sampler_init.init_state,
            model=model_sampler_init.model,
        )

        if recorded_iters_for_all_chains is None:
            recorded_iters_for_all_chains = {'iters': results['Iters']}

        chain_to_kls_mcmc[f"KL_MCMC_{chain_count}"] = results['MCMC']
        chain_to_kls_opad[f"KL_OPAD_{chain_count}"] = results['OPAD']
        chain_to_kls_opad_plus[f"KL_OPAD_plus_{chain_count}"] = results['OPAD+']

        _write_dataframe(result_path / 'Recorded_Iters_for_all_algs.csv', recorded_iters_for_all_chains)
        _write_dataframe(result_path / 'MCMC_KL_results.csv', chain_to_kls_mcmc)
        _write_dataframe(result_path / 'OPAD_KL_results.csv', chain_to_kls_opad)
        _write_dataframe(result_path / 'OPAD_plus_KL_results.csv', chain_to_kls_opad_plus)

    load_and_show_multichain_kl_plots(RESPATH=str(result_path))
    return str(result_path)


def load_and_show_multichain_kl_plots(RESPATH):
    df_iters = pd.read_csv(os.path.join(RESPATH, 'Recorded_Iters_for_all_algs.csv'), index_col=0)
    iters = df_iters['iters']
    df_mcmc = pd.read_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index_col=0)
    df_opad = pd.read_csv(os.path.join(RESPATH, 'OPAD_KL_results.csv'), index_col=0)
    df_opad_plus = pd.read_csv(os.path.join(RESPATH, 'OPAD_plus_KL_results.csv'), index_col=0)

    mcmc_color = 'blue'
    opad_color = 'red'
    opad_plus_color = 'black'
    quantiles = [0.05, 0.95]

    plt.figure()

    mcmc_mean = df_mcmc.mean(axis=1)
    opad_mean = df_opad.mean(axis=1)
    opad_plus_mean = df_opad_plus.mean(axis=1)
    mcmc_quantiles = df_mcmc.apply(lambda row: row.quantile(quantiles), axis=1)
    opad_quantiles = df_opad.apply(lambda row: row.quantile(quantiles), axis=1)
    opad_plus_quantiles = df_opad_plus.apply(lambda row: row.quantile(quantiles), axis=1)

    plt.plot(iters, mcmc_mean, color=mcmc_color, linestyle='--', label='MCMC')
    plt.plot(iters, opad_mean, color=opad_color, linestyle='--', label='OPAD')
    plt.plot(iters, opad_plus_mean, color=opad_plus_color, linestyle='--', label='OPAD+')

    plt.yscale('log')
    y_limits = plt.ylim()

    plt.fill_between(x=iters, y1=mcmc_quantiles[quantiles[0]], y2=mcmc_quantiles[quantiles[1]], color=mcmc_color, alpha=0.2)
    plt.fill_between(x=iters, y1=opad_quantiles[quantiles[0]], y2=opad_quantiles[quantiles[1]], color=opad_color, alpha=0.2)
    plt.fill_between(x=iters, y1=opad_plus_quantiles[quantiles[0]], y2=opad_plus_quantiles[quantiles[1]], color=opad_plus_color, alpha=0.2)

    plt.ylim(y_limits)
    plt.xlabel('Iterations')
    plt.ylabel('KL')
    plt.legend()
    plt.savefig(os.path.join(RESPATH, 'Confidence.png'))
    plt.show(block=True)

    plt.figure()
    for column in df_mcmc.columns:
        plt.plot(iters, df_mcmc[column], color=mcmc_color, alpha=0.4)
    for column in df_opad.columns:
        plt.plot(iters, df_opad[column], color=opad_color, alpha=0.4)
    for column in df_opad_plus.columns:
        plt.plot(iters, df_opad_plus[column], color=opad_plus_color, alpha=0.4)

    plt.yscale('log')
    plt.ylim(y_limits)
    legend_elements = [
        Line2D([0], [0], color=mcmc_color, lw=2, label='MCMC'),
        Line2D([0], [0], color=opad_color, lw=2, label='OPAD'),
        Line2D([0], [0], color=opad_plus_color, lw=2, label='OPAD+'),
    ]
    plt.legend(handles=legend_elements, loc='best')
    plt.xlabel('Iterations')
    plt.ylabel('KL')
    plt.savefig(os.path.join(RESPATH, 'Multi_chain.png'))
    plt.show(block=True)
