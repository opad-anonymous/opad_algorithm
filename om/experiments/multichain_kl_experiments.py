import os
import datetime
import numpy as np
# import pickle

import pandas as pd
import matplotlib.pyplot as plt

# import time
import csv
from om.experiments.experiment_utils import *
from om.experiments.experiment_utils import kl_compute
from matplotlib.lines import Line2D


def multi_chain_kl_compute_and_plot(
        path,
        model_sampler_init_generator,
        generator_args,
        num_mcmc_samples_per_chain,
        num_records_per_chain,
        num_independent_chains,
        description,
        do_run_mcmc_opad=True,
        do_run_dvpi=False,
        do_run_greedy_bfs=False,
        do_run_greedy_nwss=False):
    print("===========\n", description, "\n===========\n")
    chains = ['C_' + str(c) for c in range(num_independent_chains)]
    # num_mcmc_samples_per_chain = 1000

    DIRPATH = os.path.abspath(path)
    os.makedirs(DIRPATH, exist_ok=True)

    RESPATH = os.path.join(DIRPATH, datetime.datetime.now().strftime(description + '_%Y-%m-%d_%H-%M-%S'))
    os.makedirs(RESPATH, exist_ok=True)

    # Save metadata

    metadata_dict = {"Num_experiments": num_independent_chains,
                     "Description": description,
                     "num_records_per_chain": num_records_per_chain,
                     "num_mcmc_samples_per_chain": num_mcmc_samples_per_chain,
                     }
    for k, v in generator_args.items():
        metadata_dict['gen.arg.' + k] = v

    recorded_iters_for_all_chains = None
    chain_to_kls_mcmc = {}
    chain_to_kls_opad = {}
    chain_to_kls_opad_plus = {}
    chain_to_kls_rb_wr = {}

    var_recorded_iters_for_all_chains = None
    var_chain_to_kls_DPVI = {}
    var_chain_to_kls_OPAD_DPVI = {}

    back_recorded_iters_for_all_chains = None
    back_chain_to_kls = {}

    back_recorded_iters_for_all_chains3 = None
    back_chain_to_kls3 = {}

    with open(os.path.join(RESPATH, 'metadata.csv'), 'w', newline="") as csvfile:
        w = csv.DictWriter(csvfile, metadata_dict.keys())
        w.writeheader()
        w.writerow(metadata_dict)

    # t = trange(num_independent_chains, desc='Bar desc', leave=True)
    for chain_count, chain_name in enumerate(chains):
        print("{i}: Chain name: {c}".format(i=chain_count, c=chain_name))

        generator_args['seed'] = chain_count  # to ensure that the initial time is different each time...
        model_sampler_init = model_sampler_init_generator(**generator_args)

        model = model_sampler_init.model
        sampler = model_sampler_init.sampler

        init_state = model.generate_init_state()

        variational = model_sampler_init.variational
        greedy = model_sampler_init.greedy

        results_mcmc = None
        if do_run_mcmc_opad:
            results_mcmc = kl_compute(
                num_mcmc_samples=num_mcmc_samples_per_chain,
                num_records=num_records_per_chain,
                sampler=sampler,
                init_state=init_state,
                model=model)

        var_results = None
        if do_run_dvpi:
            var_results = var_kl_compute(
                num_evolution_steps=int(num_mcmc_samples_per_chain / variational.num_particles),
                num_records=num_records_per_chain,
                variational=variational,
                model=model
            )

        back_results = None
        if do_run_greedy_bfs:
            greedy.initialize('BFS', init_state=init_state)
            back_results = greedy_kl_compute(
                equivalent_of_num_mcmc_samples_per_chain=num_mcmc_samples_per_chain,
                num_records=num_records_per_chain,
                greedy=greedy,
                model=model
            )

        back_results3 = None
        if do_run_greedy_nwss:
            greedy.initialize('NWSS', init_state=init_state)
            back_results3 = greedy_kl_compute(
                equivalent_of_num_mcmc_samples_per_chain=num_mcmc_samples_per_chain,
                num_records=num_records_per_chain,
                greedy=greedy,
                model=model
            )

        if results_mcmc is not None:
            recorded_iters = results_mcmc['Iters']
            recorded_kls_mcmc = results_mcmc['MCMC']
            recorded_kls_opad = results_mcmc['OPAD']
            recorded_kls_opad_plus = results_mcmc['OPAD+']
            recorded_kls_rb_wr = results_mcmc['RB/WR']

            if recorded_iters_for_all_chains is None:
                recorded_iters_for_all_chains = {'iters': recorded_iters}

            chain_to_kls_mcmc.update({f"KL_MCMC_{chain_count}": recorded_kls_mcmc})
            chain_to_kls_opad.update({f"KL_OM1_{chain_count}": recorded_kls_opad})
            chain_to_kls_opad_plus.update({f"KL_OM2_{chain_count}": recorded_kls_opad_plus})
            chain_to_kls_rb_wr.update({f"KL_RB/WR_{chain_count}": recorded_kls_rb_wr})

            df = pd.DataFrame(recorded_iters_for_all_chains)
            df.to_csv(os.path.join(RESPATH, 'Recorded_Iters_for_all_algs.csv'), index=True)
            df = pd.DataFrame(chain_to_kls_mcmc)
            df.to_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index=True)
            df = pd.DataFrame(chain_to_kls_opad)
            df.to_csv(os.path.join(RESPATH, 'OM1_KL_results.csv'), index=True)
            df = pd.DataFrame(chain_to_kls_opad_plus)
            df.to_csv(os.path.join(RESPATH, 'OM2_KL_results.csv'), index=True)
            df = pd.DataFrame(chain_to_kls_rb_wr)
            df.to_csv(os.path.join(RESPATH, 'RB_WR_KL_results.csv'), index=True)

        if var_results is not None:
            var_recorded_iters = var_results['Evals']
            var_recorded_kls_DPVI = var_results['DPVI']
            var_recorded_kls_OPAD_DPVI = var_results['OPAD-DPVI']

            if var_recorded_iters_for_all_chains is None:
                var_recorded_iters_for_all_chains = {'iters': var_recorded_iters}

            var_chain_to_kls_DPVI.update({f"KL_DPVI_{chain_count}": var_recorded_kls_DPVI})
            var_chain_to_kls_OPAD_DPVI.update({f"KL_DPVI_{chain_count}": var_recorded_kls_OPAD_DPVI})

            df = pd.DataFrame(var_recorded_iters_for_all_chains)
            df.to_csv(os.path.join(RESPATH, 'var_Recorded_Iters_for_all_algs.csv'), index=True)
            df = pd.DataFrame(var_chain_to_kls_DPVI)
            df.to_csv(os.path.join(RESPATH, 'DPVI_KL_results.csv'), index=True)
            df = pd.DataFrame(var_chain_to_kls_OPAD_DPVI)
            df.to_csv(os.path.join(RESPATH, 'OPAD_DPVI_KL_results.csv'), index=True)

        if back_results is not None:
            back_recorded_iters = back_results['Evals']
            back_recorded_kls = back_results['Backtrack']

            if back_recorded_iters_for_all_chains is None:
                back_recorded_iters_for_all_chains = {'iters': back_recorded_iters}

            back_chain_to_kls.update({f"KL_DPVI_{chain_count}": back_recorded_kls})

            df = pd.DataFrame(back_recorded_iters_for_all_chains)
            df.to_csv(os.path.join(RESPATH, 'back_Recorded_Iters_for_all_algs.csv'), index=True)
            df = pd.DataFrame(back_chain_to_kls)
            df.to_csv(os.path.join(RESPATH, 'back_KL_results.csv'), index=True)

        if back_results3 is not None:
            back_recorded_iters3 = back_results3['Evals']
            back_recorded_kls3 = back_results3['Backtrack']

            if back_recorded_iters_for_all_chains3 is None:
                back_recorded_iters_for_all_chains3 = {'iters': back_recorded_iters3}

            back_chain_to_kls3.update({f"KL_DPVI_{chain_count}": back_recorded_kls3})

            df = pd.DataFrame(back_recorded_iters_for_all_chains3)
            df.to_csv(os.path.join(RESPATH, 'back_Recorded_Iters_for_all_algs3.csv'), index=True)
            df = pd.DataFrame(back_chain_to_kls3)
            df.to_csv(os.path.join(RESPATH, 'back_KL_results3.csv'), index=True)

    load_and_show_plots(
        RESPATH=RESPATH,
        do_plot_mcmc=do_run_mcmc_opad,
        do_plot_mcmc_opad=do_run_mcmc_opad,
        do_plot_mcmc_opad_plus=do_run_mcmc_opad,
        do_plot_mcmc_rb_wr=do_run_mcmc_opad,
        do_plot_dpvi=do_run_dvpi,
        do_plot_dvpi_plus=do_run_dvpi,
        do_plot_greedy_bfs=do_run_greedy_bfs,
        do_plot_greedy_nwss=do_run_greedy_nwss,
    )


def load_and_show_plots(RESPATH, do_plot_confidence_intervals=True,
                        do_plot_mcmc=True,
                        do_plot_mcmc_opad=True, do_plot_mcmc_opad_plus=True,
                        do_plot_mcmc_rb_wr=True,
                        do_plot_dpvi=True, do_plot_dvpi_plus=True,
                        do_plot_greedy_bfs=True,
                        do_plot_greedy_nwss=True,
                        y_limits=None
                        ):
    print("plotting '{s}'".format(s=RESPATH))
    # Quantile Plots :
    quantiles = [0.05, 0.95]

    if do_plot_mcmc or do_plot_mcmc_opad or do_plot_mcmc_opad_plus or do_plot_mcmc_rb_wr:
        df_ITERS = pd.read_csv(os.path.join(RESPATH, 'Recorded_Iters_for_all_algs.csv'), index_col=0)
        iters = df_ITERS['iters']

    if do_plot_mcmc:
        df_MCMC = pd.read_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index_col=0)
        MCMC_mean = df_MCMC.mean(axis=1)
        MCMC_quantiles = df_MCMC.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_mcmc_opad:
        df_OM1 = pd.read_csv(os.path.join(RESPATH, 'OM1_KL_results.csv'), index_col=0)
        OM1_mean = df_OM1.mean(axis=1)
        OM1_quantiles = df_OM1.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_mcmc_opad_plus:
        df_OM2 = pd.read_csv(os.path.join(RESPATH, 'OM2_KL_results.csv'), index_col=0)
        OM2_mean = df_OM2.mean(axis=1)
        OM2_quantiles = df_OM2.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_mcmc_rb_wr:
        df_RB = pd.read_csv(os.path.join(RESPATH, 'RB_WR_KL_results.csv'), index_col=0)
        RB_mean = df_RB.mean(axis=1)
        RB_quantiles = df_RB.apply(lambda row: row.quantile(quantiles), axis=1)


    if do_plot_dpvi or do_plot_dvpi_plus:
        var_df_ITERS = pd.read_csv(os.path.join(RESPATH, 'var_Recorded_Iters_for_all_algs.csv'), index_col=0)
        var_iters = var_df_ITERS['iters']

    if do_plot_dpvi:
        var_df_DPVI = pd.read_csv(os.path.join(RESPATH, 'DPVI_KL_results.csv'), index_col=0)
        var_DPVI_mean = var_df_DPVI.mean(axis=1)
        var_DPVI_quantiles = var_df_DPVI.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_dvpi_plus:
        var_df_OPAD_DPVI = pd.read_csv(os.path.join(RESPATH, 'OPAD_DPVI_KL_results.csv'), index_col=0)
        var_OPAD_DPVI_mean = var_df_OPAD_DPVI.mean(axis=1)
        var_OPAD_DPVI_quantiles = var_df_OPAD_DPVI.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_greedy_bfs:
        back_df_ITERS = pd.read_csv(os.path.join(RESPATH, 'back_Recorded_Iters_for_all_algs.csv'), index_col=0)
        back_iters = back_df_ITERS['iters']
        back_df_KL = pd.read_csv(os.path.join(RESPATH, 'back_KL_results.csv'), index_col=0)
        back_mean = back_df_KL.mean(axis=1)
        back_quantiles = back_df_KL.apply(lambda row: row.quantile(quantiles), axis=1)

    if do_plot_greedy_nwss:
        back_df_ITERS3 = pd.read_csv(os.path.join(RESPATH, 'back_Recorded_Iters_for_all_algs3.csv'), index_col=0)
        back_iters3 = back_df_ITERS3['iters']
        back_df_KL3 = pd.read_csv(os.path.join(RESPATH, 'back_KL_results3.csv'), index_col=0)
        back_mean3 = back_df_KL3.mean(axis=1)
        back_quantiles3 = back_df_KL3.apply(lambda row: row.quantile(quantiles), axis=1)

    mcmc_color = 'blue'
    om1_color = 'red'
    om2_color = 'black'
    rb_color = 'orange'

    DPVI_color = 'green'
    OPAD_DPVI_color = 'olive'

    back_color = 'cyan'

    back_color3 = 'yellow'

    if do_plot_confidence_intervals:
        plt.figure()
        plt.yscale('log')

        if do_plot_mcmc:
            plt.plot(iters, MCMC_mean, color=mcmc_color, linestyle='--', label='MCMC')

        if do_plot_mcmc_rb_wr:
            plt.plot(iters, RB_mean, color=rb_color, linestyle='-.', label='MCMC RB/WR')

        if do_plot_mcmc_opad:
            plt.plot(iters, OM1_mean, color=om1_color, linestyle='--', label='MCMC OPAD')

        if do_plot_mcmc_opad_plus:
            plt.plot(iters, OM2_mean, color=om2_color, linestyle='--', label='MCMC OPAD+')

        if do_plot_dpvi:
            plt.plot(var_iters, var_DPVI_mean, color=DPVI_color, linestyle='--', label='DPVI')

        if do_plot_dvpi_plus:
            plt.plot(var_iters, var_OPAD_DPVI_mean, color=OPAD_DPVI_color, linestyle='--', label='DPVI OPAD+')

        if do_plot_greedy_bfs:
            plt.plot(back_iters, back_mean, color=back_color, linestyle='--', label='BFS')

        # if do_plot_backtrack2:
        #     plt.plot(back_iters2, back_mean2, color=back_color2, linestyle='-', label='PFS')

        if do_plot_greedy_nwss:
            plt.plot(back_iters3, back_mean3, color=back_color3, linestyle='-', label='NWSS')  # neighbour-weighted sampling


        if y_limits is None:
            y_limits = plt.ylim()  # based on the Expectation curves only

        if do_plot_mcmc:
            plt.fill_between(x=iters, y1=MCMC_quantiles[quantiles[0]], y2=MCMC_quantiles[quantiles[1]],
                             color=mcmc_color,
                             alpha=0.2)

        if do_plot_mcmc_opad:
            plt.fill_between(x=iters, y1=OM1_quantiles[quantiles[0]], y2=OM1_quantiles[quantiles[1]], color=om1_color,
                             alpha=0.2)

        if do_plot_mcmc_opad_plus:
            plt.fill_between(x=iters, y1=OM2_quantiles[quantiles[0]], y2=OM2_quantiles[quantiles[1]], color=om2_color,
                             alpha=0.2)

        if do_plot_mcmc_rb_wr:
            plt.fill_between(x=iters, y1=RB_quantiles[quantiles[0]], y2=RB_quantiles[quantiles[1]], color=rb_color,
                             alpha=0.2)

        if do_plot_dpvi:
            plt.fill_between(x=var_iters, y1=var_DPVI_quantiles[quantiles[0]], y2=var_DPVI_quantiles[quantiles[1]],
                             color=DPVI_color, alpha=0.2)

        if do_plot_dvpi_plus:
            plt.fill_between(x=var_iters, y1=var_OPAD_DPVI_quantiles[quantiles[0]],
                             y2=var_OPAD_DPVI_quantiles[quantiles[1]],
                             color=OPAD_DPVI_color, alpha=0.2)

        if do_plot_greedy_bfs:
            plt.fill_between(x=back_iters, y1=back_quantiles[quantiles[0]], y2=back_quantiles[quantiles[1]],
                             color=back_color, alpha=0.2)

        # if do_plot_backtrack2:
        #     plt.fill_between(x=back_iters2, y1=back_quantiles2[quantiles[0]], y2=back_quantiles2[quantiles[1]],
        #                      color=back_color2, alpha=0.2)

        if do_plot_greedy_nwss:
            plt.fill_between(x=back_iters3, y1=back_quantiles3[quantiles[0]], y2=back_quantiles3[quantiles[1]],
                             color=back_color3, alpha=0.2)

        print('y_limits', y_limits)
        plt.ylim(y_limits)  # So confidence can be outside the limit

        plt.xlabel('Target evaluations')
        plt.ylabel('KL')

        # plt.title(f'KL divergence with respect to ground truth distribution: #Exp: {num_independent_chains}')
        plt.legend()
        plt.savefig(os.path.join(RESPATH, 'KL-Confidence--' + RESPATH.split("/")[-1] + '.png'))
        plt.show(block=True)
    ################################################################
    ####### PLOTTING TRACES ######
    plt.figure()

    if do_plot_mcmc:
        for i, column in enumerate(df_MCMC.columns):
            plt.plot(iters, df_MCMC[column], color=mcmc_color, alpha=0.4)

    if do_plot_mcmc_opad:
        for i, column in enumerate(df_OM1.columns):
            plt.plot(iters, df_OM1[column], color=om1_color, alpha=0.4)

    if do_plot_mcmc_opad_plus:
        for i, column in enumerate(df_OM2.columns):
            plt.plot(iters, df_OM2[column], color=om2_color, alpha=0.4)

    if do_plot_mcmc_rb_wr:
        for i, column in enumerate(df_RB.columns):
            plt.plot(iters, df_RB[column], color=rb_color, alpha=0.4)

    if do_plot_dpvi:
        for i, column in enumerate(var_df_DPVI.columns):
            plt.plot(var_iters, var_df_DPVI[column], color=DPVI_color, alpha=0.4)

    if do_plot_dvpi_plus:
        for i, column in enumerate(var_df_OPAD_DPVI.columns):
            plt.plot(var_iters, var_df_OPAD_DPVI[column], color=OPAD_DPVI_color, alpha=0.4)

    if do_plot_greedy_bfs:
        for i, column in enumerate(back_df_KL.columns):
            plt.plot(back_iters, back_df_KL[column], color=back_color, alpha=0.4)

    # if do_plot_backtrack2:
    #     for i, column in enumerate(back_df_KL2.columns):
    #         plt.plot(back_iters2, back_df_KL2[column], color=back_color2, alpha=0.4)

    if do_plot_greedy_nwss:
        for i, column in enumerate(back_df_KL3.columns):
            plt.plot(back_iters3, back_df_KL3[column], color=back_color3, alpha=0.4)

    plt.yscale('log')
    # y_limits = plt.ylim()
    # reasonable_lower_limit = max(np.median(df_OM2.values.flatten()) / 5, np.min(df_OM2.values.flatten()))
    # plt.ylim((reasonable_lower_limit, y_limits[1]))

    if y_limits is not None:
        plt.ylim(y_limits)

    # Creating custom legend entries
    legend_elements = []
    if do_plot_mcmc:
        legend_elements.append(Line2D([0], [0], color=mcmc_color, lw=2, label='MCMC'))
    if do_plot_mcmc_rb_wr:
        legend_elements.append(Line2D([0], [0], color=rb_color, lw=2, label='MCMC RB/WR'))
    if do_plot_mcmc_opad:
        legend_elements.append(Line2D([0], [0], color=om1_color, lw=2, label='MCMC OPAD'))
    if do_plot_mcmc_opad_plus:
        legend_elements.append(Line2D([0], [0], color=om2_color, lw=2, label='MCMC OPAD+'))
    if do_plot_dpvi:
        legend_elements.append(Line2D([0], [0], color=DPVI_color, lw=2, label='DPVI'))
    if do_plot_dvpi_plus:
        legend_elements.append(Line2D([0], [0], color=OPAD_DPVI_color, lw=2, label='DPVI OPAD+'))
    if do_plot_greedy_bfs:
        legend_elements.append(Line2D([0], [0], color=back_color, lw=2, label='BFS'))
    # if do_plot_backtrack2:
    #     legend_elements.append(Line2D([0], [0], color=back_color2, lw=2, label='Greedy2'))
    if do_plot_greedy_nwss:
        legend_elements.append(Line2D([0], [0], color=back_color3, lw=2, label='NWSS'))  # Neighbour-weighted stochastic search


    # Adding the legend to the plot
    plt.legend(handles=legend_elements, loc='best')

    plt.xlabel('Target evaluations')
    plt.ylabel('KL')

    plt.savefig(os.path.join(RESPATH, 'KL-MutliChain--' + RESPATH.split("/")[-1] + '.png'))
    plt.show(block=True)
    ##############################################
