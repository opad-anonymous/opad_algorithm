import os
import datetime
import numpy as np
# import pickle

import pandas as pd
import matplotlib.pyplot as plt

# import time
import csv
from om.experiments.experiment_utils import experiment1_ising1d, \
    experiment2_ising1d__high_dim, \
    experiment3_varselect_mice_data
from om.samplers.greedy import GreedyExplore
from matplotlib.lines import Line2D

from om.tools.histogram import ArrayHistogram
from om.tools.opad import OPADDistribution
from om.models.model import DiscreteDistributionModel
from tqdm import tqdm

import time
from om.experiments.utils import TimeKeeper
from om.experiments.bias_variance_shared import (
    calc_r_hat as shared_calc_r_hat,
    compute_greedy_expected_error,
    compute_mcmc_expected_error,
    load_and_show_bias_plots,
    plot_confidence_interval,
    run_multi_chain_bias_experiment,
)

def greedy_expected_func_error_per_itr_compute(
        equivalent_of_num_mcmc_samples_per_chain: int,
        num_records: int,
        greedy: GreedyExplore,
        # num_evolution_steps: int, num_records,
        # variational,
        model: DiscreteDistributionModel,
        func_state,  # a mapping from states to scalars
        ground_truth_expected_func_value,
        do_compute_rHat=True
):
    return compute_greedy_expected_error(
        axis_mode='iters',
        num_records=num_records,
        greedy=greedy,
        model=model,
        func_state=func_state,
        ground_truth_expected_func_value=ground_truth_expected_func_value,
        equivalent_num_mcmc_samples_per_chain=equivalent_of_num_mcmc_samples_per_chain,
        do_compute_rhat=do_compute_rHat,
    )


def var_expected_func_error_per_itr_compute(
        num_evolution_steps: int, num_records,
        variational,
        model: DiscreteDistributionModel,
        func_state,  # a mapping from states to scalars
        ground_truth_expected_func_value
):
    raise Exception("not sure if it is uptodate")
    recorded_every = int(num_evolution_steps / num_records) + 1
    # var_hist = ArrayHistogram()
    # opad_accepted_only = OPADDistribution(target_model=model)
    opad_all_validated_states = OPADDistribution(target_model=model)

    recorded_iters = []
    recorded_times = []
    recorded_exp_func_err_var = []  # variational
    recorded_exp_func_err_om2 = []

    # num_posterior_validations = 0

    start_time = time.perf_counter()
    for evolution_step in tqdm(range(1, num_evolution_steps + 1)):
        newly_validated_states_scores = variational.evolve_all_states_in_one_dim()
        newly_validated_states = newly_validated_states_scores.generate_all_states()
        # num_posterior_validations += variational.num_particles

        for state in newly_validated_states:
            opad_all_validated_states.add_array(state)  # todo this part can be optimized

        # print(mcmc_hist)
        if evolution_step % recorded_every == 0:
            # kl_mcmc = kl_divergence_discrete(p_empirical=var_hist, q_target=model)
            exp_func_var = variational.particles_scores.calc_expected_value(func=func_state)
            exp_func_opad = opad_all_validated_states.calc_expected_value(func=func_state)

            # in each iteration, one dimension of each particle is updated:
            recorded_iters.append(evolution_step * variational.num_particles)

            recorded_exp_func_err_var.append(exp_func_var - ground_truth_expected_func_value)
            recorded_exp_func_err_om2.append(exp_func_opad - ground_truth_expected_func_value)

    # DPVI: Discrete particle variational inference
    # Evals: no. evaluations of the posterior = #steps * #particles
    # NOTE: Evals are comparable with MCMC iterations
    return {'Evals': recorded_iters,
            'DPVI': recorded_exp_func_err_var,
            'OPAD-DPVI': recorded_exp_func_err_om2}


def mcmc_expected_func_error_per_itr_compute(num_mcmc_samples,
                                             num_records, sampler, init_state,
                                             model: DiscreteDistributionModel,
                                             func_state,  # a mapping from states to scalars
                                             ground_truth_expected_func_value,
                                             do_compute_rHat=True
                                             ):
    return compute_mcmc_expected_error(
        axis_mode='iters',
        num_records=num_records,
        sampler=sampler,
        init_state=init_state,
        model=model,
        func_state=func_state,
        ground_truth_expected_func_value=ground_truth_expected_func_value,
        num_mcmc_samples=num_mcmc_samples,
        do_compute_rhat=do_compute_rHat,
    )


def multi_chain_expected_func_compute_and_plot_itr(
        path,
        model_sampler_init_generator,
        generator_args,
        num_mcmc_samples_per_chain,
        num_records_per_chain,
        num_independent_chains,
        description,
        func_state=lambda x: sum(x),  # a function that takes a state and return a scalar -- default:sum
        ground_truth_expected_func_value_already_known=None,
        do_run_mcmc_opad=True,
        do_run_greedy_bfs=False,
        do_run_greedy_nwss=False,
        # if None, it will be computed. But for Large Symmetric Models we can set it this way
        #func_rhat=lambda x: sum(x),  # a function that takes a state and return a scalar -- default:sum todo separate func_state from func_rhat
):
    return run_multi_chain_bias_experiment(
        path=path,
        axis_mode='iters',
        model_sampler_init_generator=model_sampler_init_generator,
        generator_args=generator_args,
        num_records_per_chain=num_records_per_chain,
        num_independent_chains=num_independent_chains,
        description=description,
        func_state=func_state,
        ground_truth_expected_func_value_already_known=ground_truth_expected_func_value_already_known,
        num_mcmc_samples_per_chain=num_mcmc_samples_per_chain,
        do_run_mcmc_opad=do_run_mcmc_opad,
        do_run_greedy_bfs=do_run_greedy_bfs,
        do_run_greedy_nwss=do_run_greedy_nwss,
    )


def plot_confidece_interval(
        df_MCMC,
        df_OM1,
        df_OM2,
        var_df_DPVI,
        var_df_OPAD_DPVI,
        back_df,
        # back_df2,
        back_df3,
        iters,
        var_iters,
        back_iters,
        back_iters2,
        back_iters3,
        y_label,
        x_label,
        save_fig_address,
        mcmc_color='blue',
        om1_color='yellow',
        om2_color='black',
        DPVI_color='green',
        OPAD_DPVI_color='red',
        back_color='magenta',
        # back_color2="green",
        back_color3="orange",
        use_log_scale_y_axis=True,
        plotting_y_limits=None,
        quantiles=(0.05, 0.95)
):
    return plot_confidence_interval(
        df_MCMC=df_MCMC,
        df_OM1=df_OM1,
        df_OM2=df_OM2,
        var_df_DPVI=var_df_DPVI,
        var_df_OPAD_DPVI=var_df_OPAD_DPVI,
        back_df=back_df,
        # back_df2=back_df2,
        back_df3=back_df3,
        x_values=iters,
        var_x_values=var_iters,
        back_x_values=back_iters,
        # back_x_values2=back_iters2,
        back_x_values3=back_iters3,
        y_label=y_label,
        x_label=x_label,
        save_fig_address=save_fig_address,
        mcmc_color=mcmc_color,
        om1_color=om1_color,
        om2_color=om2_color,
        DPVI_color=DPVI_color,
        OPAD_DPVI_color=OPAD_DPVI_color,
        back_color=back_color,
        # back_color2=back_color2,
        back_color3=back_color3,
        use_log_scale_y_axis=use_log_scale_y_axis,
        plotting_y_limits=plotting_y_limits,
        quantiles=quantiles,
    )


def plot_trace_interval(
        # processed:
        df_MCMC,
        df_OM2,
        var_df_DPVI,
        var_df_OPAD_DPVI,
        back_df,
        iters,
        var_iters,
        back_iters,
        y_label,
        x_label,
        mcmc_color,
        om1_color,
        om2_color,
        DPVI_color,
        OPAD_DPVI_color,
        back_color,
        save_fig_address,
):
    plt.figure()
    # plt.title('Square Error')

    for i, column in enumerate(df_MCMC.columns):
        plt.plot(iters, df_MCMC[column], color=mcmc_color, alpha=0.4)

    # for i, column in enumerate(df_OM1.columns):
    #     plt.plot(iters, df_OM1[column], color=om1_color, alpha=0.4)

    for i, column in enumerate(df_OM2.columns):
        plt.plot(iters, df_OM2[column], color=om2_color, alpha=0.4)

    for i, column in enumerate(var_df_DPVI.columns):
        plt.plot(var_iters, var_df_DPVI[column], color=DPVI_color, alpha=0.4)

    for i, column in enumerate(var_df_OPAD_DPVI.columns):
        plt.plot(var_iters, var_df_OPAD_DPVI[column], color=OPAD_DPVI_color, alpha=0.4)

    if back_iters is not None:
        for i, column in enumerate(back_df.columns):
            plt.plot(back_iters, back_df[column], color=back_color, alpha=0.4)

    # Creating custom legend entries
    legend_elements = [
        Line2D([0], [0], color=mcmc_color, lw=2, label='MCMC'),
        # Line2D([0], [0], color=om1_color, lw=2, label='OPAD'),
        Line2D([0], [0], color=om2_color, lw=2, label='MCMC OPAD+'),
        Line2D([0], [0], color=DPVI_color, lw=2, label='DPVI'),
        Line2D([0], [0], color=OPAD_DPVI_color, lw=2, label='DPVI OPAD+'),
        # Line2D([0], [0], color=back_color, lw=2, label='Greedy'),
    ]

    # Adding the legend to the plot
    plt.legend(handles=legend_elements, loc='best')

    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)

    plt.savefig(save_fig_address)
    plt.show(block=True)
    ##############################################


def calcR_hat(df_means, df_variances):
    return shared_calc_r_hat(df_means=df_means, df_variances=df_variances)


def load_and_show_plots(RESPATH,
                        do_plot_greedy_bfs=False,
                        do_plot_greedy_nwss=True,
                        rhat_plotting_y_lim=None,
                        variance_plotting_y_lim=None,
                        confidence_quantiles=(0.05, 0.95)):
    return load_and_show_bias_plots(
        RESPATH=RESPATH,
        axis_mode='iters',
        do_plot_greedy_bfs=do_plot_greedy_bfs,
        do_plot_greedy_nwss=do_plot_greedy_nwss,
        rhat_plotting_y_lim=rhat_plotting_y_lim,
        variance_plotting_y_lim=variance_plotting_y_lim,
        confidence_quantiles=confidence_quantiles,
    )

