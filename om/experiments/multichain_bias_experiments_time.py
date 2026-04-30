import matplotlib.pyplot as plt

from om.samplers.greedy import GreedyExplore
from matplotlib.lines import Line2D

from om.models.model import DiscreteDistributionModel

from om.experiments.bias_variance_shared import (
    calc_r_hat as shared_calc_r_hat,
    compute_greedy_expected_error,
    compute_mcmc_expected_error,
    load_and_show_bias_plots,
    plot_confidence_interval,
    run_multi_chain_bias_experiment,
)


def greedy_expected_func_error_per_time_compute(
        total_sampling_time_per_chain_seconds,
        # equivalent_of_num_mcmc_samples_per_chain: int,
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
        axis_mode='time',
        num_records=num_records,
        greedy=greedy,
        model=model,
        func_state=func_state,
        ground_truth_expected_func_value=ground_truth_expected_func_value,
        total_sampling_time_per_chain_seconds=total_sampling_time_per_chain_seconds,
        do_compute_rhat=do_compute_rHat,
    )



def mcmc_expected_func_error_per_TIME_compute(total_sampling_time_per_chain_seconds,
                                              num_records, sampler, init_state,
                                              model: DiscreteDistributionModel,
                                              func_state,  # a mapping from states to scalars
                                              ground_truth_expected_func_value,
                                              do_compute_rHat=True
                                              ):
    return compute_mcmc_expected_error(
        axis_mode='time',
        num_records=num_records,
        sampler=sampler,
        init_state=init_state,
        model=model,
        func_state=func_state,
        ground_truth_expected_func_value=ground_truth_expected_func_value,
        total_sampling_time_per_chain_seconds=total_sampling_time_per_chain_seconds,
        do_compute_rhat=do_compute_rHat,
    )


def multi_chain_expected_func_compute_and_plot_time(
        path,
        model_sampler_init_generator,
        generator_args,
        # num_mcmc_samples_per_chain,
        sampling_time_per_chain_seconds,
        num_records_per_chain,
        num_independent_chains,
        description,
        func_state=lambda x: sum(x),  # a function that takes a state and return a scalar -- default:sum
        ground_truth_expected_func_value_already_known=None,
        # do_run_DPVI=False,
        do_run_mcmc_opad=True,
        do_run_greedy_bfs=False,
        do_run_greedy_nwss=False,
        # if None, it will be computed. But for Large Symmetric Models we can set it this way
        # func_rhat=lambda x: sum(x),  # a function that takes a state and return a scalar -- default:sum todo separate func_state from func_rhat
):
    return run_multi_chain_bias_experiment(
        path=path,
        axis_mode='time',
        model_sampler_init_generator=model_sampler_init_generator,
        generator_args=generator_args,
        num_records_per_chain=num_records_per_chain,
        num_independent_chains=num_independent_chains,
        description=description,
        func_state=func_state,
        ground_truth_expected_func_value_already_known=ground_truth_expected_func_value_already_known,
        sampling_time_per_chain_seconds=sampling_time_per_chain_seconds,
        do_run_mcmc_opad=do_run_mcmc_opad,
        do_run_greedy_bfs=do_run_greedy_bfs,
        do_run_greedy_nwss=do_run_greedy_nwss,
    )


def plot_confidece_interval_TIME(
        df_MCMC,
        df_OM1,
        df_OM2,
        var_df_DPVI,
        var_df_OPAD_DPVI,
        back_df,
        # back_df2,
        back_df3,
        mc_times,
        var_times,
        back_times,
        back_times2,
        back_times3,
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
        quantiles=[0.05, 0.95]
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
        x_values=mc_times,
        var_x_values=var_times,
        back_x_values=back_times,
        # back_x_values2=back_times2,
        back_x_values3=back_times3,
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
        quantiles=tuple(quantiles),
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

    # plt.yscale('log')
    # y_limits = plt.ylim()
    # reasonable_lower_limit = max(np.median(df_OM2.values.flatten()) / 5, np.min(df_OM2.values.flatten()))
    # plt.ylim((reasonable_lower_limit, y_limits[1]))

    # plt.ylim(y_limits) # todo uncomment if you want plots have same scale...

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


def load_and_show_plots_TIME(RESPATH,
                             do_plot_greedy_bfs=True,
                             do_plot_greedy_nwss=True,
                             rhat_plotting_y_lim=None,
                             variance_plotting_y_lim=None):
    return load_and_show_bias_plots(
        RESPATH=RESPATH,
        axis_mode='time',
        do_plot_greedy_bfs=do_plot_greedy_bfs,
        do_plot_greedy_nwss=do_plot_greedy_nwss,
        rhat_plotting_y_lim=rhat_plotting_y_lim,
        variance_plotting_y_lim=variance_plotting_y_lim,
    )

