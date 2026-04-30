import csv
import datetime
import os

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from om.experiments.utils import TimeKeeper
from om.tools.histogram import ArrayHistogram
from om.tools.opad import OPADDistribution
from om.tools.rb_wr import RB_WR_Distrib

AXIS_CONFIG = {
    'iters': {
        'axis_column': 'iters',
        'axis_label': 'Target evaluations',
        'main_axis_file': 'Recorded_Iters_for_all_algs.csv',
        'greedy_bfs_axis_file': 'greedy_bfs_Recorded_Iters_for_all_algs.csv',
        'greedy_nwss_axis_file': 'greedy_nwss_Recorded_Iters_for_all_algs.csv',
    },
    'time': {
        'axis_column': 'times',
        'axis_label': 'Time (sec)',
        'main_axis_file': 'Recorded_Times_for_all_algs.csv',
        'greedy_bfs_axis_file': 'greedy_bfs_Recorded_Times_for_all_algs.csv',
        'greedy_nwss_axis_file': 'greedy_nwss_Recorded_Times_for_all_algs.csv',
    },
}


def _require_axis_mode(axis_mode):
    if axis_mode not in AXIS_CONFIG:
        raise ValueError(f"Unsupported axis_mode: {axis_mode}")
    return AXIS_CONFIG[axis_mode]


def _resolve_plotting_y_limits(current_y_limits, plotting_y_limits):
    if plotting_y_limits is None:
        return current_y_limits
    if isinstance(plotting_y_limits, (tuple, list)):
        return tuple(plotting_y_limits)
    return (current_y_limits[0], plotting_y_limits)


def _value_at_fraction_of_total(series_with_axes, fraction):
    total_max = max(series_x.max() for series_x, _ in series_with_axes if len(series_x) > 0)
    threshold = fraction * total_max

    values = []
    for series_x, series_y in series_with_axes:
        if len(series_x) == 0:
            continue

        idx = next((i for i, x in enumerate(series_x) if x >= threshold), len(series_x) - 1)
        values.append(float(series_y.iloc[idx]))

    return max(values)


def _min_across_series(series_list):
    return min(float(series.min()) for series in series_list if len(series) > 0)


def compute_greedy_expected_error(
        axis_mode,
        num_records,
        greedy,
        model,
        func_state,
        ground_truth_expected_func_value,
        equivalent_num_mcmc_samples_per_chain=None,
        total_sampling_time_per_chain_seconds=None,
        do_compute_rhat=True,
):
    _require_axis_mode(axis_mode)
    num_mutations_per_evolution = greedy.num_mutations_per_evolutions()

    recorded_evals = []
    recorded_times = []
    recorded_errors = []
    recorded_means = []
    recorded_variances = []

    time_keeper = TimeKeeper()

    if axis_mode == 'iters':
        num_evolution_steps = int(equivalent_num_mcmc_samples_per_chain / num_mutations_per_evolution)
        recorded_every = int(num_evolution_steps / num_records) + 1
        evolution_steps = range(1, num_evolution_steps + 1)

        for evolution_step in tqdm(evolution_steps):
            time_keeper.start()
            greedy.evolve()
            time_keeper.stop()

            if evolution_step % recorded_every != 0:
                continue

            approx_mean = greedy.inner_opad.calc_expected_value(func=func_state)
            recorded_evals.append(evolution_step * num_mutations_per_evolution)
            recorded_times.append(time_keeper.get_passed_time())
            recorded_errors.append(approx_mean - ground_truth_expected_func_value)

            if do_compute_rhat:
                recorded_means.append(approx_mean)
                recorded_variances.append(greedy.inner_opad.calc_variance(func=func_state))

        print('Search passed time: ', time_keeper.get_passed_time())
    else:
        recording_interval = total_sampling_time_per_chain_seconds / num_records
        num_recorded_entries = 0
        passed_time = 0
        evolution_step = 0

        while passed_time <= total_sampling_time_per_chain_seconds:
            time_keeper.start()
            greedy.evolve()
            time_keeper.stop()

            evolution_step += 1
            passed_time = time_keeper.get_passed_time()

            while passed_time > recording_interval * (num_recorded_entries + 1):
                num_recorded_entries += 1
                if num_recorded_entries > num_records:
                    break

                approx_mean = greedy.inner_opad.calc_expected_value(func=func_state)
                recorded_evals.append(evolution_step * num_mutations_per_evolution)
                recorded_times.append(passed_time)
                recorded_errors.append(approx_mean - ground_truth_expected_func_value)

                if do_compute_rhat:
                    recorded_means.append(approx_mean)
                    recorded_variances.append(greedy.inner_opad.calc_variance(func=func_state))

        assert len(recorded_times) == num_records, (
            f'no. recorded times: {len(recorded_times)} but it had to be: {num_records}'
        )
        print("Search total Passed time:", passed_time)

    return {
        'Evals': recorded_evals,
        'Times': recorded_times,
        'Greedy': recorded_errors,
        'Means': recorded_means,
        'Variances': recorded_variances,
    }


def compute_mcmc_expected_error(
        axis_mode,
        num_records,
        sampler,
        init_state,
        model,
        func_state,
        ground_truth_expected_func_value,
        num_mcmc_samples=None,
        total_sampling_time_per_chain_seconds=None,
        do_compute_rhat=True,
):
    _require_axis_mode(axis_mode)
    current_state = init_state.copy()
    mcmc_hist = ArrayHistogram()
    opad_accepted_only = OPADDistribution(target_model=model)  # OPAD
    opad_with_proposals = OPADDistribution(target_model=model)  # OPAD+
    rb_distrib = RB_WR_Distrib()

    recorded_iters = []
    recorded_times = []
    recorded_mcmc_errors = []
    recorded_opad_errors = []
    recorded_opad_plus_errors = []
    recorded_rb_errors = []  # RB/WR

    recorded_mcmc_means = []
    recorded_mcmc_variances = []
    recorded_opad_means = []
    recorded_opad_variances = []
    recorded_opad_plus_means = []
    recorded_opad_plus_variances = []
    recorded_rb_means = []
    recorded_rb_variances = []

    time_keeper = TimeKeeper()

    def record_snapshot(sample_count, passed_time):
        approx_mcmc = mcmc_hist.calc_expected_value(func=func_state)
        approx_opad = opad_accepted_only.calc_expected_value(func=func_state)
        approx_opad_plus = opad_with_proposals.calc_expected_value(func=func_state)
        approx_rb = rb_distrib.calc_expected_value(func=func_state)

        recorded_iters.append(sample_count)
        recorded_times.append(passed_time)
        recorded_mcmc_errors.append(approx_mcmc - ground_truth_expected_func_value)
        recorded_opad_errors.append(approx_opad - ground_truth_expected_func_value)
        recorded_opad_plus_errors.append(approx_opad_plus - ground_truth_expected_func_value)
        recorded_rb_errors.append(approx_rb - ground_truth_expected_func_value)

        if do_compute_rhat:
            recorded_mcmc_means.append(approx_mcmc)
            recorded_opad_means.append(approx_opad)
            recorded_opad_plus_means.append(approx_opad_plus)
            recorded_rb_means.append(approx_rb)

            recorded_mcmc_variances.append(mcmc_hist.calc_variance(func=func_state))
            recorded_opad_variances.append(opad_accepted_only.calc_variance(func=func_state))
            recorded_opad_plus_variances.append(opad_with_proposals.calc_variance(func=func_state))
            recorded_rb_variances.append(rb_distrib.calc_variance(func=func_state))

    if axis_mode == 'iters':
        recorded_every = int(num_mcmc_samples / num_records) + 1

        for sample_count in tqdm(range(1, num_mcmc_samples + 1)):
            time_keeper.start()
            previous_state = current_state.copy()
            current_state, proposed_states, acceptances = sampler.next_sample_proposals_acceptances(current_state)
            time_keeper.stop()

            assert len(proposed_states) == 1
            proposed_state = proposed_states[0]
            acceptance = acceptances[0]

            mcmc_hist.add_array(current_state)
            opad_accepted_only.add_array(current_state)
            opad_with_proposals.add_array(current_state)
            opad_with_proposals.add_array(proposed_state)
            rb_distrib.add_previous_and_proposed_states(previous_state=previous_state,
                                                        proposed_state=proposed_state, acceptance_prob=acceptance)

            if sample_count % recorded_every == 0:
                record_snapshot(sample_count=sample_count, passed_time=time_keeper.get_passed_time())

        print(f"One MCMC chain total samplng time: {time_keeper.get_passed_time()} seconds")
    else:
        recording_interval = total_sampling_time_per_chain_seconds / num_records
        num_recorded_entries = 0
        passed_time = 0
        sample_count = 0

        while passed_time <= total_sampling_time_per_chain_seconds:
            time_keeper.start()
            previous_state = current_state.copy()
            current_state, proposed_states, acceptances = sampler.next_sample_proposals_acceptances(current_state)
            time_keeper.stop()

            passed_time = time_keeper.get_passed_time()
            sample_count += 1

            assert len(proposed_states) == 1
            proposed_state = proposed_states[0]
            acceptance = acceptances[0]

            mcmc_hist.add_array(current_state)
            opad_accepted_only.add_array(current_state)
            opad_with_proposals.add_array(current_state)
            opad_with_proposals.add_array(proposed_state)
            rb_distrib.add_previous_and_proposed_states(previous_state=previous_state,
                                                        proposed_state=proposed_state,
                                                        acceptance_prob=acceptance)

            while passed_time > recording_interval * (num_recorded_entries + 1):
                num_recorded_entries += 1
                if num_recorded_entries > num_records:
                    break
                record_snapshot(sample_count=sample_count, passed_time=passed_time)

        assert len(recorded_times) == num_records, (
            f'no. recorded times: {len(recorded_times)} but it had to be: {num_records}'
        )
        print('MCMC passed_time: ', passed_time)

    return {
        'Iters': recorded_iters,
        'Times': recorded_times,
        'MCMC': recorded_mcmc_errors,
        'OPAD': recorded_opad_errors,
        'OPAD+': recorded_opad_plus_errors,
        'RB': recorded_rb_errors,
        'MCMC.means': recorded_mcmc_means,
        'OPAD.means': recorded_opad_means,
        'OPAD+.means': recorded_opad_plus_means,
        'RB.means': recorded_rb_means,
        'MCMC.variances': recorded_mcmc_variances,
        'OPAD.variances': recorded_opad_variances,
        'OPAD+.variances': recorded_opad_plus_variances,
        'RB.variances': recorded_rb_variances
    }


def _write_dataframe(path, data):
    pd.DataFrame(data).to_csv(path, index=True)


def run_multi_chain_bias_experiment(
        path,
        axis_mode,
        model_sampler_init_generator,
        generator_args,
        num_records_per_chain,
        num_independent_chains,
        description,
        func_state=lambda x: sum(x),
        ground_truth_expected_func_value_already_known=None,
        num_mcmc_samples_per_chain=None,
        sampling_time_per_chain_seconds=None,
        do_run_mcmc_opad=True,
        do_run_greedy_bfs=True,
        do_run_greedy_nwss=True,
):
    axis_config = _require_axis_mode(axis_mode)
    print("===========\n", description, "\n===========\n")

    chains = [f'C_{chain_index}' for chain_index in range(num_independent_chains)]
    dirpath = os.path.abspath(path)
    main_results_path = dirpath
    os.makedirs(main_results_path, exist_ok=True)

    result_path = os.path.join(
        main_results_path,
        datetime.datetime.now().strftime(description + '_%Y-%m-%d_%H-%M-%S'),
    )
    os.makedirs(result_path, exist_ok=True)
    if axis_mode == 'time':
        print('Results will be saved in: ', result_path)

    metadata_dict = {
        "Num_experiments": num_independent_chains,
        "Description": description,
        "num_records_per_chain": num_records_per_chain,
    }
    if axis_mode == 'iters':
        metadata_dict["num_mcmc_samples_per_chain"] = num_mcmc_samples_per_chain
    else:
        metadata_dict["sampling_time_per_chain_seconds"] = sampling_time_per_chain_seconds
    for key, value in generator_args.items():
        metadata_dict[f'gen.arg.{key}'] = value

    with open(os.path.join(result_path, 'metadata.csv'), 'w', newline="") as csvfile:
        writer = csv.DictWriter(csvfile, metadata_dict.keys())
        writer.writeheader()
        writer.writerow(metadata_dict)

    primary_axis_records = None
    chain_to_mcmc_errors = {}
    chain_to_opad_errors = {}
    chain_to_opad_plus_errors = {}
    chain_to_rb_errors = {}
    chain_to_mcmc_means = {}
    chain_to_opad_means = {}
    chain_to_opad_plus_means = {}
    chain_to_rb_means = {}
    chain_to_mcmc_variances = {}
    chain_to_opad_variances = {}
    chain_to_opad_plus_variances = {}
    chain_to_rb_variances = {}

    greedy_runs = {
        'greedy_bfs': {
            'enabled': do_run_greedy_bfs,
            'strategy': 'BFS',
            'label': 'BFS',
            'axis_records': None,
            'errors': {},
            'means': {},
            'variances': {},
        },
        'greedy_nwss': {
            'enabled': do_run_greedy_nwss,
            'strategy': 'NWSS',
            'label': 'NWSS',
            'axis_records': None,
            'errors': {},
            'means': {},
            'variances': {},
        },
    }

    for chain_count, chain_name in enumerate(chains):
        print("{i}: Chain name: {c}".format(i=chain_count, c=chain_name))

        generator_args['seed'] = chain_count
        model_sampler_init = model_sampler_init_generator(**generator_args)

        model = model_sampler_init.model
        sampler = model_sampler_init.sampler
        init_state = model.generate_init_state()
        greedy = model_sampler_init.greedy

        if ground_truth_expected_func_value_already_known is None:
            message = 'computing the ground truth E[theta]...' if axis_mode == 'time' else 'computing the ground truth...'
            print(message)
            ground_truth_expected_func_value = model.calc_expected_value(func=func_state)
            ground_truth_expected_func_value_already_known = ground_truth_expected_func_value
        else:
            ground_truth_expected_func_value = ground_truth_expected_func_value_already_known

        if do_run_mcmc_opad:
            results_mcmc = compute_mcmc_expected_error(
                axis_mode=axis_mode,
                num_records=num_records_per_chain,
                sampler=sampler,
                init_state=init_state,
                model=model,
                func_state=func_state,
                ground_truth_expected_func_value=ground_truth_expected_func_value,
                num_mcmc_samples=num_mcmc_samples_per_chain,
                total_sampling_time_per_chain_seconds=sampling_time_per_chain_seconds,
            )

            axis_values = results_mcmc['Times'] if axis_mode == 'time' else results_mcmc['Iters']
            if primary_axis_records is None:
                primary_axis_records = {axis_config['axis_column']: axis_values}

            chain_to_mcmc_errors[f"KL_MCMC_{chain_count}"] = results_mcmc['MCMC']
            chain_to_opad_errors[f"KL_OPAD_{chain_count}"] = results_mcmc['OPAD']
            chain_to_opad_plus_errors[f"KL_OPAD_plus_{chain_count}"] = results_mcmc['OPAD+']
            chain_to_rb_errors[f"KL_RB_{chain_count}"] = results_mcmc['RB']

            chain_to_mcmc_means[f"means_MCMC_{chain_count}"] = results_mcmc['MCMC.means']
            chain_to_opad_means[f"means_OPAD_{chain_count}"] = results_mcmc['OPAD.means']
            chain_to_opad_plus_means[f"means_OPAD_plus_{chain_count}"] = results_mcmc['OPAD+.means']
            chain_to_rb_means[f"means_RB_{chain_count}"] = results_mcmc['RB.means']

            chain_to_mcmc_variances[f"variances_MCMC_{chain_count}"] = results_mcmc['MCMC.variances']
            chain_to_opad_variances[f"variances_OPAD_{chain_count}"] = results_mcmc['OPAD.variances']
            chain_to_opad_plus_variances[f"variances_OPAD_plus_{chain_count}"] = results_mcmc['OPAD+.variances']
            chain_to_rb_variances[f"variances_RB_{chain_count}"] = results_mcmc['RB.variances']

            _write_dataframe(os.path.join(result_path, axis_config['main_axis_file']), primary_axis_records)
            _write_dataframe(os.path.join(result_path, 'MCMC_KL_results.csv'), chain_to_mcmc_errors)
            _write_dataframe(os.path.join(result_path, 'OPAD_KL_results.csv'), chain_to_opad_errors)
            _write_dataframe(os.path.join(result_path, 'OPAD_plus_KL_results.csv'), chain_to_opad_plus_errors)
            _write_dataframe(os.path.join(result_path, 'RB_KL_results.csv'), chain_to_rb_errors)
            _write_dataframe(os.path.join(result_path, 'MCMC_means_results.csv'), chain_to_mcmc_means)
            _write_dataframe(os.path.join(result_path, 'OPAD_means_results.csv'), chain_to_opad_means)
            _write_dataframe(os.path.join(result_path, 'OPAD_plus_means_results.csv'), chain_to_opad_plus_means)
            _write_dataframe(os.path.join(result_path, 'RB_means_results.csv'), chain_to_rb_means)
            _write_dataframe(os.path.join(result_path, 'MCMC_variances_results.csv'), chain_to_mcmc_variances)
            _write_dataframe(os.path.join(result_path, 'OPAD_variances_results.csv'), chain_to_opad_variances)
            _write_dataframe(os.path.join(result_path, 'OPAD_plus_variances_results.csv'), chain_to_opad_plus_variances)
            _write_dataframe(os.path.join(result_path, 'RB_variances_results.csv'), chain_to_rb_variances)

        for run_name, run_config in greedy_runs.items():
            if not run_config['enabled']:
                continue

            if axis_mode == 'time':
                print(f"Running {run_config['label']}...")

            greedy.initialize(run_config['strategy'], init_state=init_state)
            results_greedy = compute_greedy_expected_error(
                axis_mode=axis_mode,
                num_records=num_records_per_chain,
                greedy=greedy,
                model=model,
                func_state=func_state,
                ground_truth_expected_func_value=ground_truth_expected_func_value,
                equivalent_num_mcmc_samples_per_chain=num_mcmc_samples_per_chain,
                total_sampling_time_per_chain_seconds=sampling_time_per_chain_seconds,
            )

            axis_values = results_greedy['Times'] if axis_mode == 'time' else results_greedy['Evals']
            if run_config['axis_records'] is None:
                run_config['axis_records'] = {axis_config['axis_column']: axis_values}

            run_config['errors'][f"KL_{run_name}_{chain_count}"] = results_greedy['Greedy']
            run_config['means'][f"means_{run_name}_{chain_count}"] = results_greedy['Means']
            run_config['variances'][f"variances_{run_name}_{chain_count}"] = results_greedy['Variances']

            _write_dataframe(
                os.path.join(result_path, axis_config[f'{run_name}_axis_file']),
                run_config['axis_records'],
            )
            _write_dataframe(os.path.join(result_path, f'{run_name}_KL_results.csv'), run_config['errors'])
            _write_dataframe(os.path.join(result_path, f'{run_name}_means_results.csv'), run_config['means'])
            _write_dataframe(
                os.path.join(result_path, f'{run_name}_variances_results.csv'),
                run_config['variances'],
            )

    load_and_show_bias_plots(
        RESPATH=result_path,
        axis_mode=axis_mode,
        # do_plot_rb=True,
        do_plot_greedy_bfs=do_run_greedy_bfs,
        # do_plot_backtrack2=do_run_backtrack2,
        do_plot_greedy_nwss=do_run_greedy_nwss,
    )
    return result_path


def plot_confidence_interval(
        df_MCMC,
        df_OM1,
        df_OM2,
        df_RB,
        var_df_DPVI,
        var_df_OPAD_DPVI,
        back_df,
        # back_df2,
        back_df3,
        x_values,
        var_x_values,
        back_x_values,
        # back_x_values2,
        back_x_values3,
        y_label,
        x_label,
        save_fig_address,
        mcmc_color='blue',
        om1_color='red',
        om2_color='black',
    rb_color='orange',
        DPVI_color='green',
        OPAD_DPVI_color='red',
        back_color='magenta',
        # back_color2='green',
        back_color3='orange',
        use_log_scale_y_axis=True,
        plotting_y_limits=None,
        quantiles=(0.05, 0.95),
):
    plt.figure()
    mcmc_markevery = max(1, len(x_values) // 12)

    ndim = df_MCMC.ndim
    if ndim > 1:
        mcmc_mean = df_MCMC.mean(axis=1)
        om1_mean = df_OM1.mean(axis=1)
        om2_mean = df_OM2.mean(axis=1)
        rb_mean = df_RB.mean(axis=1)
        mcmc_quantiles = df_MCMC.apply(lambda row: row.quantile(quantiles), axis=1)
        om1_quantiles = df_OM1.apply(lambda row: row.quantile(quantiles), axis=1)
        om2_quantiles = df_OM2.apply(lambda row: row.quantile(quantiles), axis=1)
        rb_quantiles = df_RB.apply(lambda row: row.quantile(quantiles), axis=1)
        linestyle = '--'
    else:
        mcmc_mean = df_MCMC
        om1_mean = df_OM1
        om2_mean = df_OM2
        rb_mean = df_RB
        linestyle = '-'

    if var_x_values is not None:
        ndim_var = var_df_DPVI.ndim
        if ndim_var > 1:
            var_dpvi_mean = var_df_DPVI.mean(axis=1)
            var_opad_dpvi_mean = var_df_OPAD_DPVI.mean(axis=1)
            var_dpvi_quantiles = var_df_DPVI.apply(lambda row: row.quantile(quantiles), axis=1)
            var_opad_dpvi_quantiles = var_df_OPAD_DPVI.apply(lambda row: row.quantile(quantiles), axis=1)
        else:
            var_dpvi_mean = var_df_DPVI
            var_opad_dpvi_mean = var_df_OPAD_DPVI

    if back_x_values is not None:
        ndim_back = back_df.ndim
        if ndim_back > 1:
            back_mean = back_df.mean(axis=1)
            back_quantiles = back_df.apply(lambda row: row.quantile(quantiles), axis=1)
        else:
            back_mean = back_df

    if back_x_values3 is not None:
        ndim_back3 = back_df3.ndim
        if ndim_back3 > 1:
            back_mean3 = back_df3.mean(axis=1)
            back_quantiles3 = back_df3.apply(lambda row: row.quantile(quantiles), axis=1)
        else:
            back_mean3 = back_df3

    plt.plot(x_values, om1_mean, color=om1_color, linestyle='--', label='MCMC OPAD')
    plt.plot(x_values, om2_mean, color=om2_color, linestyle='--', label='MCMC OPAD+')
    plt.plot(x_values, rb_mean, color=rb_color, linestyle='-.', label='MCMC RB/WR')

    if use_log_scale_y_axis:
        plt.yscale('log')
    y_limits = _resolve_plotting_y_limits(plt.ylim(), plotting_y_limits)

    if ndim > 1:
        plt.fill_between(x=x_values, y1=mcmc_quantiles[quantiles[0]], y2=mcmc_quantiles[quantiles[1]], color=mcmc_color,
                         alpha=0.2)
        plt.fill_between(x=x_values, y1=om1_quantiles[quantiles[0]], y2=om1_quantiles[quantiles[1]], color=om1_color,
                         alpha=0.2)
        plt.fill_between(x=x_values, y1=om2_quantiles[quantiles[0]], y2=om2_quantiles[quantiles[1]], color=om2_color,
                         alpha=0.2)
        plt.fill_between(x=x_values, y1=rb_quantiles[quantiles[0]], y2=rb_quantiles[quantiles[1]], color=rb_color,
                         alpha=0.2)

    if var_x_values is not None:
        plt.plot(var_x_values, var_dpvi_mean, color=DPVI_color, linestyle='--', label='DPVI')
        plt.plot(var_x_values, var_opad_dpvi_mean, color=OPAD_DPVI_color, linestyle='--', label='DPVI OPAD+')
        plt.fill_between(x=var_x_values, y1=var_dpvi_quantiles[quantiles[0]], y2=var_dpvi_quantiles[quantiles[1]],
                         color=DPVI_color, alpha=0.2)
        plt.fill_between(x=var_x_values, y1=var_opad_dpvi_quantiles[quantiles[0]],
                         y2=var_opad_dpvi_quantiles[quantiles[1]], color=OPAD_DPVI_color, alpha=0.2)

    if back_x_values is not None:
        plt.plot(back_x_values, back_mean, color=back_color, linestyle='--', label='BFS')
        if ndim_back > 1:
            plt.fill_between(x=back_x_values, y1=back_quantiles[quantiles[0]], y2=back_quantiles[quantiles[1]],
                             color=back_color, alpha=0.2)

    if back_x_values3 is not None:
        plt.plot(back_x_values3, back_mean3, color=back_color3, linestyle='--', label='NWSS')
        if ndim_back3 > 1:
            plt.fill_between(x=back_x_values3, y1=back_quantiles3[quantiles[0]], y2=back_quantiles3[quantiles[1]],
                             color=back_color3, alpha=0.2)

    plt.plot(
        x_values,
        mcmc_mean,
        color=mcmc_color,
        linestyle='--',
        linewidth=2.5,
        marker='o',
        markersize=4,
        markevery=mcmc_markevery,
        label='MCMC',
    )

    plt.ylim(y_limits)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig(save_fig_address)
    plt.show(block=True)


def calc_r_hat(df_means, df_variances):
    b_div_n = df_means.var(axis=1)
    within_chain_var = df_variances.mean(axis=1)
    v_hat = within_chain_var + b_div_n
    return (v_hat / within_chain_var) ** 0.5


def load_and_show_bias_plots(
        RESPATH,
        axis_mode,
        do_plot_greedy_bfs=False,
        # do_plot_backtrack2=False,
        do_plot_greedy_nwss=True,
        rhat_plotting_y_lim=None,
        variance_plotting_y_lim=None,
        confidence_quantiles=(0.05, 0.95),
):
    axis_config = _require_axis_mode(axis_mode)
    print("*** PLOTTING: ", RESPATH)

    main_axis_df = pd.read_csv(os.path.join(RESPATH, axis_config['main_axis_file']), index_col=0)
    x_values = main_axis_df[axis_config['axis_column']]

    df_mcmc_raw = pd.read_csv(os.path.join(RESPATH, 'MCMC_KL_results.csv'), index_col=0)
    df_om1_raw = pd.read_csv(os.path.join(RESPATH, 'OPAD_KL_results.csv'), index_col=0)
    df_om2_raw = pd.read_csv(os.path.join(RESPATH, 'OPAD_plus_KL_results.csv'), index_col=0)
    df_rb_raw = pd.read_csv(os.path.join(RESPATH, 'RB_KL_results.csv'), index_col=0)

    r_hat_mcmc = calc_r_hat(
        df_means=pd.read_csv(os.path.join(RESPATH, 'MCMC_means_results.csv'), index_col=0),
        df_variances=pd.read_csv(os.path.join(RESPATH, 'MCMC_variances_results.csv'), index_col=0),
    )
    r_hat_om1 = calc_r_hat(
        df_means=pd.read_csv(os.path.join(RESPATH, 'OPAD_means_results.csv'), index_col=0),
        df_variances=pd.read_csv(os.path.join(RESPATH, 'OPAD_variances_results.csv'), index_col=0),
    )
    r_hat_om2 = calc_r_hat(
        df_means=pd.read_csv(os.path.join(RESPATH, 'OPAD_plus_means_results.csv'), index_col=0),
        df_variances=pd.read_csv(os.path.join(RESPATH, 'OPAD_plus_variances_results.csv'), index_col=0),
    )
    r_hat_rb = calc_r_hat(
        df_means=pd.read_csv(os.path.join(RESPATH, 'RB_means_results.csv'), index_col=0),
        df_variances=pd.read_csv(os.path.join(RESPATH, 'RB_variances_results.csv'), index_col=0),
    )

    greedy_series = {
        'greedy_bfs': {'enabled': do_plot_greedy_bfs},
        'greedy_nwss': {'enabled': do_plot_greedy_nwss},
    }
    for run_name, run_cfg in greedy_series.items():
        if not run_cfg['enabled']:
            continue
        greedy_axis_df = pd.read_csv(
            os.path.join(RESPATH, axis_config[f'{run_name}_axis_file']),
            index_col=0,
        )
        run_cfg['x_values'] = greedy_axis_df[axis_config['axis_column']]
        run_cfg['raw'] = pd.read_csv(os.path.join(RESPATH, f'{run_name}_KL_results.csv'), index_col=0)
        run_cfg['rhat'] = calc_r_hat(
            df_means=pd.read_csv(os.path.join(RESPATH, f'{run_name}_means_results.csv'), index_col=0),
            df_variances=pd.read_csv(os.path.join(RESPATH, f'{run_name}_variances_results.csv'), index_col=0),
        )

    if rhat_plotting_y_lim is None:
        rhat_series_with_axes = [
            (x_values, r_hat_mcmc),
            (x_values, r_hat_om1),
            (x_values, r_hat_om2),
            (x_values, r_hat_rb),
        ]
        rhat_series = [r_hat_mcmc, r_hat_om1, r_hat_om2, r_hat_rb]
        if do_plot_greedy_bfs:
            rhat_series.append(greedy_series['greedy_bfs']['rhat'])
            rhat_series_with_axes.append(
                (greedy_series['greedy_bfs']['x_values'], greedy_series['greedy_bfs']['rhat'])
            )
        if do_plot_greedy_nwss:
            rhat_series.append(greedy_series['greedy_nwss']['rhat'])
            rhat_series_with_axes.append(
                (greedy_series['greedy_nwss']['x_values'], greedy_series['greedy_nwss']['rhat'])
            )
        rhat_plotting_y_lim = (
            _min_across_series(rhat_series),
            _value_at_fraction_of_total(rhat_series_with_axes, fraction=0.1),
        )

    mcmc_color = 'blue'
    om1_color = 'red'
    om2_color = 'black'
    rb_color = 'orange'
    dpvi_color = 'green'
    opad_dpvi_color = 'orange'
    back_color = 'cyan'
    # back_color2 = 'green'
    back_color3 = 'magenta'

    plot_confidence_interval(
        df_MCMC=r_hat_mcmc,
        df_OM1=r_hat_om1,
        df_OM2=r_hat_om2,
        df_RB=r_hat_rb,
        var_df_DPVI=None,
        var_df_OPAD_DPVI=None,
        back_df=greedy_series['greedy_bfs'].get('rhat') if do_plot_greedy_bfs else None,
        back_df3=greedy_series['greedy_nwss'].get('rhat') if do_plot_greedy_nwss else None,
        x_values=x_values,
        var_x_values=None,
        back_x_values=greedy_series['greedy_bfs'].get('x_values') if do_plot_greedy_bfs else None,
        back_x_values3=greedy_series['greedy_nwss'].get('x_values') if do_plot_greedy_nwss else None,
        y_label='$\\widehat{R}$',
        x_label=axis_config['axis_label'],
        save_fig_address=os.path.join(RESPATH, 'R-hat--' + RESPATH.split("/")[-1] + '.png'),
        mcmc_color=mcmc_color,
        om1_color=om1_color,
        om2_color=om2_color,
        rb_color=rb_color,
        DPVI_color=dpvi_color,
        OPAD_DPVI_color=opad_dpvi_color,
        back_color=back_color,
        # back_color2=back_color2,
        back_color3=back_color3,
        use_log_scale_y_axis=False,
        plotting_y_limits=rhat_plotting_y_lim,
    )

    plot_confidence_interval(
        df_MCMC=df_mcmc_raw ** 2,
        df_OM1=df_om1_raw ** 2,
        df_OM2=df_om2_raw ** 2,
        df_RB=df_rb_raw ** 2,
        var_df_DPVI=None,
        var_df_OPAD_DPVI=None,
        back_df=greedy_series['greedy_bfs'].get('raw') ** 2 if do_plot_greedy_bfs else None,
        back_df3=greedy_series['greedy_nwss'].get('raw') ** 2 if do_plot_greedy_nwss else None,
        x_values=x_values,
        var_x_values=None,
        back_x_values=greedy_series['greedy_bfs'].get('x_values') if do_plot_greedy_bfs else None,
        back_x_values3=greedy_series['greedy_nwss'].get('x_values') if do_plot_greedy_nwss else None,
        y_label='Square Error, $(\\theta - \\theta^*)^2$',
        x_label=axis_config['axis_label'],
        save_fig_address=os.path.join(RESPATH, 'SquareErrorConfidence--' + RESPATH.split("/")[-1] + '.png'),
        mcmc_color=mcmc_color,
        om1_color=om1_color,
        om2_color=om2_color,
        rb_color=rb_color,
        DPVI_color=dpvi_color,
        OPAD_DPVI_color=opad_dpvi_color,
        back_color=back_color,
        # back_color2=back_color2,
        back_color3=back_color3,
        quantiles=confidence_quantiles,
    )

    plot_confidence_interval(
        df_MCMC=df_mcmc_raw.mean(axis=1).abs(),
        df_OM1=df_om1_raw.mean(axis=1).abs(),
        df_OM2=df_om2_raw.mean(axis=1).abs(),
        df_RB=df_rb_raw.mean(axis=1).abs(),
        var_df_DPVI=None,
        var_df_OPAD_DPVI=None,
        back_df=greedy_series['greedy_bfs'].get('raw').mean(axis=1).abs() if do_plot_greedy_bfs else None,
        back_df3=greedy_series['greedy_nwss'].get('raw').mean(axis=1).abs() if do_plot_greedy_nwss else None,
        x_values=x_values,
        var_x_values=None,
        back_x_values=greedy_series['greedy_bfs'].get('x_values') if do_plot_greedy_bfs else None,
        back_x_values3=greedy_series['greedy_nwss'].get('x_values') if do_plot_greedy_nwss else None,
        y_label='Absolute Bias, $|E(\\theta - \\theta^*)|$',
        x_label=axis_config['axis_label'],
        save_fig_address=os.path.join(RESPATH, 'BiasConfidence--' + RESPATH.split("/")[-1] + '.png'),
        mcmc_color=mcmc_color,
        om1_color=om1_color,
        om2_color=om2_color,
        rb_color=rb_color,
        DPVI_color=dpvi_color,
        OPAD_DPVI_color=opad_dpvi_color,
        back_color=back_color,
        # back_color2=back_color2,
        back_color3=back_color3,
    )

    plot_confidence_interval(
        df_MCMC=df_mcmc_raw.var(axis=1),
        df_OM1=df_om1_raw.var(axis=1),
        df_OM2=df_om2_raw.var(axis=1),
        df_RB=df_rb_raw.var(axis=1),
        var_df_DPVI=None,
        var_df_OPAD_DPVI=None,
        back_df=greedy_series['greedy_bfs'].get('raw').var(axis=1) if do_plot_greedy_bfs else None,
        back_df3=greedy_series['greedy_nwss'].get('raw').var(axis=1) if do_plot_greedy_nwss else None,
        x_values=x_values,
        var_x_values=None,
        back_x_values=greedy_series['greedy_bfs'].get('x_values') if do_plot_greedy_bfs else None,
        back_x_values3=greedy_series['greedy_nwss'].get('x_values') if do_plot_greedy_nwss else None,
        y_label='$Var(\\theta)$',
        x_label=axis_config['axis_label'],
        save_fig_address=os.path.join(RESPATH, 'Var--' + RESPATH.split("/")[-1] + '.png'),
        mcmc_color=mcmc_color,
        om1_color=om1_color,
        om2_color=om2_color,
        rb_color=rb_color,
        DPVI_color=dpvi_color,
        OPAD_DPVI_color=opad_dpvi_color,
        back_color=back_color,
        # back_color2=back_color2,
        back_color3=back_color3,
        plotting_y_limits=variance_plotting_y_lim,
    )
