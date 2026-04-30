from om.models.ising1D import IsingModel1D
from om.samplers.dvpi import DiscreteParticleVariationalInference, FixedFirstDiscreteParticleVariationalInference
from om.samplers.greedy import GreedyExplore
from om.tools.histogram import ArrayHistogram
from om.tools.opad import OPADDistribution
from om.tools.rb_wr import RB_WR_Distrib
from om.samplers.mh import BiValueArrayMetropolisSampler, FixedFirstElementArraySampler
from om.tools.kl_divergence import kl_divergence_discrete
from tqdm import tqdm
import numpy as np
from om.models.var_select import standardize_add_one_column, BayesianVarSelectModel, \
    fetch_mince_nutrition_data
from om.models.model import DiscreteDistributionModel

from om.models.ising1D_high_dim import IsingModel1D_Large_Symmetric


class ModelVariationalMcmcGreedyContainer:
    def __init__(self, model, variational, sampler, greedy, init_state):
        self.model = model
        self.variational = variational

        self.sampler = sampler
        self.init_state = init_state

        self.greedy = greedy


def experiment1_ising1d(num_particles, latice_size=15, J=1, h=0.1, beta=0.5, seed=0):
    np.random.seed(seed)
    model = IsingModel1D(latice_size, J=J, h=h, beta=beta)
    variational = DiscreteParticleVariationalInference(model=model, num_particles=num_particles)

    sampler = BiValueArrayMetropolisSampler(model=model)  # MCMC
    backtrack = GreedyExplore(model=model)
    init_state = model.generate_init_state()

    return ModelVariationalMcmcGreedyContainer(model=model, variational=variational,
                                               sampler=sampler,
                                               greedy=backtrack,
                                               init_state=init_state)


def experiment2_ising1d__high_dim(num_particles, latice_size=15, J=1, beta=0.5, seed=0):
    np.random.seed(seed)
    model = IsingModel1D_Large_Symmetric(latice_size, J=J, beta=beta)
    variational = DiscreteParticleVariationalInference(model=model, num_particles=num_particles)

    sampler = BiValueArrayMetropolisSampler(model=model)  # MCMC
    backtrack = GreedyExplore(model=model)
    init_state = None

    return ModelVariationalMcmcGreedyContainer(model=model, variational=variational,
                                               sampler=sampler, greedy=backtrack, init_state=init_state)


def experiment3_varselect_mice_data(data_path, seed=0,
                                    num_variational_particles=10):  # , num_mcmc_samples=1000, num_records=100):
    np.random.seed(seed)

    X, Y, X_names = fetch_mince_nutrition_data(mice_data_path=data_path)
    X, X_names = standardize_add_one_column(X=X, X_names=X_names)
    a, b = 3.0, 1.0  # inverse-gamma hyper-params used for the prior of sigma2  #todo what is a good choice?

    n = X.shape[0]  # number of data points
    c = n  # hyper parameter used in p(beta). todo what is a good choice? length of X?
    pi = 0.5  # Bernoulli parameter todo for now

    model = BayesianVarSelectModel(X=X, y=Y, a=a, b=b, c=c, pi=pi)
    sampler = FixedFirstElementArraySampler(model=model)
    variational = FixedFirstDiscreteParticleVariationalInference(
        model=model,
        num_particles=num_variational_particles,
        possible_values=(0, 1), fixed_first_value=1)

    greedy = GreedyExplore(model=model, possible_values=(0, 1), fixed_first_value=1)

    init_state = model.generate_init_state()

    return ModelVariationalMcmcGreedyContainer(model=model, sampler=sampler,
                                               variational=variational, greedy=greedy, init_state=init_state)


def greedy_kl_compute(  # num_evolution_steps: int,
        equivalent_of_num_mcmc_samples_per_chain: int,
        num_records: int,
        greedy: GreedyExplore,
        model: DiscreteDistributionModel,
):
    """
    :param equivalent_of_num_mcmc_samples_per_chain: evolve equivalent to this much MCMC steps
    :param num_records: num times KL is computed
    :param greedy:
    :param model:
    :return:
    """
    num_mutations_per_evolution = greedy.num_mutations_per_evolutions()
    num_evolution_steps = int(equivalent_of_num_mcmc_samples_per_chain / num_mutations_per_evolution)
    recorded_every = int(num_evolution_steps / num_records) + 1

    recorded_iters = []
    recorded_kls = []

    num_posterior_validations = 0

    for evolution_step in tqdm(range(1, num_evolution_steps + 1)):

        greedy.evolve()

        num_posterior_validations += model.get_dimension()  # this is actually, the number newly generated proposals

        if evolution_step % recorded_every == 0:
            kl_backtrack = kl_divergence_discrete(p_empirical=greedy.inner_opad, q_target=model)

            recorded_iters.append(evolution_step * num_mutations_per_evolution)

            recorded_kls.append(kl_backtrack)

    # Evals: no. generated proposals = #steps * #model.dim
    # NOTE: Evals are comparable with MCMC iterations
    return {'Evals': recorded_iters, 'Backtrack': recorded_kls}


def var_kl_compute(num_evolution_steps: int,
                   num_records: int,
                   variational: DiscreteParticleVariationalInference,
                   model: DiscreteDistributionModel,
                   ):
    recorded_every = int(num_evolution_steps / num_records) + 1
    opad_all_validated_states = OPADDistribution(target_model=model)

    recorded_iters = []
    recorded_kls_var = []  # variational
    recorded_kls_om2 = []

    num_posterior_validations = 0

    for evolution_step in tqdm(range(1, num_evolution_steps + 1)):
        newly_validated_states_scores = variational.evolve_all_states_in_one_dim()
        newly_validated_states = newly_validated_states_scores.generate_all_states()
        num_posterior_validations += variational.num_particles

        for state in newly_validated_states:
            opad_all_validated_states.add_array(state)  # todo this part can be optimized

        if evolution_step % recorded_every == 0:
            kl_var = kl_divergence_discrete(p_empirical=variational.particles_scores, q_target=model)
            kl_om2 = kl_divergence_discrete(p_empirical=opad_all_validated_states, q_target=model)

            # in each iteration, one dimension of each particle is updated:
            recorded_iters.append(evolution_step * variational.num_particles)

            recorded_kls_var.append(kl_var)
            recorded_kls_om2.append(kl_om2)

    # DPVI: Discrete particle variational inference
    # Evals: no. evaluations of the posterior = #steps * #particles
    # NOTE: Evals are comparable with MCMC iterations
    return {'Evals': recorded_iters, 'DPVI': recorded_kls_var, 'OPAD-DPVI': recorded_kls_om2}

def kl_compute(num_mcmc_samples, num_records, sampler, init_state, model):
    current_state = init_state.copy()
    recorded_every = int(num_mcmc_samples / num_records) + 1
    mcmc_hist = ArrayHistogram()
    opad_accepted_only = OPADDistribution(target_model=model)
    opad_with_proposals = OPADDistribution(target_model=model)
    rb_wr = RB_WR_Distrib()

    recorded_iters = []
    recorded_kls_mcmc = []
    recorded_kls_opad = []
    recorded_kls_opad_plus = []
    recorded_kls_rb_wr = []

    for sample_count in tqdm(range(1, num_mcmc_samples + 1)):
        previous_state = current_state.copy()
        current_state, proposed_states, acceptances = sampler.next_sample_proposals_acceptances(current_state)
        assert len(proposed_states) == 1  # only one proposal per sample
        proposed_state = proposed_states[0]
        acceptance = acceptances[0]

        mcmc_hist.add_array(current_state)

        opad_accepted_only.add_array(current_state)

        opad_with_proposals.add_array(current_state)
        opad_with_proposals.add_array(proposed_state)
        rb_wr.add_previous_and_proposed_states(
            previous_state=previous_state,
            proposed_state=proposed_state,
            acceptance_prob=acceptance,
        )

        if sample_count % recorded_every == 0:
            kl_mcmc = kl_divergence_discrete(p_empirical=mcmc_hist, q_target=model)
            kl_opad = kl_divergence_discrete(p_empirical=opad_accepted_only, q_target=model)
            kl_opad_plus = kl_divergence_discrete(p_empirical=opad_with_proposals, q_target=model)
            kl_rb_wr = kl_divergence_discrete(p_empirical=rb_wr, q_target=model)

            recorded_iters.append(sample_count)
            recorded_kls_mcmc.append(kl_mcmc)
            recorded_kls_opad.append(kl_opad)
            recorded_kls_opad_plus.append(kl_opad_plus)
            recorded_kls_rb_wr.append(kl_rb_wr)

    return {
        'Iters': recorded_iters,
        'MCMC': recorded_kls_mcmc,
        'OPAD': recorded_kls_opad,
        'OPAD+': recorded_kls_opad_plus,
        'RB/WR': recorded_kls_rb_wr,
    }


