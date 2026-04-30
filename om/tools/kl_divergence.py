import numpy as np

from om.models.model import DiscreteDistributionModel


def kl_divergence_discrete(p_empirical: DiscreteDistributionModel, q_target: DiscreteDistributionModel):
    p_support = p_empirical.generate_all_states()
    p_normalization_factor = p_empirical.calc_normalization_factor()

    q_normalization_factor = q_target.calc_normalization_factor()
    log_p_normalization_factor = np.log(p_normalization_factor)
    log_q_normalization_factor = np.log(q_normalization_factor)

    kl = 0
    for state in p_support:
        p_unnormalized_prob = p_empirical.calc_unnormalized_prob(state)
        if p_unnormalized_prob <= 0:
            continue

        log_px = np.log(p_unnormalized_prob) - log_p_normalization_factor
        log_qx = -q_target.calc_neg_log_unnormalized_prob(state) - log_q_normalization_factor
        px = np.exp(log_px)
        kl += px * (log_px - log_qx)
    return kl

