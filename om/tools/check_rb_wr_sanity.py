import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from om.tools.rb_wr import RB_WR_Distrib


def _assert_close(actual, expected, message, atol=1e-12):
    if not np.isclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}: expected {expected}, got {actual}")


def test_single_step_weight_split():
    rb = RB_WR_Distrib()
    previous_state = np.array([0])
    proposed_state = np.array([1])

    rb.add_previous_and_proposed_states(
        previous_state=previous_state,
        proposed_state=proposed_state,
        acceptance_prob=0.25,
    )

    _assert_close(rb.calc_unnormalized_prob(previous_state), 0.75, "wrong weight for previous_state")
    _assert_close(rb.calc_unnormalized_prob(proposed_state), 0.25, "wrong weight for proposed_state")
    _assert_close(rb.calc_normalization_factor(), 1.0, "wrong normalization factor after one step")


def test_repeated_steps_accumulate_correctly():
    rb = RB_WR_Distrib()
    state_a = np.array([0])
    state_b = np.array([1])

    rb.add_previous_and_proposed_states(state_a, state_b, 0.25)
    rb.add_previous_and_proposed_states(state_a, state_b, 0.60)

    _assert_close(rb.calc_unnormalized_prob(state_a), 0.75 + 0.40, "wrong accumulated weight for state_a")
    _assert_close(rb.calc_unnormalized_prob(state_b), 0.25 + 0.60, "wrong accumulated weight for state_b")
    _assert_close(rb.calc_normalization_factor(), 2.0, "wrong normalization factor after two steps")


def test_expected_value_matches_rb_formula():
    rb = RB_WR_Distrib()
    state_a = np.array([0])
    state_b = np.array([1])

    rb.add_previous_and_proposed_states(state_a, state_b, 0.25)
    rb.add_previous_and_proposed_states(state_a, state_b, 0.60)

    expected_mean = ((0.75 + 0.40) * 0 + (0.25 + 0.60) * 1) / 2.0
    actual_mean = rb.calc_expected_value(func=lambda x: x[0])
    _assert_close(actual_mean, expected_mean, "wrong RB/WR expected value")


if __name__ == "__main__":
    test_single_step_weight_split()
    test_repeated_steps_accumulate_correctly()
    test_expected_value_matches_rb_formula()
    print("RB/WR sanity checks passed.")
