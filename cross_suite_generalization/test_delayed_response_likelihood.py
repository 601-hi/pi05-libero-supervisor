import unittest

import numpy as np

from delayed_response_likelihood import (
    episode_alignment_quality,
    fit_lag_bank,
    lag_marginal_nll,
)


class DelayedResponseLikelihoodTest(unittest.TestCase):
    def test_fit_recovers_delayed_response_component(self):
        rng = np.random.default_rng(7)
        command = rng.normal(size=(200, 3))
        response = np.zeros_like(command)
        response[2:] = 0.4 * command[:-2] + rng.normal(scale=0.01, size=(198, 3))
        bank = fit_lag_bank([(command, response)], [0, 1, 2, 3])
        best = max(bank, key=lambda item: item.weight)
        self.assertEqual(best.lag_steps, 2)
        self.assertAlmostEqual(best.gain, 0.4, places=2)

    def test_marginal_likelihood_prefers_matching_response(self):
        rng = np.random.default_rng(8)
        command = rng.normal(size=(200, 3))
        response = np.zeros_like(command)
        response[1:] = 0.5 * command[:-1] + rng.normal(scale=0.02, size=(199, 3))
        bank = fit_lag_bank([(command, response)], [0, 1, 2])
        history = command[20:23]
        expected = 0.5 * history[-2]
        self.assertLess(
            lag_marginal_nll(history, expected, bank),
            lag_marginal_nll(history, expected + 2.0, bank),
        )

    def test_unresponsive_excited_episode_is_rejected(self):
        rng = np.random.default_rng(9)
        command = rng.normal(size=(80, 3))
        response = rng.normal(scale=0.001, size=(80, 3))
        result = episode_alignment_quality(command, response, [0, 1, 2, 3])
        self.assertFalse(result["accepted_for_identification"])

    def test_deployable_bank_rejects_acausal_negative_lag(self):
        command = np.ones((10, 2))
        with self.assertRaisesRegex(ValueError, "causal"):
            fit_lag_bank([(command, command.copy())], [-1, 0, 1])


if __name__ == "__main__":
    unittest.main()
