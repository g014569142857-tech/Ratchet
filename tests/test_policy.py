"""Tests for the Thompson Sampling policy (the RL heart of Ratchet).

These verify real statistical behaviour, not mock call counts.
"""

import pytest

from ratchet.core.policy import ThompsonPolicy


@pytest.fixture()
def policy():
    return ThompsonPolicy(seed=42)


class TestArmRegistration:
    def test_arms_start_with_uniform_prior(self, policy):
        policy.register("a")
        assert policy.arms["a"].alpha == pytest.approx(1.0)
        assert policy.arms["a"].beta == pytest.approx(1.0)

    def test_registering_same_arm_twice_is_idempotent(self, policy):
        policy.register("a")
        policy.register("a")
        assert len(policy.arms) == 1

    def test_list_arms_returns_registered_names(self, policy):
        policy.register("a")
        policy.register("b")
        assert set(policy.list_arms()) == {"a", "b"}


class TestSelection:
    def test_select_returns_a_registered_arm(self, policy):
        policy.register("a")
        policy.register("b")
        assert policy.select() in {"a", "b"}

    def test_select_raises_when_no_arms(self, policy):
        with pytest.raises(ValueError, match="no arms"):
            policy.select()

    def test_select_is_reproducible_with_same_seed(self):
        p1, p2 = ThompsonPolicy(seed=7), ThompsonPolicy(seed=7)
        for p in (p1, p2):
            p.register("a")
            p.register("b")
            p.register("c")
        assert [p1.select() for _ in range(20)] == [p2.select() for _ in range(20)]

    def test_select_explores_rarely_chosen_arms(self, policy):
        # Production change that would break this: replacing the Beta sample in
        # select() with a greedy argmax over mean(). Thompson Sampling must keep
        # a non-zero chance of playing an untried arm.
        #
        # With "a" at Beta(201,1) the untried arm "b" wins with probability
        # 1/202 per draw, so 5000 draws make a false failure ~1e-11 likely.
        # 500 draws would have been flaky (~8% chance of never picking "b").
        policy.register("a")
        policy.register("b")
        for _ in range(200):
            policy.update("a", 1.0)
        picks = {policy.select() for _ in range(5000)}
        assert "b" in picks

    def test_exploration_rate_matches_beta_posterior(self, policy):
        # The low-but-nonzero exploration rate is the actual guarantee; assert it
        # quantitatively so a broken sampler cannot pass by accident.
        policy.register("a")
        policy.register("b")
        for _ in range(200):
            policy.update("a", 1.0)
        draws = 20000
        b_picks = sum(1 for _ in range(draws) if policy.select() == "b")
        observed_rate = b_picks / draws
        assert 0.002 < observed_rate < 0.012  # expected ~1/202 = 0.00495


class TestRewardUpdate:
    def test_positive_reward_increases_alpha(self, policy):
        policy.register("a")
        policy.update("a", 1.0)
        assert policy.arms["a"].alpha == pytest.approx(2.0)
        assert policy.arms["a"].beta == pytest.approx(1.0)

    def test_zero_reward_increases_beta(self, policy):
        policy.register("a")
        policy.update("a", 0.0)
        assert policy.arms["a"].alpha == pytest.approx(1.0)
        assert policy.arms["a"].beta == pytest.approx(2.0)

    def test_partial_reward_splits_between_alpha_and_beta(self, policy):
        policy.register("a")
        policy.update("a", 0.5)
        assert policy.arms["a"].alpha == pytest.approx(1.5)
        assert policy.arms["a"].beta == pytest.approx(1.5)

    def test_reward_is_clamped_to_unit_interval(self, policy):
        policy.register("a")
        policy.update("a", 5.0)
        policy.update("a", -5.0)
        assert policy.arms["a"].alpha == pytest.approx(2.0)
        assert policy.arms["a"].beta == pytest.approx(2.0)

    def test_update_unknown_arm_registers_it_first(self, policy):
        policy.update("ghost", 1.0)
        assert "ghost" in policy.arms

    def test_winning_arm_selected_most_often(self, policy):
        # After strong evidence, the better arm should dominate selection.
        policy.register("good")
        policy.register("bad")
        for _ in range(100):
            policy.update("good", 1.0)
            policy.update("bad", 0.0)
        picks = [policy.select() for _ in range(1000)]
        assert picks.count("good") > picks.count("bad")


class TestMeanAndPersistence:
    def test_mean_reflects_reward_history(self, policy):
        policy.register("a")
        policy.update("a", 1.0)
        policy.update("a", 1.0)
        policy.update("a", 0.0)
        # alpha=3, beta=2 -> mean = 3/5
        assert policy.mean("a") == pytest.approx(0.6)

    def test_state_roundtrips_through_dict(self):
        policy = ThompsonPolicy(seed=1)
        policy.register("a")
        policy.update("a", 1.0)
        restored = ThompsonPolicy.from_dict(policy.to_dict(), seed=1)
        assert restored.arms["a"].alpha == pytest.approx(2.0)
        assert restored.arms["a"].beta == pytest.approx(1.0)
