import gym
import numpy as np
import itertools
import random

from gym import spaces

from typing import Dict, Any, Tuple, List, Set


class Hypothesis():
    @property
    def blickets(self) -> Set[int]:
        ...

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        ...


### Base Conjunctive Hypotheses for 3 blickets ###


class ConjunctiveHypothesis:
    blickets = None
    structure = "conjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return all(c in blickets for c in cls.blickets)  # type: ignore


class ABconj(ConjunctiveHypothesis):
    blickets = set([0, 1])


class ACconj(ConjunctiveHypothesis):
    blickets = set([0, 2])


class BCconj(ConjunctiveHypothesis):
    blickets = set([1, 2])


class ABCconj(ConjunctiveHypothesis):
    blickets = set([0, 1, 2])


### Base Disjunctive Hypotheses for 3 blickets ###


class DisjunctiveHypothesis:
    blickets = None
    structure = "disjunctive"

    @classmethod
    def test(cls, blickets: Set[int]) -> bool:
        return any(c in blickets for c in cls.blickets)  # type: ignore


class Adisj(DisjunctiveHypothesis):
    blickets = set([0])


class Bdisj(DisjunctiveHypothesis):
    blickets = set([1])


class Cdisj(DisjunctiveHypothesis):
    blickets = set([2])


class ABdisj(DisjunctiveHypothesis):
    blickets = set([0, 1])


class ACdisj(DisjunctiveHypothesis):
    blickets = set([0, 2])


class BCdisj(DisjunctiveHypothesis):
    blickets = set([1, 2])


class ABCdisj(DisjunctiveHypothesis):
    blickets = set([0, 1, 2])


class CausalEnv_v0(gym.Env):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        """
        Representation of the Blicket environment, based on the exeperiments presente in the causal learning paper.

        Args:
            env_config (Dict[str, Any]): A dictionary representing the environment configuration.
                Keys: Values (Default)


        Action Space:
            => [Object A (on/off), Object B state (on/off), Object C state (on/off)]

        """

        self._n_blickets = env_config.get("n_blickets", 3)  # Start with 3 blickets
        self._reward_structure = env_config.get("reward_structure", "baseline")  # Start with baseline reward structure
        self._symbolic = env_config.get("symbolic", True)  # Start with symbolic observation space

        if self._reward_structure not in ("baseline", "quiz", "quiz-type", "quiz-typeonly"):
            raise ValueError(
                "Invalid reward structure: {}, must be one of (baseline, quiz, quiz-type, quiz-typeonly)".format(self._reward_structure)
            )

        # Setup penalties and reward structures
        self._add_step_reward_penalty = env_config.get("add_step_reward_penalty", False)
        self._add_detector_state_reward_for_quiz = env_config.get("add_detector_state_reward_for_quiz", False)
        self._step_reward_penalty = env_config.get("step_reward_penalty", 0.01)
        self._detector_reward = env_config.get("detector_reward", 1)
        self._quiz_positive_reward = env_config.get("quiz_positive_reward", 1)
        self._quiz_negative_reward = env_config.get("quiz_negative_reward", -1)
        self._max_steps = env_config.get("max_steps", 20)
        self._blicket_dim = env_config.get("blicket_dim", 3)
        self._quiz_disabled_steps = env_config.get("quiz_disabled_steps", -1)

        # Gym environment setup
        self.action_space = (
            spaces.MultiDiscrete([2] * self._n_blickets)
            if 'quiz' not in self._reward_structure
            else spaces.MultiDiscrete([2] * (self._n_blickets + 1))
        )

        if self._symbolic:
            if 'quiz' in self._reward_structure:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 2,), dtype=np.float32
                )  # The state of all of the blickets, plus the state of the detector plus the quiz indicator
            else:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1,), dtype=np.float32
                )  # The state of all of the blickets, plus the state of the detector
        else:
            self._blicket_cmap = {i: np.random.uniform(self._blicket_dim) for i in range(self._n_blickets)}
            if 'quiz' in self._reward_structure:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 2, self._blicket_dim), dtype=np.float32
                )
            else:
                self.observation_space = spaces.Box(
                    low=0, high=1, shape=(self._n_blickets + 1, self._blicket_dim), dtype=np.float32
                )

        # Add the hypothesis spaces
        self._hypotheses: List[Hypothesis] = env_config.get(
            "hypotheses",
            [
                ABconj,
                ACconj,
                BCconj,
                # ABCconj,
                Adisj,
                Bdisj,
                Cdisj,
                # ABdisj,
                # ACdisj,
                # BCdisj,
                # ABCdisj,
            ],
        )

        # Setup the environment by default
        self._current_gt_hypothesis = random.choice(self._hypotheses)

        self._steps = 0
        self._observations = 0
        self._quiz_step = None

        self.reset()

    def reset(self) -> np.ndarray:
        # Reset the color map for the blickets
        self._current_gt_hypothesis = random.choice(self._hypotheses)
        if "quiz" in self._reward_structure:
            # Reset the color map
            self._blicket_cmap = {i: np.random.uniform(self._blicket_dim) for i in range(self._n_blickets)}

        # Reset the step trackers
        self._steps = 0
        self._quiz_step = None

        # Get the baseline observation
        return self._get_observation(blickets=np.zeros(self._n_blickets))

    def _get_baseline_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._symbolic:
            return np.concatenate(
                [
                    blickets,
                    np.array([1]) if self._get_detector_state(blickets) else np.array([0]),
                ],
                axis=0,
            ) # type: ignore
        return np.concatenate(
            [
                np.stack(
                    [
                        self._blicket_cmap[i] if blickets[i] == 0 else np.zeros(self._blicket_dim)
                        for i in range(self._n_blickets)
                    ],
                    axis=0,
                ),
                np.ones((1, self._blicket_dim))
                if self._get_detector_state(blickets)
                else np.zeros((1, self._blicket_dim)),
            ],
            axis=0,
        ) # type: ignore

    def _get_quiz_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._quiz_step is not None:
            if self._symbolic:
                return np.concatenate(
                    [
                        np.array([1 if self._quiz_step == i else 0 for i in range(self._n_blickets)]),
                        np.array([0]),
                        np.array([1]),
                    ],
                    axis=0,
                )
            else:
                return np.concatenate(
                    [
                        np.stack(
                            [
                                self._blicket_cmap[i] if self._quiz_step == i else np.zeros(self._blicket_dim)
                                for i in range(self._n_blickets)
                            ],
                            axis=0,
                        ),
                        np.zeros((1, self._blicket_dim)),
                        np.ones((1, self._blicket_dim)),
                    ],
                    axis=0,
                )  # type: ignore

        if self._symbolic:
            return np.concatenate(
                [
                    blickets,  # Blickets
                    np.array([1] if self._get_detector_state(blickets) else [0]),  # Detector state
                    np.array([0]),  # Quiz indicator
                ],
                axis=0,
            )  # type: ignore
        return np.concatenate(
            [
                np.stack(
                    [
                        self._blicket_cmap[i] if blickets[i] == 1 else np.zeros(self._blicket_dim)
                        for i in range(self._n_blickets)
                    ],
                    axis=0,
                ),
                np.zeros((1, self._blicket_dim)),  # Detector state
                np.zeros((1, self._blicket_dim)),  # Quiz indicator
            ],
            axis=0,
        )  # type: ignore

    def _get_observation(self, blickets: np.ndarray) -> np.ndarray:
        if self._reward_structure == "baseline":
            return self._get_baseline_observation(blickets)
        elif 'quiz' in self._reward_structure:
            return self._get_quiz_observation(blickets)
        raise ValueError("Invalid reward structure: {}".format(self._reward_structure))

    def _get_detector_state(self, active_blickets: np.ndarray) -> bool:
        blickets_on = set()
        for i in range(len(active_blickets)):
            if active_blickets[i] == 1:
                blickets_on.add(i)
        return self._current_gt_hypothesis.test(blickets_on)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:

        observation, reward, done, info = (None, 0, False, {})

        # Generate the observations and reward
        if self._reward_structure == "baseline":
            observation = self._get_baseline_observation(action[: self._n_blickets])

            # Get the reward
            if self._add_step_reward_penalty:
                reward -= self._step_reward_penalty
            if self._get_detector_state(action[: self._n_blickets]):
                reward += self._detector_reward
            done = self._steps > self._max_steps

        elif 'quiz' in self._reward_structure:
            if self._quiz_step is not None:

                # Get the reward
                if self._reward_structure == 'quiz':
                    reward = (
                        self._quiz_positive_reward
                        if (
                            action[self._quiz_step] == 1
                            and self._quiz_step in self._current_gt_hypothesis.blickets
                            or action[self._quiz_step] == 0
                            and self._quiz_step not in self._current_gt_hypothesis.blickets
                        )
                        else self._quiz_negative_reward
                    )
                elif self._reward_structure in ('quiz-type', 'quiz-typeonly'):
                    if self._quiz_step < self._n_blickets:
                        reward = (
                        self._quiz_positive_reward
                        if (
                            action[self._quiz_step] == 1
                            and self._quiz_step in self._current_gt_hypothesis.blickets
                            or action[self._quiz_step] == 0
                            and self._quiz_step not in self._current_gt_hypothesis.blickets
                        )
                        else self._quiz_negative_reward
                    )
                    else:
                        reward = (
                            0.5
                            if (action[0] == 0 and issubclass(self._current_gt_hypothesis, ConjunctiveHypothesis) or action[0] == 1 and issubclass(self._current_gt_hypothesis, DisjunctiveHypothesis))
                            else -0.5
                        )



                # We're in the quiz phase.
                self._quiz_step += 1
                observation = self._get_quiz_observation(action[: self._n_blickets])

                if self._reward_structure in ('quiz-type', 'quiz-typeonly'):
                    if self._quiz_step > self._n_blickets:
                        done = True
                elif self._reward_structure == 'quiz':
                    if self._quiz_step >= self._n_blickets:
                        done = True
            else:
                # Check the action to see if we should go to quiz phase
                if self._steps > self._max_steps or (action[-1] == 1 and self._steps > self._quiz_disabled_steps):

                    if self._add_step_reward_penalty:
                        reward -= self._step_reward_penalty
                    if self._add_detector_state_reward_for_quiz and self._get_detector_state(
                        action[: self._n_blickets]
                    ):
                        reward += self._detector_reward

                    # Go to quiz phase
                    self._quiz_step = 0 if self._reward_structure != 'quiz-typeonly' else self._n_blickets
                    observation = self._get_quiz_observation(action[: self._n_blickets])
                else:
                    # We're in the standard action phase
                    observation = self._get_quiz_observation(action[: self._n_blickets])


        assert observation is not None

        self._steps += 1
        return observation, reward, done, info
