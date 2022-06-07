from stable_baselines.a2c.a2c import A2C
from stable_baselines.ppo2.ppo2 import PPO2
from envs.causal_env_v0 import CausalEnv_v0, ABconj, ACconj, BCconj, Adisj, Bdisj, Cdisj
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import argparse
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from functools import partialmethod



def partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def _get_environments(holdout_strategy: str, quiz_disabled_steps: int = -1, reward_structure: str = 'quiz'):

    def make_env(rank, seed=0, qd=-1):
        def _init():
            if holdout_strategy == 'none':
                env = CausalEnv_v0({"reward_structure": reward_structure, "quiz_disabled_steps": qd})
            elif holdout_strategy == 'conjunctive_train':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                        BCconj,
                    ]})
            elif holdout_strategy == 'disjunctive_train':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        Adisj,
                        Bdisj,
                        Cdisj,
                    ]})
            elif holdout_strategy == 'conjunctive_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                    ]
                })
            elif holdout_strategy == 'disjunctive_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        Adisj,
                        Bdisj,
                    ]
                })
            elif holdout_strategy == 'both_loo':
                env = CausalEnv_v0({
                    "reward_structure": reward_structure,
                    "quiz_disabled_steps": qd,
                    "hypotheses": [
                        ABconj,
                        ACconj,
                        Adisj,
                        Bdisj,
                    ]
                })
            else:
                raise ValueError('Unsupported holdout strategy: {}'.format(holdout_strategy))
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init

    def vec_env(n=4, qd=-1):
        env = DummyVecEnv([make_env(i, qd=qd) for i in range(n)])
        return env

    env = vec_env(4, qd=quiz_disabled_steps)

    if holdout_strategy == 'none':
        eval_env = CausalEnv_v0({"reward_structure": reward_structure})
    elif holdout_strategy == 'conjunctive_train':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Adisj,
                Bdisj,
                Cdisj,
            ]})
    elif holdout_strategy == 'disjunctive_train':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                ABconj,
                ACconj,
                BCconj,
            ]})
    elif holdout_strategy == 'conjunctive_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                BCconj,
            ]
        })
    elif holdout_strategy == 'disjunctive_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Cdisj,
            ]
        })
    elif holdout_strategy == 'both_loo':
        eval_env = CausalEnv_v0({
            "reward_structure": reward_structure,
            "hypotheses": [
                Cdisj,
                BCconj,
            ]
        })
    else:
        raise ValueError('Unsupported holdout strategy: {}'.format(holdout_strategy))

    return env, eval_env


def main(args):
    env, eval_env = _get_environments(args.holdout_strategy, args.quiz_disabled_steps, args.reward_structure)

    # Stop training when the model reaches the reward threshold
    # callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=3, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=None, eval_freq=5000, verbose=1, n_eval_episodes=100)

    if args.policy == 'mlp':
        policy = MlpPolicy
        save_name = f'{args.alg}_{args.policy}'
    elif args.policy == 'mlp_lstm':
        policy = partialclass(MlpLstmPolicy, n_lstm=args.lstm_units)
        save_name = f'{args.alg}_{args.policy}_{args.lstm_units}'
    elif args.policy == 'mlp_lnlstm':
        policy = partialclass(MlpLnLstmPolicy, n_lstm=args.lstm_units)
        save_name = f'{args.alg}_{args.policy}_{args.lstm_units}'
    else:
        raise ValueError('Unsupported policy: {}'.format(args.policy))

    save_name += ('_qd=' + str(args.quiz_disabled_steps)) if args.quiz_disabled_steps > 0 else ''
    save_name += ('_rs=' + str(args.reward_structure))
    save_name += ('_hs=' + str(args.holdout_strategy))

    if args.alg == 'a2c':
        model = A2C(policy, env, verbose=1, tensorboard_log="./logs/{}".format(save_name))
    elif args.alg == 'ppo2':
        model = PPO2(policy, env, verbose=1, tensorboard_log="./logs/{}".format(save_name))
    else:
        raise ValueError('Unsupported algorithm: {}'.format(args.alg))

    model.learn(
        total_timesteps=int(args.num_steps),
        callback=eval_callback
    )
    model.save(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--alg', type=str, default='a2c', help='Algorithm to use')
    parser.add_argument('--policy', type=str, default='mlp', help='Policy to use')
    parser.add_argument('--lstm_units', type=int, default=256, help='Number of LSTM units')
    parser.add_argument('--num_steps', type=int, default=int(3000000), help='Number of training steps')
    parser.add_argument('--quiz_disabled_steps', type=int, default=-1, help='Number of quiz disabled steps (-1 for no forced exploration)')
    parser.add_argument('--holdout_strategy', type=str, default='none', help='Holdout strategy')
    parser.add_argument('--reward_structure', type=str, default='quiz', help='Reward structure')
    args = parser.parse_args()
    main(args)
