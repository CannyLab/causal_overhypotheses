
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../envs'))
import argparse
from stable_baselines import A2C, PPO2
from causal_env_v0 import CausalEnv_v0
import tqdm
import numpy as np
import pickle

def main(args):
    print('Loading model...')
    # Load the model
    if args.model_path is None:
        model = None
    elif 'a2c' in args.model_path:
        model = A2C.load(args.model_path)
    elif 'ppo2' in args.model_path:
        model = PPO2.load(args.model_path)
    else:
        raise ValueError('Unknown model type {}'.format(args.model))

    # Create an environment
    env = CausalEnv_v0({
        "reward_structure":  "quiz-typeonly",
        "quiz_disabled_steps": args.quiz_disabled_steps,
    })

    # Roll out the environment for k trajectories
    print('Collecting Trajectories...')
    trajectories = []
    for i in tqdm.tqdm(range(args.num_trajectories)):
        # Reset the environment
        obs = env.reset()

        # Roll out the environment for n steps
        steps = []
        for j in range(args.max_steps):
            # Get the action from the model
            if model is not None:
                # Because the environment is stacked, we have to extract x2
                action = model.predict(np.stack([obs, obs, obs, obs]), deterministic=True)[0][0]
            else:
                action = env.action_space.sample()


            # Step the environment
            n_obs, reward, done, info = env.step(action)

            steps.append((obs, action, reward, n_obs, done))
            obs = n_obs

            # Check if the episode has ended
            if done:
                break

        observations = np.stack([step[0] for step in steps])
        next_obeservations = np.stack([step[3] for step in steps])
        actions = np.stack([step[1] for step in steps])
        rewards = np.stack([step[2] for step in steps])
        terminals = np.stack([step[4] for step in steps])
        trajectories.append({
            'observations': observations,
            'next_observations': next_obeservations,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals
        })

    # Save the trajectories
    print('Saving Trajectories...')
    with open(args.output_path, 'wb') as f:
        pickle.dump(trajectories, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect Trajectories from Causal Environments')
    parser.add_argument('--env', type=str, default='CausalEnv_v0', help='Environment to use')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model')
    parser.add_argument('--num_trajectories', type=int, default=10000, help='Number of trajectories to collect')
    parser.add_argument('--max_steps', type=int, default=30, help='Maximum number of steps per trajectory')
    parser.add_argument('--quiz_disabled_steps', type=int, default=-1, help='Number of steps to disable quiz')
    parser.add_argument('--output_path', type=str, default='trajectories.pkl', help='Path to output file')

    args = parser.parse_args()

    main(args)
