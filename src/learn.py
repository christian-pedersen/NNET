from tensorforce.execution import Runner
from tensorforce.agents import Agent, PPOAgent
from Env import NetworkEnv
import numpy as np

environment = NetworkEnv(max_solver_step=20000, number_steps_execution=10, q_off=10)


agent = Agent.create(agent='ppo',
                        environment=environment,
                        network='auto',
                        max_episode_timesteps=2000,
                        batch_size=10,
                        learning_rate=1e-3, subsampling_fraction=0.2,
                        likelihood_ratio_clipping=0.2, discount=0.99, predict_terminal_values=False,
                        exploration=0.01, variable_noise=0.01,
                        l2_regularization=0.01, entropy_regularization=0.01)


runner = Runner(agent=agent,
                environment=environment,
                max_episode_timesteps=2000)

runner.run(num_episodes=1000)

runner.close()
agent.close()
environment.close()
