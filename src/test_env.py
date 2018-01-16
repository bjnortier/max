import gym

gym.envs.register(
    id='Metalhead-v1',
    entry_point='envs.metalhead_v1:MetalheadEnvV1',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
env = gym.make('Metalhead-v1')
env.reset()
env.render()
input("Press Enter to continue...")
