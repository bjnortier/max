import gym

gym.envs.register(
    id='FullCheetah-v1',
    entry_point='envs.full_cheetah_v1:FullCheetahEnvV1',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
env = gym.make('FullCheetah-v1')
env.reset()
env.render()
input("Press Enter to continue...")
