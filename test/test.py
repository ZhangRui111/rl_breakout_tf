a = [1, 2, 3]
print(type(a))
a.insert(0, None)
print(a)


# import gym
# from shared.utils import my_print
#
# my_print('target_params_replaced', '-')
#
# env = gym.make('Breakout-v0')
# env.reset()
# for _ in range(10000):
#     env.render()
#     action = env.action_space.sample()
#     env.step(action)  # take a random action

# import retro
# import time
#
# # for game in retro.data.list_games():
# #     print(game, retro.data.list_states(game))
#
# # env = retro.make(game='StarGunner-Atari2600')
# env = retro.make(game='AdventuresOfStarSaver-GameBoy')
# env.reset()
# for i in range(10000):
#     time.sleep(0.1)
#     # if i % 2 == 0:
#     #     action = [1, 1, 1, 1, 1, 1, 1, 1, 1]
#     # else:
#     #     action = [0, 1, 1, 1, 1, 1, 1, 1, 0]
#     action = env.action_space.sample()
#     print(action)
#     _obs, _rew, done, _info = env.step(action)
#     env.render()
#     if done:
#         env.reset()
