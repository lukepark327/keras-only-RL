""" References
https://github.com/rlcode/reinforcement-learning/blob/master/3-atari/1-breakout/breakout_dqn.py
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py
"""

import numpy as np
import gym
import random
from skimage.color import rgb2gray
from skimage.transform import resize

from agents.DQN import DQNAgent
from agents.Double import DDQNAgent
from agents.Dueling import DuelingAgent, D3QNAgent

import arguments


def pre_processing(observe):
    processed_observe = np.uint8(resize(  # float --> integer (to reduce the size of replay memory)
        rgb2gray(observe),  # 210*160*3(color) --> 84*84(mono)
        (84, 84),
        mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    # arguments
    args = arguments.parser()
    EPISODES = args.e  # Number of episodes
    Double = args.double  # Double DQN
    Dueling = args.dueling  # Double DQN

    print("> Setting:", args)

    env = gym.make('BreakoutDeterministic-v4')

    if Double is True:
        if Dueling is True:
            agent = D3QNAgent(action_size=3)
        else:
            agent = DDQNAgent(action_size=3)
    if Dueling is True:  # Double is False
        agent = DuelingAgent(action_size=3)
    else:
        agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0
    for e in range(EPISODES):
        done, dead = False, False
        step, score, start_life = 0, 0, 5  # 1 episode = 5 lives

        observe = env.reset()

        # Do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            if agent.render:
                env.render()

            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)

            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            observe, reward, done, info = env.step(real_action)

            # pre-process the observation --> history
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            agent.avg_q_max += np.amax(agent.model.predict(np.float32(history / 255.))[0])

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1.)

            agent.append_sample(history, action, reward, next_history, dead)
            agent.learn()

            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                print("episode:", e,
                      "\tglobal_step:", global_step,
                      "\tmemory_len: ", len(agent.memory),
                      "\tscore:", score,
                      "\tepsilon:", round(agent.epsilon, 6),
                      "\tavg_q_max:", round(agent.avg_q_max / float(step), 6),
                      "\tavg_loss:", round(agent.avg_loss / float(step), 6))

                agent.avg_q_max, agent.avg_loss = 0.0, 0.0

        if e % 1000 == 0:
            # TODO: mkdir
            # agent.save_model("./save_model/breakout_dqn.h5")
            pass
