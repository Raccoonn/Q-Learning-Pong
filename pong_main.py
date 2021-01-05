
import os
import time
import pygame
import numpy as np
import matplotlib.pyplot as plt

from pong_tf_dqn import Agent
from pongEnvironment import pongGame





if __name__ == '__main__':

    show_game = False

    load_networks = False

    train_networks = True

    input_dims = 4

    action_Space = [-50, -25, 0, 25, 50]
    n_actions = len(action_Space)

    p1_Type = 'Agent'
    p2_Type = 'Agent'

    batch_size = 512
    fc1_dims = 512
    fc2_dims = 256

    file_1 = 'p1.h5'
    file_2 = 'p2.h5'

    # Initialize DeepQ agents for the specified players
    if p1_Type == 'Agent':
        agent_1 = Agent(alpha=0.00025, gamma=0.99, epsilon=1, epsilon_dec=0.996, epsilon_end=0.01,
                        batch_size=batch_size, input_dims=input_dims, n_actions=n_actions, mem_size=500000,
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                        fname=file_1)

    if p2_Type == 'Agent':
        agent_2 = Agent(alpha=0.0001, gamma=0.99, epsilon=1, epsilon_dec=0.996, epsilon_end=0.01,
                        batch_size=batch_size, input_dims=input_dims, n_actions=n_actions, mem_size=500000,
                        fc1_dims=fc1_dims, fc2_dims=fc2_dims,
                        fname=file_2)


    # Load networks if specified
    if load_networks == True:
        if p1_Type == 'Agent':
            agent_1.load_model(file_1)

        if p2_Type == 'Agent':
            agent_2.load_model(file_2)
  
        print('\n... Models Loaded ...\n')



    screen_Size = (1000, 600)
    framerate = 60

    env = pongGame(screen_Size, p1_Type, p2_Type, action_Space)

    if show_game == True:
        env.setupWindow(framerate)



    rally_store = []
    avg_rallies = []

    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, 100000):

        done = False
        p1_state, p2_state = env.reset()

        start_time = time.time()
        print('\n')

        while not done:
            print('Playing a game...  ' + p_syms[p_i], end='\r')
            p_i = (p_i+1) % 4

            # If game is being rendered update screen
            if show_game == True:
                e = pygame.event.poll()
                if e.type == pygame.QUIT:
                    break
                elif e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_RETURN:
                        done = False
                        p1_state, p2_state = env.reset()
                        break

                env.render()
                

            # Get actions based on player_Type
            if p1_Type == 'Human':
                p1_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]
            elif p1_Type == 'Agent':
                p1_action = agent_1.choose_action(p1_state)

            if p2_Type == 'Human':
                p2_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]
            elif p2_Type == 'Agent':
                p2_action = agent_2.choose_action(p2_state)


            # Environment takes a step, return observations, reward, and status
            p1_state_, p2_state_, p1_reward, p2_reward, done = env.step(p1_action, p2_action)


            # Train agent DeepQ Networks
            # Assign history values so it doesnt break when memcntr is low
            history_1, history_2 = None, None
            if train_networks == True:
                history_1, history_2 = None, None
                if p1_Type == 'Agent':
                    agent_1.remember(p1_state, p1_action, p1_reward, p1_state_, int(done))
                    history_1 = agent_1.learn()

                if p2_Type == 'Agent':
                    agent_2.remember(p2_state, p2_action, p2_reward, p2_state_, int(done))
                    history_2 = agent_2.learn()


            # Update state for new step
            p1_state = p1_state_
            p2_state = p2_state_



        # Store rallies for plot
        rallies = env.ball.rallies
        rally_store.append(rallies)
        avg_rallies.append(np.mean(rally_store))

        plt.figure()
        plt.plot(list(range(episode)), rally_store)
        plt.savefig('Rallies.png')
        plt.close()

        print('\n\n\nGame: ', episode)
        print('Rallies: ', rallies)
        print('Time Elapsed: ', round(time.time()-start_time, 2), '\n')
        if history_1 != None and history_2 != None:
            print('\nTraining History:\n', 'Agent 1: ', history_1.history, '\n',
                                           'Agent 2: ', history_2.history, '\n')


        # Save networks if they are also being trained
        try:
            if episode % 1 == 0 and train_networks == True:
                if p1_Type == 'Agent':
                    agent_1.save_model()

                if p2_Type == 'Agent':
                    agent_2.save_model()
                print('\n... Models Saved ...\n')
        except:
            pass



        # Load the superior model as both agents if it wins enough games
        # Reset win counter to reevaluate later
        # if episode > 1 and episode % 100 == True:
        #     win_diff = game_wins[0] - game_wins[1]
        #     if win_diff > 25:
        #         agent_2.load_model(file_1)
        #         game_wins = [0, 0]
        #         print('\n... Agent 2 Swapped ...\n')
        #     elif win_diff < -25:
        #         agent_1.load_model(file_2)
        #         game_wins = [0, 0]
        #         print('\n... Agent 1 Swapped ...\n')



    pygame.quit()