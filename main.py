
import os
import time
import pygame
import numpy as np

from dqn import Agent
from pongEnvironment import pongGame

import training_tools





if __name__ == '__main__':

    show_game = True

    load_networks = True

    train_networks = False



    filename = input('\nInput load filename:    ')

    # Constants and hyperparameters
    # Others loaded from file for better version control
    input_shape = 6

    info, env_info, save_dir, p1_type, p2_type, \
        mem_size, action_space, batch_size, h1_dims = training_tools.load_setup(filename)

    file_1 = save_dir + 'p1.h5'
    file_2 = save_dir + 'p2.h5'


    # Create storage directory
    try:
        os.mkdir(save_dir)
        print('\n... Training Directory Created ...\n')
    except:
        print('\nERROR: Directory already exists, select new name.\n')
        check = input('\nPress Y/y to continue:     ')
        if check not in ('Y', 'y'):
            quit()


    agent_info = [info, env_info]

    with open(save_dir + 'info.txt', 'w') as f:
        for l in agent_info:
            f.write(l + '\n')
    




    # Initialize Deep Q agents and memories
    if p1_type == 'Agent':
        agent_1 = Agent(lr=0.001, gamma=0.99, epsilon=0, epsilon_dec=0.9996, epsilon_min=0.01,
                        input_shape=input_shape, h1_dims=h1_dims, action_space=action_space,
                        fname=file_1)

        if train_networks == True:
            memory_1 = training_tools.ReplayBuffer(mem_size, input_shape, len(action_space))



    if p2_type == 'Agent':
        agent_2 = Agent(lr=0.001, gamma=0.99, epsilon=0, epsilon_dec=0.9996, epsilon_min=0.01,
                        input_shape=input_shape, h1_dims=h1_dims, action_space=action_space,
                        fname=file_2)

        if train_networks == True:
            memory_2 = training_tools.ReplayBuffer(mem_size, input_shape, len(action_space))

    





    # Load networks if specified
    if load_networks == True:
        if p1_type == 'Agent':
            agent_1.load_model(file_1)

        if p2_type == 'Agent':
            agent_2.load_model(file_2)
  
        print('\n... Models Loaded ...\n')



    # Initialize Pong environment
    screen_Size = (1000, 600)

    env = pongGame(screen_Size, p1_type, p2_type, action_space)

    if show_game == True:
        framerate = 60
        env.setupWindow(framerate)



    # Initialize storage lists for plots
    p1_rwd_store = []
    p2_rwd_store = []
    p1_avg = []
    p2_avg = []

    rally_store = []
    avg_rallies = []

    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, 100000):

        done = False
        frame = 0
        p1_state, p2_state = env.reset()

        p1_tot = 0
        p2_tot = 0

        start_time = time.time()
        print('\n')

        while not done:
            print('Playing a game...  ' + p_syms[p_i], end='\r')
            p_i = (p_i+1) % 4

            frame += 1

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
            if p1_type == 'Human':
                p1_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]

            elif p1_type == 'Agent':
                p1_action = agent_1.choose_action(p1_state)


            if p2_type == 'Human':
                p2_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]

            elif p2_type == 'Agent':
                p2_action = agent_2.choose_action(p2_state)


            # Environment takes a step, return observations, reward, and status
            p1_state_, p2_state_, p1_reward, p2_reward, done = env.step(p1_action, p2_action)

            # Update memory objects with states for each player
            if train_networks == True:
                memory_1.store_transition(p1_state, p1_action, p1_reward, p1_state_, int(done))
                memory_2.store_transition(p2_state, p2_action, p2_reward, p2_state_, int(done))
                 

            # Train agent DeepQ Networks
            # Assign history values so it doesnt break when memcntr is low
            history_1, history_2 = None, None
            if train_networks == True and memory_1.mem_cntr > batch_size:
                history_1, history_2 = None, None
   
                if p1_type == 'Agent':
                    history_1 = agent_1.learn(batch_size, memory_1.sample_buffer(batch_size))

                if p2_type == 'Agent':
                    history_2 = agent_2.learn(batch_size, memory_2.sample_buffer(batch_size))


            # Update state for new step
            p1_state = p1_state_
            p2_state = p2_state_

            p1_tot += p1_reward
            p2_tot += p2_reward


            # Save networks if they are also being trained
            # Moved inside game loop, will save every 1000 frames
            # Games are getting longer and need to be saved in progress
            try:
                if episode % 1 == 0 and train_networks == True and frame % 1000 == 0:
                    if p1_type == 'Agent':
                        agent_1.save_model()

                    if p2_type == 'Agent':
                        agent_2.save_model()
                    print('\n... Models Saved ...\n')
            except:
                pass



        # Print episode summary
        print('\n\n\nGame: ', episode)
        print('Rallies: ', env.ball.rallies)
        print('Time Elapsed: ', round(time.time()-start_time, 2), '\n')
        if history_1 != None and history_2 != None:
            print('\nTraining History:\n', 'Agent 1: ', history_1.history, '\n',
                                           'Agent 2: ', history_2.history, '\n')



        # Plot training progress
        if train_networks == True:

            training_tools.plot_progress(p1_tot, p2_tot,
                                         p1_rwd_store, p2_rwd_store,
                                         env.ball.rallies, rally_store, episode, avg_rallies,
                                         p1_avg, p2_avg,
                                         save_dir)



        # Save networks if they are also being trained
        if episode % 1 == 0 and train_networks == True:
            if p1_type == 'Agent':
                agent_1.save_model()

            if p2_type == 'Agent':
                agent_2.save_model()
            print('\n... Models Saved ...\n')







    pygame.quit()