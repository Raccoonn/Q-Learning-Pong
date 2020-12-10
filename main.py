
import os
import time
import pygame

from tf_dqn import Agent
from pongEnvironment import pongGame


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



if __name__ == '__main__':

    show_game = False

    load_networks = True

    train_networks = True

    action_Space = [-50, -25, 0, 25, 50]

    p1_Type = 'Agent'
    p2_Type = 'Agent'

    # Initialize DeepQ agents for the specified players
    if p1_Type == 'Agent':
        agent_1 = Agent(gamma=0.99, epsilon=0.05, alpha=0.005, input_dims=4,
                        n_actions=5, mem_size=100000, batch_size=1024, epsilon_end=0.01,
                        fc1_dims=512, fc2_dims=512, fname='p1.h5')

    if p2_Type == 'Agent':
        agent_2 = Agent(gamma=0.99, epsilon=0.1, alpha=0.001, input_dims=4,
                        n_actions=5, mem_size=100000, batch_size=1024, epsilon_end=0.01,
                        fc1_dims=512, fc2_dims=512, fname='p2.h5')


    # Load networks if specified
    if load_networks == True:
        if p1_Type == 'Agent':
            agent_1.load_model('p1.h5')

        if p2_Type == 'Agent':
            agent_2.load_model('p2.h5')
  
        print('\n... Models Loaded ...\n')



    screen_Size = (1600, 800)
    framerate = 60

    env = pongGame(screen_Size, p1_Type, p2_Type, action_Space)

    if show_game == True:
        env.setupWindow()



    game_wins = [0, 0]

    p_i, p_syms = 0, ('\\', '|', '/', '-')

    for episode in range(1, 100000):

        p1_score = 0
        p2_score = 0

        state, done = env.reset()

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
                        done = env.reset()
                        break

                env.render()
                

            # Get actions based on player_Type
            if p1_Type == 'Human':
                p1_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]
            elif p1_Type == 'Agent':
                p1_action = agent_1.choose_action(state)

            if p2_Type == 'Human':
                p2_action = pygame.mouse.get_pos()[1], pygame.mouse.get_rel()[1]
            elif p2_Type == 'Agent':
                p2_action = agent_2.choose_action(state)


            # Environment takes a step, return observations, reward, and status
            state_, p1_reward, p2_reward, done = env.step(p1_action, p2_action)

            p1_score += p1_reward
            p2_score += p2_reward


            # Train agent DeepQ Networks
            if train_networks == True:
                if p1_Type == 'Agent':
                    agent_1.remember(state, p1_action, p1_reward, state_, int(done))
                    agent_1.learn()

                if p2_Type == 'Agent':
                    agent_2.remember(state, p2_action, p2_reward, state_, int(done))
                    agent_2.learn()


            # Update state for new step
            state = state_


        # Tally wins after each episode
        if p1_score > p2_score:
            game_wins[0] += 1
        else:
            game_wins[1] += 1

        print('\n\n\nGame: ', episode)
        print('Scores and total wins:', round(p1_score), round(p2_score), game_wins)
        print('Time Elapsed: ', round(time.time()-start_time, 2), '\n')


        # Save networks if they are also being trained
        if episode % 10 == 0 and train_networks == True:
            if p1_Type == 'Agent':
                agent_1.save_model()

            if p2_Type == 'Agent':
                agent_2.save_model()
            print('\n... Models Saved ...\n')


        # Load the superior model as both agents if it wins enough games
        # Reset win counter to reevaluate later
        if episode > 1 and episode % 100 == True:
            win_diff = game_wins[0] - game_wins[1]
            if win_diff > 25:
                agent_2.load_model('p1.h5')
                game_wins = [0, 0]
                print('\n... Agent 2 Swapped ...\n')
            elif win_diff < -25:
                agent_1.load_model('p2.h5')
                game_wins = [0, 0]
                print('\n... Agent 1 Swapped ...\n')



    pygame.quit()