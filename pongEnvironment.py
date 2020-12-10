
import numpy as np
import pygame


"""
    - Combined all objects and environment to be together

    - Human paddle takes y, vy values from the mouse and updates the paddle

    - Agent paddle takes an index from the action_space, the paddle is moved
      and vy is calculated as the difference

"""




class Paddle:
    """
    Paddle Object
    """
    Height, Width = 100, 20

    def __init__(self, screen_Size, player_Type, player_Num, action_Space=None):

        self.screen_Width, self.screen_Height = screen_Size
        self.player_Type = player_Type
        self.player_Num = player_Num

        if player_Type == 'Agent':
            self.action_Space = action_Space

        self.y = self.screen_Height // 2
        self.vy = 0



    def update(self, action):
        """
        Given an action move paddle position and update velocity
            - If player_Type = "Human" action is y and vy of mouse
            - If player_Type = "Network" action is index of value in action space
        """
        if self.player_Type == 'Human':
            y_, self.vy = action

        elif self.player_Type == 'Agent':
            y_ = self.y + self.action_Space[action]
            if y_ < 0:
                y_ = 0
            elif y_ > self.screen_Height:
                y_ = self.screen_Height

            self.vy = abs(self.y - y_)

        self.y = y_



    def reset_paddle(self):
        """
        Reset paddle position for a new episode
        """
        self.y = self.screen_Height // 2
        self.vy = 0



    def show_paddle(self, screen, fgColor):
        """
        Show paddle in pygame window if the object is being drawn
        """
        if self.player_Num == 1:
            pygame.draw.rect(screen, fgColor, pygame.Rect((0, self.y-self.Height//2, self.Width, self.Height)))
        elif self.player_Num == 2:
            pygame.draw.rect(screen, fgColor, pygame.Rect((self.screen_Width-self.Width, self.y-self.Height//2, self.Width, self.Height)))






class Ball:
    Radius = 10
    V_max = 100

    def __init__(self, screen_Size, paddle_Width):
        """
        Setup ball for size of given game space
        """
        self.screen_Width, self.screen_Height = screen_Size

        # Setup x,y limits for ball position
        self.left_x = paddle_Width
        self.right_x = self.screen_Width - paddle_Width
        self.top_y = self.Radius
        self.bot_y = self.screen_Height - self.Radius

        self.x = self.screen_Width//2
        self.y = np.random.randint(1, self.screen_Height-1)

        self.vx = np.random.choice([-1, 1]) * np.random.choice([30])
        self.vy = np.random.choice([-1, 1]) * np.random.choice([5, 10, 20, 30])



    def update(self, paddle_1, paddle_2):
        """
        Update position and velocity of the ball
        """
        done = False

        p1_reward = 0
        p2_reward = 0

        # Move ball and move to edges if necessary
        x_ = self.x + self.vx
        y_ = self.y + self.vy

        if x_ < self.left_x:
            x_ = self.left_x
        elif x_ > self.right_x:
            x_ = self.right_x

        if y_ < self.top_y:
            y_ = self.top_y
        elif y_ > self.bot_y:
            y_ = self.bot_y


        # Contact with top or bottom
        if y_ == self.top_y or y_ == self.bot_y:
            self.vy = round(-0.9 * self.vy)

        # Left or right sides, update done if paddle misses ball
        if x_ == self.left_x:
            if abs(y_ - paddle_1.y) <= paddle_1.Height // 2:
                x_ += self.Radius
                self.vx = round(-0.9 * self.vx)
                self.vy += paddle_1.vy//6

                p1_reward += 300
                p2_reward -= 0
            else:
                done = True
                p1_reward -= 500
                p2_reward += 100

        elif x_ == self.right_x:
            if abs(y_ - paddle_2.y) <= paddle_2.Height // 2:
                x_ -= self.Radius
                self.vx = round(-0.9 * self.vx)
                self.vy += paddle_2.vy//6

                p1_reward -= 0
                p2_reward += 300
            else:
                done = True
                p1_reward += 100
                p2_reward -= 500

        # Update ball position and velocity if exceeded
        self.x = x_
        self.y = y_


        # Add reward based on closeness of paddle height to ball height
        p1_reward += 10 - 20 * abs((paddle_1.y - self.y) / self.screen_Height)
        p2_reward += 10 - 20 * abs((paddle_2.y - self.y) / self.screen_Height)

        if self.vx > self.V_max:
            self.vx = self.V_max   

        if self.vy > self.V_max:
            self.vy = self.V_max


        # Setup values to return state observation
        state = np.array([self.x/self.screen_Width, self.y/self.screen_Height, 
                          self.vx/self.V_max, self.vy/self.V_max])

        return state, p1_reward, p2_reward, done



    def reset_ball(self):
        """
        Reset ball function, selects randomized velocity, location (near middle line)
        """
        done = False

        # Reset position and select new speeds
        self.x = self.screen_Width//2
        self.y = np.random.randint(1, self.screen_Height-1)

        self.vx = np.random.choice([-1, 1]) * np.random.choice([20])
        self.vy = np.random.choice([-1, 1]) * np.random.choice([5, 10, 15, 20])


        # Setup values to return state observation
        state = np.array([self.x/self.screen_Width, self.y/self.screen_Height, 
                          self.vx/self.V_max, self.vy/self.V_max])

        return state




    def show_ball(self, screen, fgColor):
        """
        Show the ball in pygame window if the objects are being drawn
        """
        pygame.draw.circle(screen, fgColor, (self.x, self.y), self.Radius)

  




class pongGame:
    """
    Class for running the game, modeled after Gym with step and render methods.
    """
    def __init__(self, screen_Size, p1_Type, p2_Type, action_Space):
        """
        Initialize objects for simulation/rendering
        """
        self.screen_Width, self.screen_Height = screen_Size

        # Initialize game objects
        self.paddle_1 = Paddle(screen_Size, p1_Type, 1, action_Space)
        self.paddle_2 = Paddle(screen_Size, p2_Type, 2, action_Space)
        self.ball = Ball(screen_Size, self.paddle_1.Width)




    def setupWindow(self, framerate, bgColor="black", fgColor="green"):
        """
        Initialize pygame window, will need to be called if the scene is to be rendered.
        """
        self.framerate = framerate
        self.clock = pygame.time.Clock()
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_Width, self.screen_Height))

        self.bgColor = pygame.Color(bgColor)
        self.fgColor = pygame.Color(fgColor)      


    def render(self):
        """
        If called, render the scene in the pygame window
        """
        pygame.display.flip()
        self.screen.fill(self.bgColor)

        self.paddle_1.show_paddle(self.screen, self.fgColor)
        self.paddle_2.show_paddle(self.screen, self.fgColor)
        self.ball.show_ball(self.screen, self.fgColor)

        self.clock.tick(self.framerate)




    def step(self, p1_action, p2_action):
        """
        Take a step each frame
            - done reflects the status of the ball, if False game is over
        """
        self.paddle_1.update(p1_action)
        self.paddle_2.update(p2_action)
        
        state, p1_reward, p2_reward, done = self.ball.update(self.paddle_1, self.paddle_2)

        return state, p1_reward, p2_reward, done


    def reset(self):
        """
        Reset game, return state and done=False
        """

        self.paddle_1.reset_paddle()
        self.paddle_2.reset_paddle()

        state = self.ball.reset_ball()

        return state, False