# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 14:35:58 2020

@author: yaoxiaoyuan
"""
import time
import sys
import random
import copy
import pygame
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

class Game():
    """
    """
    def __init__(self, height, width, need_display=True):
        """
        """
        self.width = width
        self.height = height
        self.game_width = (width + 4) * 20
        self.game_height = (height + 5) * 20
        
        self.WHITE = (255, 255, 255)
        self.GRAY_3 = (64, 64, 64)
        self.GRAY_2 = (128, 128, 128)
        self.GRAY = (192, 192, 192)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        self.trans_table = {0:{0:[1,0,0], 1:[0,-1,1], 2:[1,0,0], 3:[0,1,3]},
                            1:{0:[1,0,0], 1:[0,-1,1], 2:[-1,0,2], 3:[0,-1,1]},
                            2:{0:[-1,0,2], 1:[0,-1,1], 2:[-1,0,2], 3:[0,1,3]},
                            3:{0:[1,0,0], 1:[0,1,3], 2:[-1,0,2], 3:[0,1,3]}}
        
        self.need_display = need_display
        if self.need_display == True:
            self.screen = pygame.display.set_mode((self.game_width, 
                                                   self.game_height))
            pygame.init()

        self.highest_score = 0
        self.frames = []
        self.restart()
        
        
    def restart(self):
        """
        """
        self.head = [random.randint(0, self.width-1), 
                     random.randint(0, self.height-1)]
        self.snake_location = [[self.head[0],self.head[1]]]
        self.direction = 0
        self.random_food()
        self.frames.append(np.zeros([1, self.height+2, self.width+2]))
        self.frames.append(self.get_frame(False))
        self.score = 0
        self.alive_time = 0
        if self.need_display == True:
            self.display()


    def random_food(self):
        """
        """
        grids = [[x,y] for x in range(self.width) for y in range(self.height)]
        blank = [xy for xy in grids if xy not in self.snake_location]
        if len(blank) > 0:
            self.food = random.choice(blank)


    def update(self, action):
        """
        """
        reward = 0
        dead = False
        old_state = self.get_state()
        
        if self.is_dead(action):
            reward = -10
            dead = True
        
        delta_x,delta_y,new_dir = self.trans_table[self.direction][action]
        self.head = [self.head[0] + delta_x, self.head[1] + delta_y]
        self.snake_location.append(self.head)
        self.direction = new_dir
            
        if self.head == self.food:
            reward = 5
            self.score += 1
            self.highest_score = max(self.highest_score, self.score)
            self.random_food()
        else:
            self.snake_location = self.snake_location[1:]
            
        self.frames.append(self.get_frame(dead))
        
        new_state = self.get_state()
        
        if self.need_display == True:
            self.display()

        over = False
        if len(self.snake_location) == self.width * self.height:
            over = True
        if dead == True:
            over = True

        self.alive_time += 1
        
        return reward, over, old_state, new_state, action 


    def is_dead(self, action):
        """
        """
        x,y = self.head[0], self.head[1]
        delta_x,delta_y,_ = self.trans_table[self.direction][action]
        x,y = x + delta_x, y + delta_y
        
        dead = False
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            dead = True
        
        if [x,y] in self.snake_location[1:]:
            dead = True
        
        return dead
    
    
    def draw_object(self, x, y, color, screen):
        """
        """
        pygame.draw.rect(screen, color, (40 + x * 20, 40 + y * 20, 20, 20))


    def draw_wall(self):
        """
        """
        pygame.draw.rect(self.screen, self.BLACK, 
                         (20, 20, 20, (self.height + 2)*20))
        pygame.draw.rect(self.screen, self.BLACK, 
                         (20, 20, (self.width + 2)*20, 20))
        pygame.draw.rect(self.screen, self.BLACK, 
                         (20, (self.height + 2)*20, (self.width + 2)*20, 20))
        pygame.draw.rect(self.screen, self.BLACK, 
                         ((self.width + 2)*20, 20, 20, (self.height + 2)*20))


    def draw_score(self):
        """
        """
        myfont = pygame.font.SysFont('Segoe UI', 20)
        text_score = myfont.render('SCORE: ', True, (0, 0, 0))
        text_score_number = myfont.render(str(self.score), True, (0, 0, 0))
        text_highest_score = myfont.render('HIGHEST: ', True, (0, 0, 0))
        text_highest_score_number = myfont.render(str(self.highest_score), 
                                               True, (0, 0, 0))
        self.screen.blit(text_score, (25, self.game_height - 35))
        self.screen.blit(text_score_number, (95, self.game_height - 35))
        self.screen.blit(text_highest_score, (125, self.game_height - 35))
        self.screen.blit(text_highest_score_number, (230, self.game_height - 35))


    def get_frame(self, dead):
        """
        """
        if dead == True:
            return np.ones([1, self.height+2, self.width+2])
        
        frame = np.zeros([1, self.height+2, self.width+2])
        
        #wall
        frame[:, [0,-1], :] = 1
        frame[:, :, [0,-1]] = 1

        #snake
        for i,(x,y) in enumerate(self.snake_location):
            if i == len(self.snake_location) - 1:
                frame[:,y+1,x+1] = 64 / 255
            else:
                frame[:,y+1,x+1] = 128 / 255
        
        #food
        frame[0, self.food[1]+1, self.food[0]+1] = 192 / 255
        
        return frame


    def get_state(self):
        """
        """
        state = np.zeros([1, 2, self.height+2, self.width+2])
        state[0, 0:1, :, :] = self.frames[-2]
        state[0, 1:2, :, :] = self.frames[-1]
        
        return state


    def display(self):
        """
        """
        self.screen.fill(self.WHITE)
        
        self.draw_wall()

        for i,(x,y) in enumerate(self.snake_location):
            if i == len(self.snake_location) - 1:
                self.draw_object(x, y, self.GRAY_2, self.screen)
            else:
                self.draw_object(x, y, self.GRAY, self.screen)
                
        self.draw_object(self.food[0], self.food[1], self.GRAY_3, self.screen)

        self.draw_score()

        pygame.display.update()
        pygame.display.flip()


class QNet(nn.Module):
    """
    """
    def __init__(self, width, height):
        """
        """
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        
        w = ((width - 4) + 1) // 2
        h = ((height - 4) + 1) // 2
        out_size = 64 * w * h
        self.fc1 = nn.Linear(out_size, 128)
        self.fc2 = nn.Linear(128, 4)
        

    def forward(self, s):
        """
        """
        hidden = self.conv1(s)
        hidden = F.relu(hidden)
        hidden = self.conv2(hidden)
        hidden = F.relu(hidden)
        hidden = F.max_pool2d(hidden, 2)
        hidden = self.dropout1(hidden)
        hidden = torch.flatten(hidden, 1)
        hidden = self.fc1(hidden)
        hidden = torch.tanh(hidden)
        hidden = self.dropout2(hidden)
        
        qsa = self.fc2(hidden)
            
        return qsa


class Agent():
    """
    """
    def __init__(self, height, width, batch_size, 
                 gamma, lr, grad_clip, max_memory,
                 use_cuda=True):
        """
        """
        self.memory = []
        self.use_cuda = use_cuda
        self.build_model(height+2, width+2)
        self.gamma = gamma
        self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
                lr, amsgrad=True)
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.lr = lr
        self.max_memory = max_memory
    
    
    def build_model(self, height, width):
        """
        """
        self.model = QNet(height, width)
        if self.use_cuda == True:
            self.model = self.model.cuda()
        self.target_model = copy.deepcopy(self.model)
        
    
    def memorize(self, mem):
        """
        """
        if len(self.memory) >= self.max_memory:
            self.memory = random.sample(self.memory, self.max_memory - 1)
        self.memory += [mem]
    
    
    def get_batch(self):
        """
        """
        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
        else:
            batch = self.memory
        
        reward = torch.tensor(np.array([t[0] for t in batch]),
                              dtype=torch.float)
        over = torch.tensor(np.array([float(t[1]) for t in batch]),
                           dtype=torch.float)
        old_state = torch.tensor(np.concatenate([t[2] for t in batch], 0),
                                 dtype=torch.float)
        new_state = torch.tensor(np.concatenate([t[3] for t in batch], 0),
                                 dtype=torch.float)
        action = torch.tensor(np.array([[t[4]] for t in batch]),
                              dtype=torch.long)
        
        if self.use_cuda == True:
            reward = reward.cuda()
            over = over.cuda()
            old_state = old_state.cuda()
            new_state = new_state.cuda()
            action = action.cuda()
        
        return reward, over, old_state, new_state, action
        
    
    def train(self):
        reward, over, old_state, new_state, action = self.get_batch()
        
        self.target_model.eval()
        with torch.no_grad():
            next_action = self.model(new_state).max(-1)[1].unsqueeze(-1)
            q_new = torch.gather(self.target_model(new_state), 1, next_action)

        self.model.train()
        q_old = torch.gather(self.model(old_state), 1, action)
        
        loss = F.mse_loss(q_old.flatten(), 
                          reward + self.gamma * (1 - over) * q_new.flatten())
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                       self.grad_clip)
        self.optimizer.step()
        
    
    def predict(self, state):
        """
        """
        state = torch.tensor(state, dtype=torch.float)
        if self.use_cuda == True:
            state = state.cuda()
        
        self.model.eval()
        with torch.no_grad():
            q = self.model(state)
        
        action = q.argmax(-1).item()
        
        return action
    

def train(config):
    """
    """
    game = Game(config["height"], config["width"], False)
    agent = Agent(config["width"],
                  config["height"],
                  config["batch_size"],
                  config["gamma"], 
                  config["lr"], 
                  config["grad_clip"], 
                  config["max_memory"], 
                  config["use_cuda"])
    
    if config["reload"] == True:
        agent.model.load_state_dict(
                torch.load(config["load_model"] + ".model",
                           map_location=lambda storage, loc: storage))
        agent.optimizer.load_state_dict(
                torch.load(config["load_model"] + ".optimizer",
                           map_location=lambda storage, loc: storage))
    
    epsilon = config["epsilon"]
    epsilon_dacay = config["epsilon_dacay"]
    min_epsilon = config["min_epsilon"]

    max_score = 0
    avg = []
    steps = 0
    for i in range(1, config["episode"] + 1):
        game.restart()
        while True:
            
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = agent.predict(game.get_state())
                    
            reward, over, old_state, new_state, action = game.update(action)
            
            mem = [reward, over, old_state, new_state, action]
            agent.memorize(mem)
            
            steps += 1
            
            if steps % config["train_every_n_actions"] == 0:
                if len(agent.memory) > 1000:
                    agent.train()
        
                epsilon = max(min_epsilon, epsilon - epsilon_dacay)

            if steps % config["update_target_n_actions"] == 0:
                agent.target_model = copy.deepcopy(agent.model)
            
            if over == True:
                break

        max_score = max(max_score, game.score)
        avg = avg[-100:] + [game.score]
        print("episode %d score %d, max %d, avg %d, alive_time %d, epsilon %f" % 
             (i, game.score, max_score, sum(avg)/len(avg), game.alive_time, epsilon))
            
        if i % 1000 == 0:
            torch.save(agent.model.state_dict(), 
                       config["load_model"] + ".model")
            torch.save(agent.optimizer.state_dict(), 
                       config["load_model"] + ".optimizer")


def test(config):
    """
    """
    game = Game(config["height"], config["width"])
    clock = pygame.time.Clock()
    agent = Agent(config["width"], 
                  config["height"], 
                  config["batch_size"],
                  config["gamma"], 
                  config["lr"], 
                  config["grad_clip"], 
                  config["max_memory"], 
                  config["use_cuda"])
    agent.model.load_state_dict(torch.load(config["load_model"] + ".model",
                                map_location=lambda storage, loc: storage))
    while True:
        clock.tick(5)
        
        action = agent.predict(game.get_state())
        reward, over, old_state, new_state, action = game.update(action)
        if over == True:
            game.restart()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
    

if __name__ == "__main__":

    config = {"width":8,
              "height":8,
              "use_cuda":True,
              "episode":500000,
              "gamma":0.99,
              "lr":0.0003,
              "grad_clip":5,
              "batch_size":32,
              "epsilon":0.9999,
              "epsilon_dacay":0.00001,
              "min_epsilon":0.001,
              "max_memory":25000,
              "train_every_n_actions": 4,
              "update_target_n_actions": 2000,
              "load_model":"../model/snake_8_8",
              "save_model":"../model/snake_8_8",
              "reload":False}
    if sys.argv[1] == "train":
        train(config)
    elif sys.argv[1] == "test":
        test(config)