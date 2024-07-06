#RL functions 

import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import random
from collections import deque
#creating the agent 

class Agent():
    #initialization of epsilon, gamma, training sizes and model load
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size #normailzed previous days
        self.action_size = 3 #sit, buy, sell or your 3 options 
        self.memory = deque
        self.inventory = []
        self.model_name = model_name
        #threshold constant values to drive which action to choose
        self.gamma = 0.95 
        self.epsilon = 1.0 
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model(model_name) if is_eval else  self._model()
    
    #create the model
    def _model(self): 
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model
    
    def act(self, state):
        # Check if it's not evaluation mode and if a randomly chosen number is less than epsilon
        if not self.is_eval and random.random() <= self.epsilon:
            # Return a random action (either 0=sit, 1=buy, or 2=sell)
            return random.randrange(self.action_size)
        # Predict the action values from the model for the given state
        options = self.model.predict(state)
        # Return the action with the highest value
        return np.argmax(options[0])
    
    #if memory of act gets full, we have expReplay to reset memory and 
    def expReplay(self, batch_size): 
        # Initialize an empty list to store the mini-batch
        mini_batch = []
        # Get the length of the memory deque
        l = len(self.memory)
        # Collect the last batch_size elements from memory to form the mini-batch
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            # Set the initial target to the reward
            target = reward 
            # If the episode is not done, update the target with discounted future reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            # Predict the current state's Q-values
            target_f = self.model.predict(state)
            # Update the Q-value corresponding to the taken action
            target_f[0][action] = target
            # Fit the model to the state and updated Q-values
            self.model.fit(state, target_f, epochs=1, verbose=0)
        # Decay epsilon if it is greater than its minimum value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



#now that the agent is created, let use define some basic math functions 
def formatPrice(n):
    return("-Rs." if n<0 else "Rs.") +"{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def getState(data, t, n): 
    d = t - n + 1 
    block = data[d:t+1] if d > 0 else -d * [data[0]] + data[0:t+1]
    res = []
    for i in range(n-1):
        res.append(sigmoid(block[i+1] - block[i]))
    return np.array([res])


def train_agent(agent, data, window_size, batch_size, epochs=1000):
    total_profit = 0
    data_length = len(data) - 1

    for epoch in range(1, epochs + 1):
        state = getState(data, 0, window_size + 1)
        for t in range(window_size, data_length):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buying
                agent.inventory.append(data[t])
            elif action == 2 and len(agent.inventory) > 0:  # selling
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price

            done = True if t == data_length - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print(f"Epoch {epoch}/{epochs} - Total Profit: {formatPrice(total_profit)}")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if epoch % 10 == 0:
            agent.model.save(f"model_ep{epoch}.h5")
