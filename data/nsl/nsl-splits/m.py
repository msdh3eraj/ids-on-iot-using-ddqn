import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import namedtuple, deque

import pandas as pd
class DDQNloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,predicted,actual):
        loss = ((torch.abs(target - predicted))**2).sum()
        return loss
class Darkexploss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,predicted,actual,qqvalues,logits):
        loss1 = ((torch.abs(target - predicted))**2).sum()
        darkexperienceloss = ((torch.abs(logits - qqvalues)) ** 2).sum()
        loss2 = 0.2 * darkexperienceloss
        ll= loss1+loss2
        return loss1    
        

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  # Convert input to float
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PrioritizedReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity

        self.buffer = []

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)  # Remove the oldest experience
            self.buffer.append(transition)  # Add the new experience


    def sample(self, batch_size, w=0.2):
#             losses = np.array([experience[5] for experience in self.buffer])
            losses = np.array([experience[5].detach().numpy() for experience in self.buffer])
            loss_powers = losses ** w
            probs = loss_powers / np.sum(loss_powers)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            return samples

class DoubleDQNAgent:
    def __init__(self, state_size, action_size, memory_capacity=1):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_size, action_size)
        self.target_model = QNetwork(state_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = PrioritizedReplayMemory(memory_capacity)


    def act(self, state):#bahvior policy bri to take action.

            if np.random.rand() <= self.epsilon:
                #return np.random.randint(self.action_size, size=len(state))  # Generate random actions for each state
                k=np.random.randint(self.action_size,size=1)
                if(k==0):
                  return np.array([0,1])
                else:
                  return np.array([1,0]) 
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).view(-1, 1)  # Adjust the dimensions
                q_values = self.model(torch.tensor(state_tensor))
                actions = torch.argmax(q_values, dim=1).numpy()
                return actions




    def remember(self, state, action, reward, next_state, done, loss, logits):
        transition = (state, action, reward, next_state, done, loss ,logits)

        self.memory.push(transition)





    def replay(self, batch_size, beta):
        samples = self.memory.sample(batch_size, beta)
        for sample in samples:
            state, action, reward, next_state, done, loss , logits = sample
          
            if next_state is not None:
                target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)))[0]
            else:
                target = torch.tensor(reward)    
                

            qqvalues = agent.model(torch.tensor(state))
            q_value, _ = torch.max(qqvalues, dim=1)

            loss_module = Darkexploss()
            loss = loss_module(q_value, target, qqvalues,logits)
           

# Backward pass
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # Set retain_graph=True to retain the computational graph
            self.optimizer.step()



    def target_train(self):
        self.target_model.load_state_dict(self.model.state_dict())








file_path = 'output2.csv'
data = pd.read_csv(file_path).head(100)

#features = data.iloc[:, :-1]  # Extract all columns except the last one
class_label = data.iloc[:, -1]  # Extract the last column
#feature_arrays = [np.array(data.iloc[:, i]) for i in range(len(data.columns) - 1)]

feature_arrays = []
for index, row in data.iterrows():
    feature_arrays.append(np.array(row)[1:-1])  # Exclude the last element

print(feature_arrays[2])
hh=len(feature_arrays)


# Set hyperparameters
epsilon = 0.4  # Epsilon for epsilon-greedy approach
memory_capacity = 1000  # Capacity for prioritized experience replay memory
E = 1000 # Total episodes
N = 100 # Number of feature vectors
gamma = 0.9 # Discount factor for Bellman equation
batch_size = 32  # Batch size for experience replay

# Initialize the Double DQN agent
state_size =33 # Change this value to the appropriate size from your dataset
# total = 98442
action_size = 2  # Assuming two possible actions (yes or no)
agent = DoubleDQNAgent(state_size, action_size, memory_capacity)







# Training loop
k=0
for episode in range(1, E + 1):
    
    # Shuffle feature vectors
    print("episode: ", episode)
    random.shuffle(feature_arrays)
    sum=0
    for j in range(1, N + 1):
         k=k+1
         if k % 1000 == 0:
            agent.target_train()   
         state = feature_arrays[j - 1]
         #state = state.reshape(-1,1)

         if j < N:
                next_state = feature_arrays[j]
                #next_state = next_state.reshape(-1,1)
                
         else:
                next_state = None     
               
         action = agent.act(state)  #behavior policy
#          print(action)   
         if(action[0] == 0):
          finalaction=1
         else:
          finalaction=0
#          print(finalaction) 
         if (finalaction ==  class_label[j-1]):
             reward= 1
         else:    
#             print(actual_action)
             reward = 0
         #reward = torch.from_numpy(np.where(action == actual_action, 1, 0))  # Convert the NumPy array to a PyTorch tensor
#          print(torch.tensor(next_state))
            
         if next_state is not None:
                target = reward + gamma * torch.max(agent.target_model(torch.tensor(next_state)))
         else:
                target = torch.tensor(reward)

         qqvalues= agent.model(torch.tensor(state))
         q_value= torch.max(qqvalues)
#          mm= torch.argmax(qqvalues)   
         loss_module = DDQNloss()
         loss = loss_module(q_value, target)
#          print(loss.item())
         q_valueww = agent.model(torch.tensor(state))
         actionatthistime = torch.argmax(q_valueww)
#          print("gvytfyteyg", mm)
         actualaction = class_label[j-1]
         if(actualaction==  actionatthistime):
            sum=sum+1

         #reward222 = torch.from_numpy(np.where(mm == actual_action2, 1, 0))
         #print('acuracy')
         #print(reward222.sum().item()/len(mm))
#          print(" ")   
         agent.optimizer.zero_grad()
         loss.backward()
         agent.optimizer.step()

         agent.remember(state, action, reward, next_state, False, loss, logits=qqvalues)


#     beta = 0.4
#     print("hell")# Set the beta parameter for prioritized experience replay
    #agent.replay(batch_size, beta)
    print("accuracy:")
    print(sum/98442)





















