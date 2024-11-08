from agents import Car
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        
    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQLAgent(Car):
    def __init__(self, position, orientation, state_size, action_size, goal):
        # Call parent class (Car) constructor first
        super().__init__(position, orientation, 'red')  # or whatever color you want
        
        self.goal = goal
        # Now we can access self.position since it's initialized by the parent class
        self.prev_distance = self.position.distanceTo(goal)
        
        # DQL parameters
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_frequency = 10  # How often to update target network
        self.training_step = 0
        
        # Initialize neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size=20, output_size=self.action_size).to(self.device)
        self.target_net = DQN(input_size=20, output_size=self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # Define discrete actions
        self.steering_actions = [-0.3, -0.1, 0.0, 0.1, 0.3]
        self.throttle_actions = [0.0, 0.1, 0.3]

    def get_state(self, world):
        # Get distances and angles to nearby objects
        nearby_objects = world.get_nearby_objects(self, 30)  # 30m radius
        object_features = []
        
        # Initialize with max distance for empty slots
        for _ in range(6):  # Track up to 6 nearest objects
            object_features.extend([30.0, 0.0])  # distance and angle
            
        # Fill in actual object information
        obj_idx = 0
        for obj in sorted(nearby_objects, 
                         key=lambda x: self.position.distanceTo(x.position))[:6]:
            if obj != self:
                dist = self.position.distanceTo(obj.position)
                angle = np.arctan2(obj.position.y - self.position.y, 
                                 obj.position.x - self.position.x) - self.heading
                object_features[obj_idx*2] = dist
                object_features[obj_idx*2 + 1] = angle
                obj_idx += 1
        
        # Add agent's own state and goal information
        state = np.array([
            self.position.x/120.0, self.position.y/120.0,  # Normalized position
            self.velocity.x/10.0, self.velocity.y/10.0,    # Normalized velocity
            np.sin(self.heading), np.cos(self.heading),        # Angle as sin/cos
            self.position.distanceTo(self.goal)/120.0,    # Normalized distance to goal
            np.arctan2(self.goal.y - self.position.y,      # Angle to goal
                      self.goal.x - self.position.x) - self.heading
        ] + object_features)
        
        return state

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Exploration: random action
            steering_idx = random.randint(0, 2)
            throttle_idx = random.randint(0, 2)
        else:
            # Exploitation: best action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.max(1)[1].item()
                steering_idx = action_idx // 3
                throttle_idx = action_idx % 3
        
        # Convert indices to actual control values
        steering = [-0.5, 0.0, 0.5][steering_idx]
        throttle = [0.0, 0.5, 1.0][throttle_idx]
        
        return steering, throttle
    
    def store_transition(self, state, action, reward, next_state, done):
        # Convert action to index
        steering_idx = [0.5, 0.0, -0.5].index(action[0])
        throttle_idx = [0.0, 0.5, 1.0].index(action[1])
        action_idx = steering_idx * 3 + throttle_idx
        
        self.memory.append((state, action_idx, reward, next_state, done))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.update_target_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def act(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            action = action_values.argmax().item()
        
        # Convert action index to steering and throttle
        action_idx = action
        steering_idx = action_idx // len(self.throttle_actions)
        throttle_idx = action_idx % len(self.throttle_actions)
        
        return (self.steering_actions[steering_idx], 
                self.throttle_actions[throttle_idx])
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    target = reward + self.gamma * self.target_model(next_state_tensor).max(1)[0].item()
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            target_f = self.model(state_tensor)
            target_f[0][action] = target
            
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict()) 
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def _calculate_heading_diff(self, car, center_building):
        v = car.center - center_building.center
        desired_heading = np.mod(np.arctan2(v.y, v.x) + np.pi/2, 2*np.pi)
        return np.sin(desired_heading - car.heading)