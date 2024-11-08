import numpy as np
import torch
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian, RectangleBuilding
from geometry import Point
import time
from tkinter import *
from dql_agent import DQLAgent

human_controller = False

# World settings
dt = 0.1  # time steps in seconds
world_width = 120  # meters
world_height = 120
lane_width = 3.5
sidewalk_width = 2.0

# Create world
w = World(dt, width=world_width, height=world_height, ppm=6)

# Street dimensions
street_width = 2 * lane_width  # Two lanes per direction
block_size = 40  # Size of building blocks

# Create horizontal and vertical roads (as paintings)
# Horizontal roads
for y in [world_height/4, 3*world_height/4]:
    w.add(Painting(
        Point(world_width/2, y),
        Point(world_width, 2*street_width),
        'gray20'
    ))

# Vertical roads
for x in [world_width/4, 3*world_width/4]:
    w.add(Painting(
        Point(x, world_height/2),
        Point(2*street_width, world_height),
        'gray20'
    ))

# Add lane markers
marker_width = 0.5
marker_length = 5
gap_length = 5
marker_color = 'white'

# Helper function to add dashed lines
def add_dashed_line(start_x, start_y, length, is_vertical=False):
    current_pos = 0
    while current_pos < length:
        marker_pos = min(marker_length, length - current_pos)
        if is_vertical:
            w.add(Painting(
                Point(start_x, start_y + current_pos + marker_pos/2),
                Point(marker_width, marker_pos),
                marker_color
            ))
        else:
            w.add(Painting(
                Point(start_x + current_pos + marker_pos/2, start_y),
                Point(marker_pos, marker_width),
                marker_color
            ))
        current_pos += marker_pos + gap_length

# Add lane markers for horizontal roads
for y in [world_height/4, 3*world_height/4]:
    add_dashed_line(0, y, world_width)

# Add lane markers for vertical roads
for x in [world_width/4, 3*world_width/4]:
    add_dashed_line(x, 0, world_height, is_vertical=True)

# Add buildings in the corners and middle blocks
building_positions = [
    # Corner buildings
    (block_size/2, block_size/2),
    (block_size/2, world_height - block_size/2),
    (world_width - block_size/2, block_size/2),
    (world_width - block_size/2, world_height - block_size/2),
    # Middle buildings
    (block_size/2, world_height/2),
    (world_width - block_size/2, world_height/2),
    (world_width/2, block_size/2),
    (world_width/2, world_height - block_size/2),
    (world_width/2, world_height/2)
]

for pos_x, pos_y in building_positions:
    building = RectangleBuilding(
        Point(pos_x, pos_y),
        Point(block_size - sidewalk_width, block_size - sidewalk_width),
        'gray80'
    )
    w.add(building)

# Add a car
c1 = Car(Point(world_width/4 - lane_width/2, 10), np.pi/2)
c1.max_speed = 30.0  # 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

w.render() # This visualizes the world we just constructed.

if not human_controller:
    # Training settings
    EPISODES = 1000
    batch_size = 32
    
    # Initialize DQL agent
    state_size = 6  # x, y, velocity_x, velocity_y, heading, distance_to_goal
    action_size = 15  # 5 steering actions * 3 throttle actions
    agent = DQLAgent(state_size, action_size)
    
    # Define start and goal positions
    start_pos = Point(world_width/4 - lane_width/2, 10)  # Bottom of map
    goal_pos = Point(world_width/4 - lane_width/2, world_height - 10)  # Top of map
    goal_radius = 5.0  # Success radius around goal
    
    for episode in range(EPISODES):
        # Reset the environment
        c1.center = start_pos
        c1.heading = np.pi/2  # Pointing upward
        c1.velocity = Point(0, 3.0)
        
        total_reward = 0
        steps = 0
        max_steps = 600
        
        while steps < max_steps:
            # Get current state
            distance_to_goal = np.sqrt((c1.center.x - goal_pos.x)**2 + (c1.center.y - goal_pos.y)**2)
            state = np.array([
                c1.center.x / world_width,  # Normalize positions
                c1.center.y / world_height,
                c1.velocity.x / c1.max_speed,
                c1.velocity.y / c1.max_speed,
                c1.heading / (2 * np.pi),
                distance_to_goal / world_height
            ])
            
            # Get action from agent
            steering, throttle = agent.act(state)
            c1.set_control(steering, throttle)
            
            # Advance simulation
            w.tick()
            w.render()
            
            # Calculate reward
            new_distance = np.sqrt((c1.center.x - goal_pos.x)**2 + (c1.center.y - goal_pos.y)**2)
            
            # Reward shaping
            reward = 0
            reward += (distance_to_goal - new_distance)  # Reward for getting closer to goal
            reward -= abs(steering) * 0.1  # Small penalty for steering
            
            # Check if reached goal
            done = False
            if new_distance < goal_radius:
                reward += 1000  # Big reward for reaching goal
                done = True
                print(f"Episode {episode}: Reached goal!")
            
            # Check for collisions
            if w.collision_exists():
                reward = -1000  # Big penalty for collision
                done = True
            
            # Get new state
            new_distance_to_goal = np.sqrt((c1.center.x - goal_pos.x)**2 + (c1.center.y - goal_pos.y)**2)
            next_state = np.array([
                c1.center.x / world_width,
                c1.center.y / world_height,
                c1.velocity.x / c1.max_speed,
                c1.velocity.y / c1.max_speed,
                c1.heading / (2 * np.pi),
                new_distance_to_goal / world_height
            ])
            
            # Store experience in memory
            action_idx = (agent.steering_actions.index(steering) * 
                         len(agent.throttle_actions) + 
                         agent.throttle_actions.index(throttle))
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train the network
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target network every episode
        agent.update_target_model()
        
        print(f"Episode: {episode + 1}/{EPISODES}, Score: {total_reward:.2f}, Steps: {steps}")
        
        # Save the model periodically
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), f'dql_agent_episode_{episode}.pth')

else: # Let's use the keyboard input for human control
    from interactive_controllers import KeyboardController
    c1.set_control(0., 0.) # Initially, the car will have 0 steering and 0 throttle.
    controller = KeyboardController(w)
    for k in range(600):
        c1.set_control(controller.steering, controller.throttle)
        w.tick() # This ticks the world for one time step (dt second)
        w.render()
        time.sleep(dt/4) # Let's watch it 4x
        if w.collision_exists():
            import sys
            sys.exit(0)
    w.close()