import numpy as np
import torch
from world import World
from agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from geometry import Point
import time
from tkinter import *
from dql_agent import DQLAgent

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
world_width = 120 # in meters
world_height = 120
inner_building_radius = 30
num_lanes = 2
lane_marker_width = 0.5
num_of_lane_markers = 50
lane_width = 3.5

w = World(dt, width = world_width, height = world_height, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.



# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
# A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

# To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
cb = CircleBuilding(Point(world_width/2, world_height/2), inner_building_radius, 'gray80')
w.add(cb)
rb = RingBuilding(Point(world_width/2, world_height/2), inner_building_radius + num_lanes * lane_width + (num_lanes - 1) * lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
w.add(rb)

# Let's also add some lane markers on the ground. This is just decorative. Because, why not.
for lane_no in range(num_lanes - 1):
    lane_markers_radius = inner_building_radius + (lane_no + 1) * lane_width + (lane_no + 0.5) * lane_marker_width
    lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
    for theta in np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers):
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(lane_marker_width, lane_marker_height), 'white', heading = theta))
    

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(91.75,60), np.pi/2)
c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
c1.velocity = Point(0, 3.0)
w.add(c1)

w.render() # This visualizes the world we just constructed.



if not human_controller:
    # Training settings
    EPISODES = 1000
    batch_size = 32
    
    # Initialize DQL agent
    state_size = 4  # distance_to_center, velocity, heading_diff, heading
    action_size = 15  # 5 steering actions * 3 throttle actions
    agent = DQLAgent(state_size, action_size)
    
    for episode in range(EPISODES):
        # Reset the environment
        c1.center = Point(91.75, 60)
        c1.heading = np.pi/2
        c1.velocity = Point(0, 3.0)
        
        # Add lap tracking variables
        last_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)
        lap_count = 0
        crossed_start = False
        
        total_reward = 0
        for time_step in range(600):
            # Get current state
            state = agent.get_state(c1, cb)
            
            # Track laps
            current_angle = np.arctan2(c1.center.y - world_height/2, c1.center.x - world_width/2)
            angle_diff = current_angle - last_angle
            if angle_diff > np.pi:
                angle_diff -= 2*np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2*np.pi
                
            # Check if car crossed starting position (bottom of circle)
            if current_angle > 0 and last_angle < 0 and angle_diff > 0:
                if crossed_start:
                    lap_count += 1
                    reward += 1000  # Bonus reward for completing a lap
                crossed_start = True
                
            last_angle = current_angle
            
            # Get action from agent
            steering, throttle = agent.act(state)
            c1.set_control(steering, throttle)
            
            # Advance simulation
            w.tick()
            w.render()
            
            # Calculate reward
            distance_to_center = c1.distanceTo(cb)
            desired_distance = inner_building_radius + lane_width/2
            distance_error = abs(distance_to_center - desired_distance)
            
            reward = 1.0  # Base reward for surviving
            reward -= distance_error * 0.1  # Penalty for being off-center
            reward -= abs(steering) * 0.1  # Small penalty for steering
            
            # Get new state
            next_state = agent.get_state(c1, cb)
            
            # Check if episode is done
            done = False
            if w.collision_exists():
                reward = -1000  # Big penalty for collision
                done = True
            
            # Store experience in memory
            action_idx = (agent.steering_actions.index(steering) * 
                         len(agent.throttle_actions) + 
                         agent.throttle_actions.index(throttle))
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train the network
            agent.replay(batch_size)
            
            total_reward += reward
            
            if done:
                break
        
        # Update target network every episode
        agent.update_target_model()
        
        print(f"Episode: {episode + 1}/{EPISODES}, Score: {total_reward}")
        
        # Save the model periodically
        if episode % 100 == 0:
            torch.save(agent.model.state_dict(), f'dql_agent_episode_{episode}.pth')
    
    w.close()

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
