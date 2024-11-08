import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from dql_agent import DQLAgent  # Import the DQL agent

human_controller = False

dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(dt, width = 120, height = 120, ppm = 6) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.

# top left building
w.add(Painting(Point(8.5, 106.5), Point(17, 27), 'gray80'))
w.add(RectangleBuilding(Point(7.5, 107.5), Point(15, 25)))

# bottom left building
w.add(Painting(Point(8.5, 41), Point(17, 82), 'gray80'))
w.add(RectangleBuilding(Point(7.5, 40), Point(15, 80)))

# top middle building
w.add(Painting(Point(60, 106.5), Point(70, 27), 'gray80')) 
w.add(RectangleBuilding(Point(60, 107.5), Point(66, 25))) 

# bottom middle building
w.add(Painting(Point(60, 41), Point(70, 82), 'gray80'))
w.add(RectangleBuilding(Point(60, 40), Point(66, 80)))

# top right building
w.add(Painting(Point(111.5, 106.5), Point(17, 27), 'gray80'))  
w.add(RectangleBuilding(Point(112.5, 107.5), Point(15, 25)))

# bottom right building
w.add(Painting(Point(111.5, 41), Point(17, 82), 'gray80'))  
w.add(RectangleBuilding(Point(112.5, 40), Point(15, 80)))

w.add(Painting(Point(99, 8), Point(10, 10), 'white'))

# Let's also add some zebra crossings, because why not.
w.add(Painting(Point(18, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(19, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(20, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(21, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(22, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(23, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(24, 81), Point(0.5, 2), 'white'))

# Add zebra crossings for the new intersection (similar pattern as existing ones)
w.add(Painting(Point(96, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(97, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(98, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(99, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(100, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(101, 81), Point(0.5, 2), 'white'))
w.add(Painting(Point(102, 81), Point(0.5, 2), 'white'))

# Replace c1 with DQL agent and define goal
goal = Point(99, 8)
c1 = DQLAgent(Point(20,20), np.pi/2, goal)
w.add(c1)

c2 = Car(Point(118,90), np.pi, 'blue')
c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(c2)

c3 = Car(Point(10,86), 0, 'yellow')
c3.velocity = Point(0.0,3.0) # We can also specify an initial velocity just like this.
w.add(c3)

# Add a pedestrian at the top left zebra crossing
p1 = Pedestrian(Point(28, 81), np.pi)
p1.max_speed = 10.0
w.add(p1)

# Add a pedestrian at the top right zebra crossing
p2 = Pedestrian(Point(90, 81), 0)
p2.max_speed = 10.0
w.add(p2)

w.render() # This visualizes the world we just constructed.


if not human_controller:
    num_episodes = 1000
    max_steps = 400
    for episode in range(num_episodes):        
        w.reset()
        # Reset agent position and add back to world
        c1.center = Point(20, 20)
        c1.heading = np.pi/2
        c1.velocity = Point(0, 0)
        w.add(c1)
        
        # Reset other agents
        c2.center = Point(118, 90)
        c2.heading = np.pi
        c2.velocity = Point(3.0, 0)
        w.add(c2)
        
        c3.center = Point(10, 86)
        c3.heading = 0
        c3.velocity = Point(0.0, 3.0)
        w.add(c3)
        
        p1.center = Point(28, 81)
        p1.heading = np.pi
        w.add(p1)
        
        p2.center = Point(90, 81)
        p2.heading = 0
        w.add(p2)
        
        # Reset controls for other agents
        p1.set_control(0, 0.22)
        p2.set_control(0, 0.22)
        c2.set_control(0, 0.05)
        c3.set_control(0, 0.05)
        
        episode_rewards = 0
        c1.prev_distance = c1.position.distanceTo(goal)

        w.render()
        
        print(f'\nStarting Episode {episode + 1}')
        
        for step in range(max_steps):
            state = c1.get_state(w)
            action = c1.select_action(state)
            c1.set_control(action[0], action[1])
            
            # Update other cars at specific times
            if step == 325:
                c2.set_control(-0.45, 0.3)
            elif step == 367:
                c2.set_control(0, 0.1)
            
            w.tick()
            w.render()
            #time.sleep(dt/4)
            
            reward = 0
            done = False
            
            # Check for collisions
            if w.collision_exists():
                print('Collision occurred!')
                reward = -100
                done = True
            
            # Check if goal reached
            distance_to_goal = c1.position.distanceTo(goal)
            if distance_to_goal < 5:
                print('Goal reached!')
                reward = 100
                done = True
            
            if not done:
                reward = -0.1
                reward += 0.3 * (c1.prev_distance - distance_to_goal)
            
            next_state = c1.get_state(w)
            c1.store_transition(state, action, reward, next_state, done)
            c1.train()
            
            c1.prev_distance = distance_to_goal
            episode_rewards += reward
            
            if done:
                print(f'Episode {episode + 1} finished after {step} steps. Total reward: {episode_rewards:.2f}')
                break
        
        # If episode didn't end naturally
        if not done:
            print(f'Episode {episode + 1} timed out. Total reward: {episode_rewards:.2f}')

    w.close()

else: # Let's use the steering wheel (Logitech G29) for the human control of car c1
    p1.set_control(0, 0.22) # The pedestrian will have 0 steering and 0.22 throttle. So it will not change its direction.
    c2.set_control(0, 0.35)
    
    from interactive_controllers import SteeringWheelController
    controller = SteeringWheelController(w)
    for k in range(400):
        c1.set_control(controller.steering, controller.throttle)
        w.tick() # This ticks the world for one time step (dt second)
        w.render()
        time.sleep(dt/4) # Let's watch it 4x
        if w.collision_exists():
            import sys
            sys.exit(0)