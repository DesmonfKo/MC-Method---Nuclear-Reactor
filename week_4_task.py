# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:51:41 2025

@author: mclau
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

# Function to generate a random unit vector (isotropic directions)
def random_unit_vector():
    v = np.random.normal(size=3)
    return v / np.linalg.norm(v)

# Function to generate random positions inside a sphere
def random_position_sphere(radius):
    while True:
        pos = rand.uniform(-radius, radius, 3)
        if np.linalg.norm(pos) <= radius:
            return pos

# Function to generate random positions inside a cylinder
def random_position_cylinder(radius, height):
    while True:
        x, y = rand.uniform(-radius, radius, 2)
        if x**2 + y**2 <= radius**2:
            break
    z = rand.uniform(-height / 2, height / 2)
    return np.array([x, y, z])

# Function to simulate neutron trajectories
def simulate_neutron_trajectories(boundary, size, n_neutrons=50, mean_free_path=1):
    trajectories = []
    
    for _ in range(n_neutrons):
        # Choose initial position
        if boundary == "sphere":
            position = random_position_sphere(size)
        elif boundary == "cylinder":
            position = random_position_cylinder(size[0], size[1])
        else:
            raise ValueError("Boundary must be 'sphere' or 'cylinder'")
        
        path = [position]  # Store trajectory
        
        while True:
            direction = random_unit_vector()
            d = np.random.exponential(mean_free_path)  # Sample free path
            
            new_position = position + d * direction  # Move neutron
            
            # Check if neutron has exited
            if boundary == "sphere" and np.linalg.norm(new_position) > size:
                break
            elif boundary == "cylinder" and (new_position[2] > size[1]/2 or new_position[2] < -size[1]/2 or new_position[0]**2 + new_position[1]**2 > size[0]**2):
                break
            
            path.append(new_position)
            position = new_position  # Update position
            
        trajectories.append(np.array(path))  # Store path
        
    return trajectories

# Function to plot neutron trajectories
def plot_trajectories(trajectories, boundary, size):
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6)
    if boundary_type == "sphere":
        # Extract radius
        radius = size  
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='none', label="Sphere Boundary")
    
    elif boundary_type == "cylinder":
        # Extract radius and height
        radius, height = size  
        theta = np.linspace(0, 2 * np.pi, 100)
        z = np.linspace(-height / 2, height / 2, 50)
        theta, z = np.meshgrid(theta, z)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot_surface(x, y, z, color='g', alpha=0.1, edgecolor='none', label="Cylinder Boundary")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Neutron Trajectories")

    

# Example usage

boundary_type = "cylinder"  # Change to "cylinder" to switch
size = 5 if boundary_type == "sphere" else (5, 10)  # Radius for sphere, (radius, height) for cylinder


trajectories = simulate_neutron_trajectories(boundary_type, size,1000)
plot_trajectories(trajectories, boundary_type, size)
