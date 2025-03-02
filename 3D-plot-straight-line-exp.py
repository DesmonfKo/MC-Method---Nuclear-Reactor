import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib
import random
import time



# Define global simulation parameters
n_neutrons = 500
mean_free_path = 0.05

# Function to generate random positions inside a sphere
def random_position_sphere(radius):
    pos = rand.uniform(-radius, radius, 3)
    while np.linalg.norm(pos) > radius:
        pos = rand.uniform(-radius, radius, 3)
    return pos

# Function to generate random positions inside a cylinder
def random_position_cylinder(radius, height):
    x, y = rand.uniform(-radius, radius, 2)
    while x**2 + y**2 > radius**2:
        x, y = rand.uniform(-radius, radius, 2)
    z = rand.uniform(-height / 2, height / 2)
    return np.array([x, y, z])

# Function to simulate neutron trajectories
def simulate_neutrons_batch(boundary, size, n_neutrons, mean_free_path):
    trajectories = []

    for i in range(n_neutrons):
        if boundary == "sphere":
            position = random_position_sphere(size)
        elif boundary == "cylinder":
            position = random_position_cylinder(size[0], size[1])
        else:
            raise ValueError("Boundary must be 'sphere' or 'cylinder'")

        path = [position]

        while True:
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)
            d = np.random.exponential(mean_free_path)
            new_position = position + d * direction

            if boundary == "sphere" and np.linalg.norm(new_position) > size:
                break
            elif boundary == "cylinder" and (
                new_position[2] > size[1] / 2 or 
                new_position[2] < -size[1] / 2 or 
                new_position[0]**2 + new_position[1]**2 > size[0]**2
            ):
                break

            path.append(new_position)
            position = new_position

        trajectories.append(np.array(path))

    return trajectories

# Plot trajectories
def plot_trajectories(trajectories, boundary, size, sample_size=20):
    matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1,1,1])

    sampled_trajectories = random.sample(trajectories, min(sample_size, len(trajectories)))

    for traj in sampled_trajectories:
        traj = traj[::10]  # Downsample for speed
        color = np.random.rand(3,)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, alpha=0.6, linewidth=0.8)

    if boundary == "sphere":
        radius = size
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        x = radius * np.outer(np.cos(u), np.sin(v))
        y = radius * np.outer(np.sin(u), np.sin(v))
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='b', alpha=0.1, edgecolor='none')

    elif boundary == "cylinder":
        radius, height = size
        theta = np.linspace(0, 2 * np.pi, 30)
        z = np.linspace(-height / 2, height / 2, 15)
        theta, z = np.meshgrid(theta, z)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        ax.plot_surface(x, y, z, color='g', alpha=0.1, edgecolor='none')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Neutron Trajectories\nMean Free Path = {mean_free_path}, Neutrons = {n_neutrons}, Sample = {sample_size}")

    plt.show(block=True)

# Plot histogram and log-log verification
def plot_free_path_lengths_histogram(step_lengths, sample_size=20):
    plt.figure(figsize=(12, 5))

    # Histogram of Step Lengths
    plt.subplot(1, 2, 1)
    plt.hist(step_lengths, bins=50, density=True, alpha=0.6, color='b', label="Simulated Path Lengths")

    x = np.linspace(0, max(step_lengths), 500)
    y = (1 / mean_free_path) * np.exp(-x / mean_free_path)
    plt.plot(x, y, 'r-', label="Exponential Distribution (Theory)")

    plt.xlabel("Step Length")
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of Neutron Step Lengths\nMean Free Path = {mean_free_path}, Neutrons = {n_neutrons}, Sample = {sample_size}")
    plt.legend()

    # Log-Log Verification (only one plot is generated now)
    plt.subplot(1, 2, 2)
    sorted_lengths = np.sort(step_lengths)
    survival_prob = 1 - np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

    valid = sorted_lengths > 0
    sorted_lengths = sorted_lengths[valid]
    survival_prob = survival_prob[valid]

    plt.plot(sorted_lengths[::10], np.log(survival_prob[::10]), 'bo', markersize=2, label="Log of Survival Probability")
    plt.plot(x, -x / mean_free_path, 'r-', label=f"Expected Linear Fit: y = -x/{mean_free_path:.2f}")

    plt.xlabel("Step Length")
    plt.ylabel("Log(Survival Probability)")
    plt.title("Log-Log Verification of Exponential Distribution")
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)

# Run the simulation
boundary_type = "cylinder"  # Change to "sphere" to switch
size = (5, 10) if boundary_type == "cylinder" else 5

start_time = time.time()
trajectories = simulate_neutrons_batch(boundary_type, size, n_neutrons, mean_free_path)
end_time = time.time()
print(f"Simulation time: {end_time - start_time:.2f} seconds")

# Extract step lengths (downsampled for speed)
all_step_lengths = [np.linalg.norm(traj[i+1] - traj[i]) for traj in trajectories for i in range(0, len(traj)-1, 10)]

# Plot histogram (non-interactive)

plot_free_path_lengths_histogram(all_step_lengths)

# Plot neutron trajectories (interactive)
plot_trajectories(trajectories, boundary_type, size)
