import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from scipy.constants import Avogadro as NA
NA = 6.022e23  # Avogadro's number

# Basically generate a random radius to multiply the unit vector generated, so that every position generated
# definitely within the radius;

def random_position_sphere_optimized(radius=1):
    '''
    # Generate random direction (unit vector)

    Parameters
    ----------
    radius : FLOAT, optional
        The radius of the sphere for modelling the nuclear reactor. Default is unit 1.

    Returns
    -------
    r * vec : ARRAY
        This is an array of 1 row 3 column due to setting the size as 3. Only return this if its length is smaller
        or equal to the set radius.

    '''
    # Generate a vector in the form [x, y, z]
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)
    
    # Scale to random radius^3 (to ensure uniform density)
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1, height=1):
    '''
    # Generate random direction (unit vector)

    Parameters
    ----------
    radius : FLOAT, optional
        The radius of the sphere for modelling the nuclear reactor. Default is unit 1.

    height : FLOAT, optional
        The height of the sphere for modelling the nuclear reactor. Default is unit 1.
    
    Returns
    -------
    np.array : ARRAY
        This is an array of 1 row 3 column due to setting the size as 3. Only return this if it fulfills the
        criteria.

    '''
    # Generate random angle (theta) and radial distance (r)
    theta = rand.uniform(0, 2 * np.pi)  # Uniform angle in [0, 2π)
    r = np.sqrt(rand.uniform(0, 1)) * radius  # Radial distance (corrected for uniform density)
    
    # Convert polar coordinates (r, theta) to Cartesian (x, y)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Generate z uniformly along the cylinder's height
    z = rand.uniform(-height / 2, height / 2)
    
    return np.array([x, y, z])

### By D

def random_position_sphere_optimized(radius=1):
    '''
    Generates a random position within a sphere of given radius with uniform density.

    Parameters
    ----------
    radius : float, optional
        The radius of the sphere. Default is 1.

    Returns
    -------
    np.ndarray
        A 3D vector representing the position within the sphere.
    '''
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1, height=1):
    '''
    Generates a random position within a cylinder of given radius and height with uniform density.

    Parameters
    ----------
    radius : float, optional
        The radius of the cylinder. Default is 1.
    height : float, optional
        The height of the cylinder. Default is 1.

    Returns
    -------
    np.ndarray
        A 3D vector representing the position within the cylinder.
    '''
    theta = rand.uniform(0, 2 * np.pi)
    r = np.sqrt(rand.uniform(0, 1)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height/2, height/2)
    return np.array([x, y, z])

def simulate_neutron(reactor_type='sphere', radius=1, height=1, mean_free_path=1, absorption_prob=0.5):
    '''
    Simulates the random walk of a single neutron in a reactor until it leaks or is absorbed.

    Parameters
    ----------
    reactor_type : str, optional
        The geometry of the reactor, either 'sphere' or 'cylinder'. Default is 'sphere'.
    radius : float, optional
        Radius of the reactor (sphere or cylinder). Default is 1.
    height : float, optional
        Height of the cylinder (only used if reactor_type is 'cylinder'). Default is 1.
    mean_free_path : float, optional
        Mean free path for neutron collisions. Default is 1.
    absorption_prob : float, optional
        Probability of absorption at each collision. Default is 0.5.

    Returns
    -------
    tuple (bool, bool)
        (absorbed, leaked) where each is True if the event occurred.
    '''
    # Initialize position based on reactor type
    if reactor_type == 'sphere':
        pos = random_position_sphere_optimized(radius)
    elif reactor_type == 'cylinder':
        pos = random_position_cylinder(radius, height)
    else:
        raise ValueError("Invalid reactor_type. Choose 'sphere' or 'cylinder'.")

    while True:
        # Sample step length to next collision
        step_length = rand.exponential(mean_free_path)

        # Generate random direction
        direction = rand.randn(3)
        direction /= np.linalg.norm(direction)

        # Compute maximum allowed step before exiting reactor
        if reactor_type == 'sphere':
            a = np.dot(direction, direction)
            b = 2 * np.dot(pos, direction)
            c = np.dot(pos, pos) - radius**2
            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                t_max = np.inf
            else:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b + sqrt_disc) / (2 * a)
                t2 = (-b - sqrt_disc) / (2 * a)
                t_candidates = [t for t in [t1, t2] if t > 0]
                t_max = min(t_candidates) if t_candidates else np.inf
        elif reactor_type == 'cylinder':
            x, y, z = pos
            dx, dy, dz = direction

            # Radial constraint
            a_rad = dx**2 + dy**2
            b_rad = 2 * (x * dx + y * dy)
            c_rad = x**2 + y**2 - radius**2
            disc_rad = b_rad**2 - 4 * a_rad * c_rad
            if disc_rad < 0:
                t_rad = np.inf
            else:
                sqrt_disc_rad = np.sqrt(disc_rad)
                t1 = (-b_rad + sqrt_disc_rad) / (2 * a_rad)
                t2 = (-b_rad - sqrt_disc_rad) / (2 * a_rad)
                t_rad_candidates = [t for t in [t1, t2] if t > 0]
                t_rad = min(t_rad_candidates) if t_rad_candidates else np.inf

            # Axial constraint
            if dz == 0:
                t_axial = np.inf if abs(z) <= height / 2 else 0.0
            else:
                t_upper = (height / 2 - z) / dz
                t_lower = (-height / 2 - z) / dz
                t_axial_candidates = [t for t in [t_upper, t_lower] if t > 0]
                t_axial = min(t_axial_candidates) if t_axial_candidates else np.inf

            t_max = min(t_rad, t_axial)
        else:
            raise ValueError("Invalid reactor_type.")

        # Check if step exceeds t_max
        if step_length > t_max:
            return (False, True)  # Neutron leaks
        else:
            pos += direction * step_length
            # Check absorption
            if rand.uniform() < absorption_prob:
                return (True, False)  # Neutron absorbed

def simulate_generation(N, reactor_type='sphere', radius=1, height=1, mean_free_path=1, absorption_prob=0.5):
    '''
    Simulates a generation of N neutrons and counts absorptions and leaks.

    Parameters
    ----------
    N : int
        Number of neutrons to simulate.
    reactor_type : str, optional
        Geometry of the reactor. Default is 'sphere'.
    radius : float, optional
        Radius of the reactor. Default is 1.
    height : float, optional
        Height of the cylinder (if reactor_type is 'cylinder'). Default is 1.
    mean_free_path : float, optional
        Mean free path for collisions. Default is 1.
    absorption_prob : float, optional
        Absorption probability per collision. Default is 0.5.

    Returns
    -------
    tuple (int, int)
        Number of absorbed neutrons, number of leaked neutrons.
    '''
    absorbed = 0
    leaked = 0

    for _ in range(N):
        a, l = simulate_neutron(reactor_type=reactor_type, radius=radius, height=height,
                                mean_free_path=mean_free_path, absorption_prob=absorption_prob)
        if a:
            absorbed += 1
        elif l:
            leaked += 1

    return absorbed, leaked

# Example usage:
'''
absorbed_cylinder, leaked_cylinder = simulate_generation(1000, reactor_type='cylinder',\
                                                         radius=100, height=50, mean_free_path=2.08,\
                                                         absorption_prob=0.038)
print(f"Absorbed: {absorbed_cylinder}, Leaked: {leaked_cylinder}")
'''

absorbed, leaked = simulate_generation(1000, reactor_type='sphere',\
                                       radius=5, mean_free_path=2.08,\
                                       absorption_prob=0.038)
print(f"Absorbed: {absorbed}, Leaked: {leaked}")
