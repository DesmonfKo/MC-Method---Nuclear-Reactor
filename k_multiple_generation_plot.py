
"""
Created on Tue Mar  4 12:45:18 2025

@author: mclau
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import matplotlib
from numpy.linalg import norm
from scipy.constants import Avogadro as NA
import random
import time
from scipy.optimize import curve_fit


# ========================
# 1. Input Validation Helpers
# ========================
def get_valid_input(prompt, validation_func, error_msg):
    while True:
        user_input = input(prompt)
        if validation_func(user_input):
            return user_input
        print(f"Invalid input! {error_msg}")

def validate_reactor_type(input_str):
    return input_str.lower() in ['finite', 'infinite']

def validate_geometry(input_str):
    return input_str.lower() in ['sphere', 'cylinder']

def validate_moderator(input_str):
    return input_str.capitalize() in ['H2O', 'D2O', 'Graphite']

def validate_float(input_str, min_val=0, max_val=None):
    try:
        value = float(input_str)
        valid = True
        if min_val is not None: valid &= value >= min_val
        if max_val is not None: valid &= value <= max_val
        return valid
    except ValueError:
        return False

def validate_size(input_str, geometry):
    try:
        if geometry == 'sphere':
            float(input_str)
            return True
        elif geometry == 'cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                values = list(map(float, input_str[1:-1].split(',')))
                return len(values) == 2 and all(v > 0 for v in values)
        return False
    except:
        return False

def validate_dataset(input_str):
    return input_str.lower() in ['lilley', 'wikipedia']

def validate_neutron_model(input_str):
    return input_str.lower() in ['thermal', 'fast']

# ========================
# 2. Material Class Definition
# ========================
class ReactorMaterial:
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0, nu=0, xi=0):
        self.name = name
        self.mat_type = mat_type
        self.density = density * 1e3  # kg/m³
        self.molar_mass = molar_mass * 1e-3  # kg/mol
        
        # Store original barn values
        self.sigma_s_b = sigma_s_b  
        self.sigma_a_b = sigma_a_b
        self.sigma_f_b = sigma_f_b
        
        # Convert to m² for physics calculations
        self.sigma_s = sigma_s_b * 1e-28
        self.sigma_a = sigma_a_b * 1e-28
        self.sigma_f = sigma_f_b * 1e-28
        
        self.nu = nu
        self.xi = xi

    @property
    def number_density(self):
        return (self.density / self.molar_mass) * NA

# ========================
# 3. Material Database
# ========================
# Lilley's default materials
LILLEY_U235 = ReactorMaterial(
    name="U235", mat_type='fuel',
    density=18.7, molar_mass=235,
    sigma_s_b=10, sigma_a_b=680, sigma_f_b=579,
    nu=2.42, xi=0
)

LILLEY_U238 = ReactorMaterial(
    name="U238", mat_type='fuel',
    density=18.9, molar_mass=238,
    sigma_s_b=8.3, sigma_a_b=2.72,
    sigma_f_b=0, nu=0, xi=0
)

MODERATORS = {
    'H20': ReactorMaterial(
        name="H20", mat_type='moderator',
        density=1.0, molar_mass=18.01,
        sigma_s_b=49.2, sigma_a_b=0.66,
        xi=0.920
    ),
    'D20': ReactorMaterial(
        name="D20", mat_type='moderator',
        density=1.1, molar_mass=20.02,
        sigma_s_b=10.6, sigma_a_b=0.001,
        xi=0.509
    ),
    'Graphite': ReactorMaterial(
        name="Graphite", mat_type='moderator',
        density=1.6, molar_mass=12.01,
        sigma_s_b=4.7, sigma_a_b=0.0045,
        xi=0.158
    )
}

# ========================
# 4. Modified Reactor Mixture Class
# ========================
class ReactorMixture:
    def __init__(self, u235_percent, moderator, R_mtf, u235_material, u238_material):
        if not 0 <= u235_percent <= 100:
            raise ValueError("U-235 concentration must be 0-100%")
        self.u235_frac = u235_percent / 100
        self.moderator = moderator
        self.R_mtf = R_mtf
        self.u235_material = u235_material
        self.u238_material = u238_material

    @property
    def macroscopic_scattering(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        N_fuel = N_U235 + N_U238
        N_mod = N_fuel * self.R_mtf
        return (N_U235 * self.u235_material.sigma_s + 
                N_U238 * self.u238_material.sigma_s + 
                N_mod * self.moderator.sigma_s)

    @property
    def macroscopic_absorption(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        N_fuel = N_U235 + N_U238
        N_mod = N_fuel * self.R_mtf
        return (N_U235 * self.u235_material.sigma_a + 
                N_U238 * self.u238_material.sigma_a + 
                N_mod * self.moderator.sigma_a)

    @property
    def macroscopic_fission(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        return (N_U235 * self.u235_material.sigma_f + 
                N_U238 * self.u238_material.sigma_f)

    @property
    def macroscopic_total(self):
        return self.macroscopic_scattering + self.macroscopic_absorption

    @property
    def scattering_probability(self):
        total = self.macroscopic_total
        return self.macroscopic_scattering / total if total > 0 else 0

    @property
    def absorption_probability(self):
        total = self.macroscopic_total
        return self.macroscopic_absorption / total if total > 0 else 0

    @property
    def fission_probability(self):
        return (self.macroscopic_fission / self.macroscopic_absorption 
                if self.macroscopic_absorption > 0 else 0)

# ========================
# 5. Physics Implementation
# ========================
class ReactorSimulator:
    def __init__(self, mixture, geometry, size):
        self.mixture = mixture
        self.geometry = geometry
        self.size = size
        
    @property
    def total_mean_free_path(self):
        return 1 / self.mixture.macroscopic_total
    
    def resonance_escape(self):
        N_U238 = (1 - self.mixture.u235_frac) / self.mixture.R_mtf
        sigma_s_mod = self.mixture.moderator.sigma_s_b
        term = (N_U238 / sigma_s_mod) ** 0.514
        return np.exp(-2.73 / self.mixture.moderator.xi * term)
    
    def thermal_utilization(self):
        N_U235 = self.mixture.u235_material.number_density * self.mixture.u235_frac
        N_U238 = self.mixture.u238_material.number_density * (1 - self.mixture.u235_frac)
        N_mod = (N_U235 + N_U238) * self.mixture.R_mtf
        
        fuel_abs = N_U235 * self.mixture.u235_material.sigma_a + N_U238 * self.mixture.u238_material.sigma_a
        total_abs = fuel_abs + N_mod * self.mixture.moderator.sigma_a
        return fuel_abs / total_abs
    
    def calculate_k(self, c=1):
        u235 = self.mixture.u235_material
        u238 = self.mixture.u238_material
        numerator = u235.number_density * self.mixture.u235_frac * u235.sigma_f * u235.nu
        denominator = (u235.number_density * self.mixture.u235_frac * u235.sigma_a +
                      u238.number_density * (1 - self.mixture.u235_frac) * u238.sigma_a)
        η = numerator / denominator
        return η * self.thermal_utilization() * self.resonance_escape() * c

    def first_generation_k_factor(self):
        return self.mixture.fission_probability * self.mixture.u235_material.nu

# ========================
# 6. Enhanced User Interface
# ========================
def reactor_properties():
    # Get reactor type
    reactor_type = get_valid_input(
        "Reactor type [finite/infinite]: ",
        validate_reactor_type,
        "Must be 'finite' or 'infinite'"
    ).lower()

    # Get geometry
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'"
    ).lower()

    # Get size
    size_prompt = ("Enter size in meters:\n- Sphere: radius\n- Cylinder: [radius,height]\nSize: ")
    size = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format"
    )

    if geometry == 'cylinder':
        size = list(map(float, size.strip('[]').split(',')))
    else:
        size = float(size)

    # Get moderator
    moderator = get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be Water, D2O, or Graphite"
    ).capitalize()

    # Get dataset
    dataset = get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be 'Lilley' or 'Wikipedia'"
    ).lower()

    if dataset == 'wikipedia':
        neutron_model = get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be 'Thermal' or 'Fast'"
        ).lower()

        # Create Wikipedia materials
        if neutron_model == 'thermal':
            u235 = ReactorMaterial(
                name="U235", mat_type='fuel',
                density=18.7, molar_mass=235,
                sigma_s_b=10, sigma_a_b=99+583, sigma_f_b=583,
                nu=2.42, xi=0
            )
            u238 = ReactorMaterial(
                name="U238", mat_type='fuel',
                density=18.9, molar_mass=238,
                sigma_s_b=9, sigma_a_b=2+0.00002, sigma_f_b=0.00002,
                nu=0, xi=0
            )
        else:  # fast
            u235 = ReactorMaterial(
                name="U235", mat_type='fuel',
                density=18.7, molar_mass=235,
                sigma_s_b=4, sigma_a_b=0.09+1, sigma_f_b=1,
                nu=2.42, xi=0
            )
            u238 = ReactorMaterial(
                name="U238", mat_type='fuel',
                density=18.9, molar_mass=238,
                sigma_s_b=5, sigma_a_b=0.07+0.3, sigma_f_b=0.3,
                nu=0, xi=0
            )
    else:
        neutron_model = 'N/A'
        u235 = LILLEY_U235
        u238 = LILLEY_U238

    # Get U-235 concentration
    u235_percent = float(get_valid_input(
        "U-235 concentration [0-100%]: ",
        lambda x: validate_float(x, 0, 100),
        "Must be number between 0 and 100"
    ))

    # Get R_mtf
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio (R_mtf > 0): ",
        lambda x: validate_float(x, 0.01),
        "Must be positive number"
    ))

    # Initialize components
    moderator_obj = MODERATORS[moderator]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235, u238)
    simulator = ReactorSimulator(mixture, geometry, size)

    # Calculate leakage
    c = 1 if reactor_type == 'infinite' else (1 - 0.1) * (1 - 0.2)  # Example values
    
    # Results
    print("\n=== Reactor Physics Parameters ===")
    print(f"Thermal utilization (f): {simulator.thermal_utilization():.3f}")
    print(f"Resonance escape (p): {simulator.resonance_escape():.3f}")
    print(f"Scattering probability: {mixture.scattering_probability:.3f}")
    print(f"Absorption probability: {mixture.absorption_probability:.3f}")
    print(f"Fission probability: {mixture.fission_probability:.3f}")
    print(f"Total mean free path (R_mtf={R_mtf}): {simulator.total_mean_free_path:.2e} m")
    print(f"1st_generation_k_factor: {simulator.first_generation_k_factor():.3f}")
    print(f"k_effective: {simulator.calculate_k(c):.3f}")
    
    return {
        "total_mean_free_path": simulator.total_mean_free_path,
        "reactor_type": reactor_type,
        "size": size,
        "geometry": geometry,
        "absorption_probability": mixture.absorption_probability,
        "fission_probability": mixture.fission_probability,
        "neutron_model": neutron_model,
        "moderator": moderator,
        "U35-concentration": u235_percent,
        "dataset": dataset,
        "R_mtf": R_mtf,
        

    }



# ========================
# Neutron Position Generation
# ========================
def random_position_sphere_optimized(radius=1):
    vec = rand.randn(3)
    vec /= norm(vec)
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1, height=1):
    theta = rand.uniform(0, 2 * np.pi)
    r = np.sqrt(rand.uniform(0, 1)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height/2, height/2)
    return np.array([x, y, z])

# ========================
# Neutron Simulation Core
# ========================


def simulate_neutron_trajectory(geometry='sphere', size=1, mean_free_path=1, absorption_prob=0.5, fission_prob=0.8, nu=2.42):
    if geometry == 'sphere':
        radius = size
        pos = random_position_sphere_optimized(radius)
    elif geometry == 'cylinder':
        radius, height = size
        pos = random_position_cylinder(radius, height)
    
    trajectory = [pos.copy()]
    step_lengths = []
    fission_occurred = False
    
    while True:
        direction = rand.randn(3)
        direction /= norm(direction)
        step_length = rand.exponential(mean_free_path)
        
        # Calculate maximum allowed step
        if geometry == 'sphere':
            a = np.dot(direction, direction)
            b = 2 * np.dot(pos, direction)
            c = np.dot(pos, pos) - radius**2
            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                t_max = np.inf
            else:
                t_max = min([t for t in [(-b + np.sqrt(discriminant))/(2*a),
                                       (-b - np.sqrt(discriminant))/(2*a)] if t > 0])
        elif geometry == 'cylinder':
            # Radial collision
            x, y, z = pos
            dx, dy, dz = direction
            a_rad = dx**2 + dy**2
            b_rad = 2*(x*dx + y*dy)
            c_rad = x**2 + y**2 - radius**2
            disc_rad = b_rad**2 - 4*a_rad*c_rad
            t_rad = min([t for t in [(-b_rad + np.sqrt(disc_rad))/(2*a_rad),
                                   (-b_rad - np.sqrt(disc_rad))/(2*a_rad)] if t > 0]) if disc_rad >= 0 else np.inf
            
            # Axial collision
            if dz == 0:
                t_axial = np.inf if abs(z) <= height/2 else 0
            else:
                t_upper = (height/2 - z)/dz
                t_lower = (-height/2 - z)/dz
                t_axial = min([t for t in [t_upper, t_lower] if t > 0])
            
            t_max = min(t_rad, t_axial)
        
        if step_length > t_max:
            exit_pos = pos + direction * t_max
            trajectory.append(exit_pos)
            return (False, True, trajectory, step_lengths, fission_occurred)
        else:
            new_pos = pos + direction * step_length
            trajectory.append(new_pos.copy())
            step_lengths.append(step_length)
            pos = new_pos
            if rand.uniform() < absorption_prob:
                # Check for fission
                if rand.uniform() < fission_prob:
                    fission_occurred = True
                    return (True, False, trajectory, step_lengths, fission_occurred)
                else:
                    return (True, False, trajectory, step_lengths, fission_occurred)


# ========================
# Batch Simulation and Plotting
# ========================


def plot_trajectories(trajectories, geometry, size, reactor_propeties_dictionary, n_neutrons):
    matplotlib.use('TkAgg')
    plt.figure(figsize=(8, 6))
    ax = plt.axes(projection='3d')
    ax.set_box_aspect([1,1,1])
    
    for traj in trajectories:
        if len(traj) > 1:
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.6, lw=0.8, marker='o', markersize=1)

    
    if geometry == 'sphere':
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = size * np.outer(np.cos(u), np.sin(v))
        y = size * np.outer(np.sin(u), np.sin(v))
        z = size * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='blue', alpha=0.1)
    else:
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(-size[1]/2, size[1]/2, 30)
        theta, z = np.meshgrid(theta, z)
        x = size[0] * np.cos(theta)
        y = size[0] * np.sin(theta)
        ax.plot_surface(x, y, z, color='green', alpha=0.1)
    

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    

    geometry = reactor_propeties_dictionary["geometry"]
    size = reactor_propeties_dictionary["size"]
    mean_free_path = reactor_propeties_dictionary["total_mean_free_path"]
    neutron_model = reactor_propeties_dictionary["neutron_model"]
    moderator = reactor_propeties_dictionary["moderator"]
    u235_percent = reactor_propeties_dictionary["U35-concentration"]
    R_mtf =  reactor_propeties_dictionary["R_mtf"]
    dataset = reactor_propeties_dictionary["dataset"]
    
        # Set the title with n_neutrons, U-235 concentration, and R_mtf
    plt.title(f"Neutron Simulation: {n_neutrons} Neutrons | U-235: {u235_percent:.2f}% | R_mtf: {R_mtf:.2f}", fontsize=14, fontweight='bold')
    
    caption_text = (f"Moderator: {moderator} | Neutron Model: {neutron_model} | Dataset: {dataset}\n"
                    f"Size: {size} | Mean Free Path: {mean_free_path:.3e} m")
    plt.figtext(0.5, 0.01, caption_text, ha='center', va='top', fontsize=10, wrap=True)
    plt.show(block=True)



def simulate_multiple_generations(n_generations, initial_neutrons, geometry='sphere', size=1,
                                  mean_free_path=1, absorption_prob=0.2, fission_prob=0.8, nu=2.42,
                                  max_neutrons=1e5):
    """
    Simulate multiple generations of neutrons, tracking the multiplication factor k_n.
    
    Parameters:
        n_generations (int): Number of generations to simulate.
        initial_neutrons (int): Initial number of neutrons in the first generation.
        geometry (str): Reactor geometry ('sphere' or 'cylinder').
        size (float or tuple): Size parameters for the reactor.
        mean_free_path (float): Neutron mean free path.
        absorption_prob (float): Probability of absorption per collision.
        fission_prob (float): Probability that an absorption leads to fission.
        nu (float): Average neutrons produced per fission.
        max_neutrons (int): Maximum number of neutrons per generation to prevent overflow.
    
    Returns:
        list: Multiplication factors for each generation.
    """
    k_values = []
    neutrons_current = initial_neutrons
    
    for gen in range(n_generations):
        if neutrons_current <= 0:
            k_values.append(0.0)
            continue
        
        # Simulate current generation
        fissions = 0
        for _ in range(int(neutrons_current)):
            fission, leaked, _, _, fission_occurred = simulate_neutron_trajectory(
                geometry, size, mean_free_path, absorption_prob, fission_prob, nu
            )
            if fission_occurred:
                fissions += 1
        
        # Calculate next generation's neutrons (Poisson-distributed)
        neutrons_next = np.sum(np.random.poisson(nu, fissions))
        
        # Prevent overflow
        if neutrons_next > max_neutrons:
            neutrons_next = max_neutrons
        
        # Calculate k_n for this generation
        k_n = neutrons_next / neutrons_current if neutrons_current > 0 else 0.0
        k_values.append(k_n)
        
        # Update for next generation
        neutrons_current = neutrons_next
    
    # Plotting
    plt.plot(range(n_generations), k_values)
    plt.xlabel('Generation (n)')
    plt.ylabel('Multiplication Factor (k_n)')
    plt.title('Multiplication Factor vs. Generation')
    plt.show()
    
    return k_values


# Example usage with reactor properties
if __name__ == '__main__':
    reactor_props = reactor_properties()  # Ensure this function returns 'fission_prob' and 'nu'
    
    # Extract parameters
    geometry = reactor_props['geometry']
    size = reactor_props['size']
    mean_free_path = reactor_props['total_mean_free_path']
    absorption_prob = reactor_props['absorption_probability']
    fission_prob = reactor_props['fission_probability']  # Added to reactor_properties()
    nu = 2
    
    # Simulate
    k_vals = simulate_multiple_generations(
        n_generations=40,
        initial_neutrons=1000,
        geometry=geometry,
        size=size,
        mean_free_path=mean_free_path,
        absorption_prob=absorption_prob,
        fission_prob=fission_prob,
        nu=nu,
        max_neutrons=100000
    )   
