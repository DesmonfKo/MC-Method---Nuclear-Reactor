# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 20:20:33 2025

@author: mclau
"""

# PATCH 1 of N
#####################################

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

# Avogadro's number
NA = 6.022e23

########################################
# 1. Input Validation Helpers
########################################
def get_valid_input(prompt, validation_func, error_msg):
    """
    Continuously prompt the user until validation_func returns True.
    Returns the user input (string) once it is valid.
    """
    while True:
        user_input = input(prompt).strip()
        if validation_func(user_input):
            return user_input
        print(f"Invalid input! {error_msg}")


def validate_geometry(input_str):
    return input_str.lower() in ['sphere','cylinder']


def validate_float(input_str, min_val=None, max_val=None):
    """
    If min_val is not None, enforce value >= min_val.
    If max_val is not None, enforce value <= max_val.
    """
    try:
        value = float(input_str)
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    except ValueError:
        return False


def validate_size(input_str, geometry):
    """
    We expect:
     - For 'sphere': a single float for radius
     - For 'cylinder': a 2-element list [radius, height]
    """
    try:
        if geometry == 'sphere':
            val = float(input_str)
            return (val > 0)
        elif geometry == 'cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                values = list(map(float, input_str[1:-1].split(',')))
                if len(values) == 2 and all(v>0 for v in values):
                    return True
        return False
    except:
        return False


def validate_dataset(input_str):
    return input_str.lower() in ['lilley','wikipedia']


def validate_neutron_model(input_str):
    return input_str.lower() in ['thermal','fast']


def validate_moderator(input_str):
    """
    We will allow user to type H2O, D2O, or Graphite in any case (upper/lower).
    We'll rename them internally, but just check membership ignoring case.
    """
    val = input_str.strip().lower()
    return val in ['h2o','d2o','graphite']


def validate_positive_integer(input_str):
    """
    For user input that must be a positive integer (e.g. number of neutrons, number of generations).
    """
    try:
        val = int(input_str)
        return (val > 0)
    except ValueError:
        return False

########################################
# 2. Random Position Generators
########################################
def random_position_cylinder(radius=1, height=1):
    '''
    Generate a random position uniformly within a right circular cylinder
    of radius=radius and height=height.

    Returns
    -------
    np.array of shape (3,) with [x, y, z].
    '''
    theta = rand.uniform(0, 2 * np.pi)  # uniform angle
    r = np.sqrt(rand.uniform(0, 1)) * radius  # radial distance
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height / 2, height / 2)
    return np.array([x, y, z])


def random_position_sphere_optimized(radius=1):
    '''
    Generate a random position uniformly within a sphere of given radius.

    Returns
    -------
    np.array of shape (3,).
    '''
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)   # random direction
    r = radius * (rand.uniform() ** (1/3))  # random radius^3 for uniform inside volume
    return vec * r

#####################################
# PATCH 2 of N
#####################################
class ReactorMaterial:
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name = name
        self.mat_type = mat_type
        self.density_gcc = density
        self.molar_mass_gmol = molar_mass
        self.density = density * 1e3
        self.molar_mass = molar_mass*1e-3
        self.sigma_s_b = sigma_s_b
        self.sigma_a_b = sigma_a_b
        self.sigma_f_b = sigma_f_b
        self.sigma_s = sigma_s_b * 1e-28
        self.sigma_a = sigma_a_b * 1e-28
        self.sigma_f = sigma_f_b * 1e-28
        self.nu = nu
        self.xi = xi

    @property
    def number_density(self):
        return (self.density / self.molar_mass) * NA


###############################
# Lilley's default materials
###############################
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
    sigma_f_b=0.0, nu=0, xi=0
)

MODERATORS_DICT = {
    'h2o': ReactorMaterial(
        name="H2O", mat_type='moderator',
        density=1.0, molar_mass=18.01,
        sigma_s_b=49.2, sigma_a_b=0.66,
        xi=0.920
    ),
    'd2o': ReactorMaterial(
        name="D2O", mat_type='moderator',
        density=1.1, molar_mass=20.02,
        sigma_s_b=10.6, sigma_a_b=0.001,
        xi=0.509
    ),
    'graphite': ReactorMaterial(
        name="Graphite", mat_type='moderator',
        density=1.6, molar_mass=12.01,
        sigma_s_b=4.7, sigma_a_b=0.0045,
        xi=0.158
    )
}


###############################
# Wikipedia cross-section sets
###############################
def get_wikipedia_materials(neutron_model='thermal'):
    """
    Return (u235, u238) for 'thermal' or 'fast' cross section sets.
    """
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
    return u235, u238


########################################
# ReactorMixture
########################################
class ReactorMixture:
    def __init__(self, fraction_U235, moderator, R_mtf, u235_material, u238_material):
        self.fraction_U235 = fraction_U235
        self.moderator = moderator
        self.R_mtf = R_mtf
        self.u235 = u235_material
        self.u238 = u238_material

    @property
    def macroscopic_scattering(self):
        aU235 = self.fraction_U235
        aU238 = 1.0 - aU235
        B = aU235 + aU238 + self.R_mtf
        conv = 1.0e6 * NA * 1.0e-28
        part_u235 = (aU235/B)*(self.u235.density_gcc/self.u235.molar_mass_gmol)*self.u235.sigma_s_b
        part_u238 = (aU238/B)*(self.u238.density_gcc/self.u238.molar_mass_gmol)*self.u238.sigma_s_b
        part_mod  = (self.R_mtf/B)*(self.moderator.density_gcc/self.moderator.molar_mass_gmol)*self.moderator.sigma_s_b
        return conv*(part_u235 + part_u238 + part_mod)

    @property
    def macroscopic_absorption(self):
        aU235 = self.fraction_U235
        aU238 = 1.0 - aU235
        B = aU235 + aU238 + self.R_mtf
        conv = 1.0e6 * NA * 1.0e-28
        part_u235 = (aU235/B)*(self.u235.density_gcc/self.u235.molar_mass_gmol)*self.u235.sigma_a_b
        part_u238 = (aU238/B)*(self.u238.density_gcc/self.u238.molar_mass_gmol)*self.u238.sigma_a_b
        part_mod  = (self.R_mtf/B)*(self.moderator.density_gcc/self.moderator.molar_mass_gmol)*self.moderator.sigma_a_b
        return conv*(part_u235 + part_u238 + part_mod)

    @property
    def macroscopic_fission(self):
        aU235 = self.fraction_U235
        aU238 = 1.0 - aU235
        B     = aU235 + aU238 + self.R_mtf
        conv  = 1.0e6 * NA * 1.0e-28
        part_u235 = (aU235/B)*(self.u235.density_gcc/self.u235.molar_mass_gmol)*self.u235.sigma_f_b
        part_u238 = (aU238/B)*(self.u238.density_gcc/self.u238.molar_mass_gmol)*self.u238.sigma_f_b
        return conv*(part_u235 + part_u238)

    @property
    def fission_probability(self):
        sigma_f = self.macroscopic_fission
        sigma_a = self.macroscopic_absorption
        return sigma_f / sigma_a if sigma_a > 1e-30 else 0.0

    @property
    def scattering_probability(self):
        s = self.macroscopic_scattering
        a = self.macroscopic_absorption
        denom = s + a
        return s/denom if denom>1e-30 else 0.0

    @property
    def absorption_probability(self):
        s = self.macroscopic_scattering
        a = self.macroscopic_absorption
        denom = s + a
        return a/denom if denom>1e-30 else 0.0
    

class ReflectorMaterial:
    def __init__(self, name, sigma_s_b, sigma_a_b, density_gcc, molar_mass_gmol):
        self.name = name
        self.sigma_s_b = sigma_s_b
        self.sigma_a_b = sigma_a_b
        self.density_gcc = density_gcc
        self.molar_mass_gmol = molar_mass_gmol
        self.sigma_s = sigma_s_b * 1e-28
        self.sigma_a = sigma_a_b * 1e-28
        self.number_density = (density_gcc * 1e3 / molar_mass_gmol) * NA

    @property
    def macroscopic_scattering(self):
        return self.number_density * self.sigma_s

    @property
    def macroscopic_absorption(self):
        return self.number_density * self.sigma_a

    @property
    def reflection_probability(self):
        sigma_total = self.macroscopic_scattering + self.macroscopic_absorption
        return self.macroscopic_scattering / sigma_total if sigma_total > 0 else 0.0

#####################################
# PATCH 3 of N
#####################################

def distance_to_collision(mixture):
    sigma_s = mixture.macroscopic_scattering
    sigma_a = mixture.macroscopic_absorption
    sigma_tot = sigma_s + sigma_a
    if sigma_tot <= 1e-30:
        return 1e10
    return -np.log(rand.rand()) / sigma_tot

def point_is_inside_sphere(point, radius):
    """
    Check if the 3D point is inside the sphere of given radius.
    """
    r2 = point[0]**2 + point[1]**2 + point[2]**2
    return (r2 <= radius**2)


def point_is_inside_cylinder(point, radius, height):
    """
    Check if the 3D point is inside cylinder with 'radius' and 'height'.
    Cylinder axis is z in [-height/2, height/2].
    """
    x, y, z = point
    r2 = x*x + y*y
    return (r2 <= radius**2) and (abs(z) <= height/2)


def radial_distance_sphere(point):
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)


def radial_distance_cylinder(point):
    x, y, z = point
    return np.sqrt(x*x + y*y)


def random_scatter_direction():
    costheta = 2.0*rand.rand() - 1.0
    phi = 2.0*np.pi*rand.rand()
    sintheta = np.sqrt(1.0 - costheta**2)
    return np.array([sintheta*np.cos(phi), sintheta*np.sin(phi), costheta])

def point_is_inside_core(point, geometry, core_size):
    if geometry == "sphere":
        R = core_size
        return np.linalg.norm(point) <= R
    elif geometry == "cylinder":
        R, H = core_size
        return (np.sqrt(point[0]**2 + point[1]**2) <= R) and (abs(point[2]) <= H/2)

def point_is_inside_reflector(point, geometry, core_size, reflector_thickness):
    """Check if a point is inside the reflector."""
    if geometry == "sphere":
        R_core = core_size
        R_reflector = R_core + reflector_thickness
        r = np.linalg.norm(point)
        return R_core < r <= R_reflector
    elif geometry == "cylinder":
        R_core, H_core = core_size
        R_reflector = R_core + reflector_thickness
        H_reflector = H_core + 2 * reflector_thickness
        r = np.sqrt(point[0]**2 + point[1]**2)
        z = abs(point[2])
        return (R_core < r <= R_reflector) or (H_core/2 < z <= H_reflector/2)

def simulate_first_generation(
    mixture, 
    geometry, 
    size, 
    N0, 
    average_neutrons_per_fission=2.42, 
    bins=20,
    reflector=None, 
    reflector_thickness=0.0
):
    """
    Simulate first generation with reflector support.
    """
    P_scat = mixture.scattering_probability
    P_abs = mixture.absorption_probability
    P_fis = mixture.fission_probability

    # Core dimensions
    if geometry == 'sphere':
        R_core = float(size)
        R_reflector = R_core + reflector_thickness
    else:
        R_core, H_core = float(size[0]), float(size[1])
        R_reflector = R_core + reflector_thickness
        H_reflector = H_core + 2 * reflector_thickness

    # Initialize neutron positions
    neutron_positions = np.zeros((N0, 3))
    for i in range(N0):
        if geometry == 'sphere':
            neutron_positions[i,:] = random_position_sphere_optimized(R_core)
        else:
            neutron_positions[i,:] = random_position_cylinder(R_core, H_core)

    # Reflector properties
    if reflector is not None:
        P_reflect_scatter = reflector.reflection_probability
    else:
        P_reflect_scatter = 0.0

    # Tracking variables
    absorbed_positions = []
    is_active = np.ones(N0, dtype=bool)
    leak_count = 0
    reflect_r_vals = []
    scatter_r_vals = []
    absorb_r_vals = []
    fission_r_vals = []
    reflector_interaction_count = 0

    active_indices = np.where(is_active)[0]
    while len(active_indices) > 0:
        for idx in active_indices:
            pos = neutron_positions[idx]
            while True:  # Modified: continuous tracking until absorption/leak
                d_coll = distance_to_collision(mixture)
                dirn = random_scatter_direction()
                new_pos = pos + d_coll * dirn

                # Check containment
                in_core = (point_is_inside_sphere(new_pos, R_core) if geometry == 'sphere'
                          else point_is_inside_cylinder(new_pos, R_core, H_core))
                in_reflector = False
                if not in_core and reflector is not None:
                    in_reflector = (point_is_inside_sphere(new_pos, R_reflector) if geometry == 'sphere'
                                   else (np.sqrt(new_pos[0]**2 + new_pos[1]**2) <= R_reflector
                                         and abs(new_pos[2]) <= H_reflector/2))

                # Handle different cases
                if in_core:
                    # Process core collision
                    rand_event = rand.rand()
                    if rand_event < P_scat:
                        # Scatter in core
                        radial_dist = (radial_distance_sphere(new_pos) if geometry == 'sphere'
                                      else radial_distance_cylinder(new_pos))
                        scatter_r_vals.append(radial_dist)
                        pos = new_pos
                    else:
                        # Absorption in core
                        radial_dist = (radial_distance_sphere(new_pos) if geometry == 'sphere'
                                      else radial_distance_cylinder(new_pos))
                        absorb_r_vals.append(radial_dist)
                        if rand.rand() < P_fis:
                            fission_r_vals.append(radial_dist)
                        absorbed_positions.append(new_pos)
                        is_active[idx] = False
                        break
                elif in_reflector:
                    # Handle reflector interaction
                    reflector_interaction_count += 1
                    radial_dist = (radial_distance_sphere(new_pos)) if geometry == 'sphere' else radial_distance_cylinder(new_pos)
                    if rand.rand() < P_reflect_scatter:
                        reflect_r_vals.append(radial_dist)
                        # Scatter back into core
                        pos = _adjust_to_core_boundary(new_pos, geometry, size)
                    else:
                        # Absorbed in reflector
                        leak_count += 1
                        is_active[idx] = False
                        break
                else:
                    # Permanent leakage
                    leak_count += 1
                    is_active[idx] = False
                    break

        active_indices = np.where(is_active)[0]

    # Existing binning and results code
    absorbed_count = len(absorbed_positions)
    bins_r = np.linspace(0, R_core, bins+1)
    if geometry == 'sphere':
        shell_volumes = (4/3)*np.pi*(bins_r[1:]**3 - bins_r[:-1]**3)
    else:
        H = size[1] if isinstance(size, list) else size[0][1]  # Handle cylinder size
        shell_volumes = np.pi*(bins_r[1:]**2 - bins_r[:-1]**2) * H
    # Add reflector metrics to results
    results = {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': absorbed_count,
        'leak_count': leak_count,
        'reflector_interaction_count': reflector_interaction_count,
        'bin_edges': bins_r,
        'bin_centers': 0.5*(bins_r[:-1] + bins_r[1:]),
        'scatter_density': np.histogram(scatter_r_vals, bins=bins_r)[0] / _calculate_shell_volumes(geometry, bins_r, size),
        'absorb_density': np.histogram(absorb_r_vals, bins=bins_r)[0] / _calculate_shell_volumes(geometry, bins_r, size),
        'fission_density': np.histogram(fission_r_vals, bins=bins_r)[0] / _calculate_shell_volumes(geometry, bins_r, size),
        'reflect_density': np.histogram(reflect_r_vals, bins=bins_r)[0] / shell_volumes
        
    }
    return results

def _adjust_to_core_boundary(point, geometry, core_size):
    if geometry == 'sphere':
        R = core_size
        return point * (R / np.linalg.norm(point))
    else:
        R, H = core_size
        new_pos = point.copy()
        r = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
        if r > R:
            theta = np.arctan2(new_pos[1], new_pos[0])
            new_pos[0] = R * np.cos(theta)
            new_pos[1] = R * np.sin(theta)
        if abs(new_pos[2]) > H/2:
            new_pos[2] = np.sign(new_pos[2]) * (H/2 - 1e-9)
        return new_pos

def _calculate_shell_volumes(geometry, bins_r, size):
    """Helper for volume normalization"""
    if geometry == 'sphere':
        return (4/3)*np.pi*(bins_r[1:]**3 - bins_r[:-1]**3)
    else:
        H = float(size[1])
        return np.pi*(bins_r[1:]**2 - bins_r[:-1]**2) * H

def plot_first_generation_hist(results_dict, geometry, N0):
    """
    Plot collisions vs radial distance (volume-normalized).
    Show scatter, absorb, fission lines.
    """
    r = results_dict['bin_centers']
    scat_dens = results_dict['scatter_density']
    abs_dens  = results_dict['absorb_density']
    fis_dens  = results_dict['fission_density']

    plt.figure(figsize=(7,5))
    plt.plot(r, scat_dens, '-o', color='blue',   label='Scattered')
    plt.plot(r, abs_dens,  '-o', color='orange', label='Absorbed')
    plt.plot(r, fis_dens,  '-o', color='green',  label='Fission')
    if np.sum(results_dict['reflect_density']) > 0:
        plt.plot(r, results_dict['reflect_density'], '-o', color='red', label='Reflected')
    
    if geometry=='sphere':
        plt.xlabel("Radial distance r from reactor center (m)")
    else:
        plt.xlabel("Radial distance r from central axis of cylinder (m)")
    plt.ylabel("Collision density (collisions / m^3)")
    plt.title(f"1st Generation Collision Distribution (Volume Normalized)\n{geometry.capitalize()}, N={N0} neutrons")
    plt.legend()
    plt.tight_layout()
    plt.show()

#####################################
# PATCH 4 of N
#####################################

def compute_k_factor_and_uncertainty(num_absorbed, num_initial, fission_probability, avg_neutrons_per_fission):
    if num_initial <= 0:
        return (0.0, 0.0)
    k = (num_absorbed / num_initial) * fission_probability * avg_neutrons_per_fission
    N = num_absorbed
    p = fission_probability
    stdev_fissions = np.sqrt(N * p * (1 - p)) if N > 0 and 0 <= p <= 1 else 0.0
    dk = (avg_neutrons_per_fission / num_initial) * stdev_fissions
    return (k, dk)

def simulate_generation(mixture, geometry, size, initial_positions, reflector=None, reflector_thickness=0.0):
    N = initial_positions.shape[0]
    is_active = np.ones(N, dtype=bool)
    leak_count = 0
    absorbed_positions = []

    if geometry == 'sphere':
        R_core = float(size)
        R_reflector = R_core + reflector_thickness
    else:
        R_core, H_core = float(size[0]), float(size[1])
        R_reflector = R_core + reflector_thickness
        H_reflector = H_core + 2 * reflector_thickness

    P_scat = mixture.scattering_probability
    P_fis = mixture.fission_probability
    P_reflect_scatter = reflector.reflection_probability if reflector else 0.0

    for idx in range(N):
        pos = initial_positions[idx]
        while is_active[idx]:
            d_coll = distance_to_collision(mixture)
            dirn = random_scatter_direction()
            new_pos = pos + d_coll * dirn

            in_core = point_is_inside_core(new_pos, geometry, (R_core, H_core) if geometry == 'cylinder' else R_core)
            in_reflector = False
            if not in_core and reflector:
                if geometry == 'sphere':
                    r = np.linalg.norm(new_pos)
                    in_reflector = R_core < r <= R_reflector
                else:
                    r = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                    z = abs(new_pos[2])
                    in_reflector = (R_core < r <= R_reflector) or (H_core/2 < z <= H_reflector/2)

            if in_core:
                if rand.rand() < P_scat:
                    pos = new_pos
                else:
                    absorbed_positions.append(new_pos)
                    is_active[idx] = False
                    break
            elif in_reflector:
                if rand.rand() < P_reflect_scatter:
                    pos = _adjust_to_core_boundary(new_pos, geometry, (R_core, H_core) if geometry == 'cylinder' else R_core)
                else:
                    leak_count +=1
                    is_active[idx] = False
                    break
            else:
                leak_count +=1
                is_active[idx] = False
                break

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }


def main():
    print("=== Nuclear Reactor MC Simulation ===\n")

    # Geometry input
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'."
    ).lower()

    # Size input
    if geometry == 'sphere':
        size_prompt = "Enter sphere radius (m): "
    else:
        size_prompt = "Enter cylinder [radius,height] in m, e.g. [1.0,2.0]: "
    size_str = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format."
    )
    reactor_size = parse_size(size_str, geometry)

    # Moderator input
    mod_key = get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O, D2O, or Graphite"
    ).lower()

    # Dataset selection
    dataset = get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be 'Lilley' or 'Wikipedia'."
    ).lower()

    # Nuclear data initialization
    if dataset == 'wikipedia':
        neutron_model = get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be 'Thermal' or 'Fast'"
        ).lower()
        u235_mat, u238_mat = get_wikipedia_materials(neutron_model)
    else:
        u235_mat, u238_mat = LILLEY_U235, LILLEY_U238

    # Fuel composition
    u235_percent = float(get_valid_input(
        "U-235 concentration in fuel (0-100): ",
        lambda x: validate_float(x, 0, 100),
        "Must be a number between 0 and 100"
    )) / 100.0

    # Moderator ratio
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio, R_mtf (>=0): ",
        lambda x: validate_float(x, 0.0),
        "Must be a number >= 0"
    ))

    # Reflector configuration
    use_reflector = get_valid_input(
        "Include reflector? (yes/no): ",
        lambda x: x.lower() in ['yes', 'no'],
        "Must answer 'yes' or 'no'"
    ).lower() == 'yes'
    
    reflector = None
    reflector_thickness = 0.0
    if use_reflector:
        reflector_type = get_valid_input(
            "Reflector material [Graphite/Beryllium]: ",
            lambda x: x.lower() in ['graphite', 'beryllium'],
            "Must be Graphite or Beryllium"
        ).lower()
        reflector_thickness = float(get_valid_input(
            "Reflector thickness (m): ",
            lambda x: validate_float(x, min_val=0.001, max_val=10.0),  # Now allows ≥0
            "Must be between 0.001 and 10.0 m"
        ))
        # Initialize reflector material
        if reflector_type == 'graphite':
            reflector = ReflectorMaterial(
                name="Graphite",
                sigma_s_b=4.7,
                sigma_a_b=0.0045,
                density_gcc=1.6,
                molar_mass_gmol=12.01
            )
        else:  # Beryllium
            reflector = ReflectorMaterial(
                name="Beryllium",
                sigma_s_b=6.0,
                sigma_a_b=0.001,
                density_gcc=1.85,
                molar_mass_gmol=9.01
            )

    # Create reactor mixture
    moderator_obj = MODERATORS_DICT[mod_key]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235_mat, u238_mat)

    # Display cross sections
    print("\nMacroscopic Cross Sections (SI units, m^-1):")
    print(f"  Σ_s (scattering) = {mixture.macroscopic_scattering:.8e}")
    print(f"  Σ_a (absorption) = {mixture.macroscopic_absorption:.8e}")
    print(f"  Σ_f (fission)    = {mixture.macroscopic_fission:.8e}")

    # Initial neutron count
    N0 = int(get_valid_input(
        "Enter the number of neutrons for the 1st generation: ",
        validate_positive_integer,
        "Must be a positive integer!"
    ))

    # First generation simulation
    results_1st = simulate_first_generation(
        mixture=mixture,
        geometry=geometry,
        size=reactor_size,
        N0=N0,
        average_neutrons_per_fission=2.42,
        bins=20,
        reflector=reflector,
        reflector_thickness=reflector_thickness
    )

    # Process first generation results
    absorbed_count_1st = results_1st['absorbed_count']
    leak_count_1st = results_1st['leak_count']
    
    # Calculate initial k-factor
    k1, dk1 = compute_k_factor_and_uncertainty(
        num_absorbed=absorbed_count_1st,
        num_initial=N0,
        fission_probability=mixture.fission_probability,
        avg_neutrons_per_fission=mixture.u235.nu  # Use actual nu from material
    )

    # Visualization and reporting
    plot_first_generation_hist(results_1st, geometry, N0)
    print("\n=== Results for 1st Generation ===")
    print(f"Scattering Probability = {mixture.scattering_probability:.6f}")
    print(f"Absorption Probability = {mixture.absorption_probability:.6f}")
    print(f"Fission Probability    = {mixture.fission_probability:.6f}")
    print(f"Neutrons absorbed: {absorbed_count_1st}")
    print(f"Neutrons leaked: {leak_count_1st}")
    print(f"Reflector interactions: {results_1st['reflector_interaction_count']}")
    if use_reflector:
        print(f"Reflector interactions: {results_1st['reflector_interaction_count']}")
    print(f"Initial k-factor: {k1:.6f} ± {dk1:.6f}")



    # Subsequent generations
    generations = int(get_valid_input(
        "Number of generations to simulate: ",
        validate_positive_integer,
        "Must be a positive integer"
    ))
    
    initial_positions = results_1st['absorbed_positions']
    k_values = [(k1, dk1)]
    current_initial = absorbed_count_1st  # Track scaled neutron count for next gen
    
    for gen in range(2, generations + 1):
        if current_initial == 0:
            print(f"Generation {gen}: No neutrons remaining")
            k_values.append((0.0, 0.0))
            break
        
        # Simulate generation
        results = simulate_generation(
            mixture=mixture,
            geometry=geometry,
            size=reactor_size,
            initial_positions=initial_positions,
            reflector=reflector,
            reflector_thickness=reflector_thickness
        )
        absorbed_count = results['absorbed_count']
        leak_count = results['leak_count']
        
        # Calculate k using ACTUAL counts (no scaling)
        k, dk = compute_k_factor_and_uncertainty(
            num_absorbed=absorbed_count,
            num_initial=current_initial,
            fission_probability=mixture.fission_probability,
            avg_neutrons_per_fission=mixture.u235.nu
        )
        k_values.append((k, dk))
        
        scaled = False
        # Apply population control AFTER computing k
        if absorbed_count < 100:
            if absorbed_count == 0:
                print(f"Generation {gen}: All neutrons lost. Simulation terminated.")
                break
            # Scale up by resampling
            indices = np.random.choice(absorbed_count, size=100, replace=True)
            initial_positions = results['absorbed_positions'][indices]
            current_initial_next = 100
            scaled = True
        elif absorbed_count > 11000:
            # Scale down by subsampling
            indices = np.random.choice(absorbed_count, size=11000, replace=False)
            initial_positions = results['absorbed_positions'][indices]
            current_initial_next = 11000
            scaled = True
        else:
            initial_positions = results['absorbed_positions']
            current_initial_next = absorbed_count
            scaled = False
        
        # Print results
        print(f"\nGeneration {gen}:")
        print(f"  Neutrons absorbed: {absorbed_count}")
        print(f"  Neutrons leaked: {leak_count}")
        if scaled:
            print(f"  Adjusted neutron count for next generation: {current_initial_next}")
        print(f"  k-factor: {k:.6f} ± {dk:.6f}")
        
        # Update for next generation
        current_initial = current_initial_next

    # Final visualization and analysis
    plot_k_evolution(k_values, N0)
    
    if len(k_values) >= 20:
        analyze_convergence(k_values)
    
    print("\nSimulation completed successfully!")

def parse_size(size_str, geometry):
    """Helper to parse size input"""
    if geometry == 'sphere':
        return float(size_str)
    elif geometry == 'cylinder':
        if isinstance(size_str, str):  # Add type check for safety
            s = size_str.strip()[1:-1].split(',')
            return [float(s[0]), float(s[1])]
        return size_str  # Already parsed case

def plot_k_evolution(k_values, N0):
    """Plot k-factor evolution"""
    gens = np.arange(1, len(k_values)+1)
    k = np.array([x[0] for x in k_values])
    dk = np.array([x[1] for x in k_values])
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(gens, k, yerr=dk, fmt='-o', capsize=5)
    plt.axhline(1.0, color='red', linestyle='--')
    plt.xlabel("Generation Number")
    plt.ylabel("k-factor")
    plt.title(f"Reactor Criticality Evolution (N0={N0})")
    plt.grid(True)
    plt.show()

def analyze_convergence(k_values):
    """Analyze late-generation convergence"""
    late_k = np.array([x[0] for x in k_values[10:20]])
    late_dk = np.array([x[1] for x in k_values[10:20]])
    
    valid = late_dk > 1e-12
    if np.any(valid):
        weights = 1 / (late_dk[valid]**2)
        avg_k = np.sum(late_k[valid] * weights) / np.sum(weights)
        avg_dk = 1 / np.sqrt(np.sum(weights))
        print(f"\nStabilized k-factor: {avg_k:.6f} ± {avg_dk:.6f}")
    else:
        print("\nWarning: Insufficient data for convergence analysis")


    
if __name__=="__main__":

    main()
