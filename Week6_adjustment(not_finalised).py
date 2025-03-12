# -*- coding: utf-8 -*-
"""
Example Overhaul of Reactor Simulation
- Redefining "generation" as a full random-walk simulation from start to end.
- Preserving user-friendly inputs (geometry, size, dataset, etc.).
- Storing generational k-factor values in a list and plotting them.
"""

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from scipy.constants import Avogadro as NA

# ----------------------------------------------------
# 1. Geometry Helpers: Sphere & Cylinder
# ----------------------------------------------------
def random_position_sphere_optimized(radius=1.0):
    """
    Generate a random position inside a sphere with uniform density.
    'radius' is in meters.
    """
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)
    # Scale radius by cube root transformation to ensure uniform density
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1.0, height=1.0):
    """
    Generate a random position inside a cylinder with uniform density.
    'radius' and 'height' are in meters.
    """
    theta = rand.uniform(0, 2*np.pi)
    r = np.sqrt(rand.uniform(0, 1)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height/2, height/2)
    return np.array([x, y, z])

# Helper to check if a position is inside the reactor
def is_inside_reactor(pos, geometry, size):
    if geometry == 'sphere':
        r = np.linalg.norm(pos)
        return (r <= size)
    elif geometry == 'cylinder':
        r_xy = np.sqrt(pos[0]**2 + pos[1]**2)
        return (r_xy <= size[0]) and (abs(pos[2]) <= size[1]/2)
    else:
        return False

# ----------------------------------------------------
# 2. Input Validation
# ----------------------------------------------------
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
    # For example: 'water', 'heavywater', or 'graphite'
    return input_str.lower() in ['water', 'heavywater', 'graphite']

def validate_float(input_str, min_val=0, max_val=None):
    try:
        value = float(input_str)
        valid = True
        if min_val is not None:
            valid &= (value >= min_val)
        if max_val is not None:
            valid &= (value <= max_val)
        return valid
    except ValueError:
        return False

def validate_size(input_str, geometry):
    """
    For sphere, the user enters a single float for radius (in meters).
    For cylinder, the user enters [radius,height] (both in meters).
    """
    try:
        if geometry == 'sphere':
            float(input_str)
            return True
        elif geometry == 'cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                values = list(map(float, input_str[1:-1].split(',')))
                return (len(values) == 2 and all(v > 0 for v in values))
        return False
    except:
        return False

def validate_dataset(input_str):
    return input_str.lower() in ['lilley', 'wikipedia']

def validate_neutron_model(input_str):
    return input_str.lower() in ['thermal', 'fast']

def get_positive_integer(prompt_text):
    while True:
        try:
            value = int(input(prompt_text))
            if value > 0:
                return value
            else:
                print("Value must be a positive integer!")
        except ValueError:
            print("Invalid input, please enter a positive integer.")

# ----------------------------------------------------
# 3. Material Class and Database
# ----------------------------------------------------
class ReactorMaterial:
    def __init__(self, name, mat_type, density, molar_mass,
                 sigma_s_b, sigma_a_b, sigma_f_b=0, nu=0, xi=0):
        self.name = name
        self.mat_type = mat_type
        # Convert to SI units
        self.density = density * 1e3      # kg/m³
        self.molar_mass = molar_mass * 1e-3  # kg/mol
        self.sigma_s_b = sigma_s_b
        self.sigma_a_b = sigma_a_b
        self.sigma_f_b = sigma_f_b
        # Convert barns to m²
        self.sigma_s = sigma_s_b * 1e-28
        self.sigma_a = sigma_a_b * 1e-28
        self.sigma_f = sigma_f_b * 1e-28
        self.nu = nu
        self.xi = xi

    @property
    def number_density(self):
        # N = (ρ / M) * Avogadro
        return (self.density / self.molar_mass) * NA

# Example "Lilley" data
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
    'water': ReactorMaterial(
        name="water", mat_type='moderator',
        density=1.0, molar_mass=18.01,
        sigma_s_b=49.2, sigma_a_b=0.66,
        xi=0.920
    ),
    'heavywater': ReactorMaterial(
        name="heavywater", mat_type='moderator',
        density=1.1, molar_mass=20.02,
        sigma_s_b=10.6, sigma_a_b=0.001,
        xi=0.509
    ),
    'graphite': ReactorMaterial(
        name="graphite", mat_type='moderator',
        density=1.6, molar_mass=12.01,
        sigma_s_b=4.7, sigma_a_b=0.0045,
        xi=0.158
    )
}

# ----------------------------------------------------
# 4. Reactor Mixture and Simulator
# ----------------------------------------------------
class ReactorMixture:
    def __init__(self, u235_percent, moderator, R_mtf, u235_material, u238_material):
        self.u235_frac = u235_percent / 100
        self.moderator = moderator
        self.R_mtf = R_mtf
        self.u235_material = u235_material
        self.u238_material = u238_material

    @property
    def macroscopic_scattering(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        N_mod  = (N_U235 + N_U238) * self.R_mtf
        return (N_U235*self.u235_material.sigma_s
              + N_U238*self.u238_material.sigma_s
              + N_mod*self.moderator.sigma_s)

    @property
    def macroscopic_absorption(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        N_mod  = (N_U235 + N_U238) * self.R_mtf
        return (N_U235*self.u235_material.sigma_a
              + N_U238*self.u238_material.sigma_a
              + N_mod*self.moderator.sigma_a)

    @property
    def macroscopic_fission(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        return (N_U235*self.u235_material.sigma_f
              + N_U238*self.u238_material.sigma_f)

    @property
    def average_neutrons_per_fission(self):
        """
        Weighted average ν from U235 and U238.
        """
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)

        numerator = (N_U235*self.u235_material.sigma_f*self.u235_material.nu
                   + N_U238*self.u238_material.sigma_f*self.u238_material.nu)
        denominator = (N_U235*self.u235_material.sigma_f
                     + N_U238*self.u238_material.sigma_f)
        return numerator/denominator if denominator>0 else 0.0

class ReactorSimulator:
    def __init__(self, mixture, geometry, size):
        self.mixture = mixture
        self.geometry = geometry
        self.size = size  # radius if sphere, [radius, height] if cylinder

    @property
    def total_mean_free_path(self):
        Σs = self.mixture.macroscopic_scattering
        Σa = self.mixture.macroscopic_absorption
        return 1.0 / (Σs + Σa) if (Σs+Σa)>0 else 1e9

    def scattering_probability(self):
        Σs = self.mixture.macroscopic_scattering
        Σa = self.mixture.macroscopic_absorption
        return Σs / (Σs+Σa) if (Σs+Σa)>0 else 0.0

    def absorption_probability(self):
        Σs = self.mixture.macroscopic_scattering
        Σa = self.mixture.macroscopic_absorption
        return Σa / (Σs+Σa) if (Σs+Σa)>0 else 0.0

    def fission_probability(self):
        Σf = self.mixture.macroscopic_fission
        Σa = self.mixture.macroscopic_absorption
        return Σf / Σa if Σa>0 else 0.0

# ----------------------------------------------------
# 5. Single "Generation" Simulation (New Definition)
# ----------------------------------------------------
def simulate_one_generation(simulator, N_initial):
    """
    Simulate a full random-walk of N_initial neutrons until all are absorbed or leak.
    Returns:
      stats = dict containing:
        - scattered_positions
        - fission_positions
        - absorbed_positions
        - leak_count
        - total_collisions  (scattering + fission + absorption)
        - n_absorbed
        - n_fission
        - n_scatter
    """
    # We'll store the final positions of each collision event
    scattered_positions = []
    absorbed_positions  = []
    fission_positions   = []

    # Start the simulation with an array of neutron positions
    if simulator.geometry=='sphere':
        # Generate initial positions inside sphere
        neutron_positions = [random_position_sphere_optimized(simulator.size)
                             for _ in range(N_initial)]
    else: # cylinder
        neutron_positions = [random_position_cylinder(simulator.size[0], simulator.size[1])
                             for _ in range(N_initial)]

    # Probability of scattering vs absorption
    P_scat = simulator.scattering_probability()
    P_abs  = simulator.absorption_probability()
    # Among absorption events, fraction that is fission
    frac_fission = simulator.fission_probability()
    # If an absorption event occurs, probability it is fission is frac_fission

    # We'll random walk neutrons until none remain
    active_neutrons = neutron_positions[:]  # copy
    leak_count = 0
    while len(active_neutrons) > 0:
        new_active = []
        for pos in active_neutrons:
            # random walk step
            step_len = np.random.exponential(scale=simulator.total_mean_free_path)
            direction = rand.randn(3)
            direction /= np.linalg.norm(direction)
            new_pos = pos + step_len*direction

            # check if new_pos is inside reactor
            if not is_inside_reactor(new_pos, simulator.geometry, simulator.size):
                leak_count += 1
                continue  # neutron is lost (leaked)

            # collision occurs
            r = rand.rand()
            if r < P_scat:
                # scatter
                scattered_positions.append(new_pos)
                new_active.append(new_pos)  # neutron continues
            else:
                # absorption
                # among absorption, fraction is fission
                r2 = rand.rand()
                if r2 < frac_fission:
                    # fission
                    fission_positions.append(new_pos)
                    # produce new neutrons
                    # We assume an average "nu" from mixture
                    # for simplicity, sample from Poisson(nu)
                    n_new = np.random.poisson(lam=simulator.mixture.average_neutrons_per_fission)
                    for _ in range(n_new):
                        # place new neutrons at same position
                        new_active.append(new_pos)
                else:
                    # non-fission absorption
                    absorbed_positions.append(new_pos)
                # in either fission or absorption, the original neutron is lost
        # move to next iteration
        active_neutrons = new_active

    # Summarize
    stats = {}
    stats['scattered_positions'] = np.array(scattered_positions)
    stats['absorbed_positions']  = np.array(absorbed_positions)
    stats['fission_positions']   = np.array(fission_positions)
    stats['leak_count']          = leak_count
    stats['n_absorbed']         = len(absorbed_positions)
    stats['n_scatter']          = len(scattered_positions)
    stats['n_fission']          = len(fission_positions)
    stats['total_collisions']   = stats['n_absorbed'] + stats['n_scatter'] + stats['n_fission']
    return stats

# ----------------------------------------------------
# 6. Plotting the Final Radial Distribution
# ----------------------------------------------------
def plot_radial_distribution(stats, simulator, N_initial, generation_index):
    """
    At the end of a generation, we have stats about scattered, absorbed, fission events.
    We can produce a histogram of radial distances. We'll do a simple raw histogram
    vs. radial distance for all collisions combined or separate them.
    """
    # Combine all final collision positions if you want one histogram,
    # or separate them if you want multiple lines
    scatter_r = np.linalg.norm(stats['scattered_positions'], axis=1) if len(stats['scattered_positions'])>0 else []
    absorb_r  = np.linalg.norm(stats['absorbed_positions'],  axis=1) if len(stats['absorbed_positions'])>0  else []
    fission_r = np.linalg.norm(stats['fission_positions'],   axis=1) if len(stats['fission_positions'])>0   else []

    # For simplicity, let's plot a single histogram of all collisions
    # Or do side-by-side line plots. We'll do the latter:
    bins = 20
    max_radius = 0
    if simulator.geometry=='sphere':
        max_radius = simulator.size
    else:
        max_radius = simulator.size[0]  # for cylinder

    edges = np.linspace(0, max_radius, bins+1)
    c_scatter, _ = np.histogram(scatter_r, edges)
    c_absorb,  _ = np.histogram(absorb_r,  edges)
    c_fission, _ = np.histogram(fission_r, edges)
    centers = 0.5*(edges[:-1]+edges[1:])

    plt.figure(figsize=(8,6))
    plt.plot(centers, c_scatter, marker='o', label='Scattered')
    plt.plot(centers, c_absorb,  marker='o', label='Absorbed')
    plt.plot(centers, c_fission, marker='o', label='Fission')
    plt.xlabel('Radial distance from center (m)')
    plt.ylabel('Number of collisions')
    plt.title(f'Radial Distribution of Collisions (Generation {generation_index})\n'
              f'N={N_initial} neutrons')
    plt.legend()
    plt.show()

# ----------------------------------------------------
# 7. Binomial k-Factor and Plotting
# ----------------------------------------------------
def compute_generational_k(stats, N_initial):
    """
    Example: define k = 1 - (#absorbed / N_initial).
    Then the uncertainty is sqrt(p(1-p)/N), with p = (#absorbed / N_initial).
    This is purely an example; adapt to your own definition if needed.
    """
    n_absorbed = stats['n_absorbed']
    p = n_absorbed / N_initial if N_initial>0 else 0
    k = 1.0 - p
    # binomial standard error in p
    sigma_p = np.sqrt(p*(1-p)/N_initial) if N_initial>0 else 0
    # so the error in k = error in (1-p) = sigma_p
    sigma_k = sigma_p
    return k, sigma_k

def plot_k_evolution(k_list, k_unc_list, N_initial):
    """
    Plot k-values (with error bars) vs generation index, plus a line at k=1.
    Title includes the number of neutrons used per generation.
    """
    gens = np.arange(1, len(k_list)+1)
    plt.figure()
    plt.errorbar(gens, k_list, yerr=k_unc_list, fmt='-o', label='k-factor')
    plt.axhline(y=1.0, color='r', linestyle='--', label='k=1')
    plt.xlabel('Generation index')
    plt.ylabel('k-factor')
    plt.title(f'Generational k-Factor Evolution (N={N_initial} neutrons each run)')
    plt.legend()
    plt.show()

# ----------------------------------------------------
# 8. Main Overhauled Function
# ----------------------------------------------------
def overhauled_main():
    """
    In this new version:
    - "Generation" means a full run from N neutrons until all are gone or leak.
    - We do S such generations, each time starting fresh with N neutrons.
    - We store the k-factor for each generation, then plot them with error bars.
    """
    # =========== 1) User Prompts ===========
    # Reactor type
    reactor_type = get_valid_input(
        "Reactor type [finite/infinite]: ",
        validate_reactor_type,
        "Must be 'finite' or 'infinite'"
    ).lower()

    # Geometry
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'"
    ).lower()

    # Size
    size_prompt = ("Enter size in meters:\n"
                   "- Sphere: radius\n"
                   "- Cylinder: [radius,height]\n"
                   "Size: ")
    size_input = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format"
    )
    if geometry=='cylinder':
        size = list(map(float, size_input.strip('[]').split(',')))
    else:
        size = float(size_input)

    # Moderator
    moderator_str = get_valid_input(
        "Moderator [water/heavywater/graphite]: ",
        validate_moderator,
        "Must be water, heavywater, or graphite"
    ).lower()

    # Dataset
    dataset = get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be 'Lilley' or 'Wikipedia'"
    ).lower()

    # If Wikipedia, prompt for model
    if dataset=='wikipedia':
        neutron_model = get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be 'Thermal' or 'Fast'"
        ).lower()
        # Hard-code example cross sections
        if neutron_model=='thermal':
            u235 = ReactorMaterial(
                "U235", 'fuel',
                density=18.7, molar_mass=235,
                sigma_s_b=10, sigma_a_b=99+583, sigma_f_b=583,
                nu=2.42, xi=0
            )
            u238 = ReactorMaterial(
                "U238", 'fuel',
                density=18.9, molar_mass=238,
                sigma_s_b=9, sigma_a_b=2+0.00002, sigma_f_b=0.00002,
                nu=0, xi=0
            )
        else: # fast
            u235 = ReactorMaterial(
                "U235", 'fuel',
                density=18.7, molar_mass=235,
                sigma_s_b=4, sigma_a_b=0.09+1, sigma_f_b=1,
                nu=2.42, xi=0
            )
            u238 = ReactorMaterial(
                "U238", 'fuel',
                density=18.9, molar_mass=238,
                sigma_s_b=5, sigma_a_b=0.07+0.3, sigma_f_b=0.3,
                nu=0, xi=0
            )
    else:
        # Use Lilley
        u235 = LILLEY_U235
        u238 = LILLEY_U238

    # U-235 enrichment
    u235_percent = float(get_valid_input(
        "U-235 concentration [0-100%]: ",
        lambda x: validate_float(x, 0, 100),
        "Must be 0..100"
    ))

    # R_mtf
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio (R_mtf >= 0): ",
        lambda x: validate_float(x, 0),
        "Must be >= 0"
    ))

    # N_initial
    N_initial = get_positive_integer("Number of initial neutrons (positive integer): ")

    # S => how many "generations" to run
    S = get_positive_integer("Number of generations (S): ")

    # =========== 2) Create Mixture and Simulator ===========
    moderator_obj = MODERATORS[moderator_str]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235, u238)
    simulator = ReactorSimulator(mixture, geometry, size)

    # Possibly compute a leakage factor c if finite
    c = 1.0 if (reactor_type=='infinite') else (1-0.1)*(1-0.2)

    # Print cross-section derived probabilities
    print("\n=== Reactor Physics Parameters ===")
    print(f"Scattering Probability: {simulator.scattering_probability():.3f}")
    print(f"Absorption Probability: {simulator.absorption_probability():.3f}")
    print(f"Fission Probability:    {simulator.fission_probability():.3f}")
    print(f"Total mean free path:   {simulator.total_mean_free_path:.2e} m")

    # =========== 3) Run S Generations, Each a Full Reaction ===========
    k_list = []
    k_unc_list = []
    all_stats = []

    for gen_index in range(1, S+1):
        print(f"\n--- Simulation for Generation {gen_index} ---")
        # Simulate the entire random-walk from start to finish
        stats = simulate_one_generation(simulator, N_initial)
        all_stats.append(stats)

        # Compute a generational k
        k_val, k_unc = compute_generational_k(stats, N_initial)
        # Round or format up to 6 decimals if needed
        print(f"Generation {gen_index}: k = {k_val:.6f} ± {k_unc:.6f}")
        print(f"Absorbed: {stats['n_absorbed']}   Fission: {stats['n_fission']}   "
              f"Scattered: {stats['n_scatter']}   Leaked: {stats['leak_count']}")

        # Store
        k_list.append(k_val)
        k_unc_list.append(k_unc)

        # (Optional) Plot final radial distribution of collisions for each generation
        plot_radial_distribution(stats, simulator, N_initial, gen_index)

    # =========== 4) Plot the k-factor Evolution ===========
    plot_k_evolution(k_list, k_unc_list, N_initial)

    print("\nSimulation complete. Results:")
    for i, (k_val, k_unc) in enumerate(zip(k_list, k_unc_list), start=1):
        print(f"Generation {i}: k = {k_val:.6f} ± {k_unc:.6f}")

    print("\n(You can further analyze or save these results as needed.)")

# ----------------------------------------------------
# 9. Entry Point
# ----------------------------------------------------
if __name__ == "__main__":
    overhauled_main()
