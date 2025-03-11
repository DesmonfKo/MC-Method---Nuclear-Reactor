import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from scipy.constants import Avogadro as NA

# ----------------------------------------------------
# 1. Geometry Helpers: Sphere & Cylinder
# ----------------------------------------------------
def random_position_sphere_optimized(radius=1):
    """
    Generate a random position inside a sphere with uniform density.
    'radius' is in meters.
    """
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)
    # Scale radius by cube root transformation to ensure uniform density
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1, height=1):
    """
    Generate a random position inside a cylinder with uniform density.
    'radius' and 'height' are in meters.
    """
    theta = rand.uniform(0, 2 * np.pi)
    r = np.sqrt(rand.uniform(0, 1)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height / 2, height / 2)
    return np.array([x, y, z])

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
    return input_str.capitalize() in ['H2O', 'D2O', 'Graphite']

def validate_float(input_str, min_val=0, max_val=None):
    try:
        value = float(input_str)
        valid = True
        if min_val is not None: 
            valid &= value >= min_val
        if max_val is not None: 
            valid &= value <= max_val
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
                return len(values) == 2 and all(v > 0 for v in values)
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
# 3. Material Class and Default Database
# ----------------------------------------------------
class ReactorMaterial:
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0, nu=0, xi=0):
        self.name = name
        self.mat_type = mat_type
        self.density = density * 1e3  # kg/m³
        self.molar_mass = molar_mass * 1e-3  # kg/mol
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
    'H2O': ReactorMaterial(
        name="H2O", mat_type='moderator',
        density=1.0, molar_mass=18.01,
        sigma_s_b=49.2, sigma_a_b=0.66,
        xi=0.920
    ),
    'D2O': ReactorMaterial(
        name="D2O", mat_type='moderator',
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

# ----------------------------------------------------
# 4. Reactor Mixture and Simulator Classes
# ----------------------------------------------------
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
        N_mod = (N_U235 + N_U238) * self.R_mtf
        return (
            N_U235 * self.u235_material.sigma_s +
            N_U238 * self.u238_material.sigma_s +
            N_mod * self.moderator.sigma_s
        )

    @property
    def macroscopic_absorption(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        N_mod = (N_U235 + N_U238) * self.R_mtf
        return (
            N_U235 * self.u235_material.sigma_a +
            N_U238 * self.u238_material.sigma_a +
            N_mod * self.moderator.sigma_a
        )

    @property
    def macroscopic_fission(self):
        N_U235 = self.u235_material.number_density * self.u235_frac
        N_U238 = self.u238_material.number_density * (1 - self.u235_frac)
        return (
            N_U235 * self.u235_material.sigma_f +
            N_U238 * self.u238_material.sigma_f
        )

    @property
    def average_neutrons_per_fission(self):
        numerator = (
            self.u235_material.number_density * self.u235_frac * 
            self.u235_material.sigma_f * self.u235_material.nu +
            self.u238_material.number_density * (1 - self.u235_frac) * 
            self.u238_material.sigma_f * self.u238_material.nu
        )
        denominator = (
            self.u235_material.number_density * self.u235_frac * 
            self.u235_material.sigma_f +
            self.u238_material.number_density * (1 - self.u235_frac) * 
            self.u238_material.sigma_f
        )
        return numerator / denominator if denominator != 0 else 0.0

class ReactorSimulator:
    def __init__(self, mixture, geometry, size):
        self.mixture = mixture
        self.geometry = geometry
        self.size = size  # float if sphere, [radius, height] if cylinder
        
    @property
    def total_mean_free_path(self):
        return 1 / (self.mixture.macroscopic_scattering + self.mixture.macroscopic_absorption)
    
    def scattering_probability(self):
        sigma_s = self.mixture.macroscopic_scattering
        sigma_total = sigma_s + self.mixture.macroscopic_absorption
        return sigma_s / sigma_total if sigma_total != 0 else 0.0
    
    def absorption_probability(self):
        sigma_a = self.mixture.macroscopic_absorption
        sigma_total = sigma_a + self.mixture.macroscopic_scattering
        return sigma_a / sigma_total if sigma_total != 0 else 0.0
    
    def fission_probability(self):
        sigma_f = self.mixture.macroscopic_fission
        sigma_a_total = self.mixture.macroscopic_absorption
        return sigma_f / sigma_a_total if sigma_a_total != 0 else 0.0
    
    @property
    def first_generation_k_factor(self):
        return self.fission_probability() * self.mixture.average_neutrons_per_fission
    
    def resonance_escape(self):
        """
        Simplistic approach used as a placeholder.
        We reference self.mixture.moderator for sigma_s_b and xi.
        """
        N_U238 = (1 - self.mixture.u235_frac) / self.mixture.R_mtf
        sigma_s_mod = self.mixture.moderator.sigma_s_b
        if self.mixture.moderator.xi == 0:
            return 1.0
        term = (N_U238 / sigma_s_mod) ** 0.514
        return np.exp(-2.73 / self.mixture.moderator.xi * term)
    
    def thermal_utilization(self):
        N_U235 = self.mixture.u235_material.number_density * self.mixture.u235_frac
        N_U238 = self.mixture.u238_material.number_density * (1 - self.mixture.u235_frac)
        N_mod = (N_U235 + N_U238) * self.mixture.R_mtf
        
        fuel_abs = (N_U235 * self.mixture.u235_material.sigma_a +
                    N_U238 * self.mixture.u238_material.sigma_a)
        total_abs = fuel_abs + N_mod * self.mixture.moderator.sigma_a
        return fuel_abs / total_abs if total_abs > 0 else 0.0
    
    def calculate_k(self, c=1):
        u235 = self.mixture.u235_material
        u238 = self.mixture.u238_material
        numerator = (u235.number_density * self.mixture.u235_frac *
                     u235.sigma_f * u235.nu)
        denominator = (u235.number_density * self.mixture.u235_frac * u235.sigma_a +
                       u238.number_density * (1 - self.mixture.u235_frac) * u238.sigma_a)
        if denominator == 0:
            return 0.0
        eta = numerator / denominator
        return eta * self.thermal_utilization() * self.resonance_escape() * c

# ----------------------------------------------------
# 5. Collision Simulation & k-Factor Methods
# ----------------------------------------------------
def collision_probabilities_method(self):
    sigma_s = self.mixture.macroscopic_scattering
    sigma_a = self.mixture.macroscopic_absorption
    sigma_f = self.mixture.macroscopic_fission
    sigma_total = sigma_s + sigma_a
    P_scatter = sigma_s / sigma_total if sigma_total > 0 else 0
    P_absorption = sigma_a / sigma_total if sigma_total > 0 else 0
    fission_fraction = sigma_f / sigma_a if sigma_a > 0 else 0
    P_fission = P_absorption * fission_fraction
    P_absorb_non_fission = P_absorption * (1 - fission_fraction)
    return P_scatter, P_absorb_non_fission, P_fission

ReactorSimulator.collision_probabilities = collision_probabilities_method

def random_walk(position, mean_free_path):
    step_length = np.random.exponential(scale=mean_free_path)
    direction = random_position_sphere_optimized(1)  # random unit vector
    return position + step_length * direction

def simulate_generation(neutron_positions, simulator, generation_number):
    """
    For each neutron, simulate a collision event.
    For generation > 1, update the position with a random walk step.
    Record positions for scattered, absorbed, and fission events.
    Returns new neutron positions (offspring) and event statistics.
    """
    P_scatter, P_absorb_non_fission, P_fission = simulator.collision_probabilities()
    new_neutron_positions = []
    scattered_positions = []
    absorbed_positions = []
    fission_positions = []
    offspring_counts = []
    
    for pos in neutron_positions:
        if generation_number > 1:
            pos = random_walk(pos, simulator.total_mean_free_path)
        r = np.random.rand()
        if r < P_scatter:
            # scattering
            count = 1
            scattered_positions.append(pos)
            new_neutron_positions.append(pos)
        elif r < P_scatter + P_absorb_non_fission:
            # non-fission absorption
            count = 0
            absorbed_positions.append(pos)
        else:
            # fission
            count = np.random.poisson(lam=simulator.mixture.average_neutrons_per_fission)
            fission_positions.append(pos)
            for _ in range(count):
                new_neutron_positions.append(pos)
        offspring_counts.append(count)
    
    stats = {
        'scattered_positions': np.array(scattered_positions) if scattered_positions else np.empty((0, 3)),
        'absorbed_positions': np.array(absorbed_positions) if absorbed_positions else np.empty((0, 3)),
        'fission_positions': np.array(fission_positions) if fission_positions else np.empty((0, 3)),
        'offspring_counts': np.array(offspring_counts)
    }
    return new_neutron_positions, stats

def compute_k_factor(offspring_counts):
    if len(offspring_counts) == 0:
        return 0, 0
    k = np.mean(offspring_counts)
    uncertainty = np.std(offspring_counts, ddof=1) / np.sqrt(len(offspring_counts))
    return k, uncertainty

# ----------------------------------------------------
# 6A. Volume-Normalized Histogram for Sphere
# ----------------------------------------------------
def plot_first_generation_histograms_volume_normalized_sphere(stats, reactor_radius, N_initial):
    """
    Plot a volume-normalized histogram of collisions from the first generation
    for a spherical reactor of radius 'reactor_radius' (meters).
    """
    import matplotlib.pyplot as plt

    scattered = stats['scattered_positions']
    absorbed  = stats['absorbed_positions']
    fissioned = stats['fission_positions']
    
    def radial_distances(positions):
        if positions.size == 0:
            return np.array([])
        return np.linalg.norm(positions, axis=1)

    r_scatter = radial_distances(scattered)
    r_absorb  = radial_distances(absorbed)
    r_fission = radial_distances(fissioned)
    
    # Define bins from 0 to reactor_radius
    num_bins = 20
    bins = np.linspace(0, reactor_radius, num_bins+1)
    
    # Raw counts in each bin
    counts_scatter, _ = np.histogram(r_scatter, bins=bins)
    counts_absorb,  _ = np.histogram(r_absorb,  bins=bins)
    counts_fission, _ = np.histogram(r_fission, bins=bins)
    
    # Shell volumes for spherical shells
    shell_volumes = (4.0/3.0) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    
    # Normalize by shell volume => collisions per cubic meter
    with np.errstate(divide='ignore', invalid='ignore'):
        density_scatter = counts_scatter / shell_volumes
        density_absorb  = counts_absorb  / shell_volumes
        density_fission = counts_fission / shell_volumes
    
    # Bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    
    # We'll plot side-by-side bars with partial transparency, plus connecting lines.
    alpha_val = 0.6
    offset = 0.3 * width
    
    plt.figure(figsize=(8,6))
    
    # Bar plots (side-by-side)
    plt.bar(bin_centers - offset, density_scatter, width=0.3*width, 
            color='blue', alpha=alpha_val, label='Scattered')
    plt.bar(bin_centers, density_absorb, width=0.3*width, 
            color='orange', alpha=alpha_val, label='Absorbed')
    plt.bar(bin_centers + offset, density_fission, width=0.3*width, 
            color='green', alpha=alpha_val, label='Fission')
    
    # Now overlay line plots connecting the tops of each bar group
    plt.plot(bin_centers - offset, density_scatter, color='blue', marker='o')
    plt.plot(bin_centers, density_absorb, color='orange', marker='o')
    plt.plot(bin_centers + offset, density_fission, color='green', marker='o')
    
    plt.xlabel('Radial distance r from reactor center (m)')
    plt.ylabel('Collision density (collisions / m$^3$)')
    plt.title(f'First Generation Collision Distribution (Volume Normalized)\n'
              f'Sphere, N={N_initial} neutrons')
    plt.legend()
    plt.show()

# ----------------------------------------------------
# 6B. Volume-Normalized Histogram for Cylinder
# ----------------------------------------------------
def plot_first_generation_histograms_volume_normalized_cylinder(stats, radius, height, N_initial):
    """
    Plot a volume-normalized histogram of collisions from the first generation
    for a cylindrical reactor of radius 'radius' and height 'height' (both in meters).
    """
    import matplotlib.pyplot as plt

    scattered = stats['scattered_positions']
    absorbed  = stats['absorbed_positions']
    fissioned = stats['fission_positions']
    
    def radial_distances(positions):
        if positions.size == 0:
            return np.array([])
        # radial distance in x-y plane
        r_xy = np.sqrt(positions[:,0]**2 + positions[:,1]**2)
        return r_xy

    r_scatter = radial_distances(scattered)
    r_absorb  = radial_distances(absorbed)
    r_fission = radial_distances(fissioned)
    
    # Define bins from 0 to radius
    num_bins = 20
    bins = np.linspace(0, radius, num_bins+1)
    
    # Raw counts in each bin
    counts_scatter, _ = np.histogram(r_scatter, bins=bins)
    counts_absorb,  _ = np.histogram(r_absorb,  bins=bins)
    counts_fission, _ = np.histogram(r_fission, bins=bins)
    
    # Volume of cylindrical shells:  pi*(r_{i+1}^2 - r_i^2) * height
    shell_volumes = np.pi * (bins[1:]**2 - bins[:-1]**2) * height
    
    with np.errstate(divide='ignore', invalid='ignore'):
        density_scatter = counts_scatter / shell_volumes
        density_absorb  = counts_absorb  / shell_volumes
        density_fission = counts_fission / shell_volumes

    # Bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    
    # Plot side-by-side bars + lines, partial transparency
    alpha_val = 0.6
    offset = 0.3 * width
    
    plt.figure(figsize=(8,6))
    
    plt.bar(bin_centers - offset, density_scatter, width=0.3*width, 
            color='blue', alpha=alpha_val, label='Scattered')
    plt.bar(bin_centers, density_absorb, width=0.3*width, 
            color='orange', alpha=alpha_val, label='Absorbed')
    plt.bar(bin_centers + offset, density_fission, width=0.3*width, 
            color='green', alpha=alpha_val, label='Fission')
    
    # Overlay lines
    plt.plot(bin_centers - offset, density_scatter, color='blue', marker='o')
    plt.plot(bin_centers, density_absorb, color='orange', marker='o')
    plt.plot(bin_centers + offset, density_fission, color='green', marker='o')
    
    plt.xlabel('Radial distance r from reactor center (m)')
    plt.ylabel('Collision density (collisions / m$^3$)')
    plt.title(f'First Generation Collision Distribution (Volume Normalized)\n'
              f'Cylinder, N={N_initial} neutrons')
    plt.legend()
    plt.show()

# ----------------------------------------------------
# 7. Plotting k-Factor Evolution & Parameter Optimization
# ----------------------------------------------------
def plot_k_evolution(k_factors, uncertainties):
    import matplotlib.pyplot as plt
    generations = np.arange(1, len(k_factors) + 1)
    plt.figure()
    plt.errorbar(generations, k_factors, yerr=uncertainties, fmt='-o', label='k-factor')
    plt.axhline(y=1, color='r', linestyle='--', label='k = 1')
    plt.xlabel('Generation (q)')
    plt.ylabel('k-factor')
    plt.title('Evolution of k-factor over Generations')
    plt.legend()
    plt.show()

def parameter_optimization(simulator, geometry, initial_neutron_count=1000, tolerance=0.05):
    """
    Simple grid search over radius, (height if cylinder), R_mtf, and U-235 percent
    to find parameter sets that yield k-factor ~ 1 in the first generation.
    """
    radii = np.linspace(0.5, 2.0, 4)
    heights = np.linspace(1.0, 4.0, 4) if geometry == 'cylinder' else [None]
    R_mtfs = np.linspace(0.5, 1.5, 3)
    U235_percents = [1, 3, 5, 10, 20]  # example set of enrichments
    
    matching_sets = []
    for r in radii:
        for R_mtf in R_mtfs:
            for u235_pct in U235_percents:
                if geometry == 'cylinder':
                    for h in heights:
                        size = [r, h]
                        mixture = ReactorMixture(u235_pct, simulator.mixture.moderator, R_mtf,
                                                 simulator.mixture.u235_material, simulator.mixture.u238_material)
                        sim_temp = ReactorSimulator(mixture, geometry, size)
                        # Generate initial neutrons
                        neutrons = [random_position_cylinder(r, h) for _ in range(initial_neutron_count)]
                        _, stats = simulate_generation(neutrons, sim_temp, generation_number=1)
                        k, _ = compute_k_factor(stats['offspring_counts'])
                        if abs(k - 1) < tolerance:
                            matching_sets.append({
                                'radius (m)': r,
                                'height (m)': h,
                                'R_mtf': R_mtf,
                                'U235 (%)': u235_pct,
                                'k-factor': k
                            })
                else:
                    size = r
                    mixture = ReactorMixture(u235_pct, simulator.mixture.moderator, R_mtf,
                                             simulator.mixture.u235_material, simulator.mixture.u238_material)
                    sim_temp = ReactorSimulator(mixture, geometry, size)
                    neutrons = [random_position_sphere_optimized(r) for _ in range(initial_neutron_count)]
                    _, stats = simulate_generation(neutrons, sim_temp, generation_number=1)
                    k, _ = compute_k_factor(stats['offspring_counts'])
                    if abs(k - 1) < tolerance:
                        matching_sets.append({
                            'radius (m)': r,
                            'R_mtf': R_mtf,
                            'U235 (%)': u235_pct,
                            'k-factor': k
                        })
    return matching_sets

# ----------------------------------------------------
# 8. Main Extended Simulation Function
# ----------------------------------------------------
def extended_main():
    # Prompt for reactor type
    reactor_type = get_valid_input(
        "Reactor type [finite/infinite]: ",
        validate_reactor_type,
        "Must be 'finite' or 'infinite'"
    ).lower()

    # Prompt for geometry
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'"
    ).lower()

    # Prompt for reactor size (in meters)
    size_prompt = ("Enter size in meters:\n"
                   "- Sphere: radius\n"
                   "- Cylinder: [radius,height]\n"
                   "Size: ")
    size_input = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format"
    )
    if geometry == 'cylinder':
        size = list(map(float, size_input.strip('[]').split(',')))
    else:
        size = float(size_input)

    # Prompt for moderator
    moderator = get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O, D2O, or Graphite"
    ).capitalize()

    # Prompt for dataset
    dataset = get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be 'Lilley' or 'Wikipedia'"
    ).lower()

    # Prompt for neutron model if Wikipedia
    if dataset == 'wikipedia':
        neutron_model = get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be 'Thermal' or 'Fast'"
        ).lower()
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
        u235 = LILLEY_U235
        u238 = LILLEY_U238

    # Prompt for U-235 concentration
    u235_percent = float(get_valid_input(
        "U-235 concentration [0-100%]: ",
        lambda x: validate_float(x, 0, 100),
        "Must be a number between 0 and 100"
    ))

    # Prompt for moderator-to-fuel ratio
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio (R_mtf > 0): ",
        lambda x: validate_float(x, 0.01),
        "Must be a positive number"
    ))

    # Prompt for number of initial neutrons
    N_initial = get_positive_integer("Enter the number of initial neutrons (positive integer): ")
    
    # Create mixture and simulator
    moderator_obj = MODERATORS[moderator]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235, u238)
    simulator = ReactorSimulator(mixture, geometry, size)
    
    # Compute leakage factor c if finite
    c = 1 if reactor_type == 'infinite' else (1 - 0.1)*(1 - 0.2)  # Example approach

    # Print initial reactor physics parameters
    print("\n=== Reactor Physics Parameters ===")
    print(f"Scattering Probability: {simulator.scattering_probability():.3f}")
    print(f"Absorption Probability: {simulator.absorption_probability():.3f}")
    print(f"Fission Probability: {simulator.fission_probability():.3f}")
    print(f"Total mean free path (R_mtf={mixture.R_mtf}): {simulator.total_mean_free_path:.2e} m")
    print(f"1st generation k-factor (k1): {simulator.first_generation_k_factor:.3f}")
    print(f"Thermal utilization (f): {simulator.thermal_utilization():.3f}")
    print(f"Resonance escape (p): {simulator.resonance_escape():.3f}")
    print(f"k_effective: {simulator.calculate_k(c):.3f}")

    # Prompt for number of simulation generations
    Q = get_positive_integer("Enter the number of simulation generations (positive integer): ")

    # Generate initial neutron positions
    neutrons = []
    if geometry == 'sphere':
        for _ in range(N_initial):
            neutrons.append(random_position_sphere_optimized(size))
    elif geometry == 'cylinder':
        for _ in range(N_initial):
            neutrons.append(random_position_cylinder(size[0], size[1]))

    # Simulate each generation and store k-factor results
    k_factors = []
    uncertainties = []
    generation_stats = []

    # Generation 1
    neutrons, stats = simulate_generation(neutrons, simulator, generation_number=1)
    k, unc = compute_k_factor(stats['offspring_counts'])
    k_factors.append(k)
    uncertainties.append(unc)
    generation_stats.append(stats)
    print(f"Generation 1: k-factor = {k:.3f} ± {unc:.3f}")

    # Subsequent generations
    for g in range(2, Q+1):
        if len(neutrons) == 0:
            print(f"No neutrons left at generation {g}. Simulation terminated.")
            break
        neutrons, stats = simulate_generation(neutrons, simulator, generation_number=g)
        k, unc = compute_k_factor(stats['offspring_counts'])
        k_factors.append(k)
        uncertainties.append(unc)
        generation_stats.append(stats)
        print(f"Generation {g}: k-factor = {k:.3f} ± {unc:.3f}")

    # Plot first-generation collisions
    if geometry == 'sphere':
        # Volume-normalized histogram (spherical)
        plot_first_generation_histograms_volume_normalized_sphere(
            generation_stats[0], size, N_initial
        )
    else:
        # Volume-normalized histogram (cylindrical)
        plot_first_generation_histograms_volume_normalized_cylinder(
            generation_stats[0], size[0], size[1], N_initial
        )

    # Plot k-factor evolution
    plot_k_evolution(k_factors, uncertainties)

    # ------------------------------------------------
    # Compute average k-factor from generation 11 to 20
    # ------------------------------------------------
    # (only if we have at least 20 generations)
    if len(k_factors) >= 20:
        k_subset = k_factors[10:20]  # 11th to 20th generation => indices 10..19
        avg_k_11_20 = np.mean(k_subset)
        print(f"\nAverage k-factor from generation 11 to 20: {avg_k_11_20:.3f}")
    else:
        print("\nFewer than 20 generations were simulated; no average k from gen 11-20.")

    # Optional parameter optimization
    opt_choice = input("Do you want to run parameter optimization for k-factor = 1? (Y/N): ").strip().lower()
    if opt_choice == 'y':
        matching_sets = parameter_optimization(simulator, geometry, initial_neutron_count=500, tolerance=0.05)
        if matching_sets:
            print("\nParameter sets yielding k-factor ≈ 1:")
            for s in matching_sets:
                if geometry == 'cylinder':
                    print(f"Radius: {s['radius (m)']} m, Height: {s['height (m)']} m, "
                          f"R_mtf: {s['R_mtf']}, U235: {s['U235 (%)']}%, k-factor: {s['k-factor']:.3f}")
                else:
                    print(f"Radius: {s['radius (m)']} m, R_mtf: {s['R_mtf']}, "
                          f"U235: {s['U235 (%)']}%, k-factor: {s['k-factor']:.3f}")
        else:
            print("No parameter sets found yielding k-factor ≈ 1 within tolerance.")

if __name__ == "__main__":
    extended_main()
