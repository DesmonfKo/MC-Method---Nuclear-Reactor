import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm
from scipy.constants import Avogadro as NA

# ---------------------------
# Existing Code (for reference)
# ---------------------------
# Random position functions for sphere and cylinder:
def random_position_sphere_optimized(radius=1):
    '''
    Generate a random position inside a sphere with uniform density.
    '''
    vec = rand.randn(3)
    vec /= np.linalg.norm(vec)
    # Scale radius by cube root transformation to ensure uniform density
    r = radius * (rand.uniform() ** (1/3))
    return r * vec

def random_position_cylinder(radius=1, height=1):
    '''
    Generate a random position inside a cylinder with uniform density.
    '''
    theta = rand.uniform(0, 2 * np.pi)
    r = np.sqrt(rand.uniform(0, 1)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rand.uniform(-height / 2, height / 2)
    return np.array([x, y, z])

# Input validation helpers:
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

# Material Class Definition
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

# Lilley's default materials:
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

# Modified Reactor Mixture Class
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

# Reactor Simulator Class (with some added helper methods below)
class ReactorSimulator:
    def __init__(self, mixture, geometry, size):
        self.mixture = mixture
        self.geometry = geometry
        self.size = size

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
        N_U238 = (1 - self.mixture.u235_frac) / self.mixture.R_mtf
        sigma_s_mod = self.mixture.moderator.sigma_s_b
        term = (N_U238 / sigma_s_mod) ** 0.514
        return np.exp(-2.73 / self.mixture.moderator.xi * term)

    def thermal_utilization(self):
        N_U235 = self.mixture.u235_material.number_density * self.mixture.u235_frac
        N_U238 = self.mixture.u238_material.number_density * (1 - self.mixture.u235_frac)
        N_mod = (N_U235 + N_U238) * self.mixture.R_mtf

        fuel_abs = N_U235*self.mixture.u235_material.sigma_a + N_U238*self.mixture.u238_material.sigma_a
        total_abs = fuel_abs + N_mod*self.mixture.moderator.sigma_a
        return fuel_abs / total_abs

    def calculate_k(self, c=1):
        u235 = self.mixture.u235_material
        u238 = self.mixture.u238_material
        numerator = u235.number_density * self.mixture.u235_frac * u235.sigma_f * u235.nu
        denominator = (u235.number_density * self.mixture.u235_frac * u235.sigma_a +
                      u238.number_density * (1 - self.mixture.u235_frac) * u238.sigma_a)
        η = numerator / denominator
        return η * self.thermal_utilization() * self.resonance_escape() * c

# ---------------------------
# Extended Simulation Functionality
# ---------------------------

# (a) Add a method to compute the collision outcome probabilities.
# Here we “split” absorption into non-fission and fission events.
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

# (b) For later generations we simulate a random walk step:
def random_walk(position, mean_free_path):
    step_length = np.random.exponential(scale=mean_free_path)
    # Use the existing function to get a random unit vector
    direction = random_position_sphere_optimized(1)
    return position + step_length * direction

# (c) Simulate one generation of neutron collisions.
def simulate_generation(neutron_positions, simulator, generation_number):
    """
    For each neutron in the input list, simulate a collision event.
    For generation > 1, update the position with a random walk step.
    Records positions for scattered, absorbed, and fission events.
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
            # Scattering: neutron continues (offspring = 1)
            count = 1
            scattered_positions.append(pos)
            new_neutron_positions.append(pos)
        elif r < P_scatter + P_absorb_non_fission:
            # Absorption: neutron is lost (offspring = 0)
            count = 0
            absorbed_positions.append(pos)
        else:
            # Fission: produce new neutrons. Here we sample from a Poisson with mean = average neutrons per fission.
            count = np.random.poisson(lam=simulator.mixture.average_neutrons_per_fission)
            fission_positions.append(pos)
            for _ in range(count):
                new_neutron_positions.append(pos)
        offspring_counts.append(count)
    stats = {
        'initial_positions': neutron_positions.copy(),
        'scattered_positions': np.array(scattered_positions) if scattered_positions else np.empty((0,3)),
        'absorbed_positions': np.array(absorbed_positions) if absorbed_positions else np.empty((0,3)),
        'fission_positions': np.array(fission_positions) if fission_positions else np.empty((0,3)),
        'offspring_counts': np.array(offspring_counts)
    }
    return new_neutron_positions, stats

# (d) Compute the k-factor and its uncertainty from the offspring counts.
def compute_k_factor(offspring_counts):
    if len(offspring_counts) == 0:
        return 0, 0
    k = np.mean(offspring_counts)
    uncertainty = np.std(offspring_counts, ddof=1) / np.sqrt(len(offspring_counts))
    return k, uncertainty

# (e) Plot histogram of first generation collision positions.
def plot_first_generation_histograms(stats):
    def radial_distances(positions):
        if positions is None or len(positions) == 0:
            return np.array([])
        return np.linalg.norm(positions, axis=1)
    
    # Extract radial positions
    r_scatter = radial_distances(stats.get('scattered_positions', []))
    r_absorb = radial_distances(stats.get('absorbed_positions', []))
    r_fission = radial_distances(stats.get('fission_positions', []))

    # Define radial bins and calculate volume elements for normalization
    num_bins = 50
    r_max = max(np.max(np.concatenate([r_scatter, r_absorb, r_fission], dtype=float)), 1)
    bins = np.linspace(0, r_max, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)  # Spherical shell volume elements

    # Histogram counts in each bin
    counts_scatter, _ = np.histogram(r_scatter, bins=bins)
    counts_absorb, _ = np.histogram(r_absorb, bins=bins)
    counts_fission, _ = np.histogram(r_fission, bins=bins)

    # Convert to neutron densities
    density_scatter = counts_scatter / bin_volumes
    density_absorb = counts_absorb / bin_volumes
    density_fission = counts_fission / bin_volumes

    # Plot histogram bars
    plt.figure(figsize=(7, 5))
    plt.bar(bin_centers, density_scatter, width=np.diff(bins), alpha=0.6, label='Scattered Neutrons', edgecolor='black')
    plt.bar(bin_centers, density_absorb, width=np.diff(bins), alpha=0.6, label='Absorbed Neutrons', edgecolor='black')
    plt.bar(bin_centers, density_fission, width=np.diff(bins), alpha=0.6, label='Fission Neutrons', edgecolor='black')

    # Scientific plot formatting
    plt.xlabel(r'Radial Distance from Reactor Center (m)', fontsize=12)
    plt.ylabel(r'Neutron Density (neutrons/m$^3$)', fontsize=12)
    plt.title(r'Neutron Density vs radial distance', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def plot_neutron_distribution(stats_N0, stats_N1, geometry, size):
    def radial_distances(positions, geometry):
        if len(positions) == 0:
            return np.array([])
        positions = np.array(positions)
        if geometry == 'sphere':
            return np.linalg.norm(positions, axis=1)
        elif geometry == 'cylinder':
            return np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        return np.array([])
    
    # Extract reactor radius (R) and calculate R²
    R = size if geometry == 'sphere' else size[0]
    R_squared = R**2

    # Extract radial distances for N₀ (initial) and N₁ (next generation)
    r_N0 = radial_distances(stats_N0['initial_positions'], geometry)
    r_N1 = radial_distances(stats_N1['initial_positions'], geometry)
    r_absorb = radial_distances(stats_N0['absorbed_positions'], geometry)
    r_fission = radial_distances(stats_N0['fission_positions'], geometry)

    # Define bins in r² space (10× more points for smoother curves)
    num_bins = 200  # Increased from 20 to 200
    bins_r_squared = np.linspace(0, R_squared, num_bins + 1)
    bins_r = np.sqrt(bins_r_squared)  # Convert to r bins for volume calculation
    bin_centers_r_squared = (bins_r_squared[:-1] + bins_r_squared[1:]) / 2

    # Calculate bin volumes/areas for normalization
    if geometry == 'sphere':
        bin_volumes = (4/3) * np.pi * (bins_r[1:]**3 - bins_r[:-1]**3)
    elif geometry == 'cylinder':
        height = size[1] if geometry == 'cylinder' else 1.0
        bin_volumes = np.pi * (bins_r_squared[1:] - bins_r_squared[:-1]) * height

    bin_volumes[bin_volumes == 0] = np.inf  # Avoid division by zero

    # Compute densities for N₀, N₁, absorbed, and fission
    counts_N0, _ = np.histogram(r_N0**2, bins=bins_r_squared)
    density_N0 = counts_N0 / bin_volumes

    counts_N1, _ = np.histogram(r_N1**2, bins=bins_r_squared)
    density_N1 = counts_N1 / bin_volumes

    counts_absorb, _ = np.histogram(r_absorb**2, bins=bins_r_squared)
    density_absorb = counts_absorb / bin_volumes

    counts_fission, _ = np.histogram(r_fission**2, bins=bins_r_squared)
    density_fission = counts_fission / bin_volumes

    # Plotting with smooth joint lines
    plt.figure(figsize=(10, 6))
    
    plt.plot(bin_centers_r_squared, density_N0, 'b-', linewidth=2, label='Initial Distribution (N₀)')
    plt.plot(bin_centers_r_squared, density_N1, 'r-', linewidth=2, label='Next Generation (N₁)')
    plt.plot(bin_centers_r_squared, density_absorb, 'g-', linewidth=1.5, label='Absorbed Neutrons (N₀)')
    plt.plot(bin_centers_r_squared, density_fission, 'm-', linewidth=1.5, label='Fission Neutrons (N₀)')
    
    # Reactor Boundary
    plt.axvline(R_squared, color='k', linestyle='--', label=f'Reactor Boundary ($R² = {R_squared:.2f}$ m²)')
    
    plt.xlabel(r'Radius$^2$ (m$^2$)', fontsize=14)
    plt.ylabel('Neutron Density [neutrons/m³]', fontsize=14)
    plt.title('Neutron Density vs Radius$^2$', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# (f) Plot the evolution of k-factor over generations with error bars.
def plot_k_evolution(k_factors, uncertainties):
    generations = np.arange(1, len(k_factors)+1)
    plt.figure()
    plt.errorbar(generations, k_factors, yerr=uncertainties, fmt='-o', label='k-factor')
    plt.axhline(y=1, color='r', linestyle='--', label='k = 1')
    plt.xlabel('Generation (q)')
    plt.ylabel('k-factor')
    plt.title('k-factor Evolution over Generations')
    plt.legend()
    plt.show()

# (g) Helper to get a positive integer from the user.
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

# (h) Parameter optimization via grid search:
def parameter_optimization(simulator, geometry, initial_neutron_count=1000, tolerance=0.05):
    # Define grids for radius (in meters), height (if applicable), moderator-to-fuel ratio, and U235 percentage.
    radii = np.linspace(0.5, 2.0, 4)
    heights = np.linspace(1.0, 4.0, 4) if geometry=='cylinder' else [None]
    R_mtfs = np.linspace(0.5, 1.5, 3)
    U235_percents = [1, 3, 5, 10, 20]  # in percent

    matching_sets = []
    for r in radii:
        for R_mtf in R_mtfs:
            for u235_pct in U235_percents:
                if geometry == 'cylinder':
                    for h in heights:
                        size = [r, h]
                        mixture = ReactorMixture(u235_pct, simulator.mixture.moderator, R_mtf, simulator.mixture.u235_material, simulator.mixture.u238_material)
                        sim_temp = ReactorSimulator(mixture, geometry, size)
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
                    mixture = ReactorMixture(u235_pct, simulator.mixture.moderator, R_mtf, simulator.mixture.u235_material, simulator.mixture.u238_material)
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

# ---------------------------
# Extended Main Function
# ---------------------------
def extended_main():
    # Reactor setup (similar to the original main)
    reactor_type = get_valid_input(
        "Reactor type [finite/infinite]: ",
        validate_reactor_type,
        "Must be 'finite' or 'infinite'"
    ).lower()

    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'"
    ).lower()

    size_prompt = ("Enter size in meters:\n- Sphere: radius\n- Cylinder: [radius,height]\nSize: ")
    size_input = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format"
    )
    if geometry == 'cylinder':
        size = list(map(float, size_input.strip('[]').split(',')))
    else:
        size = float(size_input)

    moderator = get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O, D2O, or Graphite"
    ).capitalize()

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
        else:
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

    u235_percent = float(get_valid_input(
        "U-235 concentration [0-100%]: ",
        lambda x: validate_float(x, 0, 100),
        "Must be number between 0 and 100"
    ))
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio (R_mtf > 0): ",
        lambda x: validate_float(x, 0.01),
        "Must be positive number"
    ))

    moderator_obj = MODERATORS[moderator]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235, u238)
    simulator = ReactorSimulator(mixture, geometry, size)

    # Print some reactor physics parameters (from the original simulation)
    c = 1 if reactor_type == 'infinite' else (1 - 0.1) * (1 - 0.2)
    print("\n=== Reactor Physics Parameters ===")
    print(f"Scattering Probability: {simulator.scattering_probability():.3f}")
    print(f"Absorption Probability: {simulator.absorption_probability():.3f}")
    print(f"Fission Probability: {simulator.fission_probability():.3f}")
    print(f"Total mean free path (R_mtf={mixture.R_mtf}): {simulator.total_mean_free_path:.2e} m")
    print(f"1st generation k-factor (k1): {simulator.first_generation_k_factor:.6f}")
    print(f"Thermal utilization (f): {simulator.thermal_utilization():.3f}")
    print(f"Resonance escape (p): {simulator.resonance_escape():.3f}")
    print(f"k_effective: {simulator.calculate_k(c):.6f}")

    # Prompt for number of simulation generations (Task 7)
    Q = get_positive_integer("Enter the number of simulation generations (positive integer): ")

    # Generate initial neutron positions within the reactor volume
    N_initial = 10000  # You can adjust this number as needed
    neutrons = []
    if geometry == 'sphere':
        for _ in range(N_initial):
            neutrons.append(random_position_sphere_optimized(size))
    elif geometry == 'cylinder':
        for _ in range(N_initial):
            neutrons.append(random_position_cylinder(size[0], size[1]))

    # Simulate each generation and store k-factor values and uncertainties (Tasks 8 & 9)
    k_factors = []
    uncertainties = []
    generation_stats = []

    # Generation 1 simulation
    neutrons, stats = simulate_generation(neutrons, simulator, generation_number=1)
    k, unc = compute_k_factor(stats['offspring_counts'])
    k_factors.append(k)
    uncertainties.append(unc)
    generation_stats.append(stats)

    print(f"Generation 1: k-factor = {k:.6f} ± {unc:.6f}")

    # Subsequent generations (random walk + collision simulation)
    for q in range(2, Q+1):
        if len(neutrons) == 0:
            print(f"No neutrons left at generation {q}. Simulation terminated.")
            break
        neutrons, stats = simulate_generation(neutrons, simulator, generation_number=q)
        k, unc = compute_k_factor(stats['offspring_counts'])
        k_factors.append(k)
        uncertainties.append(unc)
        generation_stats.append(stats)
        print(f"Generation {q}: k-factor = {k:.6f} ± {unc:.6f}")
        
        # Simulate Generation 1 (N₀ → N₀ scattered/absorbed/fission)
    neutrons_gen1, stats_gen1 = simulate_generation(neutrons, simulator, generation_number=1)
    
    # Simulate Generation 2 (N₁: random walk applied)
    neutrons_gen2, stats_gen2 = simulate_generation(neutrons_gen1, simulator, generation_number=2)
    # Plot histogram for the first generation collisions (Task 5)
    plot_first_generation_histograms(generation_stats[0])
    plot_neutron_distribution(stats_gen1, stats_gen2, geometry, size)

    # Plot k-factor evolution vs generation with error bars (Task 10)
    plot_k_evolution(k_factors, uncertainties)
    
    
    
    
    
    
    
    '''
    # Parameter optimization: sweep reactor parameters to find sets where k-factor ≈ 1 (Tasks 11 & 12)
    opt_choice = input("Do you want to run parameter optimization for k-factor = 1? (Y/N): ").strip().lower()
    if opt_choice == 'y':
        matching_sets = parameter_optimization(simulator, geometry, initial_neutron_count=500, tolerance=0.05)
        if matching_sets:
            print("\nParameter sets yielding k-factor ≈ 1:")
            for s in matching_sets:
                if geometry == 'cylinder':
                    print(f"Radius: {s['radius (m)']} m, Height: {s['height (m)']} m, R_mtf: {s['R_mtf']}, U235: {s['U235 (%)']}%, k-factor: {s['k-factor']:.6f}")
                else:
                    print(f"Radius: {s['radius (m)']} m, R_mtf: {s['R_mtf']}, U235: {s['U235 (%)']}%, k-factor: {s['k-factor']:.6f}")
        else:
            print("No parameter sets found yielding k-factor ≈ 1 within tolerance.")
    '''
if __name__ == "__main__":
    extended_main()
