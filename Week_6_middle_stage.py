## New Extension, Week 6 middle stage ##
#####################################
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
    We'll rename them internally (H2O->'water', D2O->'heavywater'), but just check membership ignoring case.
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
    """
    Container for nuclear data for one isotope (or compound).
    All cross sections in barns are stored but also converted to SI (m^2).
    """
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name = name
        self.mat_type = mat_type  # 'fuel' or 'moderator'
        # Convert from g/cc to kg/m^3 for density
        self.density = density * 1e3
        # Convert from g/mol to kg/mol for molar_mass
        self.molar_mass = molar_mass * 1e-3
        
        self.sigma_s_b = sigma_s_b  # scattering (barn)
        self.sigma_a_b = sigma_a_b  # absorption (barn)
        self.sigma_f_b = sigma_f_b  # fission (barn)
        
        # Convert barns to m^2 (1 barn = 1e-28 m^2)
        self.sigma_s = sigma_s_b * 1e-28
        self.sigma_a = sigma_a_b * 1e-28
        self.sigma_f = sigma_f_b * 1e-28
        
        self.nu = nu  # average neutrons per fission for that isotope
        self.xi = xi  # average log-energy decrement (only for moderators)

    @property
    def number_density(self):
        """
        number_density = (density / molar_mass) * NA
        Units: [atoms/m^3].
        """
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
# ReactorMixture with Revised Σ's
########################################
class ReactorMixture:
    """
    We define 3 isotopes in the mixture: U-235, U-238, and a moderator (e.g. H2O).
    Let a(U-235) = fraction_U235
        a(U-238) = (1 - fraction_U235)
        a(moderator) = R_mtf  (ratio of moderator atoms to total fuel atoms)

    Then the number density of each species i is:
        N_i = a_i * (density_i / molar_mass_i) * NA   [where a_i are relative atomic fractions]

    The macroscopic cross section for process j (scattering, absorption, fission) is:
        Σ_j = Σ_i [ N_i * σ_{i,j} ],
    where i runs over {U235, U238, moderator}.  But for fission, only the fuel contributes.

    Probability of fission = Σ_{f, fuel} / Σ_{a, total}
    Probability of absorption = Σ_{a, total} / [Σ_{a, total} + Σ_{s, total}]
    Probability of scattering = Σ_{s, total} / [Σ_{a, total} + Σ_{s, total}]
    """
    def __init__(self, fraction_U235, moderator, R_mtf, u235_material, u238_material):
        """
        fraction_U235 : fraction (0 to 1) of the fuel that is U-235 by atom fraction
        moderator     : a ReactorMaterial object (moderator)
        R_mtf         : ratio of (moderator atoms) to (total fuel atoms), can be >= 0
        """
        self.fraction_U235 = fraction_U235
        self.moderator = moderator
        self.R_mtf = R_mtf
        self.u235 = u235_material
        self.u238 = u238_material

        # Precompute number densities:
        self.N_u235 = self.fraction_U235 * self.u235.number_density
        self.N_u238 = (1 - self.fraction_U235) * self.u238.number_density
        self.N_mod  = self.R_mtf * (self.N_u235 + self.N_u238)

    @property
    def macroscopic_scattering(self):
        return ( self.N_u235 * self.u235.sigma_s 
               + self.N_u238 * self.u238.sigma_s
               + self.N_mod  * self.moderator.sigma_s
               )

    @property
    def macroscopic_absorption(self):
        return ( self.N_u235 * self.u235.sigma_a
               + self.N_u238 * self.u238.sigma_a
               + self.N_mod  * self.moderator.sigma_a
               )

    @property
    def macroscopic_fission(self):
        # Only fuel contributes to fission cross section
        return ( self.N_u235 * self.u235.sigma_f
               + self.N_u238 * self.u238.sigma_f
               )

    @property
    def fission_probability(self):
        """
        P_fission = Σ_f,fuel / Σ_a,total, if Σ_a>0; else 0.
        """
        sigma_f = self.macroscopic_fission
        sigma_a = self.macroscopic_absorption
        if sigma_a > 1e-30:
            return sigma_f / sigma_a
        else:
            return 0.0

    @property
    def scattering_probability(self):
        """
        P_scatter = Σ_s / (Σ_s + Σ_a).
        """
        sigma_s = self.macroscopic_scattering
        sigma_a = self.macroscopic_absorption
        denom = sigma_s + sigma_a
        if denom <= 1e-30:
            return 0.0
        return sigma_s / denom

    @property
    def absorption_probability(self):
        """
        P_abs = Σ_a / (Σ_s + Σ_a).
        """
        sigma_s = self.macroscopic_scattering
        sigma_a = self.macroscopic_absorption
        denom = sigma_s + sigma_a
        if denom <= 1e-30:
            return 0.0
        return sigma_a / denom

#####################################
# PATCH 3 of N
#####################################

def distance_to_collision(mixture):
    """
    Sample a random distance to collision with mean free path = 1 / (Σ_s + Σ_a).
    Exponential distribution: d = -ln(rand()) / Σ_tot
    """
    sigma_s = mixture.macroscopic_scattering
    sigma_a = mixture.macroscopic_absorption
    sigma_tot = sigma_s + sigma_a
    if sigma_tot <= 1e-30:
        return 1e10  # effectively no collisions
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
    Cylinder axis is along z from z=-height/2 to z=height/2.
    """
    x, y, z = point
    r2 = x*x + y*y
    return (r2 <= radius**2) and (abs(z) <= (height/2))


def radial_distance_sphere(point):
    """
    Return distance from origin.
    """
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)


def radial_distance_cylinder(point):
    """
    Return radial distance from cylinder axis (z-axis).
    """
    x, y, z = point
    return np.sqrt(x*x + y*y)


def random_scatter_direction():
    """
    Generate a random isotropic scattering direction in 3D.
    """
    costheta = 2.0*rand.rand() - 1.0
    phi = 2.0*np.pi*rand.rand()
    sintheta = np.sqrt(1.0 - costheta**2)
    return np.array([
        sintheta*np.cos(phi),
        sintheta*np.sin(phi),
        costheta
    ])


def simulate_first_generation(mixture, geometry, size, N0, average_neutrons_per_fission=2.42, bins=20):
    """
    Simulate the random walk of N0 neutrons for the 1st generation.
    They start uniformly distributed inside the chosen geometry.
    Track collisions (scatter/absorb/fission) until each neutron
    either is absorbed or leaks.

    Returns a dictionary with:
       'absorbed_positions': array of final absorption points
       'absorbed_count': int
       'leak_count': int
       'bin_edges': array
       'bin_centers': array
       'scatter_density': array
       'absorb_density': array
       'fission_density': array
    """
    # Probabilities from mixture
    P_scat = mixture.scattering_probability
    P_abs  = mixture.absorption_probability
    P_fis  = mixture.fission_probability

    if geometry=='sphere':
        R = float(size)
    else:
        R = float(size[0])
        H = float(size[1])

    # Initialize positions
    neutron_positions = np.zeros((N0, 3))
    for i in range(N0):
        if geometry=='sphere':
            neutron_positions[i,:] = random_position_sphere_optimized(R)
        else:
            neutron_positions[i,:] = random_position_cylinder(R, H)

    absorbed_positions = []
    is_active = np.ones(N0, dtype=bool)
    leak_count = 0

    scatter_r_vals = []
    absorb_r_vals = []
    fission_r_vals = []

    active_indices = np.where(is_active)[0]
    while len(active_indices) > 0:
        for idx in active_indices:
            pos = neutron_positions[idx]
            d_coll = distance_to_collision(mixture)
            dirn = random_scatter_direction()
            new_pos = pos + d_coll*dirn

            # check leak
            if geometry=='sphere':
                if not point_is_inside_sphere(new_pos, R):
                    is_active[idx] = False
                    leak_count += 1
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R, H):
                    is_active[idx] = False
                    leak_count += 1
                    continue

            # collision occurs
            # decide scatter vs absorption
            rand_event = rand.rand()
            if rand_event < P_scat:
                # scatter
                if geometry=='sphere':
                    scatter_r_vals.append(radial_distance_sphere(new_pos))
                else:
                    scatter_r_vals.append(radial_distance_cylinder(new_pos))
                neutron_positions[idx] = new_pos  # stays active
            else:
                # absorbed
                if geometry=='sphere':
                    r_abs = radial_distance_sphere(new_pos)
                else:
                    r_abs = radial_distance_cylinder(new_pos)
                absorb_r_vals.append(r_abs)

                # check if fission
                if rand.rand() < P_fis:
                    fission_r_vals.append(r_abs)

                absorbed_positions.append(new_pos)
                is_active[idx] = False

        active_indices = np.where(is_active)[0]

    absorbed_count = len(absorbed_positions)

    # radial binning
    bin_edges = np.linspace(0, R, bins+1)
    scatter_hist, _ = np.histogram(scatter_r_vals, bins=bin_edges)
    absorb_hist, _  = np.histogram(absorb_r_vals,  bins=bin_edges)
    fission_hist, _ = np.histogram(fission_r_vals, bins=bin_edges)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if geometry=='sphere':
        # volume of shell [r_n, r_{n+1}] = 4π/3 (r_{n+1}^3 - r_n^3)
        shell_volumes = (4./3.)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
    else:
        # cylinder ring volumes
        ring_areas = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
        shell_volumes = ring_areas * H

    scatter_density = scatter_hist / shell_volumes
    absorb_density  = absorb_hist  / shell_volumes
    fission_density = fission_hist / shell_volumes

    results = {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': absorbed_count,
        'leak_count': leak_count,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'scatter_density': scatter_density,
        'absorb_density': absorb_density,
        'fission_density': fission_density
    }
    return results


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
    """
    k = (#absorbed / #initial) * fission_probability * avg_neutrons_per_fission
    Uncertainty from binomial stdev with p = fission_probability among the 'absorbed' events.
    """
    if num_initial <= 0:
        return (0.0, 0.0)
    k = (num_absorbed / num_initial) * fission_probability * avg_neutrons_per_fission

    # binomial stdev in # of fissions among absorbed => sqrt(N*p(1-p))
    N = num_absorbed
    p = fission_probability
    stdev_fissions = 0.0
    if N>0 and p>=0 and p<=1:
        stdev_fissions = np.sqrt(N * p * (1-p))
    # fraction uncertainty => ( stdev_fissions / N ) => multiply by (avg_nu / num_initial)
    if N>0:
        dk = (avg_neutrons_per_fission / float(num_initial)) * stdev_fissions
    else:
        dk = 0.0
    return (k, dk)


def simulate_generation(mixture, geometry, size, initial_positions):
    """
    A simpler random-walk for subsequent generations.
    We start with 'initial_positions' of absorbed neutrons from previous gen.
    Track until absorbed or leaked.  We do not produce new fission neutrons
    mid-simulation; that is accounted for only in the generational k-factor formula.
    """
    N = initial_positions.shape[0]
    is_active = np.ones(N, dtype=bool)
    leak_count = 0
    absorbed_positions = []

    if geometry=='sphere':
        R = float(size)
    else:
        R = float(size[0])
        H = float(size[1])

    P_scat = mixture.scattering_probability
    P_fis  = mixture.fission_probability  # not used directly except to define what's "fission" among absorption?

    while True:
        active_indices = np.where(is_active)[0]
        if len(active_indices)==0:
            break

        for idx in active_indices:
            pos = initial_positions[idx]
            d_coll = distance_to_collision(mixture)
            dirn = random_scatter_direction()
            new_pos = pos + d_coll*dirn

            # leak?
            if geometry=='sphere':
                if not point_is_inside_sphere(new_pos, R):
                    is_active[idx] = False
                    leak_count += 1
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R, H):
                    is_active[idx] = False
                    leak_count += 1
                    continue

            # collision => scatter or absorb
            rand_event = rand.rand()
            if rand_event < P_scat:
                initial_positions[idx] = new_pos  # scattered => remain active
            else:
                # absorbed => done
                absorbed_positions.append(new_pos)
                is_active[idx] = False

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }


def main():
    print("=== Nuclear Reactor MC Simulation ===\n")

    # Geometry
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'."
    ).lower()

    # Size
    if geometry=='sphere':
        size_prompt = "Enter sphere radius (m): "
    else:
        size_prompt = "Enter cylinder [radius,height] in m, e.g. [1.0,2.0]: "
    size_str = get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry),
        f"Invalid {geometry} size format."
    )
    if geometry=='sphere':
        reactor_size = float(size_str)
    else:
        s = size_str.strip()[1:-1].split(',')
        radius = float(s[0])
        height = float(s[1])
        reactor_size = [radius, height]

    # Moderator
    mod_str = get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O, D2O, or Graphite"
    )
    mod_str_lower = mod_str.lower()
    if mod_str_lower=='h2o':
        user_mod_str = 'water'
        mod_key = 'h2o'
    elif mod_str_lower=='d2o':
        user_mod_str = 'heavywater'
        mod_key = 'd2o'
    else:
        user_mod_str = 'graphite'
        mod_key = 'graphite'

    # Dataset
    dataset = get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be 'Lilley' or 'Wikipedia'."
    ).lower()

    if dataset=='wikipedia':
        neutron_model = get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be 'Thermal' or 'Fast'"
        ).lower()
        (u235_mat, u238_mat) = get_wikipedia_materials(neutron_model)
    else:
        u235_mat = LILLEY_U235
        u238_mat = LILLEY_U238

    # U-235 fraction
    u235_percent = float(get_valid_input(
        "U-235 concentration in fuel (0-100): ",
        lambda x: validate_float(x, 0, 100),
        "Must be a number between 0 and 100"
    ))/100.0

    # R_mtf >= 0
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio, R_mtf (>=0): ",
        lambda x: validate_float(x, 0.0),
        "Must be a number >= 0"
    ))

    # Build mixture
    moderator_obj = MODERATORS_DICT[mod_key]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235_mat, u238_mat)

    # (2) Print out Σ_s, Σ_a, Σ_f up to 8 decimal places in SI units (m^-1)
    print("\nMacroscopic Cross Sections (SI units, 1/m):")
    print(f"  Σ_s (scattering) = {mixture.macroscopic_scattering:.8e}")
    print(f"  Σ_a (absorption) = {mixture.macroscopic_absorption:.8e}")
    print(f"  Σ_f (fission)    = {mixture.macroscopic_fission:.8e}")

    # Number of neutrons for 1st generation
    N0_str = get_valid_input(
        "Enter the number of neutrons (positive integer) for the 1st generation: ",
        validate_positive_integer,
        "Must be a positive integer!"
    )
    N0 = int(N0_str)

    # Create an empty array for positions (size N0,3)
    initial_positions = np.zeros((N0,3))

    # Simulate 1st generation
    results_1st = simulate_first_generation(
        mixture=mixture,
        geometry=geometry,
        size=reactor_size,
        N0=N0,
        average_neutrons_per_fission=2.42,
        bins=20
    )
    absorbed_count_1st = results_1st['absorbed_count']
    leak_count_1st = results_1st['leak_count']

    # compute k1
    (k1, dk1) = compute_k_factor_and_uncertainty(
        num_absorbed=absorbed_count_1st,
        num_initial=N0,
        fission_probability=mixture.fission_probability,
        avg_neutrons_per_fission=2.42
    )

    # plot the collisions histogram for 1st generation
    plot_first_generation_hist(results_1st, geometry, N0)

    # Display scattering, absorption, fission probabilities (6 dp, as before)
    print("\n=== Results for 1st Generation ===")
    print(f"Scattering Probability = {mixture.scattering_probability:.6f}")
    print(f"Absorption Probability = {mixture.absorption_probability:.6f}")
    print(f"Fission Probability    = {mixture.fission_probability:.6f}")
    print(f"Number absorbed       = {absorbed_count_1st}")
    print(f"Number leaked         = {leak_count_1st}")
    print(f"k1 = {k1:.6f} ± {dk1:.6f}")

    # store k in a list
    k_values = []
    k_values.append((k1, dk1))

    # ask how many generations
    S_str = get_valid_input(
        "How many generations of simulation do you want to run? ",
        validate_positive_integer,
        "Must be a positive integer."
    )
    S = int(S_str)

    # next generation initial positions = absorbed locations from 1st gen
    prev_absorbed_positions = results_1st['absorbed_positions']

    # run generation 2..S
    for gen_index in range(2, S+1):
        Nprev = prev_absorbed_positions.shape[0]
        if Nprev==0:
            print(f"\nGeneration {gen_index}: No absorbed neutrons => no new neutrons. Stopping.")
            # store a zero k
            k_values.append((0.0, 0.0))
            break

        # run a new generation
        results_this_gen = simulate_generation(
            mixture=mixture,
            geometry=geometry,
            size=reactor_size,
            initial_positions=prev_absorbed_positions
        )
        num_abs = results_this_gen['absorbed_count']
        num_leak = results_this_gen['leak_count']
        prev_absorbed_positions = results_this_gen['absorbed_positions']

        # compute k
        (kgen, dkgen) = compute_k_factor_and_uncertainty(
            num_absorbed=num_abs,
            num_initial=Nprev,
            fission_probability=mixture.fission_probability,
            avg_neutrons_per_fission=2.42
        )
        k_values.append((kgen, dkgen))
        print(f"\nGeneration {gen_index}: Absorbed={num_abs}, Leaked={num_leak},  k={kgen:.6f} ± {dkgen:.6f}")

    # plot k vs generation
    gens_list = np.arange(1, len(k_values)+1)
    k_arr = np.array([kv[0] for kv in k_values])
    dk_arr = np.array([kv[1] for kv in k_values])

    plt.figure()
    plt.errorbar(gens_list, k_arr, yerr=dk_arr, fmt='o-', label='k-factor with uncertainty')
    plt.axhline(y=1.0, color='r', linestyle='--', label='k = 1.0')
    plt.xlabel("Generation #")
    plt.ylabel("k-factor")
    plt.title(f"k-Factor Evolution, Starting with N0 = {N0}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # if S>=20, average k from gen=11..20
    if len(k_values)>=20:
        k_sub = k_arr[10:20]   # 11th..20th
        dk_sub = dk_arr[10:20]
        mask = (dk_sub>1e-12)
        if not np.any(mask):
            print("Cannot compute weighted average k for 11th..20th: all uncertainties zero.")
        else:
            w = 1.0/(dk_sub[mask]**2)
            k_avg = np.sum(k_sub[mask]*w)/np.sum(w)
            dk_avg = np.sqrt(1.0 / np.sum(w))
            print(f"\nAverage k between 11th and 20th generation = {k_avg:.6f} ± {dk_avg:.6f}")

    print("\nSimulation completed. Exiting.")


if __name__ == "__main__":
    main()
