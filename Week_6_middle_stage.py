## New Attempt: O##
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
    We will allow user to type H2O, D2O, or Graphite in any case (upper/lower),
    but we rename them in code for clarity: H2O->'water', D2O->'heavywater'.
    We'll just check membership ignoring case.
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
# 2. Random Position Generators (Task #24)
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
    Simple container for nuclear data for one isotope (or compound).
    All cross sections in barns are stored but also converted to SI (m^2).
    """
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name = name
        self.mat_type = mat_type  # 'fuel' or 'moderator'
        # Convert from g/cc to kg/m^3 for density
        self.density = density * 1e3
        # Convert from g/mol to kg/mol for molar mass
        self.molar_mass = molar_mass * 1e-3
        
        self.sigma_s_b = sigma_s_b  # scattering (barn)
        self.sigma_a_b = sigma_a_b  # absorption (barn)
        self.sigma_f_b = sigma_f_b  # fission (barn)
        
        # Convert barns to m^2
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
# Default Materials Database
###############################
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
# Wikipedia Materials (Example)
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
# 3. ReactorMixture with Revised Σ's
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
        R_mtf         : ratio of (moderator atoms) to (total fuel atoms)
        """
        self.fraction_U235 = fraction_U235
        self.moderator = moderator
        self.R_mtf = R_mtf
        self.u235 = u235_material
        self.u238 = u238_material

        # Precompute number densities:
        # a_U235 = fraction_U235
        # a_U238 = (1 - fraction_U235)
        # a_mod = R_mtf   (this means "for every 1 total fuel atom, we have R_mtf moderator atoms")
        self.N_u235 = self.fraction_U235 * self.u235.number_density
        self.N_u238 = (1 - self.fraction_U235) * self.u238.number_density

        # total_fuel_atoms_per_unit_volume = N_u235 + N_u238
        # N_mod = R_mtf * (N_u235 + N_u238)
        self.N_mod = self.R_mtf * (self.N_u235 + self.N_u238)

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
        P_fission = Σ_f,fuel / Σ_a,total.
        If Σ_a = 0, return 0.
        """
        sigma_f = self.macroscopic_fission
        sigma_a = self.macroscopic_absorption
        if sigma_a > 0.0:
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
        if denom <= 0.0:
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
        if denom <= 0.0:
            return 0.0
        return sigma_a / denom

#####################################
# PATCH 3 of N
#####################################

def distance_to_collision(mixture):
    """
    Sample a random distance to collision with mean free path = 1 / (Σ_s + Σ_a).
    We assume an exponential distribution:
        d = -ln( rand() ) / (Σ_s + Σ_a).
    """
    sigma_s = mixture.macroscopic_scattering
    sigma_a = mixture.macroscopic_absorption
    sigma_tot = sigma_s + sigma_a
    if sigma_tot <= 0.0:
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
    Cylinder axis is along z direction, from z=-height/2 to z=height/2.
    """
    x, y, z = point
    r2 = x*x + y*y
    if r2 <= radius**2 and abs(z) <= (height/2):
        return True
    return False


def radial_distance_sphere(point):
    """
    Return radius from center for a sphere.
    """
    return np.sqrt(point[0]**2 + point[1]**2 + point[2]**2)


def radial_distance_cylinder(point):
    """
    Return radial distance from cylinder axis (the z-axis).
    """
    x, y, z = point
    return np.sqrt(x*x + y*y)


def random_scatter_direction():
    """
    Generate a random isotropic scattering direction in 3D.
    We'll do it by random angles in spherical coords.
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
    We track collisions (scatter, absorb, 'fission') until each neutron
    either gets absorbed or leaks out.

    Returns
    -------
    results_dict : {
       'absorbed_positions': list of 3D coords where absorption occurred,
       'absorbed_count': int,
       'leak_count': int,
       'collision_r_bins_scatter': array of length=bins (volume-normalized),
       'collision_r_bins_absorb':  array of length=bins (volume-normalized),
       'collision_r_bins_fission': array of length=bins (volume-normalized),
       'bin_edges':                array of bin edges for radius,
       ...
    }
    """
    # Pull probabilities from the mixture
    P_scat = mixture.scattering_probability
    P_abs = mixture.absorption_probability
    P_fis = mixture.fission_probability  # fraction of the absorption that leads to fission = Σ_f / Σ_a

    # The total radius or dimensions for geometry
    if geometry == 'sphere':
        R = float(size)
    else:  # cylinder
        R = float(size[0])  # radius
        H = float(size[1])  # height

    # Arrays to hold neutron positions:
    neutron_positions = np.zeros((N0, 3))
    
    # Initialize them randomly inside the geometry
    for i in range(N0):
        if geometry == 'sphere':
            neutron_positions[i,:] = random_position_sphere_optimized(R)
        else:
            neutron_positions[i,:] = random_position_cylinder(R, H)

    absorbed_positions = []
    is_active = np.ones(N0, dtype=bool)  # track which neutrons are still active
    leak_count = 0

    # For histogram data: we want to store collision radius + event type
    # We'll accumulate in arrays, then do radial binning at the end
    scatter_r_vals = []
    absorb_r_vals = []
    fission_r_vals = []

    # While loop approach:
    active_indices = np.where(is_active)[0]
    while len(active_indices) > 0:
        # For each active neutron:
        for idx in active_indices:
            pos = neutron_positions[idx]
            # 1) Sample a distance to collision
            d_coll = distance_to_collision(mixture)
            # 2) Pick a random direction
            dirn = random_scatter_direction()
            # 3) Move the neutron
            new_pos = pos + d_coll*dirn

            # 4) Check if it has leaked
            if geometry == 'sphere':
                if not point_is_inside_sphere(new_pos, R):
                    # leaked
                    is_active[idx] = False
                    leak_count += 1
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R, H):
                    # leaked
                    is_active[idx] = False
                    leak_count += 1
                    continue
            
            # 5) Otherwise, we have a collision. Determine which event occurs.
            # Probability scattering vs absorption.
            # If we pick absorption, we further see if it is fission or normal capture
            rand_event = rand.rand()
            sigma_s = mixture.macroscopic_scattering
            sigma_a = mixture.macroscopic_absorption
            if (sigma_s + sigma_a) <= 0.0:
                # No collisions possible => effectively neutron won't do anything
                # Mark it leaked artificially
                is_active[idx] = False
                leak_count += 1
                continue

            # We can do:
            #   if rand_event < P_scat => scatter
            #   else => absorption => check if fission or not
            if rand_event < P_scat:
                # scattered
                # record collision position
                if geometry=='sphere':
                    scatter_r_vals.append(radial_distance_sphere(new_pos))
                else:
                    scatter_r_vals.append(radial_distance_cylinder(new_pos))
                # neutron continues with a new direction from here
                # so we simply update pos but remain active
                neutron_positions[idx] = new_pos
            else:
                # absorbed
                if geometry=='sphere':
                    r_abs = radial_distance_sphere(new_pos)
                    absorb_r_vals.append(r_abs)
                else:
                    r_abs = radial_distance_cylinder(new_pos)
                    absorb_r_vals.append(r_abs)

                # Check fission within absorption
                # Probability of fission given absorption is P_fis = Sigma_f_fuel / Sigma_a_total
                # We'll store the collision radius in fission_r_vals if it is a fission event
                if rand.rand() < mixture.fission_probability:
                    fission_r_vals.append(r_abs)

                # This neutron is done
                absorbed_positions.append(new_pos)
                is_active[idx] = False
        
        active_indices = np.where(is_active)[0]

    absorbed_count = len(absorbed_positions)

    # Now we do radial binning for scatter_r_vals, absorb_r_vals, fission_r_vals
    # Build bin edges from 0 to R in 'bins' segments for sphere or cylinder.
    bin_edges = np.linspace(0, R, bins+1)  # e.g. 21 edges => 20 bins
    scatter_hist, _ = np.histogram(scatter_r_vals, bins=bin_edges)
    absorb_hist, _  = np.histogram(absorb_r_vals,  bins=bin_edges)
    fission_hist, _ = np.histogram(fission_r_vals, bins=bin_edges)

    # We want volume-normalized collisions. For sphere: volume of shell [r, r+dr] = 4π r^2 dr
    # For cylinder: volume of annular region with thickness dr, cross-section = (height)*2πr
    # We'll do the midpoint approximation.

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    volumes = np.zeros_like(bin_centers)
    if geometry == 'sphere':
        # volume of shell between edges n and n+1: 4π/3 (r_{n+1}^3 - r_n^3)
        shell_volumes = (4./3.)*np.pi*(bin_edges[1:]**3 - bin_edges[:-1]**3)
        volumes = shell_volumes
    else:
        # cylinder: volume of ring = pi*(r2_{n+1}^2 - r2_n^2)*height
        # more precisely 2π * r_avg * dr * height if we do midpoint
        # We'll do difference of areas times height
        ring_areas = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
        shell_volumes = ring_areas * H
        volumes = shell_volumes

    # Convert hist counts to collisions / m^3
    # We'll do e.g. scatter_density[n] = scatter_hist[n] / volumes[n]
    scatter_density = scatter_hist / volumes
    absorb_density  = absorb_hist  / volumes
    fission_density = fission_hist / volumes

    results_dict = {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': absorbed_count,
        'leak_count': leak_count,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'scatter_density': scatter_density,
        'absorb_density': absorb_density,
        'fission_density': fission_density
    }
    return results_dict


def plot_first_generation_hist(results_dict, geometry, N0):
    """
    Make the requested histogram plot of collisions vs radial distance
    (volume-normalized).  We show scattering, absorption, and fission lines.
    """
    r = results_dict['bin_centers']
    scat_dens = results_dict['scatter_density']
    abs_dens  = results_dict['absorb_density']
    fis_dens  = results_dict['fission_density']

    plt.figure(figsize=(7,5))
    plt.plot(r, scat_dens, '-o', color='blue',   label='Scattered')
    plt.plot(r, abs_dens,  '-o', color='orange', label='Absorbed')
    plt.plot(r, fis_dens,  '-o', color='green',  label='Fission')
    plt.xlabel(f"Radial distance r from reactor center (m)" 
               if geometry=='sphere' else
               "Radial distance r from central axis of cylinder (m)")
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
    As described in Task #12 & #13:

    k = (num_absorbed / num_initial) * fission_probability * avg_neutrons_per_fission

    We also compute an uncertainty for k using the binomial distribution
    with success probability = fission_probability.
    Let p = fission_probability, N = num_absorbed.

    The fraction of successes is the fraction that 'fission' among 'absorbed',
    but we are only using p, so the standard deviation in the # of fission events
    among the absorbed is sqrt(N * p * (1-p)).
    Then the uncertainty in k is:
        Δk = avg_neutrons_per_fission * (1 / num_initial) * sqrt(N * p * (1-p))

    Because (num_absorbed / num_initial) is a factor out front as well, but note
    we treat (num_absorbed/num_initial) as if it's known exactly.  
    (In principle one might also account for uncertainty in num_absorbed, but
     the instructions only mention the binomial part.)

    Returns (k, dk).
    """
    if num_initial <= 0:
        return (0.0, 0.0)
    k = (num_absorbed / num_initial) * fission_probability * avg_neutrons_per_fission

    # binomial stdev in the count of fission among the absorbed => sqrt(N * p * (1-p))
    N = num_absorbed
    p = fission_probability
    stdev_fissions = np.sqrt(N * p * (1-p)) if (N>0 and 0<=p<=1) else 0.0
    # so the stdev in the fraction of fission events among absorbed is stdev_fissions / N
    # multiply that fraction by (avg_neutrons_per_fission / num_initial)
    if N>0:
        dk = (avg_neutrons_per_fission / float(num_initial)) * stdev_fissions
    else:
        dk = 0.0
    return (k, dk)


def main():
    print("=== Nuclear Reactor MC Simulation ===")
    # (3) Remove the infinite size option => skip reactor_type input

    # (2) We keep geometry prompt:
    geometry = get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be 'sphere' or 'cylinder'."
    ).lower()

    # (2) We keep the user prompt for size:
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
        # parse [radius,height]
        s = size_str.strip()[1:-1].split(',')
        radius = float(s[0])
        height = float(s[1])
        reactor_size = [radius, height]

    # (23) new validation for moderator => rename H2O->'water', D2O->'heavywater'
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

    # dataset
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
        # use Lilley's default
        u235_mat = LILLEY_U235
        u238_mat = LILLEY_U238

    # get U-235 fraction
    u235_percent = float(get_valid_input(
        "U-235 concentration in fuel (0-100): ",
        lambda x: validate_float(x, 0, 100),
        "Must be a number between 0 and 100"
    ))/100.0

    # get R_mtf
    R_mtf = float(get_valid_input(
        "Moderator-to-fuel ratio, R_mtf (>0): ",
        lambda x: validate_float(x, 0.0),
        "Must be a positive number"
    ))
    if R_mtf<=0:
        print("R_mtf must be > 0; exiting.")
        return

    # Build mixture
    moderator_obj = MODERATORS_DICT[mod_key]
    mixture = ReactorMixture(u235_percent, moderator_obj, R_mtf, u235_mat, u238_mat)

    # (4) user input for number of neutrons N0
    N0_str = get_valid_input(
        "Enter the number of neutrons (positive integer) for the 1st generation: ",
        validate_positive_integer,
        "Must be a positive integer!"
    )
    N0 = int(N0_str)

    # (5) create an empty array or list of same size as #neutrons (to store positions if needed)
    # We'll do it as a numpy array of shape (N0, 3).  We also do the actual random fill in the sim.
    initial_positions = np.zeros((N0,3))

    # (6) - (9) We'll do everything inside a new function for the 1st generation:
    results_1st = simulate_first_generation(
        mixture=mixture,
        geometry=geometry,
        size=reactor_size,
        N0=N0,
        average_neutrons_per_fission=2.42, # we do not alter it in code
        bins=20
    )

    # (12) compute k1
    absorbed_count_1st = results_1st['absorbed_count']
    leak_count_1st = results_1st['leak_count']

    # k1 = (#absorbed / #initial) * fission_probability * avg_number_per_fission
    # uncertain with binomial approach
    (k1, dk1) = compute_k_factor_and_uncertainty(
        num_absorbed=absorbed_count_1st,
        num_initial=N0,
        fission_probability=mixture.fission_probability,
        avg_neutrons_per_fission=2.42
    )

    # (14) plot the histogram for 1st generation collisions
    plot_first_generation_hist(results_1st, geometry, N0)

    # (15) display scattering, absorption, fission probabilities up to 6dp
    print("\n=== Results for 1st Generation ===")
    print(f"Scattering Probability = {mixture.scattering_probability:.6f}")
    print(f"Absorption Probability = {mixture.absorption_probability:.6f}")
    print(f"Fission Probability    = {mixture.fission_probability:.6f}")

    # (16) total number neutrons absorbed, and # leak
    print(f"Number absorbed = {absorbed_count_1st}")
    print(f"Number leaked   = {leak_count_1st}")

    # (12)-(13) print k1 and its uncertainty (6 dp)
    print(f"k1 = {k1:.6f} ± {dk1:.6f}")

    # (17) store k factor and uncertain value in a list
    k_values = []
    k_values.append( (k1, dk1) )

    # (18) user prompt for how many generations S
    S_str = get_valid_input(
        "How many generations of simulation do you want to run? ",
        validate_positive_integer,
        "Must be a positive integer."
    )
    S = int(S_str)

    # We already have generation #1 done.  So we do up to generation #2..S in a loop.
    # (19) create new array of size = number of neutrons absorbed in prev gen,
    # storing location of absorption points for next generation's initial positions.
    prev_absorbed_positions = results_1st['absorbed_positions']

    for gen_index in range(2, S+1):
        # The new generation starts with Nprev = number absorbed in previous gen
        Nprev = prev_absorbed_positions.shape[0]
        if Nprev==0:
            print(f"\nGeneration {gen_index}: No absorbed neutrons from previous generation => no new neutrons to track.  Stopping.")
            # fill k with 0.0
            k_values.append( (0.0, 0.0) )
            break

        # We create a function that does basically the same as simulate_first_generation,
        # but doesn't produce the radial histogram or the big dictionary, unless you want.
        # We'll just replicate code quickly here for clarity.

        # We re-use the same random-walk logic, but the initial positions are the absorption
        # points from the previous generation. We do not physically spawn 2.42 neutrons per fission,
        # as requested, but we do track scattering vs absorption vs leak the same way.

        # We'll do a simpler function inline here:
        # (20) run the simulation

        results_this_gen = simulate_generation(
            mixture=mixture,
            geometry=geometry,
            size=reactor_size,
            initial_positions=prev_absorbed_positions
        )

        num_abs = results_this_gen['absorbed_count']
        num_leak = results_this_gen['leak_count']

        # store for next iteration
        prev_absorbed_positions = results_this_gen['absorbed_positions']

        # (21) compute k for this generation:
        (kgen, dkgen) = compute_k_factor_and_uncertainty(
            num_absorbed=num_abs,
            num_initial=Nprev,
            fission_probability=mixture.fission_probability,
            avg_neutrons_per_fission=2.42
        )
        k_values.append( (kgen, dkgen) )
        print(f"\nGeneration {gen_index}: Absorbed={num_abs}, Leaked={num_leak},  k={kgen:.6f} ± {dkgen:.6f}")

    # (21) We have k-values for all done generations => we plot them with error bars
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

    # (22) If S>=20, average k from 11th to 20th generation
    if len(k_values) >= 20:
        # we have at least 20 generations
        # compute average k from gen=11..20
        # Weighted average = sum( k_i / dk_i^2 ) / sum( 1/dk_i^2 ), if dk_i>0
        # But the user only asked for "weighted uncertainty." We'll do a standard formula:
        k_sub = k_arr[10:20]   # 11th..20th => indices 10..19
        dk_sub = dk_arr[10:20]
        # We only consider entries with nonzero uncertainties
        mask = (dk_sub>1e-12)
        if not np.any(mask):
            # fallback if all zero
            print("Unable to compute average k between 11th and 20th: all uncertainties zero.")
        else:
            w = 1.0/(dk_sub[mask]**2)
            k_avg = np.sum(k_sub[mask]*w)/np.sum(w)
            # Weighted standard error:
            dk_avg = np.sqrt( 1.0 / np.sum(w) )
            print(f"\nAverage k between 11th and 20th generation = {k_avg:.6f} ± {dk_avg:.6f}")
    print("\nSimulation completed. Exiting.")


def simulate_generation(mixture, geometry, size, initial_positions):
    """
    A simpler version of the random-walk for subsequent generations.
    The initial neutron positions are given by 'initial_positions'.
    We track collisions until each neutron is absorbed or leaked.

    Returns a dictionary:
        {
          'absorbed_positions': np.array of shape(Nabs,3),
          'absorbed_count': int,
          'leak_count': int
        }
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
    # no need for binning or advanced stats here
    while True:
        active_indices = np.where(is_active)[0]
        if len(active_indices)==0:
            break

        for idx in active_indices:
            pos = initial_positions[idx]
            d_coll = distance_to_collision(mixture)
            dirn = random_scatter_direction()
            new_pos = pos + d_coll*dirn

            # check leak
            if geometry=='sphere':
                if not point_is_inside_sphere(new_pos, R):
                    is_active[idx] = False
                    leak_count+=1
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R, H):
                    is_active[idx] = False
                    leak_count+=1
                    continue

            # collision
            rand_event = rand.rand()
            if rand_event < P_scat:
                # scatter => remain active
                initial_positions[idx] = new_pos
            else:
                # absorbed => done
                absorbed_positions.append(new_pos)
                is_active[idx] = False

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }


if __name__=="__main__":
    main()
