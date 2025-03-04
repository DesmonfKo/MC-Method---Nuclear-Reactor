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


### By G
NA = 6.022e23  # Avogadro's number

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
def main():
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
        "Must be H2O, D2O, or Graphite"
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
    else:  # Lilley's data
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

if __name__ == "__main__":
    main()


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

