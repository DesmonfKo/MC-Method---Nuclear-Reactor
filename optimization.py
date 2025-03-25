############################################
# PATCH 1 of N
############################################

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
    If min_val is not None and value < min_val => fail.
    If max_val is not None and value > max_val => fail.
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
    For 'sphere': single float radius>0.
    For 'cylinder': [R,H], each>0.
    """
    try:
        if geometry == 'sphere':
            val = float(input_str)
            return (val > 0)
        elif geometry == 'cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                parts = list(map(float, input_str[1:-1].split(',')))
                if len(parts)==2 and all(p>0 for p in parts):
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
    We allow 'H2O','D2O','graphite' ignoring case.
    """
    val = input_str.strip().lower()
    return val in ['h2o','d2o','graphite']


def validate_positive_integer(input_str):
    """
    For user input that must be a positive integer (neutron count, generations).
    """
    try:
        val = int(input_str)
        return (val>0)
    except ValueError:
        return False

############################################
# PATCH 2 of N
############################################

class ReactorMaterial:
    """
    Holds nuclear data for one isotope or compound.
    density_gcc: g/cc
    molar_mass_gmol: g/mol
    cross sections in barns => also in SI
    """
    def __init__(self, name, mat_type, density, molar_mass, 
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name = name
        self.mat_type = mat_type
        
        # Original user units
        self.density_gcc = density
        self.molar_mass_gmol = molar_mass
        
        # Also SI conversions
        self.density = density*1e3       # kg/m^3
        self.molar_mass = molar_mass*1e-3  # kg/mol

        # barns => m^2
        self.sigma_s_b = sigma_s_b
        self.sigma_a_b = sigma_a_b
        self.sigma_f_b = sigma_f_b
        self.sigma_s = sigma_s_b*1e-28
        self.sigma_a = sigma_a_b*1e-28
        self.sigma_f = sigma_f_b*1e-28
        
        self.nu = nu
        self.xi = xi

    @property
    def number_density(self):
        """
        number_density = (density_kg_m3 / molar_mass_kg_mol)*NA
        """
        return (self.density / self.molar_mass)*NA


# Lilley's default
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


def get_wikipedia_materials(neutron_model='thermal'):
    """
    Return (u235, u238) for either 'thermal' or 'fast' cross-sections.
    """
    if neutron_model=='thermal':
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
    else: # fast
        u235 = ReactorMaterial(
            name="U235", mat_type='fuel',
            density=18.7, molar_mass=235,
            sigma_s_b=4, sigma_a_b=1.09, sigma_f_b=1,
            nu=2.42, xi=0
        )
        u238 = ReactorMaterial(
            name="U238", mat_type='fuel',
            density=18.9, molar_mass=238,
            sigma_s_b=5, sigma_a_b=0.37, sigma_f_b=0.3,
            nu=0, xi=0
        )
    return u235, u238


class ReactorMixture:
    """
    fraction_U235 + fraction_U238=1,
    R_mtf => ratio of moderator atoms to total fuel atoms
    Macroscopic cross sections computed from formula with a "conv factor"
    B = aU235 + aU238 + R_mtf
    We do a 'homogeneous mixture' => define an 'effective density' for mass calc.
    """
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
        return sigma_f / sigma_a if sigma_a>1e-30 else 0.0

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

    @property
    def effective_density_gcc(self):
        """
        A 'homogeneous mixture' density in g/cc
        eff_density = [aU235*density(U235) + aU238*density(U238) + Rmtf*density(moderator)] / B
        """
        aU235 = self.fraction_U235
        aU238 = 1.0 - aU235
        B = aU235 + aU238 + self.R_mtf
        top = aU235*self.u235.density_gcc + aU238*self.u238.density_gcc + self.R_mtf*self.moderator.density_gcc
        return top/B if B>1e-30 else 0.0

    def compute_total_mass_kg(self, geometry, size):
        """
        volume * (effective_density_kg_m^3).
        """
        if geometry=='sphere':
            R = float(size)
            volume_m3 = (4./3.)*np.pi*(R**3)
        else:
            R, H = float(size[0]), float(size[1])
            volume_m3 = np.pi*(R**2)*H
        dens_kg_m3 = self.effective_density_gcc*1e3
        mass_kg = volume_m3*dens_kg_m3
        return mass_kg

############################################
# PATCH 3 of N
############################################

def distance_to_collision(mixture):
    sigma_s = mixture.macroscopic_scattering
    sigma_a = mixture.macroscopic_absorption
    sigma_tot= sigma_s + sigma_a
    if sigma_tot<=1e-30:
        return 1e10
    return -np.log(rand.rand())/sigma_tot

def point_is_inside_sphere(p, R):
    return (p[0]**2 + p[1]**2 + p[2]**2)<=R**2

def point_is_inside_cylinder(p, R, H):
    x,y,z= p
    return (x*x + y*y <= R*R) and (abs(z)<=H/2)

def random_scatter_direction():
    costh = 2.*rand.rand()-1
    phi   = 2.*np.pi*rand.rand()
    sinth = np.sqrt(1.- costh*costh)
    return np.array([
        sinth*np.cos(phi),
        sinth*np.sin(phi),
        costh
    ])

def random_position_sphere_optimized(R=1.0):
    """
    uniform in sphere
    """
    v=rand.randn(3)
    v/= np.linalg.norm(v)
    r_ = R*(rand.rand()**(1./3.))
    return r_*v

def random_position_cylinder(R=1.0,H=1.0):
    theta= rand.uniform(0,2*np.pi)
    r= np.sqrt(rand.rand())*R
    x= r*np.cos(theta)
    y= r*np.sin(theta)
    z= rand.uniform(-H/2, H/2)
    return np.array([x,y,z])

def simulate_first_generation(mixture, geometry, size, N0, bins=20):
    """
    random walk for N0 neutrons => get collision distribution
    """
    P_scat= mixture.scattering_probability
    P_fis = mixture.fission_probability

    if geometry=='sphere':
        R= float(size)
    else:
        R=float(size[0]); H=float(size[1])

    # init
    positions= np.zeros((N0,3))
    for i in range(N0):
        if geometry=='sphere':
            positions[i,:]= random_position_sphere_optimized(R)
        else:
            positions[i,:]= random_position_cylinder(R,H)

    scatter_r, absorb_r, fission_r= [],[],[]
    absorbed_positions= []
    is_active= np.ones(N0, dtype=bool)
    leak_count=0

    while True:
        active_indices= np.where(is_active)[0]
        if len(active_indices)==0:
            break
        for idx in active_indices:
            pos= positions[idx]
            d= distance_to_collision(mixture)
            dirn= random_scatter_direction()
            new_pos= pos + d*dirn
            # leak?
            if geometry=='sphere':
                if not point_is_inside_sphere(new_pos, R):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R,H):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            # collision
            if rand.rand()< P_scat:
                # scatter
                rcol= np.linalg.norm(new_pos) if geometry=='sphere' else np.sqrt(new_pos[0]**2+ new_pos[1]**2)
                scatter_r.append(rcol)
                positions[idx]= new_pos
            else:
                # absorb
                rcol= np.linalg.norm(new_pos) if geometry=='sphere' else np.sqrt(new_pos[0]**2+ new_pos[1]**2)
                absorb_r.append(rcol)
                # fission?
                if rand.rand()<P_fis:
                    fission_r.append(rcol)
                absorbed_positions.append(new_pos)
                is_active[idx]=False

    # bin
    if geometry=='sphere':
        bins_r= np.linspace(0,R,bins+1)
        s_hist,_= np.histogram(scatter_r, bins=bins_r)
        a_hist,_= np.histogram(absorb_r,  bins=bins_r)
        f_hist,_= np.histogram(fission_r, bins=bins_r)
        shell_vol= (4./3.)*np.pi*(bins_r[1:]**3 - bins_r[:-1]**3)
    else:
        R_= R; H_= H
        bins_r= np.linspace(0,R_, bins+1)
        s_hist,_= np.histogram(scatter_r,bins=bins_r)
        a_hist,_= np.histogram(absorb_r, bins=bins_r)
        f_hist,_= np.histogram(fission_r,bins=bins_r)
        ring_area= np.pi*(bins_r[1:]**2 - bins_r[:-1]**2)
        shell_vol= ring_area*H_

    s_dens= s_hist/shell_vol
    a_dens= a_hist/shell_vol
    f_dens= f_hist/shell_vol

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count,
        'bin_edges': bins_r,
        'scatter_density': s_dens,
        'absorb_density':  a_dens,
        'fission_density': f_dens
    }

def plot_first_generation_hist(results_dict, geometry, N0):
    s_d= results_dict['scatter_density']
    a_d= results_dict['absorb_density']
    f_d= results_dict['fission_density']
    edges= results_dict['bin_edges']
    centers= 0.5*(edges[:-1]+edges[1:])

    plt.figure()
    plt.plot(centers, s_d, '-o', label='Scatter', color='blue')
    plt.plot(centers, a_d, '-o', label='Absorb', color='orange')
    plt.plot(centers, f_d, '-o', label='Fission', color='green')
    plt.title(f"1st Generation Collisions, N0={N0}, {geometry}")
    plt.xlabel("Radial distance from center or axis (m)")
    plt.ylabel("Collisions / m^3 (approx)")
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################
# PATCH 4 of N
############################################

def compute_k_factor_and_uncertainty(n_abs, n_init, fission_prob, nu):
    if n_init<=0:
        return (0.0, 0.0)
    k= (n_abs/n_init)*fission_prob*nu
    if n_abs>0 and 0<= fission_prob <=1:
        stdev_f= np.sqrt(n_abs*fission_prob*(1-fission_prob))
        dk= (nu/n_init)*stdev_f
    else:
        dk=0.0
    return (k, dk)


def simulate_one_generation(mixture, geometry, size, init_positions):
    """
    single 'generation' random walk => #absorbed, #leaked
    """
    N= init_positions.shape[0]
    is_active= np.ones(N, dtype=bool)
    leak_count=0
    absorbed_positions=[]
    P_scat= mixture.scattering_probability
    while True:
        active_indices= np.where(is_active)[0]
        if len(active_indices)==0:
            break
        for idx in active_indices:
            pos= init_positions[idx]
            dist= distance_to_collision(mixture)
            dirn= random_scatter_direction()
            new_pos= pos + dist*dirn
            # leak?
            if geometry=='sphere':
                R=float(size)
                if not point_is_inside_sphere(new_pos, R):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            else:
                R_,H_= float(size[0]), float(size[1])
                if not point_is_inside_cylinder(new_pos,R_,H_):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            # collision
            if rand.rand()< P_scat:
                init_positions[idx]= new_pos
            else:
                # absorbed
                absorbed_positions.append(new_pos)
                is_active[idx]=False
    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }


def run_previous_generations(mixture, geometry, size, N0, maxgen=3):
    """
    Illustrative 'previous generational simulation' 
    We'll do up to 'maxgen' generations
    """
    # 1st gen
    init_pos= np.zeros((N0,3))
    res1= simulate_first_generation(mixture, geometry, size, N0)
    nab1= res1['absorbed_count']
    (k1,dk1)= compute_k_factor_and_uncertainty(nab1, N0, mixture.fission_probability, mixture.u235.nu)
    k_list=[(k1,dk1)]
    cur_init= nab1
    cur_positions= res1['absorbed_positions']
    for g in range(2, maxgen+1):
        if cur_init<=0:
            k_list.append((0.0,0.0))
            break
        single_res= simulate_one_generation(mixture, geometry, size, cur_positions)
        nab= single_res['absorbed_count']
        (k,dk)= compute_k_factor_and_uncertainty(nab, cur_init, mixture.fission_probability, mixture.u235.nu)
        k_list.append((k,dk))
        cur_init= nab
        cur_positions= single_res['absorbed_positions']
    return (k_list, res1)


#############################
# Param Sweep for radius & fraction & Rmtf
#############################
def run_param_sweep(mixture, geometry, R, H, fraction, Rmtf, N0=500, max_generations=15, early_stop_k=1.7):
    """
    We do a small param sweep around R +/- 0.1 or so,
    fraction +/- 0.1, Rmtf +/- 10,
    each in steps => you can define. Then we run up to 15 gens each param set.
    We'll also compute total mass for each param set.
    This is just an example; you can elaborate as needed.
    """
    Rvals= np.arange(R-0.1, R+0.101, 0.05)
    frac_vals= np.arange(fraction-0.1, fraction+0.101, 0.1)
    rmtf_vals= np.arange(Rmtf-10, Rmtf+10.1, 5)
    # clamp them
    Rvals= [rv for rv in Rvals if rv>0]
    frac_vals=[fv for fv in frac_vals if fv>=0 and fv<=1]
    rmtf_vals=[rm for rm in rmtf_vals if rm>=0]

    results_dict= {}
    for rv in Rvals:
        for fv in frac_vals:
            for rm in rmtf_vals:
                # build new mixture
                from copy import deepcopy
                u235_cp= deepcopy(mixture.u235)
                u238_cp= deepcopy(mixture.u238)
                mod_cp = deepcopy(mixture.moderator)
                temp_mix= ReactorMixture(fv, mod_cp, rm, u235_cp, u238_cp)
                # geometry size
                if geometry=='sphere':
                    sizeX= rv
                else:
                    sizeX= [rv, H]

                # run up to 15 generations or until k>1.7
                # We'll do a function for that:
                klist= run_up_to_15_gens(temp_mix, geometry, sizeX, N0, max_generations, early_stop_k)
                finalk= klist[-1][0]
                finaldk= klist[-1][1]
                # compute mass
                mass_kg= temp_mix.compute_total_mass_kg(geometry, sizeX)
                # store
                results_dict[(rv,fv,rm)] = {
                    'k_list': klist,
                    'final_k': finalk,
                    'final_dk': finaldk,
                    'mass_kg': mass_kg
                }
    return results_dict


def run_up_to_15_gens(mixture, geometry, size, N0, max_gen=15, kstop=1.7):
    """
    Each gen: #absorbed => k. If k>kstop => early break
    """
    from copy import deepcopy
    # 1st
    init_pos= np.zeros((N0,3))
    res= simulate_first_generation(mixture, geometry, size, N0)
    nab= res['absorbed_count']
    (k,dk)= compute_k_factor_and_uncertainty(nab, N0, mixture.fission_probability, mixture.u235.nu)
    k_list=[(k,dk)]
    if k>kstop:
        return k_list
    cur_init= nab
    cur_pos= res['absorbed_positions']
    for g in range(2, max_gen+1):
        if cur_init<=0:
            k_list.append((0.0,0.0))
            break
        one_res= simulate_one_generation(mixture, geometry, size, cur_pos)
        nab2= one_res['absorbed_count']
        (k2,dk2)= compute_k_factor_and_uncertainty(nab2, cur_init, mixture.fission_probability, mixture.u235.nu)
        k_list.append((k2,dk2))
        if k2>kstop:
            break
        cur_init= nab2
        cur_pos= one_res['absorbed_positions']
    return k_list


#################### MAIN  ####################

def main():
    print("=== Reactor MC Simulation ===")
    # 1) Basic user input for geometry, size
    geometry_str= get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be sphere or cylinder"
    ).lower()

    if geometry_str=='sphere':
        size_prompt="Enter sphere radius (m): "
    else:
        size_prompt="Enter cylinder [R,H] in m, e.g. [1.0,2.0]: "

    size_inp= get_valid_input(
        size_prompt,
        lambda x: validate_size(x, geometry_str),
        "Invalid geometry size format"
    )
    if geometry_str=='sphere':
        size_val= float(size_inp)
    else:
        parts= size_inp.strip()[1:-1].split(',')
        size_val= [float(parts[0]), float(parts[1])]

    # 2) dataset
    dset_str= get_valid_input(
        "Dataset [Lilley/Wikipedia]: ",
        validate_dataset,
        "Must be Lilley or Wikipedia"
    ).lower()
    # 3) neutron model if wikipedia
    if dset_str=='wikipedia':
        nm_str= get_valid_input(
            "Neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be thermal or fast"
        ).lower()
        u235_mat,u238_mat= get_wikipedia_materials(nm_str)
    else:
        u235_mat= LILLEY_U235
        u238_mat= LILLEY_U238

    # 4) fraction
    frac_str= get_valid_input(
        "U-235 fraction (0..100): ",
        lambda x: validate_float(x,0,100),
        "Must be 0..100"
    )
    frac_f= float(frac_str)/100.0
    # 5) R_mtf
    rmtf_str= get_valid_input(
        "Moderator-to-fuel ratio R_mtf (>=0): ",
        lambda x: validate_float(x,0.0),
        "Must be >=0"
    )
    Rmtf= float(rmtf_str)
    # 6) moderator
    mod_str= get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O, D2O, or Graphite"
    ).lower()
    mod_obj= MODERATORS_DICT[mod_str]
    # build mixture
    mixture= ReactorMixture(frac_f, mod_obj, Rmtf, u235_mat,u238_mat)

    # print cross sections
    print("\n=== Mixture Cross Sections (SI) ===")
    print(f" Σ_s={mixture.macroscopic_scattering:.5e}, Σ_a={mixture.macroscopic_absorption:.5e}, Σ_f={mixture.macroscopic_fission:.5e}")
    # compute mass
    mass_kg= mixture.compute_total_mass_kg(geometry_str, size_val)
    m_u235= frac_f*mass_kg
    print(f"Effective density = {mixture.effective_density_gcc:.3f} g/cc => total mass ~{mass_kg:.3f} kg, including {m_u235:.3f} kg of U-235.\n")

    # 7) #neutrons
    n0_str= get_valid_input(
        "Number of neutrons for 'previous generational' sim: ",
        validate_positive_integer,
        "Must be positive integer"
    )
    N0=int(n0_str)

    # run the previous multi-generation sim
    klist, first_res= run_previous_generations(mixture, geometry_str, size_val, N0, maxgen=5)
    print("\n=== Multi-Generation (previous) Results ===")
    for i,(kk,dk) in enumerate(klist, start=1):
        print(f" Gen {i}: k={kk:.5f} ± {dk:.5f}")
    # also plot 1st gen collisions
    plot_first_generation_hist(first_res, geometry_str, N0)

    # 8) ask if we want to do optimization
    do_opt= get_valid_input(
        "Do you want to do an optimization param sweep? [yes/no]: ",
        lambda x: x.lower() in ['yes','no'],
        "Must be yes/no"
    ).lower()

    if do_opt=='no':
        print("Done, exiting.")
        return
    else:
        print("\n=== Optimization Param Sweep ===")
        # user picks a new neutron model for the sweep (like you wanted)
        nm_str2= get_valid_input(
            "In optimization, choose neutron model [Thermal/Fast]: ",
            validate_neutron_model,
            "Must be thermal or fast"
        ).lower()
        # user picks a new moderator
        mod_str2= get_valid_input(
            "Moderator for optimization [H2O/D2O/Graphite]: ",
            validate_moderator,
            "Must be H2O, D2O, Graphite"
        ).lower()

        # build new materials
        if dset_str=='wikipedia':
            # we override with new fast/thermal
            (u235n, u238n)= get_wikipedia_materials(nm_str2)
        else:
            # keep Lilley
            u235n= LILLEY_U235
            u238n= LILLEY_U238
        mod_optim= MODERATORS_DICT[mod_str2]

        # prompt user for how many neutrons in param sweep
        sweep_n_str= get_valid_input(
            "Neutrons for param sweep (ex: 500): ",
            validate_positive_integer,
            "Must be positive integer"
        )
        sweepN=int(sweep_n_str)

        # do a param sweep around the original radius, fraction, Rmtf
        # for demonstration let's do a small range function
        results= run_param_sweep(
            ReactorMixture(frac_f, mod_optim, Rmtf, u235n,u238n),
            geometry_str,
            R=size_val if geometry_str=='sphere' else size_val[0],
            H= size_val if geometry_str=='sphere' else size_val[1],
            fraction= frac_f,
            Rmtf= Rmtf,
            N0=sweepN,
            max_generations=15,
            early_stop_k=1.7
        )
        print("\nParam sweep done. We'll print a few results below:")
        # show some results
        keys_list= list(results.keys())[:10]
        for k_ in keys_list:
            resK= results[k_]
            finalK= resK['final_k']
            finalDK= resK['final_dk']
            massKG= resK['mass_kg']
            print(f"Param {k_}: finalK={finalK:.3f}±{finalDK:.3f}, mass={massKG:.2f} kg")
        
        print("\nAll done with optimization.\n")


if __name__=="__main__":
    main()
