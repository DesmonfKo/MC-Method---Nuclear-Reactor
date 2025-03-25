############################################
# PATCH 1 of N
############################################

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

NA = 6.022e23  # Avogadro's number

########################################
# 1. Input Validation Helpers
########################################
def get_valid_input(prompt, validation_func, error_msg):
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
    try:
        if geometry=='sphere':
            val = float(input_str)
            return (val>0)
        elif geometry=='cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                parts= list(map(float, input_str[1:-1].split(',')))
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
    val=input_str.lower()
    return val in ['h2o','d2o','graphite']


def validate_positive_integer(input_str):
    try:
        val=int(input_str)
        return (val>0)
    except ValueError:
        return False

############################################
# PATCH 1 of N
############################################

import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

NA = 6.022e23  # Avogadro's number

########################################
# 1. Input Validation Helpers
########################################
def get_valid_input(prompt, validation_func, error_msg):
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
    try:
        if geometry=='sphere':
            val = float(input_str)
            return (val>0)
        elif geometry=='cylinder':
            if input_str.startswith('[') and input_str.endswith(']'):
                parts= list(map(float, input_str[1:-1].split(',')))
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
    val=input_str.lower()
    return val in ['h2o','d2o','graphite']


def validate_positive_integer(input_str):
    try:
        val=int(input_str)
        return (val>0)
    except ValueError:
        return False

############################################
# PATCH 3 of N
############################################

def distance_to_collision(mixture):
    sigma_s= mixture.macroscopic_scattering
    sigma_a= mixture.macroscopic_absorption
    sigma_tot= sigma_s+ sigma_a
    if sigma_tot<=1e-30:
        return 1e10
    return -np.log(rand.rand())/ sigma_tot

def point_is_inside_sphere(p, R):
    return (p[0]**2+ p[1]**2+ p[2]**2)<=R**2

def point_is_inside_cylinder(p, R, H):
    x,y,z= p
    return (x*x+y*y<=R*R) and (abs(z)<=H/2)

def random_scatter_direction():
    costh= 2.*rand.rand()-1.
    phi= 2.*np.pi*rand.rand()
    sinth= np.sqrt(1.- costh*costh)
    return np.array([sinth*np.cos(phi), sinth*np.sin(phi), costh])

def random_position_sphere_optimized(R=1):
    v= rand.randn(3)
    v/= np.linalg.norm(v)
    r_= R*(rand.rand()**(1./3.))
    return r_*v

def random_position_cylinder(R=1,H=1):
    theta= rand.uniform(0,2*np.pi)
    r= np.sqrt(rand.rand())*R
    x= r*np.cos(theta)
    y= r*np.sin(theta)
    z= rand.uniform(-H/2,H/2)
    return np.array([x,y,z])

def simulate_first_generation(mixture, geometry, size, N0, bins=20):
    """
    random walk for N0 neutrons => get collisions distribution
    """
    P_scat= mixture.scattering_probability
    P_fis= mixture.fission_probability

    if geometry=='sphere':
        R= float(size)
    else:
        R_, H_= float(size[0]), float(size[1])

    positions= np.zeros((N0,3))
    # init
    for i in range(N0):
        if geometry=='sphere':
            positions[i,:]= random_position_sphere_optimized(R)
        else:
            positions[i,:]= random_position_cylinder(R_,H_)

    scatter_r, absorb_r, fission_r= [],[],[]
    absorbed_positions= []
    is_active= np.ones(N0, dtype=bool)
    leak_count=0

    while True:
        active_idx= np.where(is_active)[0]
        if len(active_idx)==0:
            break
        for idx in active_idx:
            pos= positions[idx]
            dist= distance_to_collision(mixture)
            dirn= random_scatter_direction()
            new_pos= pos+ dist*dirn
            # leak?
            if geometry=='sphere':
                if not point_is_inside_sphere(new_pos,R):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            else:
                if not point_is_inside_cylinder(new_pos, R_,H_):
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
                if rand.rand()< P_fis:
                    fission_r.append(rcol)
                absorbed_positions.append(new_pos)
                is_active[idx]=False

    # bin
    if geometry=='sphere':
        Rf= float(size)
        bins_r= np.linspace(0,Rf,bins+1)
        s_hist,_= np.histogram(scatter_r,bins=bins_r)
        a_hist,_= np.histogram(absorb_r,bins=bins_r)
        f_hist,_= np.histogram(fission_r,bins=bins_r)
        shell_vol= (4./3.)*np.pi*(bins_r[1:]**3- bins_r[:-1]**3)
    else:
        Rf= float(size[0]); Hf= float(size[1])
        bins_r= np.linspace(0,Rf,bins+1)
        s_hist,_= np.histogram(scatter_r,bins=bins_r)
        a_hist,_= np.histogram(absorb_r,bins=bins_r)
        f_hist,_= np.histogram(fission_r,bins=bins_r)
        ring_area= np.pi*(bins_r[1:]**2- bins_r[:-1]**2)
        shell_vol= ring_area*Hf

    s_dens= s_hist/shell_vol
    a_dens= a_hist/shell_vol
    f_dens= f_hist/shell_vol

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count,
        'bin_edges': bins_r,
        'scatter_density': s_dens,
        'absorb_density': a_dens,
        'fission_density': f_dens
    }

def plot_first_generation_hist(res_dict, geometry, N0):
    s_d= res_dict['scatter_density']
    a_d= res_dict['absorb_density']
    f_d= res_dict['fission_density']
    edges= res_dict['bin_edges']
    centers= 0.5*(edges[:-1]+ edges[1:])
    plt.figure()
    plt.plot(centers, s_d, '-o', color='blue', label='Scattered')
    plt.plot(centers, a_d, '-o', color='orange', label='Absorbed')
    plt.plot(centers, f_d, '-o', color='green', label='Fission')
    plt.title(f"1st Generation: geometry={geometry}, N0={N0}")
    plt.xlabel("Radial distance from center/axis (m)")
    plt.ylabel("Collisions / m^3")
    plt.legend()
    plt.tight_layout()
    plt.show()

############################################
# PATCH 4 of N
############################################

def compute_k_factor_and_uncertainty(n_abs, n_init, fission_prob, nu):
    if n_init<=0:
        return (0.0,0.0)
    k= (n_abs/n_init)* fission_prob* nu
    if n_abs>0 and 0<=fission_prob<=1:
        stdev_f= np.sqrt(n_abs*fission_prob*(1-fission_prob))
        dk= (nu/n_init)* stdev_f
    else:
        dk=0.0
    return (k,dk)

def simulate_one_generation(mixture, geometry, size, init_positions):
    N= init_positions.shape[0]
    is_active= np.ones(N, dtype=bool)
    leak_count=0
    absorbed_positions=[]
    P_scat= mixture.scattering_probability
    while True:
        idxs= np.where(is_active)[0]
        if len(idxs)==0:
            break
        for i_ in idxs:
            pos= init_positions[i_]
            dist= distance_to_collision(mixture)
            dirn= random_scatter_direction()
            newp= pos+ dist*dirn
            if geometry=='sphere':
                Rf= float(size)
                if not point_is_inside_sphere(newp,Rf):
                    leak_count+=1
                    is_active[i_]=False
                    continue
            else:
                Rf, Hf= float(size[0]), float(size[1])
                if not point_is_inside_cylinder(newp,Rf,Hf):
                    leak_count+=1
                    is_active[i_]=False
                    continue
            # collision
            if rand.rand()< P_scat:
                init_positions[i_]=newp
            else:
                absorbed_positions.append(newp)
                is_active[i_]=False
    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }

def run_generations(mixture, geometry, size, N0, max_gen=15, kstop=1.7, k_tolerance=0.02):
    """
    We'll do up to max_gen generations or stop early if k>kstop.
    We'll also see if we find a generation g s.t. |k-1|<k_tolerance => 'converged to near 1'.
    Return the entire k-list plus the generation index that meets near-1 condition (if any).
    """
    # 1st generation:
    init_pos= np.zeros((N0,3))
    first_res= simulate_first_generation(mixture, geometry, size, N0)
    nabs= first_res['absorbed_count']
    k1, dk1= compute_k_factor_and_uncertainty(nabs, N0, mixture.fission_probability, mixture.u235.nu)
    k_list= [(k1,dk1)]
    if abs(k1-1.)< k_tolerance:
        converged_gen= 1
    else:
        converged_gen= None
    if k1>kstop:
        return k_list, converged_gen

    cur_init= nabs
    cur_pos= first_res['absorbed_positions']

    for g in range(2, max_gen+1):
        if cur_init<=0:
            k_list.append((0.0,0.0))
            break
        single_res= simulate_one_generation(mixture, geometry, size, cur_pos)
        nab2= single_res['absorbed_count']
        k2, dk2= compute_k_factor_and_uncertainty(nab2, cur_init, mixture.fission_probability, mixture.u235.nu)
        k_list.append((k2, dk2))
        if (converged_gen is None) and (abs(k2-1.)<k_tolerance):
            converged_gen= g
        if k2>kstop:
            break
        cur_init= nab2
        cur_pos= single_res['absorbed_positions']
    return k_list, converged_gen


def run_param_sweep(mixture, geometry, baseR, baseH, baseFraction, baseRmtf,
                    reflect=False, reflect_thickness=0.0,
                    N0=500, max_generations=15, kstop=1.7, ktol=0.02):
    """
    We'll vary radius in [baseR-0.05.. baseR+0.05],
          fraction in [baseFraction-0.1.. baseFraction+0.1],
          Rmtf in [baseRmtf.. baseRmtf+10], etc. 
    reflect => if True, we do something special (but here we just label it).
    For each set, we run up to 15 gens and see if we find k near 1. 
    Then we store finalK, mass, and generation that meets near 1 if any.
    """
    # define small range for radius
    if baseR<0.05:
        Rvals= [baseR]  # can't go negative
    else:
        Rvals= np.arange(baseR-0.05, baseR+0.051, 0.01)
    Rvals= [r for r in Rvals if r>0]

    # fraction range
    fvals= np.arange(baseFraction-0.1, baseFraction+0.101, 0.1)
    fvals= [fv for fv in fvals if fv>=0 and fv<=1]

    # R_mtf range
    rmtfvals= np.arange(baseRmtf, baseRmtf+10.1, 5)
    rmtfvals= [rm for rm in rmtfvals if rm>=0]

    results= {}
    from copy import deepcopy
    for rv in Rvals:
        for fv in fvals:
            for rm in rmtfvals:
                # build new mixture object
                mix_ = ReactorMixture(fv, mixture.moderator, rm,
                                      deepcopy(mixture.u235), deepcopy(mixture.u238))
                # geometry
                if geometry=='sphere':
                    size_ = rv
                else:
                    size_ = [rv, baseH]
                # run
                klist, convGen= run_generations(mix_, geometry, size_, N0, max_gen=max_generations, kstop=kstop, k_tolerance=ktol)
                finalK= klist[-1][0]
                finalDK= klist[-1][1]
                # mass
                mass_kg= mix_.compute_total_mass_kg(geometry, size_)
                # store
                results[(rv,fv,rm, reflect)] = {
                    'k_list': klist,
                    'final_k': finalK,
                    'final_dk': finalDK,
                    'mass_kg': mass_kg,
                    'converged_gen': convGen
                }
    return results


def main():
    print("=== Reactor MC Simulation with Param Sweep & Reflector Option ===\n")

    # 1) Basic geometry
    geom_str= get_valid_input(
        "Geometry [sphere/cylinder]: ",
        validate_geometry,
        "Must be sphere/cylinder"
    ).lower()

    if geom_str=='sphere':
        sprompt="Enter sphere radius (m): "
        s_str= get_valid_input(sprompt, lambda x: validate_size(x,'sphere'), "Invalid radius")
        size_val= float(s_str)
        baseR= size_val; baseH= None
    else:
        cprompt="Enter cylinder [R,H] e.g. [1.0,2.0] (m): "
        c_str= get_valid_input(cprompt, lambda x: validate_size(x,'cylinder'),"Invalid cylinder size")
        prts=c_str.strip()[1:-1].split(',')
        R_= float(prts[0]); H_= float(prts[1])
        size_val=[R_,H_]
        baseR= R_; baseH=H_

    # 2) dataset
    ds= get_valid_input("Dataset [Lilley/Wikipedia]: ", validate_dataset, "Must be Lilley or Wikipedia").lower()
    if ds=='wikipedia':
        nm= get_valid_input("Neutron model [Thermal/Fast]: ", validate_neutron_model, "Must be thermal or fast").lower()
        u235mat,u238mat= get_wikipedia_materials(nm)
    else:
        u235mat= LILLEY_U235
        u238mat= LILLEY_U238

    # fraction
    frac_str= get_valid_input(
        "U-235 fraction in [0..100%]: ",
        lambda x: validate_float(x,0,100),
        "Must be 0..100"
    )
    frac_f= float(frac_str)/100.0

    # R_mtf
    rmtf_str= get_valid_input(
        "Moderator-to-fuel ratio (>=0): ",
        lambda x: validate_float(x,0.0),
        "Must be >=0"
    )
    Rmtf= float(rmtf_str)

    # moderator
    mod_str= get_valid_input(
        "Moderator [H2O/D2O/Graphite]: ",
        validate_moderator,
        "Must be H2O/D2O/Graphite"
    ).lower()
    mod_obj= MODERATORS_DICT[mod_str]

    # build mixture
    mixture= ReactorMixture(frac_f, mod_obj, Rmtf, u235mat,u238mat)
    # show cross sections
    print("\nCross Sections (m^-1):")
    print(f"  Σ_s={mixture.macroscopic_scattering:.6e}")
    print(f"  Σ_a={mixture.macroscopic_absorption:.6e}")
    print(f"  Σ_f={mixture.macroscopic_fission:.6e}")
    # compute mass
    mass_kg= mixture.compute_total_mass_kg(geom_str, size_val)
    mass_u235= mass_kg* frac_f
    print(f"Effective density= {mixture.effective_density_gcc:.3f} g/cc => total mass= {mass_kg:.3f} kg, U-235 mass= {mass_u235:.3f} kg\n")

    # 2) run a short multi-generation sim
    n0_str= get_valid_input("Neutrons for previous multi-gen sim: ", validate_positive_integer, "Must be int>0")
    N0= int(n0_str)
    klist, first_gen_res= run_generations(mixture, geom_str, size_val, N0, max_gen=3)
    print("\n=== Previous Generational Sim Results (up to 3 gens) ===")
    for i,(kk,dk) in enumerate(klist, start=1):
        print(f"  Gen {i}: k={kk:.3f}±{dk:.3f}")
    # also plot collisions of 1st gen
    # first_gen_res is a tuple: check
    # Actually run_generations returns (klist, convgen). We want the entire 1st gen distribution => let's just re-run:
    res_1= simulate_first_generation(mixture, geom_str, size_val, N0)
    plot_first_generation_hist(res_1, geom_str, N0)

    # 3) ask if do param sweep
    do_swp= get_valid_input("Do param sweep? [yes/no]: ", lambda x: x.lower() in ['yes','no'], "Must be yes/no").lower()
    if do_swp=='no':
        print("Done.")
        return

    # ask user if they want a reflector
    reflect_str= get_valid_input("Use a reflector? [yes/no]: ", lambda x: x.lower() in ['yes','no'],"Must be yes/no").lower()
    reflect_flag= (reflect_str=='yes')
    rthick=0.0
    if reflect_flag:
        # ask thickness
        thick_str= get_valid_input("Reflector thickness (m)? e.g. 0.05: ", lambda x: validate_float(x,0.0), "Must be >=0")
        rthick= float(thick_str)

    # param sweep
    sweep_n_str= get_valid_input("Neutrons for param sweep (ex: 500): ", validate_positive_integer, "Must be >0")
    sweepN= int(sweep_n_str)
    # do the sweep
    results= run_param_sweep(mixture, geom_str, baseR, baseH, frac_f, Rmtf,
                             reflect=reflect_flag, reflect_thickness=rthick,
                             N0=sweepN, max_generations=15, kstop=1.7, ktol=0.02)

    print("\nParam sweep done. We'll print a few results below:\n")
    # let's just show a few
    keys_list= list(results.keys())[:10]
    for param_key in keys_list:
        # param_key is (rv, fv, rm, reflectFlag)
        rv, fv, rm, refl = param_key
        final_k= results[param_key]['final_k']
        final_dk= results[param_key]['final_dk']
        mass_  = results[param_key]['mass_kg']
        cgen= results[param_key]['converged_gen']
        # print with units
        # geometry => radius (m), fraction => dimensionless, R_mtf => dimensionless
        # reflect => yes/no
        reflect_str_ = "yes" if refl else "no"
        print(f"Param (radius={rv:.3f} m, fraction={fv:.2f}, R_mtf={rm:.1f}, reflector={reflect_str_}):")
        print(f"  finalK={final_k:.3f}±{final_dk:.3f}, mass={mass_:.2f} kg, convergedGen={cgen}")
    print("\nAll done with optimization.\n")


################## END ##################
if __name__=="__main__":
    main()
