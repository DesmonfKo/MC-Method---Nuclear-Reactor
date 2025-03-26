## Oth Trial ##
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
    """
    Continuously prompt until validation_func returns True.
    Returns user input (string).
    """
    while True:
        val_in = input(prompt).strip()
        if validation_func(val_in):
            return val_in
        print(f"Invalid input! {error_msg}")


def validate_geometry(input_str):
    return input_str.lower() in ['sphere','cylinder']

def validate_float(input_str, min_val=None, max_val=None):
    """
    If min_val is not None and val<min_val => fail
    If max_val is not None and val>max_val => fail
    """
    try:
        v= float(input_str)
        if min_val is not None and v< min_val:
            return False
        if max_val is not None and v> max_val:
            return False
        return True
    except ValueError:
        return False

def validate_size(input_str, geometry):
    """
    For 'sphere': single float radius>0
    For 'cylinder': [radius,height], each>0
    """
    try:
        if geometry=='sphere':
            val= float(input_str)
            return (val>0)
        else:
            if input_str.startswith('[') and input_str.endswith(']'):
                parts= list(map(float,input_str[1:-1].split(',')))
                if len(parts)==2 and all(x>0 for x in parts):
                    return True
        return False
    except:
        return False

def validate_dataset(input_str):
    return input_str.lower() in ['lilley','wikipedia']

def validate_neutron_model(input_str):
    return input_str.lower() in ['thermal','fast']

def validate_moderator(input_str):
    v= input_str.strip().lower()
    return v in ['h2o','d2o','graphite']

def validate_positive_integer(input_str):
    try:
        val=int(input_str)
        return (val>0)
    except ValueError:
        return False


############################################
# PATCH 2 of N
############################################

def random_position_cylinder(radius=1, height=1):
    '''
    Sample a random point uniformly within a right circular cylinder.
    Returns np.array(3,).
    '''
    theta= rand.uniform(0,2*np.pi)
    r= np.sqrt(rand.uniform(0,1))* radius
    x= r*np.cos(theta)
    y= r*np.sin(theta)
    z= rand.uniform(-height/2, height/2)
    return np.array([x,y,z])

def random_position_sphere_optimized(radius=1):
    '''
    Sample a random point uniformly within a sphere of radius=radius.
    '''
    v= rand.randn(3)
    v/= np.linalg.norm(v)
    r_= radius* (rand.uniform()**(1/3))
    return v*r_

class ReactorMaterial:
    """
    For normal fuel/moderator definitions (no reflection_probability).
    """
    def __init__(self, name, mat_type, density, molar_mass,
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name= name
        self.mat_type= mat_type
        self.density_gcc= density
        self.molar_mass_gmol= molar_mass
        # convert to SI
        self.density= density*1e3
        self.molar_mass= molar_mass*1e-3
        self.sigma_s_b= sigma_s_b
        self.sigma_a_b= sigma_a_b
        self.sigma_f_b= sigma_f_b
        self.sigma_s= sigma_s_b*1e-28
        self.sigma_a= sigma_a_b*1e-28
        self.sigma_f= sigma_f_b*1e-28
        self.nu= nu
        self.xi= xi

    @property
    def number_density(self):
        return (self.density/ self.molar_mass)* NA


###############################
# Lilley's default materials
###############################
LILLEY_U235= ReactorMaterial(
    "U235","fuel",
    density=18.7, molar_mass=235,
    sigma_s_b=10, sigma_a_b=680, sigma_f_b=579,
    nu=2.42, xi=0
)
LILLEY_U238= ReactorMaterial(
    "U238","fuel",
    density=18.9, molar_mass=238,
    sigma_s_b=8.3, sigma_a_b=2.72, sigma_f_b=0.0,
    nu=0, xi=0
)

MODERATORS_DICT= {
    'h2o': ReactorMaterial("H2O","moderator",1.0,18.01,49.2,0.66,0.0,0,0.92),
    'd2o': ReactorMaterial("D2O","moderator",1.1,20.02,10.6,0.001,0.0,0,0.509),
    'graphite': ReactorMaterial("Graphite","moderator",1.6,12.01,4.7,0.0045,0.0,0,0.158)
}

def get_wikipedia_materials(neutron_model='thermal'):
    if neutron_model=='thermal':
        u235= ReactorMaterial(
            "U235","fuel",18.7,235,
            sigma_s_b=10, sigma_a_b=(99+583), sigma_f_b=583,
            nu=2.42, xi=0
        )
        u238= ReactorMaterial(
            "U238","fuel",18.9,238,
            sigma_s_b=9, sigma_a_b=(2+0.00002), sigma_f_b=0.00002,
            nu=0, xi=0
        )
    else:
        # fast
        u235= ReactorMaterial(
            "U235","fuel",18.7,235,
            sigma_s_b=4, sigma_a_b=1.09, sigma_f_b=1,
            nu=2.42, xi=0
        )
        u238= ReactorMaterial(
            "U238","fuel",18.9,238,
            sigma_s_b=5, sigma_a_b=0.37, sigma_f_b=0.3,
            nu=0, xi=0
        )
    return (u235,u238)

############################################
# PATCH 3 of N
############################################

class ReactorMixture:
    """
    fraction_U235 in [0..1].
    R_mtf => ratio of moderator to total fuel atoms
    """
    def __init__(self, fraction_U235, moderator, R_mtf, u235_material, u238_material):
        self.fraction_U235= fraction_U235
        self.moderator= moderator
        self.R_mtf= R_mtf
        self.u235= u235_material
        self.u238= u238_material

    @property
    def macroscopic_scattering(self):
        aU= self.fraction_U235
        a238= 1.- aU
        B= aU+ a238+ self.R_mtf
        conv=1.0e6* NA*1.0e-28
        partU= (aU/B)* (self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_s_b
        part238= (a238/B)* (self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_s_b
        partM= (self.R_mtf/B)* (self.moderator.density_gcc/self.moderator.molar_mass_gmol)* self.moderator.sigma_s_b
        return conv*(partU+ part238+ partM)

    @property
    def macroscopic_absorption(self):
        aU= self.fraction_U235
        a238= 1.- aU
        B= aU+ a238+ self.R_mtf
        conv=1.0e6* NA*1.0e-28
        pu= (aU/B)*(self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_a_b
        p238= (a238/B)*(self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_a_b
        pmod= (self.R_mtf/B)* (self.moderator.density_gcc/self.moderator.molar_mass_gmol)* self.moderator.sigma_a_b
        return conv*(pu+ p238+ pmod)

    @property
    def macroscopic_fission(self):
        aU= self.fraction_U235
        a238= 1.- aU
        B= aU+ a238+ self.R_mtf
        conv=1.0e6* NA*1.0e-28
        fu= (aU/B)* (self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_f_b
        f238= (a238/B)* (self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_f_b
        return conv*(fu+ f238)

    @property
    def fission_probability(self):
        fa= self.macroscopic_absorption
        ff= self.macroscopic_fission
        if fa<1e-30: return 0.0
        return ff/fa

    @property
    def scattering_probability(self):
        s= self.macroscopic_scattering
        a= self.macroscopic_absorption
        denom= s+a
        if denom<1e-30: return 0.0
        return s/ denom

    @property
    def absorption_probability(self):
        s= self.macroscopic_scattering
        a= self.macroscopic_absorption
        denom= s+a
        if denom<1e-30:return 0.0
        return a/ denom

    def compute_u235_mass_kg(self, geometry, size):
        """
        Return the mass of U235 in entire mixture, using homogeneous approach.
        """
        aU= self.fraction_U235
        B= aU+ (1.-aU)+ self.R_mtf
        if B<1e-30:
            return 0.0
        top= (aU*self.u235.density_gcc + (1.-aU)*self.u238.density_gcc + self.R_mtf*self.moderator.density_gcc)
        eff_dens_gcc= top/B
        eff_dens_kg_m3= eff_dens_gcc*1e3
        import math
        if geometry=='sphere':
            R_= float(size)
            vol= (4./3.)*math.pi*(R_**3)
        else:
            R_,H_= float(size[0]), float(size[1])
            vol= math.pi*(R_**2)* H_
        total_mass= vol* eff_dens_kg_m3
        return aU* total_mass


class ReflectorMaterial:
    """
    Separate class so we do have reflection_probability property.
    """
    def __init__(self, name, sigma_s_b, sigma_a_b, density_gcc, molar_mass_gmol):
        self.name= name
        self.sigma_s_b= sigma_s_b
        self.sigma_a_b= sigma_a_b
        self.density_gcc= density_gcc
        self.molar_mass_gmol= molar_mass_gmol
        self.sigma_s= sigma_s_b*1e-28
        self.sigma_a= sigma_a_b*1e-28
        self.number_density= (density_gcc*1e3/molar_mass_gmol)* NA

    @property
    def macroscopic_scattering(self):
        return self.number_density* self.sigma_s

    @property
    def macroscopic_absorption(self):
        return self.number_density* self.sigma_a

    @property
    def reflection_probability(self):
        s= self.macroscopic_scattering
        a= self.macroscopic_absorption
        tot= s+a
        if tot<1e-30:
            return 0.0
        return s/tot


def distance_to_collision(mixture):
    s= mixture.macroscopic_scattering
    a= mixture.macroscopic_absorption
    tot= s+a
    if tot<=1e-30:
        return 1e10
    return -np.log(rand.rand())/ tot


############################################
# PATCH 4 of N
############################################

def random_scatter_direction():
    costh = 2.*rand.rand()-1
    phi   = 2.*np.pi*rand.rand()
    sinth = np.sqrt(1.- costh*costh)
    return np.array([
        sinth*np.cos(phi),
        sinth*np.sin(phi),
        costh
    ])

def simulate_first_generation(mixture, geometry, size, N0,
                              bins=20, reflector=None, reflector_thickness=0.0):
    """
    Single-run approach for 1st generation collisions,
    forcibly storing 'reflector_interaction_count'.
    """
    import math
    scatter_r= []
    absorb_r= []
    fission_r= []
    reflect_r= []
    leak_count=0
    reflector_interaction_count=0

    P_scat= mixture.scattering_probability
    P_fis= mixture.fission_probability

    if geometry=='sphere':
        R_core= float(size)
        R_refl= R_core+ reflector_thickness
    else:
        R_core, H_core= float(size[0]), float(size[1])
        R_refl= R_core+ reflector_thickness
        H_refl= H_core+ 2.*reflector_thickness

    P_reflect= 0.0
    if reflector:
        P_reflect= reflector.reflection_probability

    positions= np.zeros((N0,3))
    for i in range(N0):
        if geometry=='sphere':
            positions[i,:]= random_position_sphere_optimized(R_core)
        else:
            positions[i,:]= random_position_cylinder(R_core,H_core)

    is_active= np.ones(N0,dtype=bool)
    absorbed_positions= []
    while True:
        idxs= np.where(is_active)[0]
        if len(idxs)==0:
            break
        for idx in idxs:
            pos= positions[idx]
            while True:
                dist= distance_to_collision(mixture)
                dirn= random_scatter_direction()
                newp= pos+ dist* dirn
                if geometry=='sphere':
                    dist2= newp[0]**2+ newp[1]**2+ newp[2]**2
                    in_core= (dist2<= R_core**2)
                    in_ref= (reflector is not None) and (dist2<= R_refl**2) and (not in_core)
                else:
                    rr= newp[0]**2+ newp[1]**2
                    in_core= (rr<=R_core**2) and (abs(newp[2])<= H_core/2)
                    in_ref= (reflector is not None) and (rr<=R_refl**2) and (abs(newp[2])<=H_refl/2) and (not in_core)

                if in_core:
                    if rand.rand()< P_scat:
                        # scatter
                        if geometry=='sphere':
                            scatter_r.append(math.sqrt(dist2))
                        else:
                            scatter_r.append(math.sqrt(rr))
                        pos= newp
                    else:
                        # absorb
                        if geometry=='sphere':
                            absorb_r.append(math.sqrt(dist2))
                        else:
                            absorb_r.append(math.sqrt(rr))
                        if rand.rand()< P_fis:
                            if geometry=='sphere':
                                fission_r.append(math.sqrt(dist2))
                            else:
                                fission_r.append(math.sqrt(rr))
                        absorbed_positions.append(newp)
                        is_active[idx]=False
                        break
                elif in_ref:
                    reflector_interaction_count+=1
                    if rand.rand()< P_reflect:
                        if geometry=='sphere':
                            reflect_r.append(math.sqrt(dist2))
                        else:
                            reflect_r.append(math.sqrt(rr))
                        pos= newp
                    else:
                        leak_count+=1
                        is_active[idx]=False
                        break
                else:
                    # leak
                    leak_count+=1
                    is_active[idx]=False
                    break

        idxs= np.where(is_active)[0]
        if len(idxs)==0:
            break

    ab_count= len(absorbed_positions)
    # bin collisions
    bins_r= np.linspace(0,R_core,bins+1) if geometry=='sphere' else np.linspace(0,R_core,bins+1)
    if geometry=='sphere':
        s_hist,_= np.histogram(scatter_r,bins=bins_r)
        a_hist,_= np.histogram(absorb_r,bins=bins_r)
        f_hist,_= np.histogram(fission_r,bins=bins_r)
        ref_hist,_= np.histogram(reflect_r,bins=bins_r)
        shellv= (4./3.)* np.pi*(bins_r[1:]**3- bins_r[:-1]**3)
    else:
        s_hist,_= np.histogram(scatter_r,bins=bins_r)
        a_hist,_= np.histogram(absorb_r,bins=bins_r)
        f_hist,_= np.histogram(fission_r,bins=bins_r)
        ref_hist,_= np.histogram(reflect_r,bins=bins_r)
        ring_area= np.pi*(bins_r[1:]**2- bins_r[:-1]**2)
        H_core= size[1]
        shellv= ring_area* H_core

    s_dens= s_hist/shellv
    a_dens= a_hist/shellv
    f_dens= f_hist/shellv
    ref_dens= ref_hist/shellv

    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': ab_count,
        'leak_count': leak_count,
        'reflector_interaction_count': reflector_interaction_count,
        'bin_edges': bins_r,
        'scatter_density': s_dens,
        'absorb_density': a_dens,
        'fission_density': f_dens,
        'reflect_density': ref_dens
    }

def simulate_generation(mixture, geometry, size, init_positions,
                       reflector=None, reflector_thickness=0.0):
    """
    Single generation random walk, simpler approach.
    """
    N= init_positions.shape[0]
    is_active= np.ones(N,dtype=bool)
    leak_count=0
    absorbed_positions=[]
    P_scat= mixture.scattering_probability

    if geometry=='sphere':
        R_core= float(size)
    else:
        R_core, H_core= float(size[0]), float(size[1])

    while True:
        idxs= np.where(is_active)[0]
        if len(idxs)==0:
            break
        for idx in idxs:
            pos= init_positions[idx]
            dist= distance_to_collision(mixture)
            dirn= random_scatter_direction()
            newp= pos+ dist* dirn
            if geometry=='sphere':
                if np.sum(newp**2)> (R_core**2):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            else:
                rr= newp[0]**2+ newp[1]**2
                if (rr> R_core**2) or (abs(newp[2])> (H_core/2)):
                    leak_count+=1
                    is_active[idx]=False
                    continue
            # collision
            if rand.rand()< P_scat:
                init_positions[idx]= newp
            else:
                absorbed_positions.append(newp)
                is_active[idx]=False
    return {
        'absorbed_positions': np.array(absorbed_positions),
        'absorbed_count': len(absorbed_positions),
        'leak_count': leak_count
    }


def compute_k_factor_and_uncertainty(n_abs, n_init, mixture):
    """
    Computes k = (n_abs/n_init) * p_fission * nu,
    with uncertainty computed from the binomial variance on the absorption fraction.
    
    Parameters:
      n_abs: number of absorptions (or effective independent absorption events)
      n_init: initial number of neutrons in this generation (the effective sample size)
      mixture: reactor mixture object with attributes fission_probability and u235.nu
    
    Returns:
      (k, dk) where dk is the computed uncertainty on k.
    """
    if n_init <= 0:
        return (0.0, 0.0)
    
    p_fission = mixture.fission_probability  # constant
    nu = mixture.u235.nu                     # constant
    
    # Measured absorption fraction
    p_abs = n_abs / n_init  
    k = p_abs * p_fission * nu
    
    # Binomial variance of the absorption fraction
    # Here, we assume that n_init is the effective number of independent events.
    var_p = p_abs * (1 - p_abs) / n_init
    dk = p_fission * nu * np.sqrt(var_p)
    
    return (k, dk)



def final_bounding_box_sweep_multi_gen(geometry, size,
                                       fmin_dec, fmax_dec,
                                       rmin, rmax,
                                       steps, mixture_builder,
                                       max_generations,
                                       n_neut_sweep,
                                       reflect_obj=None,
                                       reflect_th=0.0):
    """
    We'll do a grid => multi-generation => find # gens to converge => pick best param => store final k in K_map.
    Then we can contour it.
    """
    f_vals= np.linspace(fmin_dec, fmax_dec, steps)
    r_vals= np.linspace(rmin, rmax, steps)
    K_map= np.zeros((steps,steps), dtype=float)

    best_g=9999
    best_param=None

    for i_f, ff in enumerate(f_vals):
        for j_r, rr in enumerate(r_vals):
            # multi-generation
            mix= mixture_builder(ff, rr)
            pos= np.zeros((n_neut_sweep,3))

            # 1st generation
            res= simulate_first_generation(mix, geometry, size, n_neut_sweep,
                                           bins=20,
                                           reflector=reflect_obj,
                                           reflector_thickness= reflect_th)
            ab1= res['absorbed_count']
            # Revised call using material properties from mix
            (k1, dk1) = compute_k_factor_and_uncertainty(ab1, n_neut_sweep, mix)  # <-- Pass mix here
            k_list=[(k1, dk1)]
            if abs(k1-1.)<0.1:
                gen_count=1
                K_map[i_f,j_r]= k1
            else:
                current_init= ab1
                pos= res['absorbed_positions']
                gen_count= max_generations
                for g_ in range(2, max_generations+1):
                    if current_init<=0:
                        k_list.append((0.0,0.0))
                        break
                    g_res= simulate_generation(mix, geometry, size, pos,
                                               reflector=reflect_obj,
                                               reflector_thickness=reflect_th)
                    nab= g_res['absorbed_count']
                    (kg, dkg)= compute_k_factor_and_uncertainty(nab, current_init,
                                                               mix.fission_probability,
                                                               mix.u235.nu)
                    k_list.append((kg,dkg))
                    if abs(kg-1.)<0.1:
                        gen_count= g_
                        K_map[i_f,j_r]= kg
                        break
                    # pop control

                else:
                    # no break => didn't converge
                    last_k= k_list[-1][0]
                    K_map[i_f,j_r]= last_k

            if gen_count< best_g:
                best_g= gen_count
                final_k= k_list[-1][0]
                final_dk= k_list[-1][1]
                best_param= (ff, rr, final_k, final_dk, gen_count)

    return (K_map, best_param)


def iterative_2param_search(sim_func, geometry, size,
                            u235_mat, u238_mat, moderator_obj,
                            reflect_flag, reflect_obj, reflect_th,
                            n_points_per_iter, n_iter,
                            range_u235_percent, range_rmtf,
                            n_neutrons_each_gen,
                            n_neutrons_box_sweep,
                            max_generations=30,
                            k_tolerance=0.3):
    """
    We'll do random sampling => store => bounding box => final box => multi-gen =>
    contour with black lines => then print param sets.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    candidates_size= n_points_per_iter* n_iter
    candidates= np.zeros((candidates_size,4), dtype=float)  # [f, R, k, dk]
    idx_global=0

    fminP, fmaxP= range_u235_percent
    rmin, rmax= range_rmtf
    fmin= fminP/100.
    fmax= fmaxP/100.

    current_fmin= fmin
    current_fmax= fmax
    current_rmin= rmin
    current_rmax= rmax

    for iteration in range(n_iter):
        print(f"\n--- Iteration {iteration+1}/{n_iter}: bounding box => f=[{current_fmin*100:.2f}%..{current_fmax*100:.2f}%], Rmtf=[{current_rmin:.3f}..{current_rmax:.3f}]")
        iteration_points=[]
        for _ in range(n_points_per_iter):
            ff= np.random.uniform(current_fmin, current_fmax)
            rr= np.random.uniform(current_rmin, current_rmax)
            (kv, dkv)= sim_func(ff, rr, geometry, size,
                                u235_mat, u238_mat, moderator_obj,
                                reflect_flag, reflect_obj, reflect_th,
                                n_neutrons_each_gen, max_generations)
            iteration_points.append((ff, rr, kv, dkv))

        for p_ in iteration_points:
            if idx_global< candidates_size:
                candidates[idx_global,0]= p_[0]
                candidates[idx_global,1]= p_[1]
                candidates[idx_global,2]= p_[2]
                candidates[idx_global,3]= p_[3]
                idx_global+=1

        kept= [p_ for p_ in iteration_points if abs(p_[2]-1.)< k_tolerance]
        if not kept:
            print("No points met the requirement => can't further narrow bounding box.")
            continue
        fvals= [pp[0] for pp in kept]
        rvals= [pp[1] for pp in kept]
        new_fmin= min(fvals)
        new_fmax= max(fvals)
        new_rmin= min(rvals)
        new_rmax= max(rvals)
        df= 0.05*(new_fmax- new_fmin)
        dr= 0.05*(new_rmax- new_rmin)
        current_fmin= max(0.0, new_fmin- df)
        current_fmax= min(1.0, new_fmax+ df)
        current_rmin= max(0.0, new_rmin- dr)
        current_rmax= new_rmax+ dr
        if (current_fmax- current_fmin)<1e-4 and (current_rmax- current_rmin)<1e-3:
            print("Bounding box too small => break.")
            break

    print("\n=== Done iterative bounding. Now final bounding box multi-gen sweep. ===")
    final_fmin= current_fmin
    final_fmax= current_fmax
    final_rmin= current_rmin
    final_rmax= current_rmax

    def mixture_builder(ff, rr):
        return ReactorMixture(ff, moderator_obj, rr, u235_mat, u238_mat)

    steps= 20
    (K_map, best_param)= final_bounding_box_sweep_multi_gen(
        geometry, size,
        final_fmin, final_fmax,
        final_rmin, final_rmax,
        steps, mixture_builder,
        max_generations,
        n_neutrons_box_sweep,
        reflect_obj= reflect_obj if reflect_flag else None,
        reflect_th= reflect_th
    )

    # 7) Plot contour with black lines
    fvals= np.linspace(final_fmin, final_fmax, steps)
    rvals= np.linspace(final_rmin, final_rmax, steps)
    fig, ax= plt.subplots()
    cf= ax.contourf(fvals*100., rvals, K_map.T, levels=50, cmap='plasma')
    cbar= fig.colorbar(cf, ax=ax)
    cbar.set_label("k-value")

    diff_map= K_map-1.
    c2= ax.contour(fvals*100., rvals, diff_map.T, levels=[-0.1,0.1], colors='black')
    # add label => black lines => legend below x-axis
    try:
        for c_ in c2.collections:
            c_.set_label("Enclosed region where |k-1| < 0.1")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15))
    except IndexError:
        print("Warning: no contour lines found for |k-1|=0.1 => skipping black line legend.")

    ax.set_xlabel("U-235 Concentration (%)")
    ax.set_ylabel("R_mtf")
    ax.set_title("Final bounding box multi-gen sweep: k-value & region |k-1|<0.1")
    plt.tight_layout()
    plt.show()

    # 9) Print param sets => final bounding param + geometry, size, mass, # gen, final k±dk
    if best_param is None:
        print("\nNo param set found that converged within maxGenerations.\n")
    else:
        fDec, rVal, finalK, finalDK, genC= best_param
        print("\n=== Best param set with fewest generations to converge (|k-1|<0.1) ===")
        print(f"  U-235 fraction= {fDec*100:.2f}%, R_mtf= {rVal:.3f}, convergedGen= {genC}")
        print(f"  finalK={finalK:.6f} ± {finalDK:.6f}")
        # compute mass
        best_mix= mixture_builder(fDec, rVal)
        mass_kg= best_mix.compute_u235_mass_kg(geometry, size)
        print(f"  geometry= {geometry}, size= {size}, => massOfU235= {mass_kg:.3f} kg")

def plot_k_vs_generation(mixture, geometry, size, N0, max_generations=10,
                         reflector=None, reflector_thickness=0.0):
    """
    Simulates successive generations and plots the k-value vs. generation number.
    
    Parameters:
      mixture: ReactorMixture instance.
      geometry: 'sphere' or 'cylinder'.
      size: reactor size (e.g., radius for sphere or [radius, height] for cylinder).
      N0: initial number of neutrons.
      max_generations: maximum number of generations to simulate.
      reflector: optional reflector object.
      reflector_thickness: thickness of reflector (if applicable).
    """
    k_values = []         # to store k for each generation
    dk_values = []        # to store uncertainty (dk) for each generation, optional
    generations = []      # generation numbers
    
    # --- First Generation ---
    res = simulate_first_generation(mixture, geometry, size, N0,
                                    bins=20,
                                    reflector=reflector,
                                    reflector_thickness=reflector_thickness)
    n_abs = res['absorbed_count']
    # Compute k and uncertainty (dk) for the first generation
    k, dk = compute_k_factor_and_uncertainty(n_abs, N0, mixture)
    k_values.append(k)
    dk_values.append(dk)
    generations.append(1)
    
    # Use absorbed positions from the first generation as the initial positions for subsequent generations
    pos = res['absorbed_positions']
    current_init = n_abs

    # --- Subsequent Generations ---
    for gen in range(2, max_generations + 1):
        if current_init <= 0:
            print("No more active neutrons at generation", gen)
            break

        gen_res = simulate_generation(mixture, geometry, size, pos,
                                      reflector=reflector,
                                      reflector_thickness=reflector_thickness)
        n_abs = gen_res['absorbed_count']
        k_gen, dk_gen = compute_k_factor_and_uncertainty(n_abs, current_init, mixture)
        k_values.append(k_gen)
        dk_values.append(dk_gen)
        generations.append(gen)
        
        # Prepare for next generation:
        pos = gen_res['absorbed_positions']
        current_init = n_abs

    # --- Plotting ---
    plt.figure(figsize=(8, 6))
    # If you want to include uncertainties as error bars:
    plt.errorbar(generations, k_values, yerr=dk_values, marker='o', linestyle='-', capsize=5)
    plt.xlabel("Generation")
    plt.ylabel("k-value")
    plt.title("k-value vs. Generation")
    plt.grid(True)
    plt.show()

def main():
    print("=== Final Merged Code: Black Lines, Legend Outside, 2D Search w/NameError Fixed ===")

    # Single-run includes fraction & Rmtf
    geo_in= get_valid_input("Geometry [sphere/cylinder]: ",
                            validate_geometry,
                            "Must be sphere/cylinder").lower()
    if geo_in=='sphere':
        s_in= get_valid_input("Enter sphere radius (m): ",
                              lambda x: validate_size(x,'sphere'),
                              "Invalid radius")
        reactor_size= float(s_in)
    else:
        s_in= get_valid_input("Enter cylinder [radius,height], e.g. [1.0,2.0]: ",
                              lambda x: validate_size(x,'cylinder'),
                              "Invalid cylinder size")
        prt= s_in.strip()[1:-1].split(',')
        reactor_size= [float(prt[0]), float(prt[1])]

    ds_str= get_valid_input("Dataset [Lilley/Wikipedia]: ",
                            validate_dataset,
                            "Must be Lilley/Wikipedia").lower()
    if ds_str=='wikipedia':
        nm= get_valid_input("Neutron model [Thermal/Fast]: ",
                            validate_neutron_model,
                            "Must be thermal/fast").lower()
        u235_mat,u238_mat= get_wikipedia_materials(nm)
    else:
        u235_mat= LILLEY_U235
        u238_mat= LILLEY_U238

    mod_str= get_valid_input("Moderator [H2O/D2O/Graphite]: ",
                             validate_moderator,
                             "Must be H2O,D2O,Graphite").lower()
    moderator_obj= MODERATORS_DICT[mod_str]

    frac_str= get_valid_input("U-235 concentration [0..100%] for single-run: ",
                              lambda x: validate_float(x,0,100),
                              "Must be in 0..100")
    frac_percent= float(frac_str)
    frac_dec= frac_percent/100.

    rmtf_str= get_valid_input("R_mtf(>=0) for single-run: ",
                              lambda x: validate_float(x,0.0),
                              "Must be >=0")
    rmtf_val= float(rmtf_str)

    reflect_s= get_valid_input("Use reflector in single-run? [yes/no]: ",
                               lambda x:x.lower() in ['yes','no'],
                               "Must be yes/no").lower()
    reflect_flag= (reflect_s=='yes')
    reflect_obj=None
    reflect_th=0.0
    if reflect_flag:
        thick_in= get_valid_input("Reflector thickness (m): ",
                                  lambda x: validate_float(x,0.0),
                                  "Must be >=0")
        reflect_th= float(thick_in)
        rmat_s= get_valid_input("Reflector material [Graphite/Beryllium]: ",
                                lambda x: x.lower() in ['graphite','beryllium'],
                                "Must be 'graphite' or 'beryllium'").lower()
        if rmat_s=='graphite':
            reflect_obj= ReflectorMaterial("Graphite",4.7,0.0045,1.6,12.01)
        else:
            reflect_obj= ReflectorMaterial("Beryllium",6.0,0.001,1.85,9.01)

    n0_s= get_valid_input("Number of neutrons for single-run test: ",
                          validate_positive_integer,"Must be int>0")
    N0= int(n0_s)

    # Single-run
    single_mix= ReactorMixture(frac_dec, moderator_obj, rmtf_val, u235_mat, u238_mat)
    single_res= simulate_first_generation(single_mix, geo_in, reactor_size, N0,
                                          bins=20,
                                          reflector= reflect_obj,
                                          reflector_thickness= reflect_th)
    ab_single= single_res['absorbed_count']
    # Updated to use mixture
    (k_sing, dk_sing) = compute_k_factor_and_uncertainty(ab_single, N0, single_mix)
    print(f"\nSingle-run => fraction= {frac_percent:.2f}%, R_mtf= {rmtf_val:.3f}, absorbed= {ab_single}")
    print(f"  => k= {k_sing:.6f} ± {dk_sing:.6f}")
        # Uncertainty verification with repeated simulations
    verify_uncertainty = get_valid_input("\nVerify uncertainties with repeated simulations? [yes/no]: ",
                                         lambda x: x.lower() in ['yes', 'no'],
                                         "Please enter yes or no").lower()
    plot_k_vs_generation(single_mix, geo_in, reactor_size, N0)
    if verify_uncertainty == 'yes':
        num_repeats = int(get_valid_input("Enter number of repeats for uncertainty check: ",
                                           validate_positive_integer,
                                           "Must be a positive integer"))
        k_values = []
        dk_values = []
        print(f"Running {num_repeats} trials...")
        for i in range(num_repeats):
            # Run the same simulation again
            trial_result = simulate_first_generation(single_mix, geo_in, reactor_size, N0,
                                                     bins=20,
                                                     reflector=reflect_obj,
                                                     reflector_thickness=reflect_th)
            ab_trial = trial_result['absorbed_count']
            k_trial, dk_trial = compute_k_factor_and_uncertainty(ab_trial, N0, single_mix)
            k_values.append(k_trial)
            dk_values.append(dk_trial)
            print(f"Trial {i+1}: k = {k_trial:.4f} ± {dk_trial:.4f}")
    
        # Calculate empirical variance of k
        empirical_variance = np.var(k_values, ddof=1)
        average_dk_squared = np.mean([dk**2 for dk in dk_values])
    
        print("\nUncertainty Verification Results:")
        print(f"Empirical Variance of k: {empirical_variance:.6f}")
        print(f"Average of Computed dk^2: {average_dk_squared:.6f}")
        print(f"Ratio (Empirical / Average dk^2): {empirical_variance / average_dk_squared:.2f}")
    print("\nNow do the 2-Parameter iterative bounding box approach + final bounding box sweep.\n")

    # iterative search input
    nRand_s= get_valid_input("Number of random points per iteration: ",
                             validate_positive_integer,
                             "Must be int>0")
    nRand= int(nRand_s)
    nIter_s= get_valid_input("Number of bounding box iterations: ",
                             validate_positive_integer,"Must be int>0")
    nIter= int(nIter_s)
    fRange_in= get_valid_input("U-235% range [min,max], e.g. [0,100]: ",
                               lambda x:x.startswith('[') and x.endswith(']'),
                               "Must be [min,max]")
    spF= fRange_in.strip()[1:-1].split(',')
    fmin= float(spF[0]); fmax= float(spF[1])
    rRange_in= get_valid_input("R_mtf range [min,max], e.g. [0,10]: ",
                               lambda x:x.startswith('[') and x.endswith(']'),
                               "Must be [min,max]")
    spR= rRange_in.strip()[1:-1].split(',')
    rmin= float(spR[0]); rmax= float(spR[1])

    kTol_in= get_valid_input("Tolerance for k, e.g. 0.3: ",
                             lambda x: validate_float(x,0.0),
                             "Must be float>=0")
    kTol= float(kTol_in)

    nIterGen_in= get_valid_input("Neutrons per generation in iterative search: ",
                                 validate_positive_integer,
                                 "Must be int>0")
    nIterGen= int(nIterGen_in)

    nBoxSweep_in= get_valid_input("Neutrons per generation in final bounding box sweep: ",
                                  validate_positive_integer,"Must be int>0")
    nBoxSweep= int(nBoxSweep_in)

    maxGen_in= get_valid_input("Max generation for optimization (30 recommended): ",
                               validate_positive_integer,
                               "Must be int>0")
    maxGen= int(maxGen_in)

    # define sim_func => single generation => compute k
    def my_sim_func(fracDec, Rval, geometry, size,
                    u235_mat, u238_mat,
                    moderator_obj,
                    reflect_flag, reflect_obj, reflect_th,
                    n_neutrons, max_generations):
        # build mixture
        mm= ReactorMixture(fracDec, moderator_obj, Rval, u235_mat, u238_mat)
        # single generation => simulate_first_generation
        res_= simulate_first_generation(mm, geometry, size, n_neutrons,
                                        bins=20,
                                        reflector= reflect_obj if reflect_flag else None,
                                        reflector_thickness= reflect_th)
        ab_= res_['absorbed_count']
        # Use the full mixture object
        (k_, dk_) = compute_k_factor_and_uncertainty(ab_, n_neutrons, mm)  # <-- mm is ReactorMixture
        return (k_, dk_)

    # run the iterative approach
    iterative_2param_search(
        sim_func= my_sim_func,
        geometry= geo_in,
        size= reactor_size,
        u235_mat= u235_mat,
        u238_mat= u238_mat,
        moderator_obj= moderator_obj,
        reflect_flag= reflect_flag,
        reflect_obj= reflect_obj,
        reflect_th= reflect_th,
        n_points_per_iter= nRand,
        n_iter= nIter,
        range_u235_percent=(fmin,fmax),
        range_rmtf=(rmin,rmax),
        n_neutrons_each_gen= nIterGen,
        n_neutrons_box_sweep= nBoxSweep,
        max_generations= maxGen,
        k_tolerance= kTol
    )

    print("\nAll done with everything!\n")



if __name__=="__main__":
    main()
