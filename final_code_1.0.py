# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:33:28 2025

@author: 1218k
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:06:15 2025

@author: 1218k
"""

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
        val_in = input(prompt).strip()
        if validation_func(val_in):
            return val_in
        print(f"Invalid input! {error_msg}")

def validate_geometry(input_str):
    return input_str.lower() in ['sphere','cylinder']

def validate_float(input_str, min_val=None, max_val=None):
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
    return input_str.strip().lower() in ['h2o','d2o','graphite']

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
    theta= rand.uniform(0,2*np.pi)
    r= np.sqrt(rand.uniform(0,1))* radius
    x= r*np.cos(theta)
    y= r*np.sin(theta)
    z= rand.uniform(-height/2, height/2)
    return np.array([x,y,z])

def random_position_sphere_optimized(radius=1):
    v= rand.randn(3)
    v/= np.linalg.norm(v)
    r_= radius*(rand.uniform()**(1/3))
    return v*r_

class ReactorMaterial:
    def __init__(self, name, mat_type, density, molar_mass,
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name= name
        self.mat_type= mat_type
        self.density_gcc= density
        self.molar_mass_gmol= molar_mass
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
        return (self.density/self.molar_mass)* NA

# Lilley default
LILLEY_U235= ReactorMaterial("U235","fuel",18.7,235,10,680,579,2.42,0)
LILLEY_U238= ReactorMaterial("U238","fuel",18.9,238,8.3,2.72,0.0,0,0)

MODERATORS_DICT= {
    'h2o': ReactorMaterial("H2O","moderator",1.0,18.01,49.2,0.66,0.0,0,0.92),
    'd2o': ReactorMaterial("D2O","moderator",1.1,20.02,10.6,0.001,0.0,0,0.509),
    'graphite': ReactorMaterial("Graphite","moderator",1.6,12.01,4.7,0.0045,0.0,0,0.158)
}

def get_wikipedia_materials(neutron_model='thermal'):
    if neutron_model=='thermal':
        u235= ReactorMaterial("U235","fuel",18.7,235,10,(99+583),583,2.42,0)
        u238= ReactorMaterial("U238","fuel",18.9,238,9,(2+0.00002),0.00002,0,0)
    else:
        # fast
        u235= ReactorMaterial("U235","fuel",18.7,235,4,1.09,1,2.42,0)
        u238= ReactorMaterial("U238","fuel",18.9,238,5,0.37,0.3,0,0)
    return (u235,u238)


############################################
# PATCH 3 of N
############################################

class ReactorMixture:
    def __init__(self, fraction_U235, moderator, R_mtf, u235_material, u238_material):
        self.fraction_U235= fraction_U235
        self.moderator= moderator
        self.R_mtf= R_mtf
        self.u235= u235_material
        self.u238= u238_material

    @property
    def macroscopic_scattering(self):
        conv=1.0e6* NA*1.0e-28
        aU= self.fraction_U235
        B= aU+(1.-aU)+ self.R_mtf
        partU= (aU/B)* (self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_s_b
        part238= ((1.-aU)/B)* (self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_s_b
        partM= (self.R_mtf/B)* (self.moderator.density_gcc/self.moderator.molar_mass_gmol)* self.moderator.sigma_s_b
        return conv*(partU+part238+partM)

    @property
    def macroscopic_absorption(self):
        conv=1.0e6*NA*1.0e-28
        aU= self.fraction_U235
        B= aU+(1.-aU)+ self.R_mtf
        pu= (aU/B)* (self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_a_b
        p238= ((1.-aU)/B)* (self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_a_b
        pmod= (self.R_mtf/B)* (self.moderator.density_gcc/self.moderator.molar_mass_gmol)* self.moderator.sigma_a_b
        return conv*(pu+ p238+ pmod)

    @property
    def macroscopic_fission(self):
        conv=1.0e6*NA*1.0e-28
        aU= self.fraction_U235
        B= aU+(1.-aU)+ self.R_mtf
        fu= (aU/B)*(self.u235.density_gcc/self.u235.molar_mass_gmol)* self.u235.sigma_f_b
        f238= ((1.-aU)/B)*(self.u238.density_gcc/self.u238.molar_mass_gmol)* self.u238.sigma_f_b
        return conv*(fu+ f238)

    @property
    def fission_probability(self):
        fa= self.macroscopic_absorption
        ff= self.macroscopic_fission
        if fa<1e-30:return 0.0
        return ff/fa

    @property
    def scattering_probability(self):
        s= self.macroscopic_scattering
        a= self.macroscopic_absorption
        tot= s+a
        if tot<1e-30:return 0.0
        return s/tot

    @property
    def absorption_probability(self):
        s= self.macroscopic_scattering
        a= self.macroscopic_absorption
        tot= s+a
        if tot<1e-30:return 0.0
        return a/tot

    def compute_u235_mass_kg(self, geometry, size):
        import math
        aU= self.fraction_U235
        B= aU+(1.-aU)+ self.R_mtf
        if B<1e-30:return 0.0
        top= aU*self.u235.density_gcc+(1.-aU)*self.u238.density_gcc+ self.R_mtf*self.moderator.density_gcc
        eff_dens_gcc= top/B
        eff_dens_kg_m3= eff_dens_gcc*1e3
        if geometry=='sphere':
            R_= float(size)
            vol= (4./3.)* math.pi*(R_**3)
        else:
            R_,H_= float(size[0]), float(size[1])
            vol= math.pi*(R_**2)* H_
        return aU*(vol* eff_dens_kg_m3)

class ReflectorMaterial:
    def __init__(self, name, mat_type, density, molar_mass,
                 sigma_s_b, sigma_a_b, sigma_f_b=0.0,
                 nu=0.0, xi=0.0):
        self.name= name
        self.mat_type= mat_type
        self.density_gcc= density
        self.molar_mass_gmol= molar_mass
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

    @property
    def macroscopic_scattering(self):
        return self.number_density*self.sigma_s

    @property
    def macroscopic_absorption(self):
        return self.number_density*self.sigma_a

    @property
    def reflection_probability(self):
        tot= self.macroscopic_scattering+ self.macroscopic_absorption
        if tot<1e-30:return 0.0
        return self.macroscopic_scattering/tot

def distance_to_collision(mixture):
    tot= mixture.macroscopic_scattering+ mixture.macroscopic_absorption
    if tot<1e-30:
        return 1e10
    return -np.log(rand.rand())/ tot


############################################
# PATCH 4 of N
############################################

def random_scatter_direction():
    costh= 2.*rand.rand()-1.
    phi= 2.*np.pi* rand.rand()
    sinth= np.sqrt(1.- costh*costh)
    return np.array([sinth*np.cos(phi), sinth*np.sin(phi), costh])

def simulate_first_generation(mixture, geometry, size, N0,
                              bins=20, reflector=None, reflector_thickness=0.0):
    """
    Single-run approach for 1st generation collisions,
    forcibly storing 'reflector_interaction_count'.
    """
    # (unchanged code)
    # ...
    import math
    scatter_r=[]
    absorb_r=[]
    fission_r=[]
    reflect_r=[]
    leak_count=0
    reflector_interaction_count=0

    P_scat = mixture.scattering_probability
    P_fis = mixture.fission_probability

    if geometry=='sphere':
        R_core= float(size)
        R_refl= R_core+ reflector_thickness
    else:
        R_core, H_core = float(size[0]), float(size[1])
        R_refl = R_core+ reflector_thickness
        H_refl = H_core+ 2.*reflector_thickness

    P_reflect = 0.0
    if reflector:
        P_reflect = reflector.reflection_probability

    positions= np.zeros((N0,3))
    for i in range(N0):
        if geometry=='sphere':
            positions[i,:]= random_position_sphere_optimized(R_core)
        else:
            positions[i,:]= random_position_cylinder(R_core,H_core)

    is_active= np.ones(N0,dtype=bool)
    absorbed_positions=[]
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
                    in_core= (rr<= R_core**2) and (abs(newp[2])<= H_core/2)
                    in_ref= (reflector is not None) and (rr<= R_refl**2) and (abs(newp[2])<= H_refl/2) and (not in_core)
                if in_core:
                    if rand.rand()< P_scat:
                        if geometry=='sphere':
                            scatter_r.append(math.sqrt(dist2))
                        else:
                            scatter_r.append(math.sqrt(rr))
                        pos= newp
                    else:
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
                    leak_count+=1
                    is_active[idx]=False
                    break
        idxs= np.where(is_active)[0]
        if len(idxs)==0:
            break

    ab_count= len(absorbed_positions)
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
        import math
        ring_area= math.pi*(bins_r[1:]**2- bins_r[:-1]**2)
        H_core= size[1]
        shellv= ring_area*H_core

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
                if np.sum(newp**2)> R_core**2:
                    leak_count+=1
                    is_active[idx]=False
                    continue
            else:
                rr= newp[0]**2+ newp[1]**2
                if rr> R_core**2 or abs(newp[2])> (H_core/2):
                    leak_count+=1
                    is_active[idx]=False
                    continue
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
    if n_init<=0:
        return (0.0,0.0)
    p_fission= mixture.fission_probability
    nu= mixture.u235.nu
    p_abs= n_abs/ n_init
    k_= p_abs*p_fission*nu
    var_p= p_abs*(1.-p_abs)/ n_init
    dk_= p_fission* nu* np.sqrt(var_p)
    return (k_, dk_)

def plot_k_vs_generation(mixture, geometry, size, N0, max_generations=20,
                         reflector=None, reflector_thickness=0.0):
    kvals=[]
    dkvals=[]
    gens=[]
    first_= simulate_first_generation(mixture, geometry, size, N0,
                                      bins=20,
                                      reflector=reflector,
                                      reflector_thickness=reflector_thickness)
    nab= first_['absorbed_count']
    (k1,dk1)= compute_k_factor_and_uncertainty(nab, N0, mixture)
    kvals.append(k1)
    dkvals.append(dk1)
    gens.append(1)
    pos= first_['absorbed_positions']
    curr_init= nab

    for g_ in range(2, max_generations+1):
        if curr_init<=0:
            print("No more active neutrons at generation",g_)
            break
        res_= simulate_generation(mixture, geometry, size, pos,
                                  reflector= reflector,
                                  reflector_thickness= reflector_thickness)
        nab2= res_['absorbed_count']
        (kk, dkk)= compute_k_factor_and_uncertainty(nab2, curr_init, mixture)
        kvals.append(kk)
        dkvals.append(dkk)
        gens.append(g_)

        pos= res_['absorbed_positions']
        curr_init= nab2

    plt.figure(figsize=(8,6), dpi=1000)
    plt.errorbar(gens, kvals, yerr=dkvals, fmt='-o', capsize=5)
    plt.axhline(1.0, color='red', linestyle='--')
    plt.xlabel("Generation #")
    plt.ylabel("k-value")
    plt.title("k-value vs Generation")
    plt.grid(True)
    plt.show()


############################################
# PATCH 5 of N (Modified to store final_k + final_dk)
############################################

def final_bounding_box_sweep_multi_gen(geometry, size,
                                       fmin_dec, fmax_dec,
                                       rmin, rmax,
                                       steps, mixture_builder,
                                       max_generations,
                                       n_neut_sweep,
                                       reflect_obj=None,
                                       reflect_th=0.0):
    """
    We'll do a grid => multi-generation => store final param => produce a param_list,
    also store 'posterior_samples' => (f,r,k,dk) that converge with |k-1|<0.1 so we keep the final DK too.
    We'll pick best param => fewest generations.
    """
    import numpy as np

    f_vals= np.linspace(fmin_dec, fmax_dec, steps)
    r_vals= np.linspace(rmin, rmax, steps)
    K_map= np.zeros((steps,steps), dtype=float)

    best_gen= 999999
    best_param= None  # store (f, r, final_k, final_dk, gen)
    param_list= []
    posterior_samples= []  # store (f, r, final_k, final_dk)

    for i_f, ff in enumerate(f_vals):
        for j_r, rr in enumerate(r_vals):
            # multi-generation approach for each (ff, rr)
            mix= mixture_builder(ff, rr)
            pos= np.zeros((n_neut_sweep,3))
            res= simulate_first_generation(mix, geometry, size, n_neut_sweep,
                                           bins=20,
                                           reflector= reflect_obj,
                                           reflector_thickness= reflect_th)
            ab= res['absorbed_count']
            (k1, dk1)= compute_k_factor_and_uncertainty(ab, n_neut_sweep, mix)
            final_k= k1
            final_dk= dk1
            gen_count= max_generations+1

            if abs(k1-1.)<0.1:
                gen_count=1
                K_map[i_f,j_r]= k1
            else:
                current_init= ab
                pos= res['absorbed_positions']
                for g_ in range(2, max_generations+1):
                    if current_init<=0:
                        break
                    simres= simulate_generation(mix, geometry, size, pos,
                                                reflector= reflect_obj,
                                                reflector_thickness= reflect_th)
                    nab= simres['absorbed_count']
                    (kg, dkg)= compute_k_factor_and_uncertainty(nab, current_init, mix)
                    final_k= kg
                    final_dk= dkg
                    if abs(kg-1.)<0.1:
                        gen_count= g_
                        K_map[i_f,j_r]= kg
                        break
                    pos= simres['absorbed_positions']
                    current_init= nab

            # store in param_list => (f, r, final_k, final_dk, gen_count)
            param_list.append( (ff, rr, final_k, final_dk, gen_count) )

            if gen_count< best_gen:
                best_gen= gen_count
                best_param= (ff, rr, final_k, final_dk, gen_count)

            # If we have converged => store in posterior => (f, r, final_k, final_dk)
            if gen_count<= max_generations:
                posterior_samples.append( (ff, rr, final_k, final_dk) )

    return (K_map, best_param, param_list, posterior_samples)

def iterative_2param_search(sim_func, geometry, size,
                            u235_mat, u238_mat, moderator_obj,
                            reflect_flag, reflect_obj, reflect_th,
                            n_points_per_iter, n_iter,
                            range_u235_percent, range_rmtf,
                            n_neutrons_each_gen,
                            n_neutrons_box_sweep,
                            max_generations=30,
                            k_tolerance=0.3):
    import numpy as np
    import matplotlib.pyplot as plt

    candidates_size= n_points_per_iter* n_iter
    candidates= np.zeros((candidates_size,4), dtype=float)
    idx_global=0

    fminP, fmaxP= range_u235_percent
    rmin_, rmax_= range_rmtf
    fmin= fminP/100.
    fmax= fmaxP/100.

    current_fmin= fmin
    current_fmax= fmax
    current_rmin= rmin_
    current_rmax= rmax_

    # iteration prints
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

    def mixture_builder(ff, rr):
        return ReactorMixture(ff, moderator_obj, rr, u235_mat, u238_mat)

    steps= 20
    (K_map, best_param, param_list, posterior_samples)= final_bounding_box_sweep_multi_gen(
        geometry, size,
        current_fmin, current_fmax,
        current_rmin, current_rmax,
        steps, mixture_builder,
        max_generations,
        n_neutrons_box_sweep,
        reflect_obj= reflect_obj if reflect_flag else None,
        reflect_th= reflect_th
    )

    # Plot the final bounding box results
    fvals_ = np.linspace(current_fmin, current_fmax, steps)
    rvals_ = np.linspace(current_rmin, current_rmax, steps)
    fig, ax= plt.subplots()
    cf= ax.contourf(fvals_*100., rvals_, K_map.T, levels=50, cmap='plasma')
    cbar= fig.colorbar(cf, ax=ax)
    cbar.set_label("k-value")
    diff_map= K_map-1.
    c2= ax.contour(fvals_*100., rvals_, diff_map.T, levels=[-0.1,0.1], colors='black')
    try:
        for c_ in c2.collections:
            c_.set_label("Enclosed region where |k-1|<0.1")
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15))
    except:
        pass

    plt.figure(figsize=(8,6), dpi=1000)
    ax.set_xlabel("U-235 concentration(%)")
    ax.set_ylabel("R_mtf")
    ax.set_title("Final bounding box multi-gen => k-value & region |k-1|<0.1")
    plt.tight_layout()
    plt.show()

    if best_param is None:
        print("\nNo param found that converged within maxGenerations.\n")
        return

    # best_param => (f, r, final_k, final_dk, gen_count)
    fb, rb, kb, dkb, gb= best_param
    print("\n=== Best param by FEWEST generations => (|k-1|<0.1) ===")
    print(f"  U-235 fraction= {fb*100:.2f}%, R_mtf= {rb:.3f}, convergedGen= {gb}")
    print(f"  finalK={kb:.6f} ± {dkb:.6f}")
    best_mix= mixture_builder(fb, rb)
    mass_best= best_mix.compute_u235_mass_kg(geometry, size)
    print(f"  geometry= {geometry}, size= {size}, => massOfU235= {mass_best:.3f} kg\n")

    # param with minimal |k-1| => we also want finalK ± finalDK
    best_diff= 999999
    param_knear= None  # (f, r, k, dk, gen)
    for (ff, rr, kk, dkk, gg) in param_list:
        dd= abs(kk-1.)
        if dd< best_diff:
            best_diff= dd
            param_knear= (ff, rr, kk, dkk, gg)

    if param_knear is not None:
        fk, rk, kk2, dk2, gg2= param_knear
        print("=== Best param by MINIMIZING |k-1| among final bounding box grid ===")
        print(f"  fraction= {fk*100:.2f}%, R_mtf= {rk:.3f}, finalK= {kk2:.6f} ± {dk2:.6f}, gen= {gg2}")
        mix_k= mixture_builder(fk, rk)
        mass_k= mix_k.compute_u235_mass_kg(geometry, size)
        print(f"  => massOfU235= {mass_k:.3f} kg\n")
        print(">>> Comparison between the two priorities:\n"
              f"    By fewest generations => fraction= {fb*100:.2f}%, R_mtf= {rb:.3f}, k= {kb:.3f}±{dkb:.3f}, gen= {gb}\n"
              f"    By minimal |k-1|      => fraction= {fk*100:.2f}%, R_mtf= {rk:.3f}, k= {kk2:.3f}±{dk2:.3f}, gen= {gg2}\n")
    else:
        print("No param found to do minimal |k-1| approach.\n")
        return

    # pass for |k-1|<0.001
    strict_list= []
    for (ff, rr, kk, dkk, gg) in param_list:
        if abs(kk-1.)< 0.001:
            strict_list.append( (ff, rr, kk, dkk, gg) )
    if len(strict_list)==0:
        print("No param satisfies |k-1|<0.001.\n")
    else:
        best_g2=999999
        bestp2= None
        for (ff, rr, kk, dkk, gg) in strict_list:
            if gg< best_g2:
                best_g2= gg
                bestp2= (ff, rr, kk, dkk, gg)
        if bestp2:
            fs2, rs2, ks2, dks2, gs2= bestp2
            print("=== Among those with |k-1|<0.001, param with FEWEST generations is:")
            print(f"  fraction= {fs2*100:.2f}%, R_mtf= {rs2:.3f}, k= {ks2:.6f} ± {dks2:.6f}, gen= {gs2}")
            mix_strict= mixture_builder(fs2,rs2)
            mass_s= mix_strict.compute_u235_mass_kg(geometry, size)
            print(f"  => massOfU235= {mass_s:.3f} kg\n")
            print(">>> Comparison with the prior best param:\n"
                  f"    prior => fraction= {fb*100:.2f}%, R_mtf= {rb:.3f}, k= {kb:.3f}±{dkb:.3f}, gen= {gb}\n"
                  f"    strict => fraction= {fs2*100:.2f}%, R_mtf= {rs2:.3f}, k= {ks2:.3f}±{dks2:.3f}, gen= {gs2}\n")

    # Posterior-based uncertainties => all that converge => store (f, r, k, dk)
    # we do not have separate "k" usage for the posterior mass or R_mtf, but we can store them
    # to compute the sample-based Rmtf & mass. We do not do k's uncertainty from the posterior
    # but you can adapt if you want. We'll keep your old approach of R_mtf & mass
    if len(posterior_samples)<2:
        print("\nNo or only 1 convergent sample => cannot do posterior-based R_mtf & mass uncertainties.\n")
        return

    # gather R_mtf, mass
    rmtf_arr= []
    mass_arr= []
    for (ff, rr, kk, dkk) in posterior_samples:
        mm_= mixture_builder(ff, rr)
        mass_= mm_.compute_u235_mass_kg(geometry, size)
        rmtf_arr.append(rr)
        mass_arr.append(mass_)

    rmtf_mean= np.mean(rmtf_arr)
    rmtf_std= np.std(rmtf_arr, ddof=1)
    mass_mean= np.mean(mass_arr)
    mass_std= np.std(mass_arr, ddof=1)

    print("\n=== Posterior (Sample-Based) Uncertainties ===")
    print(f"  R_mtf= {rmtf_mean:.3f} ± {rmtf_std:.3f}")
    print(f"  massOfU235= {mass_mean:.3f} ± {mass_std:.3f} kg")

############################################
# PATCH 6: main()
############################################

def main():
    print("=== Full code (with finalK±dk) for both prioritizations, posterior uncertainties, etc. ===")

    geo_in= get_valid_input("Geometry [sphere/cylinder]: ", validate_geometry, "Must be sphere/cylinder").lower()
    if geo_in=='sphere':
        rad_s= get_valid_input("Enter sphere radius (m): ", lambda x: validate_size(x,'sphere'), "Invalid radius")
        reactor_size= float(rad_s)
    else:
        rad_s= get_valid_input("Enter cylinder [radius,height]: ",
                               lambda x: validate_size(x,'cylinder'),
                               "Invalid cylinder size")
        prt= rad_s.strip()[1:-1].split(',')
        reactor_size= [float(prt[0]), float(prt[1])]

    ds= get_valid_input("Dataset [Lilley/Wikipedia]: ",
                        validate_dataset,"Must be Lilley/Wikipedia").lower()
    if ds=='wikipedia':
        nm= get_valid_input("Neutron model [Thermal/Fast]: ",
                            validate_neutron_model,
                            "Must be thermal/fast").lower()
        u235_mat, u238_mat= get_wikipedia_materials(nm)
    else:
        u235_mat, u238_mat= LILLEY_U235, LILLEY_U238

    mod_= get_valid_input("Moderator [H2O/D2O/Graphite]: ",
                          validate_moderator,
                          "Must be H2O,D2O,Graphite").lower()
    moderator_obj= MODERATORS_DICT[mod_]

    frac_s= get_valid_input("U-235 concentration [0..100%] for single-run: ",
                            lambda x: validate_float(x,0,100),
                            "Must be in [0..100]")
    frac_dec= float(frac_s)/100.

    rmtf_s= get_valid_input("R_mtf(>=0) for single-run: ",
                            lambda x: validate_float(x,0.0),
                            "Must be >=0")
    rmtf_val= float(rmtf_s)

    refl_s= get_valid_input("Use reflector? [yes/no]: ",
                            lambda x:x.lower() in ['yes','no'],
                            "Must be yes/no").lower()
    reflect_flag= (refl_s=='yes')
    reflect_obj=None
    reflect_th=0.0
    if reflect_flag:
        thick_in= get_valid_input("Reflector thickness (m): ",
                                  lambda x: validate_float(x,0.0),
                                  "Must be >=0")
        reflect_th= float(thick_in)
        rmat_s= get_valid_input("Reflector material [Graphite/Beryllium]: ",
                                lambda x:x.lower() in ['graphite','beryllium'],
                                "Must be 'graphite' or 'beryllium'").lower()
        if rmat_s=='graphite':
            reflect_obj= ReactorMaterial("Graphite","reflector",1.6,12.01,4.7,0.0045,0.0,0,0)
        else:
            reflect_obj= ReactorMaterial("Beryllium","reflector",1.85,9.01,6.0,0.001,0.0,0,0)

    n0_s= get_valid_input("Number of neutrons for single-run test: ",
                          validate_positive_integer,"Must be int>0")
    N0= int(n0_s)

    # single-run
    single_mix= ReactorMixture(frac_dec, moderator_obj, rmtf_val, u235_mat, u238_mat)
    first_res= simulate_first_generation(single_mix, geo_in, reactor_size, N0,
                                         bins=20,
                                         reflector= reflect_obj,
                                         reflector_thickness= reflect_th)
    ab_single= first_res['absorbed_count']
    (k_sing, dk_sing)= compute_k_factor_and_uncertainty(ab_single, N0, single_mix)
    print(f"\nSingle-run => fraction= {frac_dec*100:.2f}%, R_mtf= {rmtf_val:.3f}, absorbed= {ab_single}")
    print(f"  => k= {k_sing:.6f} ± {dk_sing:.6f}")

    # optional generation plot
    ans_= get_valid_input("\nPlot k vs generation? [yes/no]: ",
                          lambda x:x.lower() in ['yes','no'],
                          "Must be yes/no").lower()
    if ans_=='yes':
        plot_k_vs_generation(single_mix, geo_in, reactor_size, N0)

    print("\nNow do the 2-Parameter iterative bounding box approach.\n")

    # iterative search input
    nRand_s= get_valid_input("Number of random points per iteration: ", validate_positive_integer,"Must be>0")
    nIter_s= get_valid_input("Number of bounding box iterations: ", validate_positive_integer,"Must be>0")
    nRand= int(nRand_s)
    nIter= int(nIter_s)
    fRange_s= get_valid_input("U-235% range [min,max], e.g. [0,100]: ",
                              lambda x:x.startswith('[') and x.endswith(']'),
                              "Must be [min,max]")
    spF= fRange_s.strip()[1:-1].split(',')
    fmin_= float(spF[0]); fmax_= float(spF[1])

    rRange_s= get_valid_input("R_mtf range [min,max], e.g. [0,10]: ",
                              lambda x:x.startswith('[') and x.endswith(']'),
                              "Must be [min,max]")
    spR= rRange_s.strip()[1:-1].split(',')
    rmin_= float(spR[0]); rmax_= float(spR[1])

    kTol_s= get_valid_input("Tolerance for k (e.g. 0.3): ",
                            lambda x: validate_float(x,0.0),
                            "Must be >=0")
    kTol= float(kTol_s)
    nIterGen_s= get_valid_input("Neutrons per generation in iterative search: ",
                                validate_positive_integer,"Must be>0")
    nIterGen= int(nIterGen_s)
    nBoxSweep_s= get_valid_input("Neutrons per generation in final bounding box sweep: ",
                                 validate_positive_integer,"Must be>0")
    nBoxSweep= int(nBoxSweep_s)
    maxGen_s= get_valid_input("Max generation for optimization(30 rec): ",
                              validate_positive_integer,"Must be>0")
    maxGen= int(maxGen_s)

    def my_sim_func(fracDec, Rval, geometry, size,
                    u235_mat, u238_mat,
                    moderator_obj,
                    reflect_flag, reflect_obj, reflect_th,
                    n_neutrons, max_generations):
        mm= ReactorMixture(fracDec, moderator_obj, Rval, u235_mat, u238_mat)
        fres= simulate_first_generation(mm, geometry, size, n_neutrons,
                                        bins=20,
                                        reflector= reflect_obj if reflect_flag else None,
                                        reflector_thickness= reflect_th)
        ab_ = fres['absorbed_count']
        (k_, dk_)= compute_k_factor_and_uncertainty(ab_, n_neutrons, mm)
        return (k_, dk_)

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
        range_u235_percent= (fmin_, fmax_),
        range_rmtf= (rmin_, rmax_),
        n_neutrons_each_gen= nIterGen,
        n_neutrons_box_sweep= nBoxSweep,
        max_generations= maxGen,
        k_tolerance= kTol
    )

    print("\nAll done with everything!\n")

if __name__=="__main__":
    main()
