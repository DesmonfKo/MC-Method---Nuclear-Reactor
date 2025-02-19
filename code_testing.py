import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import scipy as sp
plt.rcParams['text.usetex'] = True
'''
x=5
y=4
print(x+y)
# test for branching
# Try this
'''
def generalisation_N_dimension(dimension=2, radius=1, sample_size_N=10000):
    '''
    This function calculate abritary positive integer dimension of
    hypervolume sphere encased inside a unit confinement, using MC method.

    Important Info:
    -Unit Sphere -> Unit Radius
    -Concrete mathematics of Hypersphere:https://en.wikipedia.org/wiki/N-sphere

    Parameters
    ----------
    dimension : INT, optional
        This will be the positive integer dimension of the hypervolume. The
        default is 2.
    
    radius : Float, optional
        Radius of the sphere, default is 1 of arbitrary unit.
    
    sample_size_N : INT, optional
        This is the size of the sampling population. The default is 10000.

    Returns
    -------
    None

    '''
    total_point_sample = rand.exponential(0.01, size=(sample_size_N, dimension))
    # The above variable will output an array, we need to filter out the
    # summation of the elements in each row which is greater than 1

    # First square all the elements of the sample;
    # Then compute sum of squares for each sample (vectorized),
    # summation over every column of each row using the axis=1
    squared_samples = total_point_sample ** 2
    dummy_holding_list = np.sum(squared_samples, axis=1)

    # Find indices where sum of squares ≤ 1. The [0] takes out the first
    # element in the output of np.where(a tuple), which is a list of indices.
    in_circle = np.where(dummy_holding_list <= (radius)**2)[0]

    # The hypervolume sphere has radius R=1, with the exact generic volume
    # V_dim = pi**(dim/2) / gamma_func(n/2+1); This serves as a comparison
    # Then we can use the volume using the ratio of points as previouly done,
    # and that the encased confinement having a volume of (2*R)**N.
    estimation_of_hypersphere_volume = (2**dimension) \
        * len(in_circle) / sample_size_N

    exact_value_of_hypersphere_volume = (np.pi**(dimension/2))*(radius**dimension) / \
        sp.special.gamma(dimension/2+1)

    print(f'Estimation of hypervolume in {dimension}D =\
          {estimation_of_hypersphere_volume:.8f}')
    print(f'Exact value of hypervolume in {dimension}D =\
          {exact_value_of_hypersphere_volume:.8f}')

    return None


Three_D_sampling = generalisation_N_dimension(3, 5, 100)