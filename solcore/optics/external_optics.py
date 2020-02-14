from solcore.structure import Layer, Junction, TunnelJunction

import numpy as np
import types
from .optics_utilities import calculate_absorption, absorbed



def solve_external_optics(solar_cell, options):
    """ Uses the reflection, transmission and absorption of a solar cell from
    external data, preparing the structure for further calculations. The
    external data that must be provided in the solar cell definition are:

    external_reflected: a function that provides the fraction of reflected light 
                        at the specified wavelengths (in m)
    external_absorbed: a function that provides an array with the differential 
                        absorption at a depth Z (in m) at the specified wavelengths.

    Note that both are functions of a single variable - wavelength in the first
    case and position in the second - and that the latter provides as output an
    array at the wavelengths indicated in the options.

    Or either:

    RAT: a dictionary containing  the R, A, T numpy arrays {'R':R, 'A':A', T':T}
    absorbed_matrix: a matrix containing the absorption as A[wl,z]. 

    Note that both are based on arrays, therefore the user should provide them with
    the same spectral, and z spacing as the simulation. It is indicated in the options.

    :param solar_cell:
    :param options:
    :return:
    """

    wl = options.wavelength
    # We include the shadowing losses
    initial = (1 - solar_cell.shading) if hasattr(solar_cell, 'shading') else 1
    if hasattr(solar_cell, 'absorbed_matrix'):
        solve_external_optics_matrix(solar_cell, options)
        return 

    # We try to get the external attributes
    try:
        solar_cell.reflected = solar_cell.external_reflected * initial
        diff_absorption = solar_cell.external_absorbed * initial
    except AttributeError as err:
        raise err

    # We calculate the total amount of light absorbed in the solar cell, 
    #integrating over its whole thickness with a step of 1 nm
    # or using a user define mesh
    all_z = options.position
    all_absorbed = np.trapz(diff_absorption(all_z), all_z)

    # Each building block (layer or junction) needs to have access to the absorbed light in its region.
    # We update each object with that information.
    for j in range(len(solar_cell)):
        solar_cell[j].diff_absorption = diff_absorption
        solar_cell[j].absorbed = types.MethodType(absorbed, solar_cell[j])

    solar_cell.transmitted = (1 - solar_cell.external_reflected - all_absorbed) * initial
    solar_cell.absorbed = all_absorbed * initial


def solve_external_optics_matrix(solar_cell, options):
    """ Uses the reflection, transmission and absorption of a solar cell from
    external data, preparing the structure for further calculations. The
    external data that must be provided in the solar cell class as attributes
    are:

    RAT: a dictionary containing  the R, T, A numpy arrays {'R':R, 'T':T, 'A':A}
    absorbed_matrix: a matrix containing the absorption as A[wl,z]. 

    Note that both are based on arrays, therefore the user should provide them with
    the same spectral, and z spacing as the simulation indicated in the options.

    :param solar_cell: 
    :param options: 
    :return:
    """
    wl = options.wavelength
    # We include the shadowing losses
    initial = (1 - solar_cell.shading) if hasattr(solar_cell, 'shading') else 1
    try:
        RAT = solar_cell.RAT 
        absorbed_matrix = solar_cell.absorbed_matrix 
    except AttributeError as err:
        raise err

    position = options.position * 1e9
    # delta = (position[1] - position[0])
    delta = 1./np.diff(position)
    absorption_power = absorbed_matrix[:,:] * delta # Errors??
    output = {'position':   position, 
              'absorption': absorption_power,
             }
    diff_absorption, all_absorbed = calculate_absorption(output)

    for j in range(len(solar_cell)):
        solar_cell[j].diff_absorption = diff_absorption
        solar_cell[j].absorbed = types.MethodType(absorbed, solar_cell[j])
        
    R, A, T = RAT['R'], RAT['A'], RAT['T']
    
    solar_cell.reflected = R * initial
    solar_cell.absorbed = A * initial
    solar_cell.transmitted = T * initial
    return 0
