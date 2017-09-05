from solcore.structure import Layer, Junction
import solcore.analytic_solar_cells as ASC

import numpy as np
import types
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def solve_beer_lambert(solar_cell, options):
    wl = options.wavelength
    wl_m = wl * 1e-9
    fraction = np.ones(wl.shape)
    # absorbed = np.zeros(wl.shape)

    # We include the shadowing losses
    if hasattr(solar_cell, 'shading'):
        fraction *= (1 - solar_cell.shading)

    # And the reflexion losses
    if hasattr(solar_cell, 'reflectivity') and solar_cell.reflectivity is not None:
        solar_cell.reflected = solar_cell.reflectivity(wl_m)
        fraction *= (1 - solar_cell.reflected)
    else:
        solar_cell.reflected = np.zeros(fraction.shape)

    # Now we calculate the absorbed and transmitted light. We first get all the relevant parameters from the objects
    widths = []
    alphas = []
    offset = 0
    for j, layer_object in enumerate(solar_cell):

        # Attenuation due to absorption in the AR coatings or any layer in the front that is not part of the junction
        if type(layer_object) is Layer:
            widths.append(layer_object.width)
            alphas.append(layer_object.material.alpha(wl_m))

        # For each junction, we calculate the absorbance
        elif type(layer_object) is Junction:

            if solar_cell[j].kind in ['PDD', 'DA']:
                junction_width = 0
                for i, layer in enumerate(layer_object):
                    junction_width += layer.width
                    widths.append(layer.width)
                    alphas.append(layer.material.alpha(wl_m))

                solar_cell[j].width = junction_width

            elif solar_cell[j].kind == '2D':
                print('Warning: A junction of kind "2D" found. Junction ignored in the optics calculation!')

            elif solar_cell[j].kind == 'DB':
                # DB junctions do not often have a width and an absorption coefficient so we set an arbitrary width
                # and back calculate the absorption coefficient from the absorptance, which needs to be provided
                ASC.absorptance_detailed_balance(solar_cell[j])

                if hasattr(layer_object, 'width'):
                    w = layer_object.width
                else:
                    w = 1e-6  # 1 µm
                    solar_cell[j].width = w

                def alf(x):
                    return -1 / w * np.log(np.maximum(1 - layer_object.absorptance(x), 1e-3))

                solar_cell[j].alpha = alf

                widths.append(w)
                alphas.append(alf(wl))

            else:
                raise ValueError(
                    'ERROR in "solar_cell_solver":\n\tJunction {} has an invalid "kind". It must be "PDD", "DA", "2D" or "DB".')

            solar_cell[j].offset = offset
            offset += layer_object.width

    # With all this information, we are ready to calculate the absorbed light
    diff_absorption, transmitted, all_absorbed = calculate_absorption_beer_lambert(widths, alphas, fraction)

    # Each building block (layer or junction) needs to have access to the absorbed light in its region.
    # We update each object with that information.
    for j in range(len(solar_cell)):
        solar_cell[j].diff_absorption = diff_absorption
        solar_cell[j].absorbed = types.MethodType(absorbed, solar_cell[j])

    solar_cell.transmitted = transmitted
    solar_cell.absorbed = all_absorbed


def absorbed(self, z):
    out = self.diff_absorption(self.offset + z).T * (z < self.width)
    return out.T


def calculate_absorption_beer_lambert(widths, alphas, fraction):
    # Number of spectral elements
    N = len(alphas[0])

    cum_widths = np.cumsum([0] + widths)
    widths = np.array(widths)

    # At any given position, the absorption per unit length is alpha * exp(-alpha*z) but we have to remove all light
    # absorbed above it:
    OD = [np.zeros(N)]
    for i in range(len(widths)):
        OD.append(OD[i] + alphas[i] * widths[i])

    # After getting al the alphas and widths, we need to create a function that takes as arguments the depth z and
    # returns the differential fraction of absorbed light at that position.
    def diff_absorption(z):
        # First we find to which layer a given z belong
        idx = cum_widths.searchsorted(z) - 1
        idx = np.maximum(idx, 0)

        # Now the distance to the begining of that layer
        z_local = z - cum_widths[idx]

        # And finally, we calculate a 2D array of the absorption per unit length vs position and wavelength
        output = np.zeros((len(z), N))
        inside = np.sum(z < cum_widths[-1])
        for k in range(inside):
            loc = idx[k]
            output[k] = alphas[loc] * np.exp(-OD[loc] - alphas[loc] * z_local[k])

        return output * fraction

    all_absorbed = (1 - np.exp(-OD[-1])) * fraction
    trans = np.exp(-OD[-1]) * fraction

    return diff_absorption, trans, all_absorbed


def calculate_absorptance_PDD_and_DA(junction, wl, fraction):
    widths = []
    alphas = []
    for i, layer in enumerate(junction):
        try:
            widths.append(layer.width)
            alphas.append(layer.material.alpha(wl))
        except Exception as err:
            # We are, most likely, in the presence of a QW structure. It needs to be already processed,
            # otherwise the calculation will fail.
            # We skip this part, for now.
            print(err.args)
            raise

    cum_widths = np.cumsum([0] + widths[:-1])
    widths = np.array(widths)

    # At any given position, the absorption per unit length is alpha * exp(alpha*z) but we have to remove all light
    # absorbed above it:
    OD = [np.zeros(alphas[0].shape)]
    for i in range(len(widths) - 1):
        OD.append(OD[i] + alphas[i] * widths[i])

    # After getting al the alphas and widths, we need to create a function that takes as arguments the depth z and
    # returns the differential fraction of absorbed light at that position.
    def diff_absorption(z):

        # First we find to which layer a given z belong
        idx = cum_widths.searchsorted(z) - 1
        idx = np.maximum(idx, 0)

        # Now the distance to the begining of that layer
        z_local = z - cum_widths[idx]

        # If we go beyond the limit of the function, the absorption must be zero, so:
        too_far = 1 * (z < (cum_widths[-1] + widths[-1]))

        # And finally, we calculate a 2D array of the absorption per unit lenght vs position and wavelength
        output = np.zeros((len(z), len(wl)))
        for i in range(len(z) - 1):
            loc = idx[i]
            output[i] = alphas[loc] * np.exp(-OD[loc] - alphas[loc] * z_local[i])
            output[i] = output[i] * too_far[i]

        return output * fraction

    trans = np.exp(-OD[-1] - alphas[-1] * widths[-1])

    return diff_absorption, trans


def calculate_absorptance_DB(junction, wl, fraction):
    # First, we calculate the absorptance of the cell
    ASC.absorptance_detailed_balance(junction)

    def absorption(new_wl):
        # The fraction of light absorbed in the junction
        f = np.interp(new_wl, wl, fraction, left=fraction[0], right=fraction[-1])
        return junction.absorptance(new_wl) * f

    trans = absorption(wl)

    return absorption, trans
