#!/usr/bin/env python3
"""Testing the external asborber using a matrix
"""
import numpy as np
import scipy as sp

from solcore.solar_cell import SolarCell
from solcore.structure import Layer, Junction, TunnelJunction
from solcore import material
from solcore.light_source import LightSource
from solcore.solar_cell_solver import solar_cell_solver

## We can use the derivative in this case but we are going 
## use the absorption, as typically optical softwares will give it.
def single_pass(material, wl, d, I0=1.):
    """It returns the absorption using the Beer-Lambert law,
    to change the number of passes d=d0*N
    Args:
        material (solcore.material):  Complex refractive index (n + 1j * K)
        wl (np.array): Wavelength (wl)
        d (np.float): Material thickness (microns)
        I0 (np.float/np.array): Intensity at the interface.
    Returns:
        np.array
    """
    K = material.k(wl)
    alpha = 4.0 * np.pi * K / wl
    return I0 * (1. - np.exp(-alpha * d))

def convert_cell_to_layers(solar_cell):
    """ It converts a solar_cell object to a list of layers.
    Args:
        solar_cell (solcore.SolarCell)
    Return:
        [(solcore.Layer), ...]
    """
    all_layers = []
    for j, layer_object in enumerate(solar_cell):
        if type(layer_object) is Layer:
            all_layers.append(layer_object)
        elif type(layer_object) in [TunnelJunction, Junction]:
            for i, layer in enumerate(layer_object):
                all_layers.append(layer)
    return all_layers



## Define a simple GaAs cell

nm = 1e-9
T = 298
NP = 128
meshpoints = -400
fine = 1 *nm
ultrafine = 0.1 *nm
coarse = 20 * nm

wl = np.linspace(300, 1000, NP)*1e-9
window_top = material('AlInP')(T=T, Nd=1e24, Al=0.53, 
                              electron_mobility=1e-2, hole_mobility=1e-4)
n_GaAs = material('GaAs')(T=T, Nd=1e24,)
p_GaAs = material('GaAs')(T=T, Na=3e23,)
i_GaAs = material('GaAs')(T=T)
bsf_bottom = material('AlGaAs')(T=T, Al=0.3, Na=1e24)
contact_layer = material('AlGaAs')(T=T, Al=0.3, Na=1e25) 


# # Defining the device

jl = ( [Layer(width=  15 * nm, material=window_top, role="Window")]+
       [Layer(width=  50 * nm, material=n_GaAs, role="Emitter")] +
       [Layer(width= 1000/4 * nm, material=p_GaAs, role="Base")] +
       [Layer(width= 1000/4 * nm, material=p_GaAs, role="Base")] +
       [Layer(width= 1000/4 * nm, material=p_GaAs, role="Base")] +
       [Layer(width= 1000/4 * nm, material=p_GaAs, role="Base")] +
       [Layer(width= 100 * nm,  material=bsf_bottom, role="BSF")])

junction1 = Junction(jl, sn=1e6, sp=1e6, T=T, kind='PDD')

device = SolarCell([Layer(width=5 * nm, material=window_top, role="window"),
                    junction1,
                    Layer(width=200 *nm, material=contact_layer, role='contact')
                    ],
                    T=T, substrate=i_GaAs)


light_source = LightSource(source_type='standard', 
                           version='AM1.5d', x=wl,
                           output_units='photon_flux_per_m', 
                           concentration=1)
user_options={'light_source': light_source, 
               'wavelength': wl,
               'optics_method': 'external',
             }

layers = convert_cell_to_layers(device)
thickness = []
for lay in layers:
    thickness.append(lay.width)
thickness = np.array(thickness)

# +
Z = np.arange(0, sum(thickness), 1e-10)
user_options['position'] = Z
A = np.zeros((wl.shape[0], Z.shape[0]))
step = np.diff(Z)[0]
print(step)
I0 = np.ones_like(wl)

for i, lay in enumerate(layers):
    lut = (Z > thickness[:i].sum()) & (Z <= thickness[:i].sum() + lay.width)
    zi = Z[lut]
    print(lay.material, zi.min(), zi.max())
    A_l = single_pass(lay.material, wl, step)        

    for j, zj in enumerate(zi):
        zp = ((Z-zj) ** 2).argmin() #finding the position in the matrix
        np.testing.assert_equal(Z[zp], zj) 
        A[:, zp] = A_l * I0
        I0 = I0 - A_l * I0

# ## Calc device
# ### Calc QE

if user_options['optics_method'] == 'external':
    RAT = {'R':0, 'T':1-A.sum(axis=1), 'A':A.sum(axis=1)}
    device.RAT = RAT
    device.absorbed_matrix = A

solar_cell_solver(device, 'qe', user_options=user_options)

# Calc JV

vint = np.linspace(-4, 4, 600)
V = np.linspace(-2, 0, 300)
iv_options = {'T_ambient': 298,  'voltages': V, 'light_iv': True,
               'mpp': True, 'internal_voltages': vint,
               'optics_method': None }
user_options.update(iv_options)
solar_cell_solver(device, 'iv', user_options=user_options)

# +
labels=['GaAs-SJ']
for i, j in enumerate(device.junction_indices):
    QE_max = device[j].eqe(wl).max()
    print('QE_max {0}: {1:.20f}'.format(labels[i],device[j].eqe(wl).max()))
# np.testing.assert_equal(QE_max, 0.99838562803199837337, decimal=10)

jsc = device.iv['Isc']
voc = device.iv['Voc']
eta = device.iv.Eta

print('Jsc (mA/cm2) {:.20f}'.format(-jsc*1e-1))
# np.testing.assert_almost_equal(jsc, -252.2510673798636204879,
                               # decimal=10)

print('voc (V) {:.20f}'.format(-voc))
# np.testing.assert_almost_equal(voc, -1.13980595142133656061,
                               # decimal=10)

print('Eff (%) {:.20f}'.format(eta*100))
# np.testing.assert_almost_equal(eta, 0.284755491739901174248,
                               # decimal=10)

# Results using external
# QE_max GaAs-SJ: 0.99838562803199837337
# Jsc (mA/cm2) 25.22510673798636204879
# voc (V) 1.13980595142133656061
# Eff (%) 28.47554917399011742418

# Results using the BL
# QE_max GaAs-SJ: 0.99475377049951996256
# Jsc (mA/cm2) 25.13176350846979190123
# voc (V) 1.13970728883838945400
# Eff (%) 28.36734189137227701849

# # ## Calc device using beer lambert
# # ### Calc QE
# delattr(device, 'RAT')
# delattr(device, 'absorbed_matrix')

# user_options ['optics_method'] = 'BL'
# user_options['recalculate_absorption']= True
# solar_cell_solver(device, 'qe', user_options=user_options)

# # Calc JV

# vint = np.linspace(-4, 4, 600)
# V = np.linspace(-2, 0, 300)
# iv_options = {'T_ambient': 298,  'voltages': V, 'light_iv': True,
               # 'mpp': True, 'internal_voltages': vint,
               # 'optics_method': None }
# user_options.update(iv_options)
# solar_cell_solver(device, 'iv', user_options=user_options)

# labels=['GaAs-SJ']
# for i, j in enumerate(device.junction_indices):
    # QE_max = device[j].eqe(wl).max()
    # print('QE_max {0}: {1:.20f}'.format(labels[i],device[j].eqe(wl).max()))
# # np.testing.assert_equal(QE_max, 0.99783656125021946703,
                        # # decimal=1)

# jsc = device.iv['Isc']
# voc = device.iv['Voc']
# eta = device.iv.Eta

# print('Jsc (mA/cm2) {:.20f}'.format(-jsc*1e-1))
# # np.testing.assert_almost_equal(jsc, -251.98650008723070925,
                               # # decimal=1)

# print('voc (V) {:.20f}'.format(-voc))
# # np.testing.assert_almost_equal(voc, -1.139777987001644010,
                               # # decimal=1)

# print('Eff (%) {:.20f}'.format(eta*100))
# # np.testing.assert_almost_equal(eta, 0.28444879453268324454,
                               # # decimal=1)
