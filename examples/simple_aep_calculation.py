# -*- coding: utf-8 -*-
"""Calculate the gross and net AEP for a one-turbine layout

Notes
-----
The wind_resource module was originally written assuming that the user has
WAsP resource data. Thus, to use the module we must re-create much of the
necessary parameters for complex terrain, even for a simple example. Please
seee the lines in Step 1 for more detail.

Author
------
Jenni Rinker
rink@dtu.dk
"""

import numpy as np
import xarray as xr

from topfarm.aep import AEP_load
from topfarm.wake_model import WakeModel
from topfarm.wind_farm import WindFarmLayout
from topfarm.wind_resource import WindResourceNodes


# =============================================================================
# STEP 1: Define wind resource information

# specify the spatial values that define our wind resource
x = np.array([569649.])  # east-west location(s) of resource
y = np.array([2835530.])  # north-south location(s) of resource
elev = np.array([0.])  # elevation of resource
heights = np.array([85.0])  # height(s) of resource
sectors = np.array([1.])  # number of sectors

# specify the wind resource values
A = np.array([7.3])  # first Weibull parameter
k = np.array([1.5])  # second Weibull parameter
f = np.array([100.])  # frequency of that sector/bin
spd_up = np.array([1.0])  # speed-up due to complex terrain
dev = np.array([0.0])  # deviation due to complex terrain
inflow_angle = np.array([0.0])  # inflow angle due to complex terrain
tke = np.array([0.0])  # turbulence kinetic energy
alpha = np.array([0.2])  # shear parameter
rho = np.array([1.225])  # air density

# necessary values for our xarray dataset
dims = ('n', 'z', 'sec')  # hard-code the xarray dataset dimensions
n = np.arange(x.size)
coords = {'z': heights, 'sec': sectors, 'n': n}

# convert our input data to xarrays, which we need for AEP calculations
elev_da = xr.DataArray(elev, coords={'n': n}, dims=('n'))
x_da = xr.DataArray(x, coords={'n': n}, dims=('n',))
y_da = xr.DataArray(y, coords={'n': n}, dims=('n',))
A_da = xr.DataArray(A.reshape((1,) * len(dims)),
                    coords=coords, dims=dims)
k_da = xr.DataArray(k.reshape((1,) * len(dims)),
                    coords=coords, dims=dims)
f_da = xr.DataArray(f.reshape((1,) * len(dims)),
                    coords=coords, dims=dims)
spd_up_da = xr.DataArray(spd_up.reshape((1,) * len(dims)),
                         coords=coords, dims=dims)
dev_da = xr.DataArray(dev.reshape((1,) * len(dims)),
                      coords=coords, dims=dims)
inflow_angle_da = xr.DataArray(inflow_angle.reshape((1,) * len(dims)),
                               coords=coords, dims=dims)
tke_da = xr.DataArray(tke.reshape((1,) * len(dims)),
                      coords=coords, dims=dims)
alpha_da = xr.DataArray(alpha.reshape((1,) * len(dims)),
                        coords=coords, dims=dims)
rho_da = xr.DataArray(rho.reshape((1,) * len(dims)),
                      coords=coords, dims=dims)

# assemble everything into an overall dataset
ds = xr.Dataset({'A': A_da, 'k': k_da, 'f': f_da, 'elev': elev_da,
                 'x': x_da, 'y': y_da, 'spd_up': spd_up_da, 'deviation':
                 dev_da, 'inflow_angle': inflow_angle_da, 'tke_amb': tke_da,
                 'alpha': alpha_da, 'rho': rho_da})

# instantiate topfarm class using dataset
site_conditions = WindResourceNodes(ds)

# =============================================================================
# STEP 2: Define wind farm layout

wfl_path = '../topfarm/tests/data/simple_layout.yml'  # path to YAML file
wind_farm = WindFarmLayout(yml_path=wfl_path)

# =============================================================================
# STEP 3: Define wake model

wake_model = WakeModel()  # N.O. Jensen by default

# =============================================================================
# STEP 4: Calculate AEP

aep = AEP_load(site_conditions, wind_farm, wake_model)  # set up model
aep_gross, aep_net = aep.cal_AEP_load()  # calculate AEP
print(f'Gross AEP: {aep_gross[0]:.4e}')
print(f'Net AEP:   {aep_net[0]:.4e}')
