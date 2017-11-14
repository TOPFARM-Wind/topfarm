import os

import numpy as np
import pytest
import shutil
import xarray as xr

from topfarm.wind_resource import WindResource


# =============================================================================
# data for running tests

heights = np.array([50.0, 75.0, 100.0])
sectors = np.arange(1, 13)


A = np.array([[6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2],
              [7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4],
              [8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6]])

k = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
              [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
              [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])

f = np.array([[0.0, 1.0, 12.0, 13.0, 4.0, 9.0, 16.0,
               7.0, 8.0, 9.0, 10.0, 10.0],
              [0.0, 1.0, 12.0, 13.0, 4.0, 9.0, 16.0,
               7.0, 8.0, 9.0, 10.0, 10.0],
              [0.0, 1.0, 12.0, 13.0, 4.0, 9.0, 16.0,
               7.0, 8.0, 9.0, 10.0, 10.0]])

ws_mean = np.array([[6.10, 6.20, 6.30, 6.40, 6.50, 6.60,
                     6.70, 6.80, 6.90, 7.00, 7.10, 7.20],
                    [6.5900, 6.6803, 6.7706, 6.8609,
                     6.9511, 7.0414, 7.1317, 7.2220,
                     7.3122, 7.4025, 7.4928, 7.5831],
                    [7.5329, 7.6215, 7.7101, 7.7988,
                     7.8874, 7.9760, 8.0647, 8.1533,
                     8.2419, 8.3305, 8.4116, 8.5078]])

A_da = xr.DataArray(A, coords={'z': heights, 'sec': sectors},
                    dims=('z', 'sec'))
k_da = xr.DataArray(k, coords={'z': heights, 'sec': sectors},
                    dims=('z', 'sec'))
f_da = xr.DataArray(f, coords={'z': heights, 'sec': sectors},
                    dims=('z', 'sec'))

ds = xr.Dataset({'A': A_da, 'k': k_da, 'f': f_da})

wr = WindResource(ds)

# =============================================================================


@pytest.yield_fixture(scope='session', autouse=True)
def del_tmp():
    """create/delete tmp dir before/after tests run"""
    if os.path.isdir('tmp'):
        shutil.rmtree('tmp')
    os.mkdir('tmp')  # before tests
    yield  # do tests
    shutil.rmtree('tmp')  # after tests


def test_init():
    """verify wind resource dataset matches fed-in dataset"""
    assert wr._ds.equals(ds)


def test_pickle():
    """test dumping to/loading from pickle file"""
    wr.to_pickle('tmp/test.pkl')  # dump to pickle
    wr_pkl = WindResource.from_pickle('tmp/test.pkl')  # load from pickle
    os.remove('tmp/test.pkl')  # delete tmp dir
    assert wr._ds.equals(wr_pkl._ds)


def test_netcdf():
    """test dumping to/loading from netcdf file"""
    wr.to_netcdf('tmp/test.nc')
    wr_nc = WindResource.open_dataset('tmp/test.nc')
    assert wr._ds.equals(wr_nc._ds)
    wr_nc._ds.close()  # TODO: this manual closing is a bug in xr. fix later.
    os.remove('tmp/test.nc')


def test_ws_mean():
    """test the distribution means match the theory"""
    np.testing.assert_allclose(wr.ws_mean.values, ws_mean,
                               rtol=1e-01)
