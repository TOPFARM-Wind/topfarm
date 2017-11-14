import os
import glob
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from scipy import spatial
from scipy import special

_NUMERIC_KINDS = set('buifc')


class WindResource(object):
    """Class containing frequency distribution in terms of Weibull A and k

    Examples
    --------

    >>> A_data = xr.DataArray(np.random.random((3, 12)) * 10.0,
                              coords={'z': np.array([50, 60, 80]),
                                      'sec': np.arange(1, 13)},
                              dims=('z','sec'))
    >>> k_data = xr.DataArray(np.random.random((3, 12)) + 1.0,
                              coords={'z': np.array([50, 60, 80]),
                                      'sec': np.arange(1, 13)},
                              dims=('z','sec'))
    >>> ds = xr.Dataset({'A': A_data, 'k': k_data})
    >>> wr = WindResource(ds)
    >>> print(wr)
        <xarray.Dataset>
        Dimensions:  (sec: 12, z: 3)
        Coordinates:
          * z        (z) int64 50 60 80
          * sec      (sec) int64 1 2 3 4 5 6 7 8 9 10 11 12
        Data variables:
        A        (z, sec) float64 2.661 7.526 1.919 5.179 0.6142 8.067 ...
        k        (z, sec) float64 1.519 1.459 1.883 1.332 1.289 1.681 ...
        F


    Parameters
    ----------

    ds: xarray.Dataset
        Xarray Dataset object. Should contain the following variables:
        sectorwise frequencies (f), Weibull A and k (A, k) for each height
        and sector.


    Attributes
    ----------

    coords: list:str
        List of coordinates represented in dataset

    data_vars: list:str
        List of variables in dataset

    ws_mean: array:float
        Mean wind speed calculated from Weibull A's and k's

    freq_per_degree
       Sector frequencies in 'per degree' values.


    Methods
    -------

    from_pickle(pickle_file):
        @classmethod to open serialized (pickled) WindResource objects

    open_dataset(netcdf_file):
        @classmethod to open a WindResource file stored in netcdf .nc

    to_netcdf(args, kwargs):
        xarray.Dataset.to_netcdf method

    to_pickle(file_name):
        convert internal xarray.Dataset to pickle object and save to file.

    sel(args, kwargs):
        xarray.Dataset.sel method

    isel(args, kwargs):
        xarray.Dataset.isel method

    ws_moment(n=1.0):
        Calculate the n'th moment

    ws_pdf(ws_bins=np.linspace(0.5, 29.5, 30)):
        Convert Weibull A's and k's to Probability Density Function (PDF) for a
        range of windspeeds defined by ws_bins


    Notes
    -----

    _ds: xarray dataset
        Main data container in the class. Private to the class.
        Variables and meta-data is included.

        Dimensions and Coordinates

        - z: heights
        - sec: number of sectors

        Variables

        - A_tot, k_tot: array(sec):float
            All-sector A and k values

        - f: array(z, sec):float
            Sector-wise frequencies

        - A, k: array(z, sec):float
            Sector-wise A and k values


    """

    def __init__(self, ds):

        self._ds = ds
        self.coords = ds.coords
        self.data_vars = ds.data_vars

    @classmethod
    def open_dataset(cls, *args, **kwargs):
        return cls(xr.open_dataset(*args, **kwargs))

    @classmethod
    def from_pickle(cls, file_name):
        with open(file_name, 'rb') as f:
            ds = pickle.load(f)
        return cls(ds)

    def __getattr__(self, attr):
        if attr in self._ds:
            return self._ds[attr]
        else:
            return self._ds.attrs[attr]

    def __getitem__(self, item):
        if item in self._ds:
            return self._ds[item]
        else:
            return self._ds.attrs[item]

    def __str__(self):
        return self._ds.__str__()

    def __repr__(self):
        return self._ds.__repr__()

    def sel(self, *args, **kwargs):
        return self._ds.sel(*args, **kwargs)

    def isel(self, *args, **kwargs):
        return self._ds.isel(*args, **kwargs)

    def to_netcdf(self, *args, **kwargs):
        return self._ds.to_netcdf(*args, **kwargs)

    def to_pickle(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self._ds, f, protocol=-1)

    def ws_moment(self, n=1.0):
        return self.A**n * special.gamma(1.0 + n / self.k)

    def ws_pdf(self, ws_bins=np.arange(0.0, 31.0)):
        """
        Wind speed Probability Density Function (PDF)

        Calculated from the Weibull parameters (A, k)

        Parameters
        ----------
        ws_bins: array:float
            Wind speed bins where the probability should be calculated.

        Returns
        -------
        ws_pdf: array:float
            Probabilities for each of the wind speeds in the input.

        """

        k = self.k.values[..., np.newaxis]
        A = self.A.values[..., np.newaxis]

        return k/A*(ws_bins/A)**(k-1.0)*np.exp(-(ws_bins/A)**k)

    def ws_alpha(self):
        """
        Estimate the shear coefficient (alpha).
        Requires data at several heights.

        NOT IMPLEMENTED YET...

        """
        pass

    @property
    def freq_per_degree(self):
        return self['f'].values * self.dims['sec'] / 360.0

    @property
    def ws_mean(self):
        return self.ws_moment(n=1.0)


class WindResourceNodes(WindResource):
    """Extension of WindResource class to contain many WindResource nodes
    with no limitation to the structure of the layout of the nodes.

    Parameters
    ----------
    ds: xarray:dataset
        The dimensions of the dataset is expected to be (n, z, sec):
        node-number (n), height (z), and sector (sec).

    Methods
    -------

    get_site_conditions(locations, wdirs)
        Returns data at requested locations (x,y,z) for requested wind
        directions (wdirs).

    interp_to_positions(positions, method='linear')
        Interpolate a WindResourceNodes instance to new (x,y) nodes.
        Only works if the requsted positions are within the span of
        the existing positions.

    """

    def __init__(self, ds):

        super(WindResourceNodes, self).__init__(ds)

    def get_site_conditions(self, locations, wdirs):
        """

        Examples
        --------

        Parameters
        ----------
        locations: array:float
            locations in (x, y, z) space.
            z is height above terrain.
            Dimensions are (n, 3) where n is the sample size and
            the second dimension is position in x, y, and z space.

        wdirs: array:float
            Wind direction
            Dimension is (n) where n is the sample size.

        Returns
        -------
        ds: xarray.Dataset
            Data container.
            Contains: spd_up, tur, ver, A, k, f, alpha, ti

        Notes
        -----
        This method can potentially achieve a speed-up in cases
        where the same positions are requested many times, by
        caching a map between the locations and the indicies of
        the turbine (number)

        """
        locations = np.asarray(locations, dtype=float)
        wdirs = np.asarray(wdirs, dtype=float)

        def _sanitize_input(param, dims=1):
            """
            Private method to make sure that the input has the
            correct structure and type.
            """
            if not isinstance(param, (np.ndarray)):

                if isinstance(param, (tuple, list)):
                    param = np.array(list(param))
                elif isinstance(param, (float, int)):
                    param = np.array([param])

            if len(param.shape) == 0:
                param = np.reshape(param, (1,) * dims)

            if len(param.shape) < dims:
                param = param[np.newaxis, :]

            return param

        def _wd_to_isec(wd_in, nsec):
            """
            Private method to convert wind direction to
            sector index
            """
            wd = wd_in.copy()
            sec_width = 360.0 / nsec
            wd += sec_width/2
            wd = np.mod(wd, 360.0)
            return np.floor_divide(wd, sec_width).astype(int)

        nsec = self._ds.dims['sec']
        locations = _sanitize_input(locations, dims=2)
        wdirs = _sanitize_input(wdirs, dims=1)

        n, _ = locations.shape

        if len(wdirs.shape) != 1:
            raise ValueError('Expected wdirs to be one-dimensional!')

        if ((n > 1) and (len(wdirs) == 1)):
            wdirs = wdirs * np.ones([n])

        if n != len(wdirs):
            raise ValueError('Expected the same number of samples' +
                             ' for locations and wdirs. OR len(wdirs) = 1')

        # First we subset our dataset. Only variables that are listed below
        # AND are in the dataset is used...
        vars_out = ['x', 'y', 'z', 'elev', 'A', 'k', 'f', 'spd_up',
                    'deviation', 'inflow_angle', 'tke_amb', 'tke_tot',
                    'alpha', 'rho']
        ds = self._ds[[v for v in vars_out if v in self._ds.data_vars]]

        # Existing locations in the dataset
        locs_ex = np.stack([ds['x'].values,
                            ds['y'].values],
                           axis=1)

        # requested locations
        locs_req = locations[:, :2]

        # Find distance and index for n to the nearest turbine
        dist_arr, indn_arr = spatial.KDTree(locs_ex).query(locs_req)

        if any(dist_arr > 20.0):
            print('Warning! some points are more than 20 ' +
                  'meters from target locations...')

        # Existing heights in the dataset
        heights_ex = ds['z'].values

        # Requested heights
        heights_req = locations[:, 2]

        # Find indicies of nearest heights
        indz_arr = np.argmin(np.abs(heights_ex[:, np.newaxis] -
                                    heights_req[np.newaxis, :]), axis=0)

        # Find distances of nearest heights
        zdist = np.min(np.abs(heights_ex[:, np.newaxis] -
                              heights_req[np.newaxis, :]), axis=0)
        if any(zdist > 5.0):
            print('Warning! some points are more than 5 meter ' +
                  'above or below the requested location')

        # Find indices for dimension sec
        inds_arr = _wd_to_isec(wdirs, nsec)
        if ((n > 1) and (len(inds_arr) == 1)):
            inds_arr = inds_arr * np.ones(len(inds_arr))

        # Aggregate the data at the requested points
        ds_loc_list = []
        for i in range(n):

            indn = indn_arr[i]
            inds = inds_arr[i]
            indz = indz_arr[i]

            # Select and subset the dataset by the indicies and add to list
            ds_loc_list.append(ds.isel(n=indn, sec=inds, z=indz))

        # concatenate everything and force the right dimensions
        ds_out = xr.concat(ds_loc_list, dim='n')

        # Add wind directions to returned data.
        ds_out['wd'] = xr.DataArray(wdirs,
                                    coords=ds_out.coords,
                                    dims=ds_out.dims)

        # Calculate frequency per degree.
        ds_out['freq_per_degree'] = ds_out['f'] * nsec / 360.0

        return ds_out

    @property
    def locations(self):
        return self._ds[['x', 'y', 'z', 'elev']].to_dataframe()

    def interp_to_positions(self, positions, method='linear'):
        """
        Interpolate WindResourceGrid to new x,y positions

        Parameters
        ----------
        positions : Array_like(n, 2)
            Array of n number of x,y positions to interpolate to


        Returns
        -------
        obj:WindResourceNodes:
            Unstructured container of WindResoure nodes

        """

        if all(d in self._ds.dims for d in ['x', 'y']):
            ds = self._ds.copy().stack(n=('x', 'y'))
        else:
            ds = self._ds.copy()

        x = ds['x'].values
        y = ds['y'].values

        points = np.stack([x, y], axis=1)

        if len(positions.shape) == 1:
            positions = positions[np.newaxis, :]

        n, _ = positions.shape

        n_coord = [('n', np.arange(1, n + 1))]

        ds_out = xr.Dataset()
        for var in ds.data_vars:

            da = ds[var]

            if da.dtype.kind not in _NUMERIC_KINDS:
                continue

            dims = ['n'] + [d for d in da.dims if d != 'n']

            values = da.transpose(*dims).values

            new_values = griddata(points, values, positions, method=method)

            coords = n_coord + [(d, ds.coords[d]) for d in dims
                                if d not in ['n']]

            ds_out[var] = xr.DataArray(new_values, coords)

        ds_out.attrs = ds.attrs

        ds_out['x'] = xr.DataArray(positions[:, 0],
                                   coords=n_coord,
                                   dims=('n'))

        ds_out['y'] = xr.DataArray(positions[:, 1],
                                   coords=n_coord,
                                   dims=('n'))

        return WindResourceNodes(ds_out)


class WindResourceGrid(WindResourceNodes):
    """WindResource points in a structured (Grid) format.
    """

    def __init__(self, ds):

        super(WindResourceGrid, self).__init__(ds)

    @classmethod
    def from_wasp_rsf(cls, path, globstr='*.rsf'):
        """ Read wasp resource grid files formatted as .rsf

        Parameters
        ----------
        path: str
            path to file or directory containing WAsP .rsf file(s)

        globstr: str
            string that is used to glob files if path is a directory.

        Notes
        -----
        * 1-10 Text string (10 characters) identifying the site/WT
        * 11-20 X-coordinate (easting) of the site [m]
        * 21-30 Y-coordinate (northing) of the site [m]
        * 31-38 Z-coordinate (elevation) of the site [m]
        * 39-43 Height above ground level [m a.g.l.]
        * 44-48 Weibull A-parameter for the total distribution [ms-1]
        * 49-54 Weibull k-parameter for the total distribution
        * 55-69 Power density [Wm-2] or power production [Why-1]
        * 70-72 Number of sectors
        * 73-76 Frequency of occurrence for sector #1 [%·10]
        * 77-80 Weibull A-parameter for sector #1 [ms-1·10]
        * 81-85 Weibull k-parameter for sector #1 [·100]
        * 86-98 As columns 73-85, but for sector #2
        * ...
        * ...
        * ...
        * 216-228 As columns 73-85, but for sector #12

        """

        if os.path.isfile(path):
            rsf_files = [path]
        elif os.path.isdir(path):
            rsf_files = sorted(glob.glob(os.path.join(path, globstr)))
        else:
            raise Exception('Path was neither file nor directory...')

        first = True
        for rsf_file in rsf_files:

            df = pd.read_csv(rsf_file, sep='\s+', header=None)

            nsec = int(df.loc[0, 8])  # Get the number of sectors

            header = ['Name', 'x', 'y', 'elev', 'z', 'A_tot',
                      'k_tot', 'power_density', 'nsec']

            for i in range(1, nsec + 1):
                header += f'f_{i} A_{i} k_{i}'.split()

            df.columns = header

            # Set multiIndex for dataframe
            df.set_index(['x', 'y', 'z'], inplace=True)

            ds = df.to_xarray()

            nx = ds.dims['x']
            ny = ds.dims['y']
            nz = ds.dims['z']

            for var in ['f', 'A', 'k']:

                data = np.zeros([nx, ny, nz, nsec])

                # Aggregate sectorwise values and
                # drop redundant data arrays
                for i in range(nsec):
                    data[..., i] = ds[f'{var}_{i+1}'].values
                    ds = ds.drop(f'{var}_{i+1}')

                # Normalize and rescale according to specification
                # See Doc-string
                if var == 'f':
                    data = data / np.sum(data, axis=3)[..., np.newaxis]
                if var == 'A':
                    data /= 10.0
                if var == 'k':
                    data /= 100.0

                da = xr.DataArray(data,
                                  coords={'x': ds.coords['x'],
                                          'y': ds.coords['y'],
                                          'z': ds.coords['z'],
                                          'sec': np.arange(1, nsec+1)},
                                  dims=('x', 'y', 'z', 'sec'))
                ds[var] = da

            # Drop redundant Name and nsec Data-array
            ds.attrs['Name'] = ds['Name'].values[0, 0, 0]
            ds = ds.drop(['Name', 'nsec'])

            if first:
                ds_combined = ds
                first = False
            else:
                ds_combined = xr.concat([ds_combined, ds], dim='z')

        ds_combined = ds_combined.sortby('z')

        return cls(ds_combined)

    @classmethod
    def from_wasp_grd(cls, path, globstr='*.grd'):
        """
        Reader for WAsP .grd resource grid files.

        Parameters
        ----------
        path: str
            path to directory containing WAsP files

        globstr: str
            string that is used to glob files if path is a directory.

        Returns
        -------
        obj:WindResourceGrid

        Examples
        --------
        >>> from topfarm.wind_resource import WindResourceGrid
        >>> path = '../topfarm/tests/data/WAsP_grd/'
        >>> wrg = WindResourceGrid.from_wasp_grd(path)
        >>> print(wrg)
            <xarray.Dataset>
            Dimensions:            (sec: 12, x: 20, y: 20, z: 3)
            Coordinates:
              * sec                (sec) int64 1 2 3 4 5 6 7 8 9 10 11 12
              * x                  (x) float64 5.347e+05 5.348e+05 ...
              * y                  (y) float64 6.149e+06 6.149e+06 ...
              * z                  (z) float64 10.0 40.0 80.0
            Data variables:
                flow_inc           (x, y, z, sec) float64 1.701e+38 ...
                ws_mean            (x, y, z, sec) float64 3.824 3.489 ...
                meso_rgh           (x, y, z, sec) float64 0.06429 0.03008 ...
                obst_spd           (x, y, z, sec) float64 1.701e+38 ...
                orog_spd           (x, y, z, sec) float64 1.035 1.039 1.049 ...
                orog_trn           (x, y, z, sec) float64 -0.1285 0.6421 ...
                power_density      (x, y, z, sec) float64 77.98 76.96 193.5 ...
                rix                (x, y, z, sec) float64 0.0 0.0 0.0 0.0 ...
                rgh_change         (x, y, z, sec) float64 6.0 10.0 10.0 ...
                rgh_spd            (x, y, z, sec) float64 1.008 0.9452 ...
                f                  (x, y, z, sec) float64 0.04021 0.04215 ...
                tke                (x, y, z, sec) float64 1.701e+38 ...
                A                  (x, y, z, sec) float64 4.287 3.837 5.752 ...
                k                  (x, y, z, sec) float64 1.709 1.42 1.678 ...
                flow_inc_tot       (x, y, z) float64 1.701e+38 1.701e+38 ...
                ws_mean_tot        (x, y, z) float64 5.16 6.876 7.788 5.069 ...
                power_density_tot  (x, y, z) float64 189.5 408.1 547.8 ...
                rix_tot            (x, y, z) float64 0.0 0.0 0.0 9.904e-05 ...
                tke_tot            (x, y, z) float64 1.701e+38 1.701e+38 ...
                A_tot              (x, y, z) float64 5.788 7.745 8.789 ...
                k_tot              (x, y, z) float64 1.725 1.869 2.018 ...
                elev               (x, y) float64 37.81 37.42 37.99 37.75 ...

        """

        def _rename_var(var):
            """
            Function to rename WAsP variable names to short hand name
            """
            _rename = {
                'Flow inclination': 'flow_inc',
                'Mean speed': 'ws_mean',
                'Meso roughness': 'meso_rgh',
                'Obstacles speed': 'obst_spd',
                'Orographic speed': 'orog_spd',
                'Orographic turn': 'orog_trn',
                'Power density': 'power_density',
                'RIX': 'rix',
                'Roughness changes': 'rgh_change',
                'Roughness speed': 'rgh_spd',
                'Sector frequency': 'f',
                'Turbulence intensity': 'tke',
                'Weibull-A': 'A',
                'Weibull-k': 'k',
                'Elevation': 'elev'}

            return _rename[var]

        def _read_grd(filename):

            def _parse_line_floats(f):
                return [float(i) for i in f.readline().strip().split()]

            def _parse_line_ints(f):
                return [int(i) for i in f.readline().strip().split()]

            with open(filename, 'rb') as f:
                # file_id = f.readline().strip().decode()  # not sure needed
                nx, ny = _parse_line_ints(f)
                xl, xu = _parse_line_floats(f)
                yl, yu = _parse_line_floats(f)
                zl, zu = _parse_line_floats(f)
                values = np.genfromtxt(f)

            xarr = np.linspace(xl, xu, nx)
            yarr = np.linspace(yl, yu, ny)

            return xarr, yarr, values

        def _rsf_files_by_height(files):

            def _is_int(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            def _height_from_name(s):
                return [int(c) for c in re.split('([0-9]+)', s)
                        if _is_int(c)][-1]

            file_dict = defaultdict(list)

            for f in files:
                height = _height_from_name(f)
                file_dict[height].append(f)

            return file_dict

        if os.path.isfile(path):
            rsf_files = [path]
        elif os.path.isdir(path):
            rsf_files = sorted(glob.glob(os.path.join(path, globstr)))
        else:
            raise Exception('Path was neither file nor directory...')

        rsf_files_dict = _rsf_files_by_height(rsf_files)

        pattern = r'Sector (\w+|\d+) \s+ Height (\d+)m \s+ ([a-zA-Z0-9- ]+)'

        elev_avail = False
        first = True
        for height, rsf_files_subset in rsf_files_dict.items():

            first_at_height = True
            for rsf_file in rsf_files_subset:

                match = re.findall(pattern, os.path.basename(rsf_file))[0]

                if len(match) != 3:
                    raise ValueError('Something is wrong with the name of' +
                                     f' file: {os.path.basename(rsf_file)}')

                sector, _, var_name = match

                var_name = _rename_var(var_name)

                xarr, yarr, values = _read_grd(rsf_file)

                da = _read_grd(rsf_file)

                if sector == 'All':

                    # Only 'All' sector has the elevation files.
                    # So here we make sure that, when the elevation file
                    # is read, it gets the right (x,y) coords/dims.
                    if var_name == 'elev':
                        elev_avail = True
                        elev_vals = values
                        elev_coords = {'x': xarr,
                                       'y': yarr}
                        elev_dims = ('x', 'y')
                        continue

                    else:
                        var_name += '_tot'

                        coords = {'x': xarr,
                                  'y': yarr,
                                  'z': np.array([float(height)])}

                        dims = ('x', 'y', 'z')

                        da = xr.DataArray(values[..., np.newaxis],
                                          coords=coords,
                                          dims=dims)

                else:

                    coords = {'x': xarr,
                              'y': yarr,
                              'z': np.array([float(height)]),
                              'sec': np.array([int(sector)])}

                    dims = ('x', 'y', 'z', 'sec')

                    da = xr.DataArray(values[..., np.newaxis, np.newaxis],
                                      coords=coords,
                                      dims=dims)

                if first_at_height:
                    ds_tmp = xr.Dataset({var_name: da})
                    first_at_height = False
                else:
                    ds_tmp = xr.merge([ds_tmp, xr.Dataset({var_name: da})])

            if first:
                ds = ds_tmp
                first = False
            else:
                ds = xr.concat([ds, ds_tmp], dim='z')

        if elev_avail:
            ds['elev'] = xr.DataArray(elev_vals,
                                      coords=elev_coords,
                                      dims=elev_dims)

        return cls(ds)


def gwcfile_to_ds(file_name):
    """
    Read WAsP Generalized Wind Climate files (.gwc/.lib).

    Parameters
    ----------
    file_name: str
        path to file

    Returns
    -------
    obj:WindResource
        WindResource class

    Examples
    --------
    >>> from topfarm.wind_resource import gwcfile_to_ds
    >>> gwc_file = '../topfarm/tests/data/Waspdale.lib'
    >>> ds = gwcfile_to_ds(gwc_file)
    >>> print(ds)
        <xarray.Dataset>
        Dimensions:  (sec: 12, z: 5, z0: 4)
        Coordinates:
          * sec      (sec) int64 1 2 3 4 5 6 7 8 9 10 11 12
          * z        (z) float64 10.0 25.0 50.0 100.0 200.0
          * z0       (z0) float64 0.0 0.03 0.1 0.4
        Data variables:
            f        (sec, z, z0) float32 2.86 2.29 2.84 3.61 2.86 2.29 ...
            k        (sec, z, z0) float32 2.28 1.94 1.84 1.87 2.36 2.1 ...
            A        (sec, z, z0) float32 5.51 3.45 3.3 2.81 6.03 4.14 ...
        Attributes:
            header:   Waspdale
    """

    def _parse_line_ints(f):
        return [int(i) for i in f.readline().strip().split()]

    def _parse_line_floats(f):
        return [float(i) for i in f.readline().strip().split()]

    with open(file_name, 'rb') as f:

        header = f.readline().strip().decode()

        nz0, nz, nsec = _parse_line_ints(f)
        z0arr = _parse_line_floats(f)
        zarr = _parse_line_floats(f)

        freq = np.zeros([nsec, nz, nz0], dtype='f')
        k = np.zeros([nsec, nz, nz0], dtype='f')
        A = np.zeros([nsec, nz, nz0], dtype='f')

        for i in range(nz0):

            freq_vals = np.array(_parse_line_floats(f))[:, np.newaxis]
            freq[:, :, i] = freq_vals * np.ones([nsec, nz])

            for j in range(nz):

                A[:, j, i] = _parse_line_floats(f)
                k[:, j, i] = _parse_line_floats(f)

    coords = {'sec': np.arange(1, nsec+1), 'z': zarr, 'z0': z0arr}
    dims = ('sec', 'z', 'z0')

    da_f = xr.DataArray(freq, coords=coords, dims=dims)
    da_k = xr.DataArray(k, coords=coords, dims=dims)
    da_A = xr.DataArray(A, coords=coords, dims=dims)

    ds = xr.Dataset(data_vars={'f': da_f,
                               'k': da_k,
                               'A': da_A},
                    attrs={'header': header})

    return ds


if __name__ == '__main__':

    A_data = xr.DataArray(np.random.random((3, 3, 3, 12)) * 10.0,
                          coords={'x': np.array([0.0, 1.0, 2.0]),
                                  'y': np.array([-10.0, 0.0, 10.0]),
                                  'z': np.array([50, 60, 80]),
                                  'sec': np.arange(1, 13)},
                          dims=('x', 'y', 'z', 'sec'))

    k_data = xr.DataArray(np.random.random((3, 3, 3, 12)) + 1.0,
                          coords={'x': np.array([0.0, 1.0, 2.0]),
                                  'y': np.array([-10.0, 0.0, 10.0]),
                                  'z': np.array([50, 60, 80]),
                                  'sec': np.arange(1, 13)},
                          dims=('x', 'y', 'z', 'sec'))

    ds = xr.Dataset({'A': A_data, 'k': k_data})

    wrg = WindResourceGrid(ds)

    print(wrg)
