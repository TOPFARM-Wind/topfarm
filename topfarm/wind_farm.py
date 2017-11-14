"""Classes and functions related to overall definition of wind turbine layout
"""

from copy import deepcopy
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel import yaml
from ruamel.yaml import YAML
import plotly.offline as offline
import plotly.graph_objs as go

from topfarm.wind_resource import WindResourceNodes

def check_struc(d1, d2,
                errors=[], level='wf'):
    """Recursively check struct of dictionary 2 to that of dict 1

    Arguments
    ---------
    d1 : dict
        Dictionary with desired structure
    d2 : dict
        Dictionary with structre to check
    errors : list of str, optional
        Missing values in d2. Initial value is [].
    level : str, optional
        Level of search. Inital value is 'wf' (wind farm) for top-level
        dictionary.

    Returns
    -------
    errors : list of str
        Missing values in d2.
    """
    for k1, v1 in d1.items():  # loop through keys and values in first dict
        if k1 not in d2.keys():  # if key doesn't exist in d2
            errors.append('{} not in dictionary'.format('.'.join([level,k1])))
        elif isinstance(v1, dict):  # otherwise, if item is a dict, recurse
            errors = check_struc(v1, d2[k1],
                                 errors=errors,  # pass in accumulated errros
                                 level='.'.join([level, k1]))  # change level
    return errors


class WindFarmLayout(dict):
    """Layout of wind farm

    """

    def __init__(self, yml_path=None):
        """load from yml_path first, then add wf_dict
        """
        super().__init__()

        self._wf_dict = {}
        if yml_path is not None:
            self.load_yml(yml_path)

    def check_self(self, skel_path):
        """Check wind farm dictionary structure/contents

        This should be run before attempting any AEP calculations.
        """
        wf_dict = self._wf_dict

        # check the structure of the dictionary
        with open(skel_path, 'r') as skel_file:
            skel_dict = YAML().load(skel_file)
        errors = check_struc(skel_dict, wf_dict)

        # check datatype assumptions
        try:

            # check turbine types
            wt_type_names = []
            for i_wt_type, wt_type_dict in enumerate(wf_dict['turbine_types']):

                # do number of columns in power_curves/c_t_curves match the
                #   number of listed control strategies?
                num_strats = len(wt_type_dict['control']['strategies'])
                if (np.array(wt_type_dict['power_curves']).shape[1] != \
                                                    num_strats + 1) or \
                    (np.array(wt_type_dict['c_t_curves']).shape[1] != \
                                                    num_strats + 1):
                    errors.append('Wind turbine type {} '.format(i_wt_type) + \
                                  'has mismatched dimensions with control ' + \
                                  'strategies/power curves/thrust curves.')
                wt_type_names.append(wt_type_dict['name'])

            # check layout
            for i_wt, wt_dict in enumerate(wf_dict['layout']):

                # is turbine type in list of defined turbine types?
                if wt_dict['turbine_type'] not in wt_type_names:
                    errors.append('Wind turbine {} has an '.format(i_wt) + \
                                  'undefined turbine type' + \
                                  ' \"{}\"'.format(wt_dict['turbine_type']))

                # does position have three elements?
                if len(wt_dict['position']) != 3:
                    errors.append('Wind turbine {} has a '.format(i_wt) + \
                                  'position that is not 3 elements')

                # is the control strategy in the turbine type?
                if wt_dict['strategy'] not in \
                    wf_dict['turbine_types'][wt_type_names.index(
                        wt_dict['turbine_type'])]['control']['strategies']:
                    errors.append('Wind turbine {} has a '.format(i_wt) + \
                                  'strategy not listed for that turbine type')

        except KeyError:
            pass

        return errors

    def get_ct(self, i_wt, u):
        """Calculate wind turbine thrust coefficient

        Parameters
        ----------
        i_wt : int
            Index of wind turbine in wind farm layout.
        u : wind speed
            Wind speed in m/s.

        Returns
        -------
        ct : float
            Thrust coefficient.
        """
        try:
            len(u)
        except TypeError:
            u = np.array(u).reshape(1)
        # load relevant data from class structure
        wt_type = self._wf_dict['layout'][i_wt]['turbine_type']  # turb type
        wt_conf = self._wf_dict['layout'][i_wt]['strategy']  # cntrl confg
        wt_dict = [d for d in self._wf_dict['turbine_types'] \
                     if d['name'] == wt_type][0]  # info of turbine type
        u_cutin = wt_dict['cut_in_wind_speed']  # cut in wsp
        u_cutout = wt_dict['cut_out_wind_speed']  # cut out wsp
        ct_idle = wt_dict['c_t_idle']

        # load ct curve data
        i_conf = wt_dict['control']['strategies'].index(wt_conf)
        ct_curve = np.array(wt_dict['c_t_curves'])[:, i_conf + 1]
        u_ctcurve = np.array(wt_dict['c_t_curves'])[:, 0]
        minu_ctcurve, maxu_ctcurve = u_ctcurve.min(), u_ctcurve.max()

        # assign ct values
        ct = np.full(u.shape, np.nan)  # initialize to NaNs
        mask_1 = u < u_cutin  # below cut-in
        mask_2 = np.logical_and(u_cutin <= u, u < minu_ctcurve)  # cutin, umin
        mask_3 = np.logical_and(minu_ctcurve <= u, u <= maxu_ctcurve)  # umn,mx
        mask_4 = np.logical_and(maxu_ctcurve < u, u <= u_cutout)  # umax, ctout
        mask_5 = u_cutout < u  # above cut-out
        ct[mask_1] = ct_idle  # idle thrust below cut-in
        ct[mask_5] = ct_idle  # idle thrust above cut-out
        ct[mask_2] = np.interp(u[mask_2], [u_cutin, minu_ctcurve],
                               [0, ct_curve[u_ctcurve.argmin()]])
        ct[mask_4] = np.interp(u[mask_4], [maxu_ctcurve, u_cutout],
                               [ct_curve[u_ctcurve.argmax()], 0])
        ct[mask_3] = np.interp(u[mask_3], u_ctcurve, ct_curve)

        # return a float if we were originally given a float
        if len(ct) == 1:
            ct = float(ct)

        return ct

    def get_summary(self):
        """Pandas dataframe with info for aep/wake model calculations

        Returns
        -------
        layout : pd.DataFrame
            Pandas dataframe with wind turbine locations (x, y, z), rotor
            diameter in m (d), hub height in m (h), rater power in kW
            (p_rated), index of wind turbine type (-), and control strategy.
            Dataframe index is wind turbine index.
        """

        wt_names = [wt['turbine_type'] for wt in self._wf_dict['layout']]
        wt_types = [wt_type['name'] for wt_type in \
                     self._wf_dict['turbine_types']]
        col_names = ['x', 'y', 'z', 'd', 'h', 'p_r', 'i_type', 'strat']
        layout = pd.DataFrame(np.full((len(wt_names), len(col_names)), np.nan),
                              columns=col_names)  # initialize dataframe
        for i_wt, wt in enumerate(self._wf_dict['layout']):
            i_type = wt_types.index(wt['turbine_type'])
            wt_dict = self._wf_dict['turbine_types'][i_type]
            hub_height = wt_dict['hub_height']
            rotor_diameter = wt_dict['rotor_diameter']
            rated_power = wt_dict['rated_power']
            strat = wt['strategy']
            layout.iloc[i_wt] = wt['position'] + \
                [rotor_diameter, hub_height, rated_power, i_type, strat]
        layout.index.name = 'i_wt'

        return layout

    def get_power(self, i_wt, u):
        """Calculate wind turbine power in kW

        Parameters
        ----------
        i_wt : int
            Index of wind turbine in wind farm layout.
        u : int, float, or iterable
            Wind speed(s) in m/s.

        Returns
        -------
        p : float
            Wind turbine power in kW.
        """

        try:
            len(u)
        except TypeError:
            u = np.array(u).reshape(1)

        # load relevant data from class structure
        wt_type = self._wf_dict['layout'][i_wt]['turbine_type']  # turb type
        wt_conf = self._wf_dict['layout'][i_wt]['strategy']  # cntrl confg
        wt_dict = [d for d in self._wf_dict['turbine_types'] \
                     if d['name'] == wt_type][0]  # info of turbine type
        u_cutin = wt_dict['cut_in_wind_speed']  # cut in wsp
        u_cutout = wt_dict['cut_out_wind_speed']  # cut out wsp

        # load power curve data
        i_conf = wt_dict['control']['strategies'].index(wt_conf)
        p_curve = np.array(wt_dict['power_curves'])[:, i_conf + 1]
        u_pcurve = np.array(wt_dict['power_curves'])[:, 0]
        minu_pcurve, maxu_pcurve = u_pcurve.min(), u_pcurve.max()

        # assign power values
        p = np.full(u.shape, np.nan)  # initialize to NaNs
        mask_1 = u < u_cutin  # below cut-in
        mask_2 = np.logical_and(u_cutin <= u, u < minu_pcurve)  # ctin, umin
        mask_3 = np.logical_and(minu_pcurve <= u, u <= maxu_pcurve)  # umin,max
        mask_4 = np.logical_and(maxu_pcurve < u, u <= u_cutout)  # umax, ctout
        mask_5 = u_cutout < u  # above cut-out
        p[mask_1] = 0  # no power below cut-in
        p[mask_5] = 0  # no power above cut-out
        p[mask_2] = np.interp(u[mask_2], [u_cutin, minu_pcurve],
                              [0, p_curve[u_pcurve.argmin()]])
        p[mask_4] = p_curve[u_pcurve.argmax()] # p_u_max -> u_cutout
        p[mask_3] = np.interp(u[mask_3], u_pcurve, p_curve)

        # return a float if we were originally given a float
        if len(p) == 1:
            p = float(p)

        return p

    def load_yml(self, yml_path):
        """Load wind turbine type from yaml file into class

        Parameters
        ----------
        yml_path : str
            Path to yaml file to load.
        """

        with open(yml_path, 'r') as yml_file:
            yml_dict = YAML().load(yml_file)
        self._wf_dict = yml_dict

    def plot_layout(self, ax=None,
                    method='matplotlib', legend=True):
        """Plot wind turbine layout

        Parameters
        ----------
        fig: matplotlib figure handle, optional
            Figure to plot into
        method: str, optional
            Plotting method. Default is 'matplotlib'. Options: matplotlib.
        """

        _allowed_methods = ('matplotlib', 'plotly',)
        if method not in _allowed_methods:
            raise ValueError('Plot method {} not supported.'.format(method))

        if method == 'matplotlib':
            if not ax:  # create new figure if fig handle not passed in
                fig, ax = plt.subplots(1, figsize=(6,6))

            # plot wind turbine positions
            layout = self.get_summary()
            for i_type in sorted(set(layout['i_type'])):
                layout_type = layout[lambda df: df['i_type'] == i_type]
                type_name = self._wf_dict['turbine_types'][int(i_type)]['name']
                ax.scatter(layout_type.x,
                           layout_type.y,
                           label=type_name)  # plot wind turbine positions

            # plot met mast locations if in _wf_dict
            if 'metmasts' in self._wf_dict.keys():
                metmast_locs = np.vstack((t['position'] for t in \
                                          self._wf_dict['metmasts']))
                ax.plot(metmast_locs[:,0], metmast_locs[:,1],
                         'xk',
                         label='Met mast')  # plot met mast locations

            # prettify axes
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Wind farm: {}'.format(
                                self._wf_dict['plant_data']['name']))

            # create legend
            if legend:
                ax.legend()

        if method == 'plotly':
            """
            Use plotly to make a .html plot - which can be viewed in
            a browser
            """

            names = [wt['name'] for wt in self._wf_dict['layout']]
            positions = np.array([wt['position'] for wt in
                                  self._wf_dict['layout']])
            text = ['%s<br>x=%8.1f<br>y=%8.1f<br>' % (name, pos[0], pos[1])
                    for name, pos in zip(names, positions)]

            data = [go.Scatter(x = positions[:,0],
                               y = positions[:,1],
                               mode = 'markers',
                               marker = {'symbol': 'y-down-open',
                                         'size': 12.0},
                               text = text)]

            layout = {'title': 'Wind farm',
                      'font': dict(size=16),
                      'xaxis': dict(title='x'),
                      'yaxis': dict(title='y'),}

            offline.plot({'data': data, 'layout': layout})


    def save_yml(self, yml_path):
        """Save wind turbine type to yaml file

        Parameters
        ----------
        yml_path: str
            Path to yaml file to save.
        """
        with open(yml_path, 'w') as yml_file:
            try:
                YAML().dump(self._wf_dict, yml_file)
            except:
                yaml.dump(self._wf_dict, yml_file)

    def copy_and_update(self, new_values):
        """Create a new instance of the farm with updated values

        Note that the original object is not updated. Any NaN values in
        new_values will be skipped. Turbines can be removed by adding columns
        of all NaNs, and new turbines can be added by adding a wind turbine
        index greater than the current largest wind turbine index.

        Arguments
        ---------
        new_values : pd.DataFrame
            Pandas dataframe with values to update in the copied wind farm
            instance. Index is wind turbine index. Possible columns are
            'name', 'x', 'y'. 'z' (position of wind turbine base), 'i_type'
            (index of turbine type), and 'strat' (turbine control strategy).

        Returns
        -------
        new_wfl : WindFarmLayout object
            Object with updated values. All other values are copied from
            original wind farm layout.
        """

        # check column values in new_values
        _allowed_cols = ('name', 'x', 'y', 'z', 'i_type', 'strat')
        if any([s not in _allowed_cols for s in new_values.columns]):
            raise ValueError('Unpermitted column name in new_values. ' + \
                             '(Allowable options are ' + str(_allowed_cols))

        new_wfl = deepcopy(self)  # initialize wind turbine layout as a copy

        # create new turbines if given and remove those entries from new_values
        n_wt = len(self._wf_dict['layout'])
        new_wt_idcs = new_values.index[new_values.index >= n_wt]
        for i_new_wt in new_wt_idcs:
            new_wt_values = new_values.loc[i_new_wt]
            new_wt_dict = {'name' : str(new_wt_values['name']),
                           'position' : list(new_wt_values[['x','y','z']]),
                           'turbine_type' : new_wfl._wf_dict['turbine_types']\
                               [int(new_wt_values.i_type)]['name'],
                           'strategy' : str(new_wt_values.strat)}
            new_wfl._wf_dict['layout'].append(new_wt_dict)
        new_values = new_values.iloc[new_values.index < n_wt]

        # delete turbines
        del_wt_idcs = pd.isnull(new_values).all(1).nonzero()[0]
        if del_wt_idcs.size:
            [new_wfl._wf_dict['layout'].pop(i) for i in \
                 list(sorted(del_wt_idcs))[::-1]]
        new_values = new_values.dropna(how='all')

        # loop through existing turbines
        for i_wt, row in new_values.iterrows():
            new_wt_vals = row.dropna()
            if 'name' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['name'] = new_wt_vals['name']
            elif 'x' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['position'][0] = \
                    new_wt_vals['x']
            elif 'y' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['position'][1] = \
                    new_wt_vals['y']
            elif 'z' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['position'][2] = \
                    new_wt_vals['z']
            elif 'i_type' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['turbine_type'] = \
                    new_wfl._wf_dict['turbine_types'][\
                                    int(new_wt_vals['i_type'])]['name']
            elif 'strat' in new_wt_vals.index:
                new_wfl._wf_dict['layout'][i_wt]['strategy'] = \
                    new_wt_vals['strat']

        return new_wfl


    def get_num_turbine_types(self):
        """ Return the number of turbine types that can be chosen from.
        """
        return len(self._wf_dict['turbine_types'])
