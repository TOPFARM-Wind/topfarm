# -*- coding: utf-8 -*-

import numpy as np


class AEP_load():
    """ Calculate AEP (and fatigue load) of a given wind farm.

    Parameters
    ----------
    site_conditions: class object (WindResource class)
        Storing the terrain flow and wind resource data, and providing a
        function to get the site conditions of given location(s) for certain
        inflow wind direction.

    wind_farm: class object (WindFarm class)
        Defining the design of the wind farm, i.e., the layout and turbine type
        and hub height information for all turbines.

    wake_model: class object (WakeModel class)
        Specifying the wake models to use in the calculation. The default ones
        are N.O. Jensen for wake deficit and G.C. Larsen for turbulence.

    ws_binned: array:float, np.linspace(1, 30, num=30)
        Discretized wind speed bins at a reference height above the ground for
        far field inflow. ) [m/s]

    wd_binned: array:float, np.linspace(0, 330, num=12)
        Discretized wind direction bins at a reference height above the ground
        for far field inflow [deg]. This should be equally spaced and contains
        at least two wind directions, as we derive the sector width from it.

    height_ref: float, 85
        Reference height above the ground for defining the far field inflow
        condition [m].

    k_star: float, 0.075
        Wake decay parameter used in the N.O. Jensen wake model for the
        wake_model [-].

    availability: float, 1.0
        Availability factor for the wind farm [-].

    num_evals: integer, 0
        Number of evluations of .cal_AEP_load() [-].

    z0: float, 0.001
        Used to transfer far field inflow wind speeds between different heights
        above the ground [m].

    Returns
    -------
    wind2load: wind2load class
        Specifying the load model for calculating mean equivalent fatigue loads
        for each turbine at different channels. Currently it is the surrogate
        load model implemented. Note that when wind2load is not provided as
        input, this class will only calculate AEP.


    Methods
    -------
    reset_num_evals()
        Reset the number of AEP (and load) evalutions to zero.

    cal_AEP_load(cal_load=True)
        Calculate gross AEP, net AEP (and mean load) of this wind farm.
    """

    def __init__(self,
                 site_conditions,  # include terrain and wind resource
                 wind_farm,  # defines the design of wind farm
                 wake_model,  # wake model to use
                 wind2load=None,  # optional input, calc only AEP when None
                 ws_binned=np.linspace(1, 30, num=30),  # [m/s]
                 wd_binned=np.linspace(0, 330, num=12),  # [deg]
                 height_ref=85,  # [m]
                 k_star=0.075,  # wake decay parameter
                 availability=1.0,  # availability factor for the wind farm
                 num_evals=0,    # number of evluations of .cal_AEP_load()
                 z0=0.001  # used to transfer wind speed between diff height
                 ):
        """ ws_binned, wd_binned and height_ref defines a set of discretized
        ideal far field inflow condition at the a height above the ground
        (height_ref), which controls and specifies the bin sizes of wind speed
        and wind direction in the AEP calculation.
        """
        # inputed attributes
        self.site_conditions = site_conditions
        self.wind_farm = wind_farm
        self.wake_model = wake_model
        self.wind2load = wind2load
        self.ws_binned = ws_binned
        self.wd_binned = wd_binned
        self.height_ref = height_ref
        self.k_star = k_star
        self.availability = availability
        self.num_evals = num_evals
        self.z0 = z0

        # calculated attributes (storing necessary data during calculation)
        self.num_ws_bins = len(ws_binned)
        self.num_wd_bins = len(wd_binned)
        self.wf_design = wind_farm.get_summary().values  # [x, y, z, D, H, P]
        self.num_turbines = len(self.wf_design[:, 0])
        if self.wind2load is not None:
            self.num_channels = self.wind2load.num_channel

        # Extend the wind dir bins to include an extra dir for integration
        self.num_wd_bins = self.num_wd_bins + 1
        self.wd_binned = np.hstack(
                (self.wd_binned,
                 self.wd_binned[-1] + self.wd_binned[1] - self.wd_binned[0]))
        # note we assume wd_binned is equally spaced.

        # index i, k, l denotes the ith turbine, the kth wind speed and the lth
        # wind direction, which is defined in self.ws_binned and self.wd_binned
        shape_ikl = [self.num_turbines, self.num_ws_bins, self.num_wd_bins]
        self.local_ws_ideal_ikl = np.zeros(shape_ikl)
        self.local_wd_ideal_ikl = np.zeros(shape_ikl)
        self.local_power_ideal_ikl = np.zeros(shape_ikl)
        self.local_ws_real_ikl = np.zeros(shape_ikl)
        self.local_wd_real_ikl = np.zeros(shape_ikl)
        self.local_power_real_ikl = np.zeros(shape_ikl)
        self.local_pdf_ikl = np.zeros(shape_ikl)
        self.local_TI_real_ikl = np.zeros(shape_ikl)
        self.local_Ct_ikl = np.zeros(shape_ikl)

        if self.wind2load is not None:
            self.load_iklm = np.zeros([self.num_turbines,
                                       self.num_ws_bins,
                                       self.num_wd_bins,
                                       self.num_channels])

    def reset_num_evals(self):
        """ Reset the number of evaluation to 0.
        """
        self.num_evals = 0

    def cal_AEP_load(self, cal_load=True):
        """ Calculate gross AEP, net AEP (and mean load) of this wind farm.

        If wind2load is provided and load calculation is turned on (cal_load=
        True), returns (AEP_gross, AEP_net and mean_loads).

        If wind2load is not provided or load calculation is turned off
        (cal_load=False), returns (AEP_gross, AEP_net)

        This is the vectorized version.

        Parameters
        ----------
        cal_load: boolean (default: True)
            A flag to turn on/off the load calculation.

        Returns
        -------
        AEP_gross: array:float
            Gross AEP values of each turbine in the wind farm.

        AEP_net: array:float
            Net AEP values of each turbine in the wind farm.

        mean_loads: array:float
            Mean equivalent fatigue loads for each turbine at different
            channels. Note this one is only returned when the wind2load
            instance is provided and the load calculation is turned on.
        """
        num_hrs_a_year = 8760

        #######################################################################
        # Step 1. Get and store sector (wind direction) wise site conditions
        speed_up_il = np.zeros([self.num_turbines, self.num_wd_bins])
        turning_il = np.zeros([self.num_turbines, self.num_wd_bins])
        inclination_il = np.zeros([self.num_turbines, self.num_wd_bins])
        Weibull_A_il = np.zeros([self.num_turbines, self.num_wd_bins])
        Weibull_k_il = np.zeros([self.num_turbines, self.num_wd_bins])
        frequency_il = np.zeros([self.num_turbines, self.num_wd_bins])
        turbulence_il = np.zeros([self.num_turbines, self.num_wd_bins])
        wind_shear_il = np.zeros([self.num_turbines, self.num_wd_bins])
        rho_il = np.zeros([self.num_turbines, self.num_wd_bins])

        for l_wd in range(self.num_wd_bins):
            wd = self.wd_binned[l_wd]
            # For this general inflow direcion, get wind resource and terrain
            # effect information for all turbine locations.

            conditions = self.site_conditions.get_site_conditions(
                self.wf_design[:, (0, 1, 4)], wd)

            speed_up_il[:, l_wd] = conditions['spd_up'].values
            turning_il[:, l_wd] = conditions['deviation'].values
            inclination_il[:, l_wd] = conditions['inflow_angle'].values
            Weibull_A_il[:, l_wd] = conditions['A'].values
            Weibull_k_il[:, l_wd] = conditions['k'].values
            frequency_il[:, l_wd] = conditions['freq_per_degree'].values
            turbulence_il[:, l_wd] = conditions['tke_amb'].values
            wind_shear_il[:, l_wd] = conditions['alpha'].values
            rho_il[:, l_wd] = conditions['rho'].values

        #######################################################################
        # Step 2. Calculate ideal local flow field (ws, wd) and related pdf
        for l_wd in range(self.num_wd_bins):
            wd = self.wd_binned[l_wd]

            for k_ws in range(self.num_ws_bins):
                ws = self.ws_binned[k_ws]

                for i_wt in range(self.num_turbines):
                    H_hub = self.wf_design[i_wt, 4]

                    # calculate local wind speed and wind direction without
                    # wake effects by considering terrain effect
                    local_ws_ideal = (ws * np.log(H_hub / self.z0) /
                                      np.log(self.height_ref / self.z0) *
                                      speed_up_il[i_wt, l_wd])

                    local_wd_ideal = (wd + turning_il[i_wt, l_wd])

                    self.local_ws_ideal_ikl[i_wt, k_ws, l_wd] = local_ws_ideal
                    self.local_wd_ideal_ikl[i_wt, k_ws, l_wd] = local_wd_ideal

                    # calculating related pdf for all wind speeds
                    self.local_pdf_ikl[i_wt, k_ws, l_wd] = \
                        self.cal_pdf_Weibull(self.local_ws_ideal_ikl[i_wt,
                                                                     k_ws,
                                                                     l_wd],
                                             Weibull_A_il[i_wt, l_wd],
                                             Weibull_k_il[i_wt, l_wd]) \
                        * frequency_il[i_wt, l_wd]

        #######################################################################
        # Step 3. Calculate ideal local Ct and power
        for i_wt in range(self.num_turbines):
            self.local_power_ideal_ikl[i_wt, :, :] = (
                    self.wind_farm.get_power(i_wt,
                                             self.local_ws_ideal_ikl[i_wt, :,
                                                                     :]))

            self.local_Ct_ikl[i_wt, :, :] = (
                    self.wind_farm.get_ct(i_wt,
                                          self.local_ws_ideal_ikl[i_wt, :,
                                                                  :]))

        #######################################################################
        # Step 4. Calculate real local wind speed and turbulence intensity

        # assuming same wake decay coefficients for all turbines
        k_star_list = [self.k_star] * self.num_turbines

        (self.local_ws_real_ikl, self.local_TI_real_ikl) = self.wake_model.cal_wake(
                    self.wf_design[:, 0],   # [x_i]
                    self.wf_design[:, 1],   # [y_i]
                    self.wf_design[:, 4],   # [H_i]]
                    self.wf_design[:, 3],   # [D_i]
                    self.local_ws_ideal_ikl,
                    self.local_wd_ideal_ikl,
                    self.local_Ct_ikl,
                    turbulence_il,
                    k_star_list)

        #######################################################################
        # Step 5. Calculate real power of each turbine
        for i_wt in range(self.num_turbines):
            self.local_power_real_ikl[i_wt, :, :] = (
                    self.wind_farm.get_power(i_wt,
                                             self.local_ws_real_ikl[i_wt, :,
                                                                    :]))

        #######################################################################
        # Step 6. Calculate loads

        # if the wind2load is available and load calculation is turned on
        if (self.wind2load is not None) and cal_load:
            # for the single value 0 dimension vectorized calculation
            # for l_wd in range(self.num_wd_bins):
            #     for k_ws in range(self.num_ws_bins):
            #         for i_wt in range(self.num_turbines):
            #
            #             self.load_iklm[i_wt, k_ws, l_wd, :] = (
            #                 self.wind2load.load_calculation(
            #                     self.local_ws_real_ikl[i_wt, k_ws, l_wd],
            #                     self.local_TI_real_ikl[i_wt, k_ws, l_wd],
            #                     wind_shear_il[i_wt, l_wd],
            #                     inclination_il[i_wt, l_wd],
            #                     rho_il[i_wt, l_wd],
            #                     self.wf_design[i_wt, 4],      # H
            #                     self.wf_design[i_wt, 3],      # D
            #                     self.wf_design[i_wt, 5]))     # P_rated
            # 2d dimensional data passing of vectorized calculation
            for i_wt in range(self.num_turbines):
                self.load_iklm[i_wt, :, :, :] = (
                    self.wind2load.load_calculation_2d(
                        self.local_ws_real_ikl[i_wt, :, :],
                        self.local_TI_real_ikl[i_wt, :, :],
                        np.tile(wind_shear_il[i_wt, :],
                                (self.num_ws_bins, 1)),
                        np.tile(inclination_il[i_wt, :],
                                (self.num_ws_bins, 1)),
                        np.tile(rho_il[i_wt, :], (self.num_ws_bins, 1)),
                        self.wf_design[i_wt, 4],  # H
                        self.wf_design[i_wt, 3],  # D
                        self.wf_design[i_wt, 5]))  # P_rated

        #######################################################################
        # Step 7. Calculate mean power and AEP values using numerical integ.

        delta_ws = (self.local_ws_ideal_ikl[:, 1:, 1:] -
                    self.local_ws_ideal_ikl[:, :-1, 1:])

        delta_wd = (self.local_wd_ideal_ikl[:, 1:, 1:] -
                    self.local_wd_ideal_ikl[:, 1:, :-1])

        pdf_array = (self.local_pdf_ikl[:, 1:, 1:] +
                     self.local_pdf_ikl[:, :-1, 1:]) / 2

        power_ideal_array = (self.local_power_ideal_ikl[:, 1:, 1:] +
                             self.local_power_ideal_ikl[:, :-1, 1:]) / 2

        power_real_array = (self.local_power_real_ikl[:, 1:, 1:] +
                            self.local_power_real_ikl[:, :-1, 1:]) / 2

        mean_power_ideal = np.sum(
            delta_ws * delta_wd * pdf_array * power_ideal_array, (1, 2))

        mean_power_real = np.sum(
            delta_ws * delta_wd * pdf_array * power_real_array, (1, 2))

        AEP_gross = num_hrs_a_year * mean_power_ideal * self.availability
        AEP_net = num_hrs_a_year * mean_power_real * self.availability

        #######################################################################
        # Step 8. Calculate loads

        # if the wind2load is available and load calculation is turned on
        if (self.wind2load is not None) and cal_load:
            load_array = (self.load_iklm[:, 1:, 1:, :]
                          + self.load_iklm[:, :-1, 1:, :])/2
            slope_array = self.wind2load.pce_slopes
            mean_loads = np.zeros([self.num_turbines, self.num_channels])

            for m_channel in range(self.num_channels):
                mean_loads[:, m_channel] = (np.sum(
                    delta_ws * delta_wd * pdf_array *
                    (load_array[:, :, :, m_channel] **
                     slope_array[m_channel]), (1, 2))
                     / self.wind2load.frequence) ** \
                    (1 / slope_array[m_channel])

        #################################################################
        # updating number of evluations
        self.num_evals = self.num_evals + 1

        if (self.wind2load is not None) and cal_load:
            return (AEP_gross, AEP_net, mean_loads)
        else:
            return (AEP_gross, AEP_net)

    def cal_AEP_load_naive(self, cal_load=True):
        """ Calculate gross AEP, net AEP and mean load of this wind farm

        This is the naive version with all in for loops. This naive version is
        only kept here for understanding the process and possible testing
        usages.
        """
        num_hrs_a_year = 8760

        #######################################################################
        # Step 1. Calculate flow field, pdf , power and load

        for l_wd in range(self.num_wd_bins):
            wd = self.wd_binned[l_wd]
            # For this general inflow direcion, get wind resource and terrain
            # effect information for all turbine locations. All thess lists are
            # one dimensional array with self.num_turbs elements.

            conditions = self.site_conditions.get_site_conditions(
                self.wf_design[:, (0, 1, 4)], wd)

            speed_up_list = conditions['spd_up'].values
            turning_list = conditions['deviation'].values
            inclination_list = conditions['inflow_angle'].values
            Weibull_A_list = conditions['A'].values
            Weibull_k_list = conditions['k'].values
            frequency_list = conditions['freq_per_degree'].values
            turbulence_list = conditions['tke_amb'].values
            wind_shear_list = conditions['alpha'].values
            rho_list = conditions['rho'].values

            for k_ws in range(self.num_ws_bins):
                ws = self.ws_binned[k_ws]
                Ct_list = np.zeros(self.num_turbines)

                ###########################################################
                # ideal case: without wake effects
                for i_wt in range(self.num_turbines):
                    H_hub = self.wf_design[i_wt, 4]

                    # calculate local wind speed and wind direction without
                    # wake effects by considering terrain effect
                    local_ws_ideal = (ws * np.log(H_hub / self.z0) /
                                      np.log(self.height_ref / self.z0) *
                                      speed_up_list[i_wt])
                    local_wd_ideal = (wd + turning_list[i_wt])

                    self.local_ws_ideal_ikl[i_wt, k_ws, l_wd] = local_ws_ideal
                    self.local_wd_ideal_ikl[i_wt, k_ws, l_wd] = local_wd_ideal

                    # calculating relating pdf and power
                    self.local_pdf_ikl[i_wt, k_ws, l_wd] = (
                        (self.cal_pdf_Weibull(
                            local_ws_ideal,
                            Weibull_A_list[i_wt],
                            Weibull_k_list[i_wt])) *
                        frequency_list[i_wt])

                    self.local_power_ideal_ikl[i_wt, k_ws, l_wd] = (
                        self.wind_farm.get_power(i_wt, local_ws_ideal))

                    # get Ct
                    Ct_list[i_wt] = self.wind_farm.get_ct(i_wt,
                                                          local_ws_ideal)

                ###########################################################
                # Real case: with wake effects
                # caculate wake influced flow field
                k_star_list = [self.k_star] * self.num_turbines

                (ws_eff, TI_eff) = self.wake_model.cal_wake(
                    self.wf_design[:, 0],
                    self.wf_design[:, 1],
                    self.wf_design[:, 4],
                    self.wf_design[:, 3],
                    self.local_ws_ideal_ikl[:, k_ws, l_wd],
                    self.local_wd_ideal_ikl[:, k_ws, l_wd],
                    Ct_list,
                    turbulence_list,
                    k_star_list)

                self.local_ws_real_ikl[:, k_ws, l_wd] = ws_eff

                if (self.wind2load is not None) and cal_load:

                    for i_wt in range(self.num_turbines):
                        self.local_power_real_ikl[i_wt, k_ws, l_wd] = (
                            self.wind_farm.get_power(i_wt,
                                                     ws_eff[i_wt]))

                        # Calculate laod for each WT
                        wind_condition = np.array([ws_eff[i_wt],
                                                   TI_eff[i_wt],
                                                   wind_shear_list[i_wt],
                                                   inclination_list[i_wt],
                                                   rho_list[i_wt]])
                        turbine_parameter = np.array([self.wf_design[i_wt, 4],
                                                      self.wf_design[i_wt, 3],
                                                      self.wf_design[i_wt, 5]])
                        print(cal_load)
                        self.load_iklm[i_wt, k_ws, l_wd, :] = (
                            self.wind2load.load_calculation(
                                wind_condition,
                                turbine_parameter))

        #######################################################################
        # Step 4. Calculate mean power and AEP values using numerical integ.
        delta_ws = (self.local_ws_ideal_ikl[:, 1:, 1:] -
                    self.local_ws_ideal_ikl[:, :-1, 1:])

        delta_wd = (self.local_wd_ideal_ikl[:, 1:, 1:] -
                    self.local_wd_ideal_ikl[:, 1:, :-1])

        pdf_array = (self.local_pdf_ikl[:, 1:, 1:] +
                     self.local_pdf_ikl[:, :-1, 1:]) / 2

        power_ideal_array = (self.local_power_ideal_ikl[:, 1:, 1:] +
                             self.local_power_ideal_ikl[:, :-1, 1:]) / 2

        power_real_array = (self.local_power_real_ikl[:, 1:, 1:] +
                            self.local_power_real_ikl[:, :-1, 1:]) / 2

        mean_power_ideal = np.sum(
            delta_ws * delta_wd * pdf_array * power_ideal_array, (1, 2))

        mean_power_real = np.sum(
            delta_ws * delta_wd * pdf_array * power_real_array, (1, 2))

        AEP_gross = num_hrs_a_year * mean_power_ideal
        AEP_net = num_hrs_a_year * mean_power_real

        #######################
        # mean loads
        if (self.wind2load is not None) and cal_load:
            load_array = (self.load_iklm[:, 1:, 1:, :]
                          + self.load_iklm[:, :-1, 1:, :])/2

            mean_loads = np.zeros([self.num_turbines, self.num_channels])

            for m_channel in range(self.num_channels):
                mean_loads[:, m_channel] = np.sum(
                    delta_ws * delta_wd * pdf_array *
                    load_array[:, :, :, m_channel], (1, 2))

        if (self.wind2load is not None) and cal_load:
            return (AEP_gross, AEP_net, mean_loads)
        else:
            return (AEP_gross, AEP_net)

    def cal_pdf_Weibull(self, ws, Weibull_A, Weibull_k):
        """ calculate pdf of a given wind speed based on Weibull distribution.

        Parameters
        ----------
        ws: array:float
            Wind speed [m/s]

        Weibull_A: array:float
            Scale parameter of Weibull distribution [m/s]
        Weibull_k: array:float
            Shape parameter of Weibull distribution [-].

        Returns
        -------
        pdf: array:float
            Probability density function calculated using Weibull distribution.

        """
        pdf = ((Weibull_k / Weibull_A) * (ws / Weibull_A) ** (Weibull_k - 1) *
               np.exp(-(ws / Weibull_A) ** Weibull_k))

        return pdf
