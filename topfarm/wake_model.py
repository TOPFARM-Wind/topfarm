# -*- coding: utf-8 -*-

import numpy as np


class WakeModel():
    """ Compute wake effects

    Uses the NO Jensen wake model and GC Larsen turbulence model
    """
    def __init__(self,
                 wake_model='NO_Jensen',
                 turbulence_model='GC_Larsen_turb'):
        self.wake_model = wake_model
        self.turbulence_model = turbulence_model

    def cal_wake(self, x, y, H, D, ws, wd, Ct, TI, k):
        """ Wrapper method for calculating wake effct.
        """
        if self.wake_model == 'NO_Jensen':
            num_turbines,num_ws_bins,num_wd_bins = ws.shape
            shape_ikl = [num_turbines, num_ws_bins, num_wd_bins]
            local_ws_real_ikl = np.zeros(shape_ikl)
            local_TI_real_ikl = np.zeros(shape_ikl)
            
            for l_wd in range(num_wd_bins):
                for k_ws in range(num_ws_bins):
                    # calculate effective wind speed and turbulence intensity
                    (ws_eff, TI_eff) = self.NO_Jensen(x, y, H, D, ws[:, k_ws, l_wd], 
                        wd[:, k_ws, l_wd], Ct[:, k_ws, l_wd], TI[:, l_wd], k,
                        turbulence_model=self.turbulence_model)
    
                    local_ws_real_ikl[:, k_ws, l_wd] = ws_eff
                    local_TI_real_ikl[:, k_ws, l_wd] = TI_eff
                    
            return local_ws_real_ikl, local_TI_real_ikl
        else:
            raise ValueError('The required wake model has not been \
                             implemented!')

    def NO_Jensen_naive(self, x, y, H, D, ws, wd, Ct, TI, k,
                        turbulence_model):
        """ Calculate effective wind speed

        Calculate effective wind speed (using classical N.O. Jensen wake
        model) and effective turbulence intensity (default: using G.C. Larsen
        model for wake induced turbulence) for a group of turbines, each has
        possibly different hub-height, diameter, inflow wind speed, wind
        direction, ambient turbulence intensity and wake decay coefficient.

        Parameters
        ----------
        x: array:float
            x coordinates [m]

        y: array:float
            y coordinates [m]

        H: array:float
            hub-heights [m]

        D: array:float
            rotor diameters [m]

        ws: array:float
            Local inflow wind speed [m/s]

        wd: array:float
            local inflow wind direction [deg] (N = 0, E = 90, S = 180, W = 270)

        Ct: array:float
            thrust coefficients [-]

        TI: array:float
            local ambient turbulence intensity [-]

        k: array:float
            local wake decay coefficient [-]

            Note: all the above inputs should 1d arrays of the same size.

        turbulence_model: string
            Name of the used model for accounting wake induced turbulence.

        Returns
        -------
        ws_eff: array:float
            Effective wind speed [m/s]

        TI_eff: array:float
            Effective turbulence intensity [-]
        """
        # make sure all input data are transformed into float array
        x, y, H, D, ws, wd, Ct, TI, k = (np.array(x, dtype='float64'),
                                         np.array(y, dtype='float64'),
                                         np.array(H, dtype='float64'),
                                         np.array(D, dtype='float64'),
                                         np.array(ws, dtype='float64'),
                                         np.array(wd, dtype='float64'),
                                         np.array(Ct, dtype='float64'),
                                         np.array(TI, dtype='float64'),
                                         np.array(k, dtype='float64'))

        num_turbines = len(x)  # number of turbines
        Ar = np.pi*(D/2)**2  # rotor areas
        ws_eff = np.zeros_like(ws)  # effective ws

        TI_eff = np.zeros_like(TI)  # effective turbulence intensity

        # Assuming the caculation is for a given general inflow condition and
        # the difference of local wind direction between turbines are limited
        wd_mean = np.mean(wd)
        wd_range = max(wd) - min(wd)
        if wd_range > 30:
            raise ValueError('Warning: maximal local wind direction difference'
                             + 'between turbines is beyond 30 deg, calculation'
                             + 'will be problemtic')

        # rotate the coordinate so that wd_mean represents wind along x axis
        cos_mean = np.cos((wd_mean - 270)*np.pi/180.0)
        sin_mean = np.sin((wd_mean - 270)*np.pi/180.0)

        # pre-calulate cosine and sine values for each turbine
        wd = (wd - 270)*np.pi/180.0
        cos_WTs = np.cos(wd)
        sin_WTs = np.sin(wd)

        # assuming downwind order of turbines is determined by wd_mean,
        # if wd_range is reasonably small
        down_order = np.argsort(x*cos_mean + y*sin_mean)

        for i_upWT in range(num_turbines-1):
            # find the current upwind WT
            i_WT = down_order[i_upWT]

            for j_downWT in range(i_upWT+1, num_turbines):
                # find the current downwind WT
                j_WT = down_order[j_downWT]

                # calculate downwind and crosswind distance beween two WTs,
                # based on inflow wind direction of upwind WT: wd[i_WT]
                dist_down = x[j_WT] - x[i_WT]
                dist_cross = y[j_WT] - y[i_WT]

                dist_down, dist_cross = (
                        (dist_down*cos_WTs[i_WT] + dist_cross*sin_WTs[i_WT]),
                        (dist_cross*cos_WTs[i_WT] - dist_down*sin_WTs[i_WT]))

                # accounting also the difference in hub heights
                dist_cross = np.sqrt(dist_cross**2 + (H[i_WT] - H[j_WT])**2)

                # Calculate wake deficit dV = (1-sqrt(1-Ct))/(1+k*dx/R)**2
                if dist_down > 0:
                    # Calculate wake deficit
                    wake_radius = k[i_WT]*dist_down + D[i_WT]/2

                    wake_deficit = (
                        (1 - np.sqrt(1 - Ct[i_WT])) /
                        (1 + k[i_WT] * dist_down * 2 / D[i_WT]) ** 2)

                    # Calculate added turbulence
                    if turbulence_model == 'GC_Larsen_turb':
                        TI_add_from_upWT = self.cal_added_turblence_GCL(
                                                         dist_down,
                                                         D[i_WT],
                                                         Ct[i_WT])
                    else:
                        raise ValueError('Only GC Larsen model for wake' +
                                         '-induced turbulence model')

                    # accounting partial wake by coefficient A_ol/Ar
                    A_ol = self.cal_overlapping_area(wake_radius,
                                                     D[j_WT] / 2,
                                                     dist_cross)

                    # power 2 of the effective wake deficit
                    wake_def2 = (A_ol / Ar[j_WT] * wake_deficit) ** 2

                    ws_eff[j_WT] += wake_def2

                    # only consider the largest added turbulence
                    TI_add_from_upWT *= (A_ol/Ar[j_WT])
                    TI_eff[j_WT] = max(TI_add_from_upWT, TI_eff[j_WT])

        # Calculate effective wind speed
        ws_eff = ws - np.sqrt(ws_eff)
        TI_eff = np.sqrt(TI**2 + TI_eff**2)

        return (ws_eff, TI_eff)

    def NO_Jensen(self, x, y, H, D, ws, wd, Ct, TI, k,
                  turbulence_model):
        """ Calculate effective wind speed (using classical N.O. Jensen wake
        model) and effective turbulence intensity (default: using G.C. Larsen
        model for wake induced turbulence) for a group of turbines, each has
        possibly different hub-height, diameter, inflow wind speed, wind
        direction, ambient turbulence intensity and wake decay coefficient.

        This is the partially vectorized version, which vectorizes the
        calculation of all downwind turbines instead of using a loop. It makes
        the code around 5 times faster than the naive version
        NO_Jensen_naive().

        Parameters
        ----------
        x: array:float
            x coordinates [m]

        y: array:float
            y coordinates [m]

        H: array:float
            hub-heights [m]

        D: array:float
            rotor diameters [m]

        ws: array:float
            Local inflow wind speed [m/s]

        wd: array:float
            local inflow wind direction [deg] (N = 0, E = 90, S = 180, W = 270)

        Ct: array:float
            thrust coefficients [-]

        TI: array:float
            local ambient turbulence intensity [-]

        k: array:float
            local wake decay coefficient [-]

        Note: all the above inputs should 1d arrays of the same size.

        turbulence_model: string
            Name of the used model for accounting wake induced turbulence.

        Returns
        -------
        ws_eff: array:float
            Effective wind speed [m/s]

        TI_eff: array:float
            Effective turbulence intensity [-]

        """
        # make sure all input data are transformed into float array
        x, y, H, D, ws, wd, Ct, TI, k = (np.array(x, dtype='float64'),
                                         np.array(y, dtype='float64'),
                                         np.array(H, dtype='float64'),
                                         np.array(D, dtype='float64'),
                                         np.array(ws, dtype='float64'),
                                         np.array(wd, dtype='float64'),
                                         np.array(Ct, dtype='float64'),
                                         np.array(TI, dtype='float64'),
                                         np.array(k, dtype='float64'))

        num_turbines = len(x)              # number of turbines
        Ar = np.pi*(D/2)**2                # rotor areas
        ws_eff = np.zeros_like(ws)         # effective ws

        TI_eff = np.zeros_like(TI)         # effective turbulence intensity

        # Assuming the caculation is for a given general inflow condition and
        # the difference of local wind direction between turbines are limited
        wd_mean = np.mean(wd)
        wd_range = max(wd) - min(wd)
        if wd_range > 30:
            raise ValueError('Warning: maximal local wind direction difference'
                             + 'between turbines is beyond 30 deg, calculation'
                             + 'will be problemtic')

        # rotate the coordinate so that wd_mean represents wind along x axis
        cos_mean = np.cos((wd_mean - 270)*np.pi/180.0)
        sin_mean = np.sin((wd_mean - 270)*np.pi/180.0)

        # pre-calulate cosine and sine values for each turbine
        wd = (wd - 270)*np.pi/180.0
        cos_WTs = np.cos(wd)
        sin_WTs = np.sin(wd)

        # assuming downwind order of turbines is determined by wd_mean,
        # if wd_range is reasonably small
        down_order = np.argsort(x*cos_mean + y*sin_mean)

        for i_upWT in range(num_turbines-1):
            # find the current upwind WT
            i_WT = down_order[i_upWT]

            # index of downwind WTs
            j_WTs = down_order[i_upWT+1:]

            # calculate downwind and crosswind distance beween two WTs,
            # based on inflow wind direction of upwind WT: wd[i_WT]
            dist_down = x[j_WTs] - x[i_WT]
            dist_cross = y[j_WTs] - y[i_WT]

            dist_down, dist_cross = (
                    (dist_down*cos_WTs[i_WT] + dist_cross*sin_WTs[i_WT]),
                    (dist_cross*cos_WTs[i_WT] - dist_down*sin_WTs[i_WT]))

            # accounting also the difference in hub heights
            dist_cross = np.sqrt(dist_cross**2 + (H[i_WT] - H[j_WTs])**2)

            wake_radius = np.zeros_like(dist_down)
            wake_deficit = np.zeros_like(dist_down)

            index_downwind = dist_down > 0

            wake_radius[index_downwind] = (k[i_WT]*dist_down[index_downwind] +
                                           D[i_WT]/2)
            wake_deficit[index_downwind] = (
                    (1 - np.sqrt(1 - Ct[i_WT])) /
                    (1 + k[i_WT] * dist_down[index_downwind]
                     * 2 / D[i_WT]) ** 2)

            # Calculate added turbulence
            if turbulence_model == 'GC_Larsen_turb':
                TI_add_from_upWT = self.cal_added_turblence_GCL_vector(
                                                         dist_down,
                                                         D[i_WT],
                                                         Ct[i_WT])
            else:
                raise ValueError('Only GC Larsen model for wake' +
                                 '-induced turbulence model')

            # accounting partial wake by coefficient A_ol/Ar
            A_ol = self.cal_overlapping_area(wake_radius,
                                             D[j_WTs]/2,
                                             dist_cross)

            # power 2 of the effective wake deficit
            wake_def2 = (A_ol/Ar[j_WTs]*wake_deficit)**2

            ws_eff[j_WTs] += wake_def2

            # only consider the largest added turbulence
            TI_add_from_upWT *= (A_ol/Ar[j_WTs])

            TI_eff[j_WTs] = np.where(TI_add_from_upWT > TI_eff[j_WTs],
                                     TI_add_from_upWT,
                                     TI_eff[j_WTs])

        # Calculate effective wind speed
        ws_eff = ws - np.sqrt(ws_eff)
        TI_eff = np.sqrt(TI**2 + TI_eff**2)

        return (ws_eff, TI_eff)

    def cal_added_turblence_GCL(self, x, D, Ct):
        """ Calculate the added turbulence intensity at downstream distance
        x at the wake of a turbine.

        Parameters
        ----------
        x: float -> downwind distance [m]
        D: float -> rotor diameter [m]
        Ct: float -> thrust coefficient [-]

        Ouput
        ----------
        TI_add: float -> added turbulence intensity [-]
        """
        TI_add = 0
        if x > 2*D:
            TI_add = 0.29*np.sqrt(1 - np.sqrt(1-Ct))/(x/D)**(1/3)
        return TI_add

    def cal_added_turblence_GCL_vector(self, x, D, Ct):
        """ Calculate the added turbulence intensity at downstream distance
        x at the wake of a turbine.

        Vectorized version to account multiple downwind distances.

        Parameters
        ----------
        x: float array -> downwind distance [m]
        D: float -> rotor diameter [m]
        Ct: float -> thrust coefficient [-]

        Returns
        -------
        TI_add: float -> added turbulence intensity [-]
        """
        TI_add = np.zeros_like(x)
        index = x > 2*D

        TI_add[index] = 0.29*np.sqrt(1 - np.sqrt(1-Ct))/(x[index]/D)**(1/3)

        return TI_add

    def cal_overlapping_area(self, R1, R2, d):
        """ Calculate the overlapping area of two circles with radius R1 and
        R2, centers distanced d.

        Parameters
        ----------
        R1: float/float array -> ridus of the first circle [m]

        R2: float/float array -> ridus of the second circle [m]

        d: float/float array -> distance between two centers [m]

        Returns
        -------
        A_ol: float/float array -> overlapping area [m2]

        The calculation formula can be found in Eq. (A1) of :
            [Ref] Feng J, Shen WZ, Solving the wind farm layout optimization
            problem using Random search algorithm, Reneable Energy 78 (2015)
            182-192
        Note that however there are typos in Equation (A1), '2' before alpha
        and beta should be 1.
        """
        # treat all input as array
        R1, R2, d = np.array(R1), np.array(R2), np.array(d),
        A_ol = np.zeros_like(R1)
        p = (R1 + R2 + d)/2.0

        # make sure R_big >= R_small
        Rmax = np.where(R1 < R2, R2, R1)
        Rmin = np.where(R1 < R2, R1, R2)

        # full wake cases
        index_fullwake = (d <= (Rmax - Rmin))
        A_ol[index_fullwake] = np.pi*Rmin[index_fullwake]**2

        # partial wake cases
        index_partialwake = np.logical_and(d > (Rmax - Rmin),
                                           d < (Rmin + Rmax))

        alpha = np.arccos(
           (Rmax[index_partialwake]**2.0 + d[index_partialwake]**2
            - Rmin[index_partialwake]**2)
           / (2.0 * Rmax[index_partialwake]
              * d[index_partialwake]))

        beta = np.arccos(
           (Rmin[index_partialwake]**2.0 + d[index_partialwake]**2
            - Rmax[index_partialwake]**2)
           / (2.0 * Rmin[index_partialwake] * d[index_partialwake]))

        A_triangle = np.sqrt(p[index_partialwake] *
                             (p[index_partialwake]
                              - Rmin[index_partialwake]) *
                             (p[index_partialwake]
                              - Rmax[index_partialwake]) *
                             (p[index_partialwake]
                              - d[index_partialwake]))

        A_ol[index_partialwake] = (alpha * Rmax[index_partialwake] ** 2
                                   + beta * Rmin[index_partialwake] ** 2
                                   - 2.0 * A_triangle)

        return A_ol


if __name__ == '__main__':
    import time

    wake_model = WakeModel()

    # simple tests
    # two WTs in a line (5D away)
    x = [0, 400]
    y = [0, 0]
    D = [80, 90]
    H = [100, 110]
    ws = [8., 9.]
    wd = [280, 275]
    Ct = [0.8, 0.9]
    TI = [0.1, 0.15]
    k = [0.075]*len(x)
    (ws_eff, TI_eff) = wake_model.cal_wake(x, y, H, D, ws, wd, Ct, TI, k)

    print('Inflow wind speeds:')
    print(ws)
    print('Effective wind speeds:')
    print(ws_eff)

    # random wind farm
    num_turbines = 10
    x = np.random.rand(num_turbines)*4000
    y = np.random.rand(num_turbines)*4000
    D = 50 + np.random.rand(num_turbines)*100
    H = 70 + np.random.rand(num_turbines)*100
    ws = 8 + np.random.rand(num_turbines)
    wd = 90 + np.random.rand(num_turbines)*10
    Ct = 0.8 + np.random.rand(num_turbines)/10
    TI = 0.1 + np.random.rand(num_turbines)/5
    k = [0.075]*num_turbines
    (ws_eff, TI_eff) = wake_model.cal_wake(x, y, H, D, ws, wd, Ct, TI, k)

    print('Inflow wind speeds:')
    print(ws)
    print('Effective wind speeds:')
    print(ws_eff)

    # test time consumption
    num_turbines = 25
    x = np.random.rand(num_turbines)*4000
    y = np.random.rand(num_turbines)*4000
    D = 50 + np.random.rand(num_turbines)*100
    H = 70 + np.random.rand(num_turbines)*100
    ws = 8 + np.random.rand(num_turbines)
    wd = 90 + np.random.rand(num_turbines)*10
    Ct = 0.8 + np.random.rand(num_turbines)/10
    TI = 0.1 + np.random.rand(num_turbines)/5
    k = [0.075]*num_turbines
    (ws_eff, TI_eff) = wake_model.cal_wake(x, y, H, D, ws, wd, Ct, TI, k)

    t1 = time.time()
    for k_eval in range(360*30):
        ws = 8 + np.random.rand(num_turbines)
        wd = 90 + np.random.rand(num_turbines)*10
        Ct = 0.8 + np.random.rand(num_turbines)/10
        TI = 0.1 + np.random.rand(num_turbines)/5

        (ws_eff, TI_eff) = wake_model.cal_wake(x, y, H, D, ws, wd, Ct, TI, k)

    t2 = time.time()

    print(t2-t1)
