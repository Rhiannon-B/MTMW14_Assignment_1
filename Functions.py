import Parameters as p
import numpy as np
from matplotlib import pyplot as plt
import random as r


def time_param(periods, timesteps):
    time = periods * 41  # in months
    dt = (time / timesteps) / 30  # remove months
    dt_non_dim = dt / 2  # non-dimensionalised dt
    nt = timesteps * 30  # resolve months
    time_array = np.linspace(0, time, nt)

    return nt, dt_non_dim, time_array


def depth_f(hw, Te, r, a, b, xi1):
    return -r*hw - a*b*Te - a*xi1


def temp_f(hw, Te, gamma, en, b, xi1, xi2, c):
    R = gamma*b - c
    return R*Te + gamma*hw - en*(hw + b*Te)**3 + gamma*xi1 + xi2


def mu_f(mu0, mu_annual, time, tau):
    mu_new = mu0 * (1 + mu_annual*np.cos(2*np.pi*time/tau - 5*np.pi/6))

    return mu_new


def wind_stress(f_annual=0, f_random=0, time=0, tau=0, tau_cor=0, dt=0):

    w = r.uniform(-1, 1)
    xi = f_annual*np.cos(2*np.pi*time/tau) + f_random*w*tau_cor/dt

    return xi


def runge_kutta(periods, timesteps, mu0, en=0, mu_annual=0,
                f_annual=0, f_random=None, tau_cor=1/30, depth0=0, temp0=1.125):

    # The model is incredibly sensitive to changes in mu
    # I am in the process of diagnosing this

    nt = time_param(periods, timesteps)[0]
    dt_nondim = time_param(periods, timesteps)[1]
    dt = dt_nondim*p.time_scale

    depth = np.empty(nt)
    depth[0] = depth0/p.depth_scale

    temp = np.empty(nt)
    temp[0] = temp0/p.temp_scale

    # Only used if mu_annual /= 0
    b = p.b0*mu0

    # Only used if f_annual /= 0
    xi1 = p.xi1

    tau = 12/p.time_scale

    # Testing mu for annual variation
    # It stays between 0.6 and 0.9, I don't know why it gives such poor results

    for i in range(nt-1):

        if mu_annual != 0:
            mu = mu_f(mu0, mu_annual, i*dt, tau)
            b = p.b0*mu
            #mu_array[i] = mu

        if f_annual != 0:
            xi1 = wind_stress(f_annual, f_random, i*dt, tau, tau_cor, dt)

        depth_k1 = depth_f(depth[i], temp[i], p.r, p.a, b, xi1)
        temp_k1 = temp_f(depth[i], temp[i], p.gamma, en, b, xi1, p.xi2, p.c)

        depth_k2 = depth_f(depth[i] + depth_k1*dt_nondim/2, temp[i] + temp_k1*dt_nondim/2, p.r, p.a, b, xi1)
        temp_k2 = temp_f(depth[i] + depth_k1*dt_nondim/2, temp[i] + temp_k1*dt_nondim/2, p.gamma, en, b, xi1, p.xi2, p.c)

        depth_k3 = depth_f(depth[i] + depth_k2*dt_nondim/2, temp[i] + temp_k2*dt_nondim/2, p.r, p.a, b, xi1)
        temp_k3 = temp_f(depth[i] + depth_k2*dt_nondim/2, temp[i] + temp_k2*dt_nondim/2, p.gamma, en, b, xi1, p.xi2, p.c)

        depth_k4 = depth_f(depth[i] + depth_k3*dt_nondim, temp[i] + temp_k3*dt_nondim, p.r, p.a, b, xi1)
        temp_k4 = temp_f(depth[i] + depth_k3*dt_nondim, temp[i] + temp_k3*dt_nondim, p.gamma, en, b, xi1, p.xi2, p.c)

        depth[i+1] = depth[i] + dt_nondim*(depth_k1 + 2*depth_k2 + 2*depth_k3 + depth_k4)/6
        temp[i+1] = temp[i] + dt_nondim*(temp_k1 + 2*temp_k2 + 2*temp_k3 + temp_k4)/6
        #print(depth_k1, depth_k2, depth_k3, depth_k4)

    depth_redim = p.depth_scale*depth.copy()
    temp_redim = p.temp_scale*temp.copy()

    return depth_redim, temp_redim


def rk_ensemble(periods, timesteps, mu0, en=0, mu_annual=0,
                f_annual=0, f_random=None, tau_cor=1/30, depth0=0, temp0=1.125, ensembles=0, ensembles_r_control=0.05):

    # The model is incredibly sensitive to changes in mu
    # I am in the process of diagnosing this

    nt = time_param(periods, timesteps)[0]
    dt_nondim = time_param(periods, timesteps)[1]
    dt = dt_nondim*p.time_scale

    # Only used if mu_annual /= 0
    b = p.b0*mu0

    # Only used if f_annual /= 0
    xi1 = p.xi1

    tau = 12/p.time_scale

    # Treats the initial run as a single "ensemble", requiring a minimum value
    # of ensembles of 1
    ensembles += 1

    # Initializing arrays - these are used both for the case where there are and
    # are not any ensembles
    depth_ensemble = np.empty((ensembles, nt))
    temp_ensemble = np.empty((ensembles, nt))

    # Setting initial conditions for the first pass of the code
    # The first scheme does not include any perturbations
    depth_ensemble[0, 0] = depth0/p.depth_scale
    temp_ensemble[0, 0] = temp0/p.temp_scale

    # Initializing amplitude values that will be used to control perturbations
    # in the ensembles
    depth_amp = 0
    temp_amp = 0

    for ens in range(ensembles):

        if ens != 0:
            depth_pert = ensembles_r_control*(depth_amp * r.uniform(-1, 1))
            temp_pert = ensembles_r_control*(temp_amp * r.uniform(-1, 1))
            depth_ensemble[ens, 0] = depth0/p.depth_scale + depth_pert
            temp_ensemble[ens, 0] = temp0/p.temp_scale + temp_pert

        for i in range(nt-1):

            if mu_annual != 0:
                mu = mu_f(mu0, mu_annual, i*dt, tau)
                b = p.b0*mu

            if f_annual != 0:
                xi1 = wind_stress(f_annual, f_random, i*dt, tau, tau_cor, dt)

            depth_k1 = depth_f(depth_ensemble[ens, i], temp_ensemble[ens, i], p.r, p.a, b, xi1)
            temp_k1 = temp_f(depth_ensemble[ens, i], temp_ensemble[ens, i], p.gamma, en, b, xi1, p.xi2, p.c)

            depth_k2 = depth_f(depth_ensemble[ens, i] + depth_k1*dt_nondim/2, temp_ensemble[ens, i] + temp_k1*dt_nondim/2, p.r, p.a, b, xi1)
            temp_k2 = temp_f(depth_ensemble[ens, i] + depth_k1*dt_nondim/2, temp_ensemble[ens, i] + temp_k1*dt_nondim/2, p.gamma, en, b, xi1, p.xi2, p.c)

            depth_k3 = depth_f(depth_ensemble[ens, i] + depth_k2*dt_nondim/2, temp_ensemble[ens, i] + temp_k2*dt_nondim/2, p.r, p.a, b, xi1)
            temp_k3 = temp_f(depth_ensemble[ens, i] + depth_k2*dt_nondim/2, temp_ensemble[ens, i] + temp_k2*dt_nondim/2, p.gamma, en, b, xi1, p.xi2, p.c)

            depth_k4 = depth_f(depth_ensemble[ens, i] + depth_k3*dt_nondim, temp_ensemble[ens, i] + temp_k3*dt_nondim, p.r, p.a, b, xi1)
            temp_k4 = temp_f(depth_ensemble[ens, i] + depth_k3*dt_nondim, temp_ensemble[ens, i] + temp_k3*dt_nondim, p.gamma, en, b, xi1, p.xi2, p.c)

            depth_ensemble[ens, i+1] = depth_ensemble[ens, i] + dt_nondim*(depth_k1 + 2*depth_k2 + 2*depth_k3 + depth_k4)/6
            temp_ensemble[ens, i+1] = temp_ensemble[ens, i] + dt_nondim*(temp_k1 + 2*temp_k2 + 2*temp_k3 + temp_k4)/6

        if ens == 0:
            depth_amp = np.amax(depth_ensemble)
            temp_amp = np.amax(temp_ensemble)

    depth_redim = p.depth_scale*depth_ensemble.copy()
    temp_redim = p.temp_scale*temp_ensemble.copy()

    return depth_redim, temp_redim


def plot(scheme, time):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(scheme[0], scheme[1])
    ax1.set_title("Phase relation between thermocline depth and temperature", fontsize="10")
    ax1.set_xlabel("Thermocline depth anomaly[m]")
    ax1.set_ylabel("Temperature anomaly [K]")

    ax3 = ax2.twinx()
    ax2.plot(time, scheme[0], c='green', label="Depth")
    ax3.plot(time, scheme[1], c='orangered', label="Temperature")
    ax2.set_ylabel("Depth anomaly [m]", c='green')
    ax3.set_ylabel("Temperature anomaly [K]", c='orangered')
    ax2.set_xlabel("Time [months]")

    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.tight_layout()
    plt.show()


def ensemble_plot(ensemble_array, num_of_ensembles, periods, nt):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i in range(num_of_ensembles):
        ax1.plot(time_param(periods, nt)[2], ensemble_array[0][i])
        ax2.plot(time_param(periods, nt)[2], ensemble_array[1][i])

    ax1.set_title("Thermocline Depth Anomaly in %s Ensembles" %num_of_ensembles)
    ax1.set_xlabel("Time [months]")
    ax1.set_ylabel("Depth anomaly [m]")
    ax2.set_title("Thermocline Temperature Anomaly in %s Ensembles"
                  %num_of_ensembles)
    ax2.set_xlabel("Time [months]")
    ax2.set_ylabel("Temperature anomaly [K]")

    fig.set_figheight(5)
    fig.set_figwidth(10)
    fig.tight_layout()
    plt.show()
