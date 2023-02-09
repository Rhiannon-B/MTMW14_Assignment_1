import Parameters as p
import Functions as f

periods = 1
nt_nondim = 100
mu = 2/3

f.plot(f.runge_kutta(periods, nt_nondim, mu), f.time_param(periods, nt_nondim)[2])
