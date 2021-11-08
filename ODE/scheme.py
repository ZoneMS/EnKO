import numpy as np

class Scheme(object):
    def __init__(self, dt, timestep, system, seed=121, xp_type="numpy"):
        self.dt = dt
        self.timestep = timestep
        self.system = system
        if xp_type=="numpy":
            self.xp = np
        elif xp_type=="cupy":
            import cupy
            self.xp = cupy
        self.xp.random.seed(seed)


    def _forward(self, system):
        pass


    def perfect_simulation(self, initial_x):
        whole_x = self.xp.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        whole_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(s*self.dt, current_x)
            whole_x[s+1] = current_x

        return whole_x


    def noise_added_simulation(self, initial_x, sys_sd, obs_sd):
        """Add observation noise
        """
        true_x = self.xp.zeros((self.timestep, len(initial_x)))
        obs_x = self.xp.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        true_x[0] = current_x
        obs_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(s*self.dt, current_x) + self.xp.random.normal(0, sys_sd, size=len(initial_x))
            true_x[s+1] = current_x
            obs_x[s+1] = current_x + self.xp.random.normal(0, obs_sd, size=len(initial_x))

        return true_x, obs_x
    
    
    def general_noise_added_simulation(self, initial_x, sys_noise_type="normal", obs_noise_type="normal", 
                                      sys_noise_param=[0.,1.], obs_noise_param=[0.,1.]):
        """Add observation noise
        """
        true_x = self.xp.zeros((self.timestep, len(initial_x)))
        obs_x = self.xp.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        true_x[0] = current_x
        obs_x[0] = current_x
        if sys_noise_type=="normal":
            sys_noise_func = self.xp.random.normal
        elif sys_noise_type=="exponential":
            sys_noise_func = self.xp.random.exponential

        for s in range(self.timestep-1):
            current_x = self._forward(s*self.dt, current_x) + sys_noise_func(*sys_noise_param, size=len(initial_x))
            true_x[s+1] = current_x
            obs_x[s+1] = current_x + self.xp.random.normal(*obs_noise_param, size=len(initial_x))

        return true_x, obs_x


    def noise_variant_simulation(self, initial_x, sd_rate):
        true_x = self.xp.zeros((self.timestep, len(initial_x)))
        obs_x = self.xp.zeros((self.timestep, len(initial_x)))
        current_x = initial_x.copy()
        true_x[0] = current_x
        obs_x[0] = current_x

        for s in range(self.timestep-1):
            current_x = self._forward(s*self.dt, current_x)
            true_x[s+1] = current_x
            obs_x[s+1] = current_x + self.xp.random.normal(0,
                                                sd_rate*self.xp.sqrt(self.xp.absolute(true_x[s+1] - true_x[s])),
                                                size=len(initial_x))

        return true_x, obs_x


    def get_transition_function(self, t=0):
        def transition_function(x, noise):
            return self._forward(x, t) + noise
        return transition_function




class EulerScheme(Scheme):
    def __init__(self, dt, timestep, system, seed=121, xp_type="numpy"):
        super(EulerScheme, self).__init__(dt, timestep, system, seed, xp_type)


    def _forward(self, t, x):
        return x + self.system.f(t, x) * self.dt



class ModifiedEulerScheme(Scheme):
    def __init__(self, dt, timestep, system, seed=121, xp_type="numpy"):
        super(ModifiedEulerScheme, self).__init__(dt, timestep, system, seed, xp_type)


    def _forward(self, t, x):
        k1 = self.system.f(t, x)
        k2 = self.system.f(t + self.dt, x + k1*self.dt)
        return x + self.dt * (k1 + k2) / 2




class RungeKuttaScheme(Scheme):
    def __init__(self, dt, timestep, system, seed=121, xp_type="numpy"):
        super(RungeKuttaScheme, self).__init__(dt, timestep, system, seed, xp_type)

    def _forward(self, t, x):
        k1 = self.system.f(t, x)
        k2 = self.system.f(t + self.dt/2, x + k1*self.dt/2)
        k3 = self.system.f(t + self.dt/2, x + k2*self.dt/2)
        k4 = self.system.f(t + self.dt, x + k3*self.dt)
        return x +  (k1 + 2*k2 + 2*k3 + k4) * self.dt / 6