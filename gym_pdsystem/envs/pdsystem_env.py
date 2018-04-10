import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

import utilsq as ut
import functions as fnc

COEFF = ct.COEFF

C_TRANSPORT = ct.C_TRANSPORT
C_LEVELS = ct.C_LEVELS

p0_GLOBAL = ct.p0_GLOBAL

P1_GLOBAL = ct.P1_GLOBAL
P2_GLOBAL = ct.P2_GLOBAL

M_GLOBAL = ct.M_GLOBAL

NOT_DELIVERYING_PENALTY = P2_GLOBAL #to be equivalent/same importance as having 0 stock or surpassing max capacity levels



class PDSystemEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        """
        Default problem for n = 5 and k = 2
        """
        
        # Tanks' information
        self.n = 5 
        self.tank_ids = list(range(1,n+1))
        self.tank_max_loads =  np.array([100., 200, 100., 800., 200.])
        C_max = np.array([ [ 100],[200],[100],[800],[200] ])
        #tank_current_loads = np.full(n,0)
        self.tank_consumption_rates =  np.array([5.] * n)

        self.load_level_percentages = np.array([ #b , c, e
                                                [0.02, 0.31, 0.9],
                                                [0.01, 0.03, 0.9],
                                                [0.05, 0.16, 0.9],
                                                [0.07, 0.14, 0.85],
                                                [0.08, 0.26, 0.9]
                                                   ])
        self.tank_current_loads = self.tank_max_loads.copy()             
        self.tank_levels = np.multiply(C_max,self.load_level_percentages)


        # Trucks' information
        self.k = 2
        self.truck_ids = list(range(k))
        self.truck_max_loads = np.array([70.,130.])
        self.truck_current_loads = truck_max_loads.copy()
        self.truck_current_positions =  np.array([self.n] * k)
        self.truck_fractions_deliverable =  np.array([ np.array([1.]), 
                                                  np.array([1.])
                                                ]) # we for now we only allow to deliver all t
        
        # World/Environment information

        self.graph = ut.simple_graph(self.n+1)
        self.w =  np.array([32., 159., 162., 156.,156., 0.])
        self.weights_matrix = ut.simple_weights(n+1, w)
        
                
        ######


        self.dt=.05
        self.viewer = None
        
        ### Actions
        self.a_low = 0
        sel.a_high = self.truck_max_loads.copy()
        self.a_shape = (2,6)

        self.action_space = spaces.Box(low=self.a_low, high=self.a_high, shape= self.a_shape)
        
        self.a_high_clip = np.full(a_shape, 1) * self.truck_max_loads.reshape(2,1)
        
        ### States
        self.s_low = 0
        sel.s_high = self.tank_max_loads.copy()
        self.s_shape = (1,5)

        self.observation_space = spaces.Box(low=self.s_low, high=self.s_high, shape = self.s_shape)

        
        
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self):
        
        def R_transport(self, coeff, w, u):
            return( coeff * np.sum(w*u) )
    
        def R_levels(self, p0 = p0_GLOBAL, M = M_GLOBAL, P1 = P1_GLOBAL,  P2 = P2_GLOBAL): #STILL TO DECIDE THE DEFAULT VALUES 

            R = 0

            for i in range(self.n):
                percentages = self.load_level_percentages[i]
                b = percentages[0]
                c = percentages[1]
                e = percentages[2]

                a = b/10
                f = 1-(1-e)/10
                d = p0*e+(1-p0)*c

                C_max = self.tank_max_loads[i]
                x = self.state[i]

                R = R + fnc.R_lvl(x, C_max, a,b,c,d,e,f,P1,P2,M)

            return(R)  

        R_total = C_LEVELS * R_levels() #- C_TRANSPORT * self.R_transport(COEFF, w_t, u_t) 
        #+ trucks_not_deliverying * NOT_DELIVERYING_PENALTY
        
        return R_total

    def step(self,u):
        """
        u == ((l_11, l_12,...,l_1n, l_1n+1), (l_21, l_22, ..., l_2n, l_2n+1)) (if k = 2 trucks)
        """
        #th, thdot = self.state # th := theta

        #dt = self.dt
        self.last_state = self.state.copy()

        u = np.clip(u, self.a_low, self.a_high_clip) #[0]
        self.last_a = u # for rendering
        
        # Go to next state
        #newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        #newth = th + newthdot*dt
        #newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        visited_ids = np.argmax(u , axis = 1)
        
        for i in range(self.k):
            tank_visited = visited_ids[i]
            if tank_visited != self.n:
                self.state[tank_visited] = np.min(self.state[tank_visited] - u[i][tank_visited] 
                                                  , self.tank_max_loads[tank_visited])
        costs = self.reward()

        return self._get_obs(), -costs, False, {} # WITH THE MINUS?

    def reset(self):
       
        #self.truck_current_positions =  np.array([self.n] * k)

        for tank_levels in zip(self.tank_levels):
                a = tank_levels[0]
                b = tank_levels[-1]
                current_load = np.random.randint(a+1,b)*1.0
                self.tank_current_loads[i] = current_load 
                
        self.state = self.tank_current_loads   
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            import numpy as np
            import matplotlib.pyplot as plt
        plt.axis([0, 3, 0, 1.2])


        index = np.array([i for i in range(3)]) + 0.5
        plt.bar(index, 1, color = "Black")
        colors = np.random.uniform(0,1,size = 3)
        plt.bar(index, colors, color = "blue")
            

        #self.viewer.add_onetime(self.img)
        #self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        #if self.last_u:
         #   self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)
        return(plt.show())
        #return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def render2(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
           # self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        #if self.last_u:
         #   self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)