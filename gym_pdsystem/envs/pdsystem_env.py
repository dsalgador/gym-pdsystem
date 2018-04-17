import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
from os import path

#import constants as ct
#import envs.constants as ct
import gym_pdsystem.utils.utilsq as ut
import gym_pdsystem.utils.constants as ct
import gym_pdsystem.utils.functions as fnc


# Example n=5, k = 2

TANK_MAX_LOADS = np.array([100., 200, 100., 800., 200.])
LEVEL_PERCENTAGES = np.array([ #b , c, e
                                                [0.02, 0.31, 0.9],
                                                [0.01, 0.03, 0.9],
                                                [0.05, 0.16, 0.9],
                                                [0.07, 0.14, 0.85],
                                                [0.08, 0.26, 0.9]
                                                   ])

TRUCK_MAX_LOADS = np.array([70.,130.])

GRAPH_WEIGHTS = np.array([32., 159., 162., 156.,156., 0.])

# Example k = 1
TRUCK_MAX_LOADS = np.array([50.])

DISCRETE = True

class PDSystemEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 1
    }

    def __init__(self, tank_max_loads = TANK_MAX_LOADS, 
    					level_percentages = LEVEL_PERCENTAGES,
    					truck_max_loads = TRUCK_MAX_LOADS,
    					graph_weights = GRAPH_WEIGHTS):
        """
        Default problem for n = 5 and k = 2
        """
        
        # Tanks' information
        self.tank_max_loads = tank_max_loads 

        self.n = len(self.tank_max_loads) 
        self.tank_ids = list(range(1,self.n+1))
        C_max = np.array([ [load] for load in self.tank_max_loads ])
        #tank_current_loads = np.full(n,0)
        self.tank_consumption_rates =  np.array([5.] * self.n)

        self.load_level_percentages = level_percentages

        for i in range(self.n):
        	self.tank_consumption_rates[i] = self.tank_max_loads[i] * (self.load_level_percentages[i][0] + self.load_level_percentages[i][1])/2.0

        self.tank_current_loads = self.tank_max_loads.copy()             
        self.tank_levels = np.multiply(C_max,self.load_level_percentages)


        # Trucks' information
        self.truck_max_loads = truck_max_loads

        self.k = len(self.truck_max_loads)
        self.truck_ids = list(range(self.k))
        self.truck_current_loads = self.truck_max_loads.copy()
        self.truck_current_positions =  np.array([self.n] * self.k)
        self.truck_fractions_deliverable =  np.array([ np.array([1.]), 
                                                  np.array([1.])
                                                ]) # we for now we only allow to deliver all t
        
        # World/Environment information

        self.graph = ut.simple_graph(self.n+1)
        self.w =  graph_weights
        self.weights_matrix = ut.simple_weights(self.n+1, self.w)
        
                
        ######
        #self.dt=.05
        self.viewer = None
        
        ### Actions
        self.a_shape = (self.k,self.n+1) # we have removed +1
        self.a_high =np.full(self.a_shape, 1) * self.truck_max_loads.reshape(self.k,1) #.reshape(self.a_shape)
        self.a_low = np.full( self.a_shape , 0)

        self.action_space = spaces.Box(low=self.a_low, high=self.a_high) #, shape= self.a_shape)
        
        self.a_high_clip = np.full(self.a_shape, 1) * self.truck_max_loads.reshape(self.k,1)
        
        ### States
        self.s_shape = (1,self.n)
        self.s_high = self.tank_max_loads.copy().reshape(self.s_shape)
        self.s_low = np.full( self.s_shape , 0)

        self.observation_space = spaces.Box(low=self.s_low, high=self.s_high) #, shape = self.s_shape)
      
        self.seed()

        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self, trucks_not_deliverying):
        
        def R_transport(coeff, w, u):
            return( coeff * np.sum(w*u) )
    
        def R_levels(p0 = ct.p0_GLOBAL, M = ct.M_GLOBAL, P1 = ct.P1_GLOBAL,  P2 = ct.P2_GLOBAL): #STILL TO DECIDE THE DEFAULT VALUES 

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

        R_total = ct.C_LEVELS * ( R_levels() + trucks_not_deliverying * ct.NOT_DELIVERYING_PENALTY ) #- ct.C_TRANSPORT * R_transport(ct.COEFF, w_t, u_t)
        
        return R_total

    def step(self,u):

        if not DISCRETE:
            """
            u == ((l_11, l_12,...,l_1n, l_1n+1), (l_21, l_22, ..., l_2n, l_2n+1)) (if k = 2 trucks)
            """
        
            u = u.reshape(self.a_shape)
            self.last_state = self.state.copy()

            u = np.clip(u, self.a_low, self.a_high_clip.reshape(self.a_shape)) #[0]
            self.last_u = u.reshape(u.shape[0] * u.shape[1],) # for rendering
            
            # Go to next state
            visited_ids = np.argmax(u , axis = 1)

            trucks_not_deliverying = 0

            for i in range(self.k):
                tank_visited = visited_ids[i]

                if tank_visited != self.n:
                	#self.state[tank_visited] = np.minimum(self.state[tank_visited] + u[i][tank_visited], self.tank_max_loads[tank_visited])
                	hypothetical_next_tank_state = self.state[tank_visited] + self.truck_max_loads[i]
                	if hypothetical_next_tank_state <= self.tank_max_loads[tank_visited]:
                			self.state[tank_visited] = hypothetical_next_tank_state
                	else:	
                    		trucks_not_deliverying = trucks_not_deliverying + 1
     
            # Tanks lower its load due to consumption rates
            self.state = np.maximum(0, self.state - self.tank_consumption_rates)
                    
            costs = self.reward(trucks_not_deliverying)

            #termination = False
            #Terminate if some tank is empty
             #if len(np.nonzero(self.state)[0]) != self.n:
            	#print(len(np.nonzero(self.state)[0]))
            	#termination = True
            return self._get_obs(), costs, False, {} # WITH THE MINUS?

        else:
                return self.step_discrete(u)




        



    def step_discrete(self,u):
        """
        u == (tank_1, tank_2  ) (if k = 2 trucks) tank_i \in {0,...,n-1,n}, n means stay in the depot
        """
    
        #u = u.reshape(self.a_shape)
        self.last_state = self.state.copy()

        #u = np.clip(u, self.a_low, self.a_high_clip.reshape(self.a_shape)) #[0]
        #self.last_u = u.reshape(u.shape[0] * u.shape[1],) # for rendering
        
        # Go to next state
        #visited_ids = np.argmax(u , axis = 1)

        trucks_not_deliverying = 0

        for i in range(self.k):
            print('action', u[i])
            tank_visited = u[i]

            if tank_visited != self.n:
            	#self.state[tank_visited] = np.minimum(self.state[tank_visited] + u[i][tank_visited], self.tank_max_loads[tank_visited])
            	hypothetical_next_tank_state = self.state[tank_visited] + self.truck_max_loads[i]
            	if hypothetical_next_tank_state <= self.tank_max_loads[tank_visited]:
            			self.state[tank_visited] = hypothetical_next_tank_state
            	else:	
                		trucks_not_deliverying = trucks_not_deliverying + 1
 
        # Tanks lower its load due to consumption rates
        self.state = np.maximum(0, self.state - self.tank_consumption_rates)
                
        costs = self.reward(trucks_not_deliverying)

        termination = False
        #Terminate if some tank is empty
         #if len(np.nonzero(self.state)[0]) != self.n:
        	#print(len(np.nonzero(self.state)[0]))
        	#termination = True

        return self._get_obs(), costs, termination, {} # WITH THE MINUS?    

    def reset(self):
       
        #self.truck_current_positions =  np.array([self.n] * k)

        for i, tank_levels in enumerate(self.tank_levels):
                a = tank_levels[0]
                b = tank_levels[-1]
                #current_load = 0.75 * (a+b)/2.0# np.random.randint(a+1,b, size =(1,)) GIVES A STRANGE ERROR
                current_load = np.random.random() * (b - a-1) + a+1 #np.random.randint(a+1,b)
                self.tank_current_loads[i] = current_load * 1.0
                
        self.state = self.tank_current_loads# / self.tank_max_loads 
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        bl_y = -screen_height/2.0 + 50
        br_y = bl_y
        bl_x = -screen_width/2.0
        br_x = screen_width/2.0

        tankwidth = 30.0
        tankheight = screen_height

        tanky = 40# TOP OF CART
        tankx = 50

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            #.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            l,r,t,b = -tankwidth/2, tankwidth/2, tankheight/2, -tankheight/2

            #tank = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            tank = rendering.FilledPolygon([(0,0), (0,tankheight), (tankwidth,tankheight), (tankwidth,0)])

            self.tanktrans = rendering.Transform()
            tank.add_attr(self.tanktrans)
            tank.set_color(.5,.5,.8)
            self.viewer.add_geom(tank)


            #self.track = rendering.Line( (0,100), (screen_width, 100 ))
            self.track = rendering.Line( (0,100), (screen_width, 100 ))

            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)


        if self.state is None: return None    
        
        y = self.state[0]
        #self.tanktrans.set_scale(tankx, x)
        #self.tanktrans.set_scale(tankx, 13)
        #self.tanktrans.set_translation(tankx, tanky)
        #self.tanktrans.set_translation(0, 0)
        self.tanktrans.set_scale(1,y/100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer: self.viewer.close()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)