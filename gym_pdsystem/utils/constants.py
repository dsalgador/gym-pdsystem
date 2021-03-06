import numpy as np

COEFF = 0.0075 * 580/1000 * 11.86

C_TRANSPORT = 0 # 0 implies no transport contribution
C_LEVELS = 10**(-6)

global p0_GLOBAL
global P1_GLOBAL
global P2_GLOBAL
global M_GLOBAL

p0_GLOBAL = 0.7

P1_GLOBAL = -10**3
P2_GLOBAL = -10**6

M_GLOBAL = 10**1

NOT_DELIVERYING_PENALTY = P2_GLOBAL #to be equivalent/same importance as having 0 stock or surpassing max capacity levels

CASE = 1
STOCHASTIC = False
STOCHASTIC_PERCENTAGE = 0.10

if CASE  == 1:
	###########################################################
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
	#DISCRETE = False
	DISCRETE = True

	############################################################
elif CASE == 2:

	###########################################################
	# Example n=9, k = 3

	TANK_MAX_LOADS = np.array([100., 200, 100., 800., 200., 500., 300., 800., 300.])
	LEVEL_PERCENTAGES = np.array([ #b , c, e
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.08, 0.26, 0.9],
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85]
	                                                   ])

	TRUCK_MAX_LOADS = np.array([70.,130., 210.])

	GRAPH_WEIGHTS = np.array([32., 159., 162., 156.,156., 32., 159., 162., 156., 0.])
	DISCRETE = True
	############################################################
	############################################################
elif CASE == 3:

	###########################################################
	# Example n=11, k = 3

	TANK_MAX_LOADS = np.array([100., 200, 100., 800., 200., 500., 300., 800., 300.,600.,900.])
	LEVEL_PERCENTAGES = np.array([ #b , c, e
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.08, 0.26, 0.9],
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.01, 0.03, 0.9],
	                                                [0.01, 0.03, 0.9]

	                                                   ])

	TRUCK_MAX_LOADS = np.array([70.,130.,210.])

	GRAPH_WEIGHTS = np.array([32., 159., 162., 156.,156., 32., 159., 162., 156.,150.,150., 0.])
	DISCRETE = True
	############################################################
elif CASE == 4:
	###########################################################
	# Example n=12, k = 3

	TANK_MAX_LOADS = np.array([100., 200, 100., 800., 200., 500., 300., 800., 300.,600.,900.,700.])
	LEVEL_PERCENTAGES = np.array([ #b , c, e
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.08, 0.26, 0.9],
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.01, 0.03, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.07, 0.14, 0.85]


	                                                   ])

	TRUCK_MAX_LOADS = np.array([70.,130.,210.])

	GRAPH_WEIGHTS = np.array([32., 159., 162., 156.,156., 32., 159., 162., 156.,150.,150.,150., 0.])
	DISCRETE = True
	############################################################
elif CASE == 5:
	###########################################################
	# Example n=12, k = 4

	TANK_MAX_LOADS = np.array([100., 200, 100., 800., 200., 500., 300., 800., 300.,600.,900.,700.])
	LEVEL_PERCENTAGES = np.array([ #b , c, e
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.08, 0.26, 0.9],
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9],
	                                                [0.07, 0.14, 0.85],
	                                                [0.01, 0.03, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.07, 0.14, 0.85]

	                                                   ])

	TRUCK_MAX_LOADS = np.array([70.,130.,250., 200.])

	GRAPH_WEIGHTS = np.array([32., 159., 162., 156.,156., 32., 159., 162., 156.,150.,150.,150., 0.])
	DISCRETE = True
	############################################################	
else:

	#####################################
	# Example k = 1, n = 3
	TANK_MAX_LOADS = np.array([100., 200, 100.])
	LEVEL_PERCENTAGES = np.array([ #b , c, e
	                                                [0.02, 0.31, 0.9],
	                                                [0.01, 0.03, 0.9],
	                                                [0.05, 0.16, 0.9]
	                                                   ])
	TRUCK_MAX_LOADS = np.array([50.])
	GRAPH_WEIGHTS = np.array([32., 159., 162., 0.])

	DISCRETE = True
	########################################