COEFF = 0.0075 * 580/1000 * 11.86

C_TRANSPORT = 0.0 # 0 implies no transport contribution
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
