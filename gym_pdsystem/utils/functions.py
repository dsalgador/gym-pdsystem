import numpy as np


def str_line1(x,b,c,P1):
    m = 1
    n = -c
    return(-P1*(m*x+n)/(c-b))

def str_line2(x,c,d,M):
    m = 1
    n = -c
    return(M*(m*x+n)/(d-c))

def str_line3(x,d,e,M):
    m = 1
    n = -e
    return(M*(m*x+n)/(d-e))

def exp1(x,a,b,P1,P2):
    A = -P1*(P2/P1)**(1/(1-b/a))
    l1 = (1/a-1/b)**(-1) * np.log(P2/P1)
    return(-A*np.exp(l1/x))

def exp2(x,e,f,P1,P2):
    B = -P1*(P2/P1)**( 1/(1-(1-e)/(1-f)) )
    l2= (1/(1-f)-1/(1-e))**(-1) * np.log(P2/P1)
    return(-B * np.exp(l2/(1-x)) ) 


def R_lvl(x, C_max, a,b,c,d,e,f,P1,P2,M):
    if(x>C_max):
        raise Exception("Load x is greater than C_max in R_levels.")
    x = x/C_max
    
    conditions = [(x>=0) & (x < a),
              (x>=a) & (x<b),
              (x>=b) & (x<c),
              (x>=c) & (x<d),
              (x>=d) & (x<e),
              (x>=e) & (x<f),
              (x>f) & (x<=1)]

    functions = [P2,
             lambda x: exp1(x,a,b,P1,P2),
             lambda x: str_line1(x,b,c,P1),
             lambda x: str_line2(x,c,d,M),
             lambda x: str_line3(x,d,e,M),
             lambda x: exp2(x,e,f,P1,P2),
             P2]
    
    return(np.piecewise(x,conditions, functions))