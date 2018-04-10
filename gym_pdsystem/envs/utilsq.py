import numpy as np
import pickle
from matplotlib import animation, rc
import matplotlib.pyplot as plt

def is_key(dic, key):
    if key in dic:
        return(True)
    else: return(False)

def simple_graph(n: int):    
        A = np.zeros((n,n))
        A[(n-1),0:n] = 1
        A = A.astype(int)
        return(A)
    
def simple_weights(n: int, w: float):    
        W = np.full((n,n), np.inf)
        W[(n-1),:] = w
        return(W)    

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def create_system_animation(visualization_steps, n_iterations, skip = 1):
    
    def barlist(n, visualization_steps = visualization_steps): 
        return visualization_steps[n][2]

    fig=plt.figure()

    N=int(n_iterations/skip) #Number of frames
    x=visualization_steps[0][0]+1

    plt.bar(x,visualization_steps[0][1], color = 'black')
    barcollection = plt.bar(x,barlist(0), color = 'blue')

    def animate(i):
        y=barlist(i+1)
        for i, b in enumerate(barcollection):
            b.set_height(y[i])

    anim=animation.FuncAnimation(fig,animate,repeat=False,blit=False,frames=N-1,
                                 interval=100)
    return(anim)    