import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from similaritymeasures import dtw

def get_str(lst):
    x,y = [int(i) for i in lst.split('-')]
    return '%d-%d'%(min(x,y), max(x,y))

class similarity:
    def __init__(self, d):
        self.d = d

    def calc_similarity(self, a, b):
        x = list(range(24))
        curve_a = np.array([[i,j] for i,j in zip(x,self.d[get_str(a)])])
        curve_b = np.array([[i,j] for i,j in zip(x,self.d[get_str(b)])])
        return dtw(curve_a, curve_b)[0]

    def plt_curves(self, x, fig_name=1):
        a, b = x[:2]
        if fig_name: plt_(self.d[a], self.d[b], fig_name=[fig_name, x[2], id])

def plt_(a,b, fig_name=''):
    plt.figure(figsize=(10,6))
    x = list(range(1,25))
    mark_size = 5
    plt.plot(x , a, color='green', marker='o', linestyle='--', linewidth=2, label='Curve_1', ms = mark_size)
    plt.plot(x , b, color='red', marker='^', linestyle='-', linewidth=2, label='Curve_2', ms = mark_size)

    plt.legend(markerscale=1.5, scatterpoints=10, fontsize=14)
    if fig_name[0]:plt.savefig('%-.1f %d.pdf'%(fig_name[1], fig_name[2]), bbox_inches='tight')
    plt.show()



def get_hours_curve_data(frame):
    lst = [0]*24
    g = frame.groupby('Start Hour')
    for h,group in g:
        lst[h]=group.shape[0]
    return lst

