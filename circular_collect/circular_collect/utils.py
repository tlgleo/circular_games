import numpy as np
from scipy.stats import expon, uniform, gamma
import math


"""Functions for radial distribution for ball position around the players"""
def radial_dist(size, r_mean):
    """Exponential and uniform distribution for r and theta"""
    r = expon.rvs(scale=r_mean, size=size)
    theta = uniform.rvs(0, 2*math.pi, size=size)
    return r, theta

def gamma_dist(t, tau_remove_balls, a = 3):
    proba = gamma.cdf(t-tau_remove_balls, a=a, loc=0, scale=1)
    return np.random.binomial(1,proba)

def expo_dist(t, tau_remove_balls):
    proba = expon.cdf(t-tau_remove_balls)
    return np.random.binomial(1,proba)

def sample_position(pos, r_mean):
    """Sample of size 1 from position pos = (i,j)"""
    r, theta = radial_dist(1, r_mean)
    r = 1 + r[0]
    theta = theta[0]
    delta = (r * math.cos(theta), r * math.sin(theta))
    delta = np.round(delta)
    output_sample = (int(pos[0]+delta[0]) , int(pos[1]+delta[1]))
    return output_sample

def sample_positions(pos, size, r_mean):
    """Sample of size >=1 from position pos = (i,j)"""
    r, theta = radial_dist(size, r_mean)
    new_r = 1 + r
    (i,j) = pos
    sample_deltas = [(r_i * math.cos(theta_i), r_i * math.sin(theta_i)) for (r_i, theta_i) in zip(new_r, theta)]
    sample_deltas = np.round(sample_deltas)
    output_sample = [(int(i+x) , int(j+y)) for (x,y) in sample_deltas]
    return output_sample


CIRC_PROBAS_4P_1C = \
    [[0.25,0.50,0.25,0.00],
      [0.00,0.25,0.50,0.25],
      [0.25,0.00,0.25,0.50],
      [0.50,0.25,0.00,0.25]
      ]

#print(CIRC_PROBAS_4P_1C)

i_p = 0
#print(np.random.choice(4, p=CIRC_PROBAS_4P_1C[i_p]))


