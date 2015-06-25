import numpy as np
from sympy import Matrix
import random as random
import math 
import pylab as plt
from numpy import linalg as LA
from convexhullcoeff_cvx import convexhullcoeff_cvx
from RIP_cond_norm import RIP_cond_norm
from sparse_vec import sparse_vec
from convex_hull_volume import convex_hull_volume
from lasso_cvx import lasso_cvx
import cvxpy as cvx
from RIP_cond_norm import RIP_cond_norm
import matplotlib.animation as animation
import cvxpy as cvx
from cvxpy import *

def sparse_vec(n, N, K, mu, sigma, p):
	"n : Dimensionality"
	"N : Number of examples"
	"K : Sparsity"
	"mu : mean"
	"sigma : Variance"
	"p : Hist of index"

	X_al = np.zeros((n,N))
	Y_nal = np.zeros((n,N))
	sample = np.arange(0,n)
	sample_index_al = random.sample(sample, p)
	sample_index_nal = random.sample(sample, n)
	
	for i in range(0, N):
		index = np.random.choice(sample_index_al, K, replace = False)
		index2 = np.random.choice(sample_index_nal, K, replace = False)
		#X[index, i] =  np.random.normal(mu, sigma, n)[: K]
		X_al[index, i] =  random.randint(1, mu)
		Y_nal[index2, i] =  random.randint(1, mu)
		#X[index, i] =  1
	return X_al, Y_nal

def w_representation(X, k_alpha):
	N = X.shape[1]
	sample = np.arange(0,N)
	index = random.sample(sample, k_alpha)
	alpha = np.zeros((N,1))
	alpha[index] = 1
	w = np.dot(X , alpha)
	return w, alpha

def upper_bound(w1, w2, R1, R2, epsilon):
	u_b = (1 - epsilon) *(np.dot(w1.T , w2)) + (R1 + R2) * epsilon 
	return u_b

def lower_bound(w1, w2, R1, R2, epsilon):
	l_b = (1 + epsilon) *( np.dot(w1.T , w2)) - (R1 + R2) * epsilon 
	return l_b

def norm_2(X):
	X = np.reshape(X,[-1])
	norm = np.dot(X.T, X)
	return norm

def cvx_Ab(w, A):
	"min { ||w - A * b||^2 }"
	r = w.shape[0]
	c = A.shape[1]
	b = cvx.Variable(c,1)
	Objective = cvx.Minimize(0.5 * sum_squares(w - A * b))
	prob = cvx.Problem(Objective, Constraints)
	prob.solve()
	output = b.value
	return output

n = 1000
N = 100
mu = 10
sigma = 0
K = 5
"In this case sparsity K change"
#for K in range(2, 100):
x_al,y_nal = sparse_vec(n, N, K, mu, sigma, K)

x_al = np.array(x_al)
y_nal = np.array(y_nal)
w = w_representation(x_al, 100)
w = np.array(w)
b_al = cvx_Ab(w, x_al)
