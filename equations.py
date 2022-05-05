import numpy as np

class Leader:
    def __init__(self, zeta=0):
        self.zeta = np.array([[zeta]])
        self.A = np.array([[0.8607, 0],[0.0929, 1]])
        self.B = np.array([[0.1393, 0.0929], [0.0071, 0.0048]])
        self.C = np.array([[1, 0], [0, 1]])
        self.D = np.array([[0, 0], [0, 0]])
    
    def step(self, prev_state, vref):
        u_k = np.append(vref, self.zeta, axis=1)
        # Leader Difference Equation: x(k+1) = A*x(k) + B*u_k
        next_state = np.matmul(self.A, prev_state.T) + np.matmul(self.B, u_k.T) 
        return next_state.T

class Follower:
    def __init__(self):
        self.A = np.array([[0.9002, -0.0950],[0.0950, 0.9952]])
        self.B = np.array([[-0.0950, 0.0950, 0.0950], [-0.0048, 0.0048, 0.0048]])
        self.C = np.array([[1, 0], [0, 1]])
        self.D = np.array([[0, 0, 0], [0, 0, 0]])
    
    def step(self, prev_state, leader_state, delta):
        u_k = np.append(delta, leader_state, axis=1)
        # Leader Difference Equation: x(k+1) = A*x(k) + B*u_k
        next_state = np.matmul(self.A, prev_state.T) + np.matmul(self.B, u_k.T)
        return next_state.T

class NewSystem:
    def __init__(self):
        self.A = np.array([[1, 2],[-2, 1]])
        self.B = np.array([[0.5], [1]])
        self.K = np.array([[1.4720, -1.0360]])
    
    def step(self, prev_state, pred):
        u_k = np.matmul(self.K, prev_state.T)
        # Leader Difference Equation: x(k+1) = A*x(k) + B*u_k
        next_state = np.matmul(self.A, prev_state.T) + np.matmul(self.B, u_k) + pred.T
        # u_k_1 = np.matmul(self.K, next_state)
        # next_sa = np.concatenate((next_state, u_k_1), axis=0)
        # return next_sa.T
        return next_state.T