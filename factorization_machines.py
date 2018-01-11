import numpy as np
import scipy.sparse as sp
import math
from scipy.special import expit as sigmoid
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin

class FactorizationMachine(BaseEstimator):
  def __init__(self, mini_batch=10, n_iter=10, rank=8, random_state=123, l2_reg_w=0.1, l2_reg_V=0.1, init_step_size=0.1, eps=1.0e-8):
    self.mini_batch = 10
    self.n_iter = n_iter
    self.random_state = random_state
    self.rank = rank
    # self.warm_start = False
    self.l2_reg_w = l2_reg_w
    self.l2_reg_V = l2_reg_V
    self.step_size = init_step_size
    self.copy_X = copy_X
    self.eps = eps

  def fit(self, X, y):
    size, n_features = X.shape

    g2_w0 = eps
    g2_w = np.zeros(n_features) + self.eps
    g2_V = np.zeros([n_features, self.rank]) + self.eps

    self.w0 = np.random.normal(size=1)
    self.w = np.random.normal(size=n_features)
    self.V = np.random.normal(size=(n_features, self.rank))

    for i in range(self.n_iter): 
      for X, y in make_batch(X, y, self.mini_batch):
        y_err = (sigmoid(self.predict(X)*y) - 1) * y

        self.w0, g2_w0 = adagrad(self.w0, self.g2_w0, y_err)

        y_err = np.repeat(y_err, n_features, axis=1)
        self.w, g2_w = adagrad(self.w, self.g2_w, y_err * X)

        y_err = np.repeat(np.reshape(y_err, [-1, n_features, 1]), (1 ,1, self.rank))
        
        expandedX = np.repeat(np.reshape(X, [-1, n_features, 1], (1, 1, self.rank)))
        gradient_V = np.matmul(X, self.V)
        gradient_V = np.repeat(np.reshape(gradient_V, [-1, 1, self.rank]), (1, n_features, self.rank))
        gradient_V -= self.V * expandedX
        gradient_V *= expandedX
        self.V, g2_V = adagrad(self.V, self.g2_V, y_err * gradient_V)

    return self

  def gradient_V(self, X):
    Vscore = X * np.matmul(X, self.V)
    Vscore = np.sum(Vscore, axis=1)

    return self.w0 + np.matmul(X, w) + Vscore

  def adagrad(self, w, g2, g):
    g = np.sum(g)
    g2 += g*g
    w -= step_size * g / math.sqrt(g2)
    return w, g2

  def predict_proba(self, X):
    return sigmoid(self.predict(X))

  def predict(self, X):
    Vscore = np.matmul(X, self.V) - np.matmil(np.power(X, 2), np.power(self.V, 2))
    Vscore = np.sum(Vscore, axis=1)

    return self.w0 + np.matmul(X, w) + Vscore

def make_batch(X, y, batch):
  size = math.ceil(X.shape[0] / batch)
  for i in range(size):
    start = i * batch
    end = (i+1) * batch
    X_ = X[start:end,:]
    y_ = y[start:end,:]
    yield X_, y_
