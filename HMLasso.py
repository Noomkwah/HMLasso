# -*- coding: utf-8 -*-
"""file_04_HMLasso.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ldozo1jkpXx1aD0vcVKkrEmQ_YZV_VaV

# Implementation of the Lasso With High Missing Rate.

The goal of this notebook is to implement the lasso with high missing rate described [here](https://www.ijcai.org/proceedings/2019/0491.pdf).

## Imports
"""

# Imports
import numpy as np
import pandas as pd

import cvxpy as cp
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

"""## HMLasso"""

ERRORS_HANDLING = "raise"

class HMLasso():
  """
  Lasso regularization that performs well with high missing rate.

  Implemented according to the related article 'HMLasso: Lasso with High Missing
  Rate' by Masaaki Takada1, Hironori Fujisawa and Takeichiro Nishikawa.
  Link to the article: https://www.ijcai.org/proceedings/2019/0491.pdf

  ------------
  Common uses: Once fitted, the HMLasso can provide linear predictions. 
  It can also be used to select variables of interest from the given data. This 
  second goal can be achieved through selection of variables whose coefficient
  is almost (or equal to) zero.

  Please note that no metric is implemented in this class for now. 
  See sklearn.metrics.mean_squared_error or like for useful metrics.

  ------------
  Common error: During the fitting HMLasso.fit(X, y), errors such as
  'ArpackNoConvergence: ARPACK error -1: No convergence' may occur. It comes 
  from the fact that the underlying solver used did not successfully assess
  the positive-semidefiniteness of the inner variable Sigma_opt. If you are 
  sure that Sigma_opt is PSD (which is likely to be the case in normal uses of 
  the estimator), you can add the two following lines:
  "
  from file_04_HMLasso import ERRORS_HANDLING
  ERRORS_HANDLING = 'ignore'
  "
  If the problem persists, then maybe praying god is the only remaining thing
  you can do.

  ------------
  Parameters:
      mu : float/int, default=1.0: the hyperparameter that control how
      parcimonious the model shall be. The larger mu is, the greater the
      regularization will be (hence the calculated beta_opt might 
      present more nullified coefficients). mu must be positive.
      
      alpha : float/int, default=1: the hyperparameter that control weights
      importance. Be wary that setting alpha > 5 can make convergence way
      slower, as the weights become closer and closer to 0 and as the numerical
      solver has more and more trouble converging.
      One may prefer setting alpha in the range [0., 3.]. Common values
      of alpha are 0., 0.5, 1. with the latter experimentally delivering best
      performances. alpha must be positive.
      See source article for more.

      fit_intercept : bool, default=True: 

      verbose : bool, default=False: control whether the verbose is dispayed.
      Set verbose = True for it to be printed.
  
  ------------
  Methods:
      fit(self, X, y, errors="ignore"):
        Fit the HMLasso on (X, y)
        X, the features, must be a mean-centered numpy array of shape (n, p)
        y, the labels, must be a vector of shape (n, 1) or (n,)

        Do not return anything. However, once the fitting is done, one can
        use 'predict' method to predict any given output using the linear model.
      
      predict(self, X):
        Predict using linear model.
        Return the predicted vector.
  
  ------------
  Constants:
      beta_opt: the estimator.


  """

  global ERRORS_HANDLING

  def __init__(self, mu=1, alpha=1, fit_intercept=True, verbose=False):

    assert isinstance(mu, (int, float)), "mu must be a number."
    assert isinstance(alpha, (int, float)), "alpha must be a number."
    assert isinstance(fit_intercept, bool), "fit_intercept must be a boolean."
    assert isinstance(verbose, bool), "verbose must be a boolean."
    assert mu >= 0, "mu must be a positive number."
    assert alpha >= 0, "alpha must be a positive number."

    self.mu = mu
    self.alpha = alpha
    self.verbose = verbose
    
    self.n = None
    self.p = None
    self.S_pair = None
    self.rho_pair = None
    self.R = None
    self.Sigma_opt = None
    self.beta_opt = None

    self.isFirstProblemSolved = False
    self.isSecondProblemSolved = False # Unused at the moment.
    self.isFitted = False
  
  def predict(self, X):
    """
    Predict using the linear model.

    ------------
    Parameters:
        X : 2D numpy array

    Returns:
        y : 1D numpy array
    """

    assert self.isFitted, "The model has not yet been fitted."
    assert X.shape[1] == self.p, f"Given data is of dimension {X.shape[1]}. Must have dimension {self.p})."
    assert not np.isnan(X).any(), "Input contains NaN."

    return np.dot(X, self.beta_opt)
  
  def fit(self, X, y):
    """
    Fit the HMLasso on (X, y).

    ------------
    Parameters:
        X : 2D numpy array, shape (n,p). It corresponds to the features, and
        must be mean-centered.
        y : 1D numpy array, shape (n,1) or (n,). It corresponds to the labels.

    Returns:
        None
    """
    
    assert type(X) == np.ndarray, "Features are not a numpy array."
    assert type(y) == np.ndarray, "Labels are not a numpy array"
    assert X.shape[0] == y.shape[0], "Features and labels shapes are not compatibles."
    assert len(y.shape) == 1, "Labels are not a vector."

    self.n, self.p = X.shape    
    self.__verify_centering__(X)
    self.S_pair, self.rho_pair, self.R = self.__impute_params__(X, y)
    self.Sigma_opt = self.__solve_first_problem__()

    # It appears that, due to floating points exceptions, Sigma_opt is not always
    # Positive semidefinite. Hence, we shall check it.
    eigenvalues = np.linalg.eig(self.Sigma_opt)[0]
    min_eigenvalue = min(eigenvalues)
    if min_eigenvalue < 0:
      print(f"[Warning] Sigma_opt is not PSD, its minimum eigenvalue is {min_eigenvalue}. Error handled by adding {-min_eigenvalue} to each eigenvalue.")
      self.Sigma_opt = self.Sigma_opt - min_eigenvalue * np.eye(self.p, self.p)
    
    # Sigma_opt may ill-typed data: some coefficients may appear comlex-valued
    # while they are not. We fix this issue. Example: 5.23 + 0.0j becomes 5.23.
    self.Sigma_opt = np.real(self.Sigma_opt)

    if ERRORS_HANDLING == "ignore":
      self.Sigma_opt = cp.psd_wrap(self.Sigma_opt)
    
    self.beta_opt = self.__solve_second_problem__()

    self.isFitted = True

    if self.verbose:
      print("Model fitted.")

  def __verify_centering__(self, X, tolerance=1e-8):
    for col in range(self.p):
      current_mean = X[:, col].mean()
      if abs(current_mean) > tolerance:
        raise Exception(f"Data is not centered: column {col} has mean of {current_mean}")
  
  def __impute_params__(self, X, y):

    if self.verbose:
      print("[Imputing parameters] Starting...")

    Z = np.nan_to_num(X)
    Y = (Z != 0).astype(int)
    R = np.dot(Y.T, Y)
    if self.verbose:
      print("[Imputing parameters] R calculated.")

    rho_pair = np.divide(np.dot(Z.T, y), R.diagonal(), out=np.zeros((self.p,)), where=(R.diagonal()!=0))
    if self.verbose:
      print("[Imputing parameters] rho_pair calculated.")

    S_pair = np.divide(np.dot(Z.T, Z), R, out=np.zeros((self.p, self.p)), where=(R!=0))
    if self.verbose:
      print("[Imputing parameters] S_pair calculated.")

    R = R / self.n

    if self.alpha > 5:
      print("[Warning] The hyperparameter alpha={} is large (greater than 5), which might make convergence way slower.")
    R = np.power(R, self.alpha)

    if self.verbose:
      print("[Imputing parameters] Parameters imputed.")

    return S_pair, rho_pair, R


  def __solve_first_problem__(self):
    
    assert self.S_pair is not None, "Pairwise covariance matrix of features is not determined."
    assert self.rho_pair is not None, "Pairwise covariance vector of features and labels is not determined."
    assert self.R is not None, "Weights are not determined."

    if self.verbose:
      print("[First Problem] Starting...")

    Sigma = cp.Variable((self.p, self.p), PSD = True) # Variable to optimize
    obj = cp.Minimize(cp.sum_squares(cp.multiply(self.R, Sigma-self.S_pair))) # Objective to minimize
    constraints = [Sigma >> 0] # Constraints: We want Sigma to be positive semi-definite.
    if self.verbose:
      print("[First Problem] Objective and constraints well-defined.")

    # Solve the optimization problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=self.verbose)
    if self.verbose:
      print(f"[First Problem] Problem status: {prob.status}.")
    if self.verbose:
      print("[First Problem] Problem solved.")

    self.isFirstProblemSolved = True

    return Sigma.value

  def __solve_second_problem__(self):
    
    assert self.S_pair is not None, "Pairwise covariance matrix of features is not determined."
    assert self.rho_pair is not None, "Pairwise covariance vector of features and labels is not determined."
    assert self.R is not None, "Weights are not determined."
    assert self.isFirstProblemSolved, " First optimization problem has not been solved."
    assert self.Sigma_opt is not None, "Sigma_opt is unknown. First optimization problem might have not been solved."

    if self.verbose:
      print("[Second Problem] Starting...")

    beta = cp.Variable(self.p) # Variable to optimize
    obj = cp.Minimize(0.5 * cp.quad_form(beta, self.Sigma_opt) - self.rho_pair.T @ beta + self.mu * cp.norm1(beta)) # Objective to minimize
    constraints = [] # Constraints
    if self.verbose:
      print("[Second Problem] Objective and constraints well-defined.")

    # Solve the optimization problem
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=self.verbose)
    if self.verbose:
      print(f"[Second Problem] Problem status: {prob.status}.")
    if self.verbose:
      print("[Second Problem] Problem solved.\n")
    
    self.isSecondProblemSolved = True

    return beta.value

"""## Test"""

def get_Xy(n, p, replace_rate=0.3):
  X = 100*np.random.rand(n,p) # Generate random X
  y = 7*X[:, 0] - 2 * X[:, 1] + 5 * X[:, 2] + 19 * X[:, 3] + 6*X[:, 4]
  
  indices = np.full(X.shape, False, bool)
  mask = np.random.choice([False, True], size=X.shape, p=((1 - replace_rate), replace_rate))
  X[mask] = np.nan

  return X, y

if __name__ == "__main__":
  count = 0
  for i in range(100):
    try:
      X, y = get_Xy(10000, 20, 0.4)

      scaler = StandardScaler(with_std=False)
      X_scaled = scaler.fit_transform(X)
      lasso = HMLasso(mu=100, alpha=1, verbose=False)
      lasso.fit(X_scaled, y)
      X_test, y_test = get_Xy(10000, 20, replace_rate=0.)
    except:
      count += 1
  print(count)

isinstance("a", (int, float))