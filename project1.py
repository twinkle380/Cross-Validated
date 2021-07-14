import pandas as pd
import numpy as np
from scipy import stats
np.random.seed(8)

# CSV file should be in the same directory as the current one
TITANIC_PATH = "titanic.csv"
titanic = pd.read_csv(TITANIC_PATH)

X = titanic.drop("Survived", axis=1).values
y = titanic["Survived"].copy().values.reshape(-1,1)

class Estimator():
  def __init__(self, max_iter=100, tol=1e-4, eps=1e-7):
    """
    Initialising the Estimator

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations to be performed
    tol : float
        Error tolerance to be used as a stopping condition
    eps : float
        Epsilon value to handle division by zero and zero in logarithm cases     
    """
    self.coef = None                # to store beta vector (ndarray)
    self.max_iter = max_iter
    self.tol = tol
    self.num_iter = None            # number of iterations taken while training (int)
    self.log_likelihoods = None     # to store likelihood value at each iteration (ndarray)
    self.eps = eps
    self.errs = None                # to store error value in beta at each iteration (ndarray)
  
  def cdf(self, X):
    """
    Cumulative Distribution Function of Standard Normal Distribution

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    cdf : ndarray
        CDF value at X
    """
    cdf = stats.norm.cdf(X)
    return cdf

  def pdf(self, X):
    """
    Probability Distribution Function of Standard Normal Distribution

    Parameters
    ----------
    X : ndarray

    Returns
    -------
    pdf : ndarray
        PDF value at X
    """
    pdf = stats.norm.pdf(X)
    return pdf

  def log_likelihood(self, X, y, beta):
    """
    Calculate log-likelihood value for the data points

    Parameters
    ----------
    X : ndarray
        Data points
    y : ndarray
        Data labels
    beta : ndarray
        Feature weights

    Returns
    -------
    likelihood : float
        Log-likelihood value
    """
    q = 2*y-1
    likelihood = np.sum(np.log(np.clip(self.cdf((q*np.matmul(X, beta))), self.eps, 1)))
    return likelihood

  def log_likelihood_gradient(self, X, y, beta):
    """
    Calculate gradient vector of the log-likelihood function

    Parameters
    ----------
    X : ndarray
        Data points
    y : ndarray
        Data labels
    beta : ndarray
        Feature weights

    Returns
    -------
    gradient : ndarray
        Gradient vector
    """
    q = 2*y-1
    X_beta = np.matmul(X, beta)
    gradient = np.matmul(X.T, (q*self.pdf(X_beta))/np.clip(self.cdf(q*X_beta), self.eps, 1-self.eps))
    return gradient

  def log_likelihood_hessian(self, X, y, beta):
    """
    Calculate hessian matrix of the log-likelihood function

    Parameters
    ----------
    X : ndarray
        Data points
    y : ndarray
        Data labels
    beta : ndarray
        Feature weights

    Returns
    -------
    hessian : ndarray
        Hessian matrix
    """
    q = 2*y-1
    X_beta = np.matmul(X, beta)
    lmbd = q*self.pdf(X_beta)/(np.clip(self.cdf(q*X_beta), self.eps, 1-self.eps))
    hessian = np.matmul(X.T, -lmbd*(lmbd+X_beta)*X)
    return hessian

  def probability(self, X, beta=None):
    """
    Calculate probability of belonging to the positive class

    Parameters
    ----------
    X : ndarray
        Data points
    beta : ndarray
        Feature weights

    Returns
    -------
    probability : ndarray
        Probability vector
    """
    if beta is None:
      beta = self.coef
    probability = self.cdf(np.matmul(X, beta))
    return probability

  def predict(self, X, beta=None, threshold=0.5):
    """
    Predicting class for the given data points

    Parameters
    ----------
    X : ndarray
        Data points
    beta : ndarray
        Feature weights
    threshold : float (0 to 1)
        Probability value less than this value is considered a negative case 

    Returns
    -------
    y_pred : ndarray
        Prediction vector
    """
    if beta is None:
      beta = self.coef
    y_pred = np.array(self.probability(X, beta) >= threshold).astype(np.int)
    return y_pred

  def fit(self, X, y):
    """
    Finding the optimum value of beta (MLE)

    Parameters
    ----------
    X : ndarray
        Data points
    y : ndarray
        Data labels
    """
    iter = 0
    err = np.inf
    m, n = np.shape(X)
    log_likelihoods = []
    beta = np.zeros((n, 1))
    errs = []
    while(err > self.tol and iter < self.max_iter):
      iter = iter+1
      beta_prev = np.copy(beta)
      log_likelihoods.append(self.log_likelihood(X, y, beta_prev))
      beta_grad = self.log_likelihood_gradient(X, y, beta_prev)
      beta_hess = self.log_likelihood_hessian(X, y, beta_prev)
      beta = beta_prev - np.matmul(np.linalg.inv(beta_hess), beta_grad)
      err = np.sum(np.square(beta-beta_prev))
      errs.append(err)
    self.coef = beta
    self.num_iter = iter
    self.log_likelihoods = np.array(log_likelihoods)
    self.errs = np.array(errs)

model = Estimator(tol=1e-6)
model.fit(X, y)
print("beta:", model.coef)


# Part 3
jack = np.array([1, 1, 20, 0, 0, 7.5])
rose = np.array([1, 0, 19, 1, 1, 512])
jack_and_rose = np.array([jack, rose])
jack_and_rose_prob = model.probability(jack_and_rose)
print("Probability of survival of Jack:", jack_and_rose_prob[0, 0])
print("Probability of survival of Rose:", jack_and_rose_prob[1, 0])