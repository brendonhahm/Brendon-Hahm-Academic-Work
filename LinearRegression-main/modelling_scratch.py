import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import statistics
import scipy
from scipy.stats import t

def correlation(xList, yList):
    """
    :param xList:
    :param yList:
    :return:
    """
    if len(xList) != len(yList):
        print('Lengths must be the same for both arrays')
    else:

        x_bar = sum(xList)/len(xList)
        y_bar = sum(yList)/len(yList)

        x_std = statistics.stdev(xList)
        y_std = statistics.stdev(yList)

        summation = 0
        for i in range(len(xList)):
            summation += ((xList[i] - x_bar)/x_std) * ((yList[i] - y_bar)/y_std)

        return (1/(len(xList)-1)) * summation


def mse(y, y_hat):
    """
    Returns the mean squared error of actual values and predicted values
    :param y: array
    :param y_hat: array
    :return: float
    """
    return np.square(np.subtract(y,y_hat)).mean()

def predict(X, coefs, bias):
    """
    Returns predicted values given data, coefficients of model, and bias term of model
    :param X: 2x2 numpy array
    :param coefs: array
    :return: int
    """
    return np.dot(X, coefs) + bias


def gradients(X, y, y_hat):
    error = y_hat - y
    dw = (1/len(X)) * np.dot(X.T, error)
    db = (1/len(X)) * np.sum(error)

    return dw, db


def linear_regression(X, y, lr, n):

    #initializing variables for iteration
    coefs = np.zeros(X.shape[1])
    bias = 0

    #fitting with iteration
    for _ in range(n):
        #make predictions
        y_hat = predict(X, coefs, bias)

        #get gradients
        dw, db = gradients(X, y, y_hat)

        #update coefs
        coefs -= lr*dw
        bias -= lr*db
        print(coefs)

    return coefs, bias


def ols(X, y):
    #coefficient list
    coefficients = []
    #constant initialization into matrix
    ones = np.ones(shape=X.shape[0]).reshape(-1,1)
    X = np.concatenate((ones, X), 1)

    #linear algebra solution fitting
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    coefficients = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    bias = coefficients[0]
    coefs = coefficients[1:]

    return coefs, bias



class Model:
    def __init__(self, model, df, VOI, controls, resp, alpha = .05, interactions = None, IV_var = None, dummies = None, learning_rate = .01, n_iterations = 1000):
        self.model = model
        self.df = df
        self.VOI = VOI
        self.controls = controls
        self.resp = resp
        self.alpha = alpha
        self.interactions = interactions
        self.IV_var = IV_var
        self.dummies = dummies

        #creating interaction terms
        interaction_labels = []
        if self.interactions is not None:
            for inter in self.interactions:
                self.df[f'{inter[0]}x{inter[1]}'] = self.df[inter[0]] * self.df[inter[1]]
                interaction_labels.append(f'{inter[0]}x{inter[1]}')

        #creating dummy variables
        dummy_labels = []
        if self.dummies is not None:
            for dum in self.dummies:
                for col in self.df:
                    if col == dum:
                        vals = list(set(self.df[col]))

                        for val in vals:
                            self.df[val] = [1 if x == val else 1 for x in self.df[col]]
                        dummy_labels.append(val)

        #creating feature matrix and response array
        feature_labels = [self.VOI] + controls + interaction_labels + dummy_labels
        feature_matrix = self.df[feature_labels].to_numpy()
        self.feature_labels = feature_labels
        self.feature_matrix = feature_matrix

        y = self.df[self.resp]



        #defining labels
        self.labels = feature_labels + [self.resp]

        #fitting linear regression
        if model == 'Linear':
            coefs, bias = linear_regression(feature_matrix, y, learning_rate, n_iterations)
            self.VOI_val = coefs[0]
            self.predict = predict(feature_matrix, coefs, bias)
            self.mse = mse(y, self.predict)
            self.r2 = 1 - self.mse/np.var(y)
            self.bias = bias
        #fitting OLS
        if model == 'OLS':
            coefs, bias = ols(feature_matrix, y)
            self.VOI_val = coefs[0]
            self.constant = bias
            self.predict = predict(feature_matrix, coefs, bias)
            self.mse = mse(y, self.predict)
            self.r2 = 1 - self.mse/np.var(y)
            self.bias = bias

        #getting standard errors of betas
        cov_matrix = np.linalg.inv(feature_matrix.transpose().dot(feature_matrix))
        beta_standard_errors = cov_matrix.diagonal()

        #recording all coefficient values and standard errors
        coefficient_dict = {}
        beta_se_dict = {}
        for i in range(len(self.labels)-1):
            #for each label
            label = self.labels[i]

            #setting coefficient value
            coef = coefs[i]
            coefficient_dict[label] = coef

            #setting standard error values for each coefficient
            se = beta_standard_errors[i]
            beta_se_dict[label] = se

        self.coefs = coefficient_dict
        self.b_se = beta_se_dict

    def __str__(self):
        return f'{self.model} model measuring effect of {self.VOI} on {self.resp}'


    # useful methods
    def corr_matrix(self):
        return self.df[self.labels].corr()

    def corr(self, var1, var2):
        return scipy.stats.pearsonr(self.df[var1], self.df[var2])[0]
        #return correlation(list(self.df[var1]), list(self.df[var2]))

    def plot_y_on_var(self, var, univariate = 'yes'):

        #plot data points
        plt.plot(self.df[var], self.df[self.resp], 'bo')


        #plot regression line if univariate == 'yes'
        if univariate == 'yes':
            coefs, bias = ols(self.df[[var]], self.df[self.resp])
            predictions = predict(self.df[[var]], coefs, bias)
            plt.plot(self.df[var], predictions)
        #plot regression line if univariate == 'no'
        else:
            plt.plot(self.df[var], self.predict)

        #formatting
        plt.title(f'Plot of {self.resp} on {var}')
        plt.ylabel(f'{self.resp}')
        plt.xlabel(f'{var}')
        plt.tight_layout()

    def residual_plot(self, var):
        sns.residplot(x = var, y = self.resp, data = self.df)
        plt.show()

        #formatting
        plt.title(f'Residual plot of residuals on predicted')
        plt.ylabel('Residuals')
        plt.xlabel('Predicted')
        plt.tight_layout()

    def pvalue(self, var):
        beta = self.coefs[var]
        se = self.b_se[var]
        n = len(self.feature_matrix)
        k = len(self.feature_labels)
        t_stat = (beta - 0)/(se/math.sqrt(n))

        #degrees of freedom
        dof = n-k-1

        #find p value from t stat
        pvalue = 2*(1 - t.cdf(abs(t_stat), dof))

        return pvalue

    def is_significant(self, var):
        pvalue = self.pvalue(var)

        #compare pvalue with alpha
        return pvalue <= self.alpha
