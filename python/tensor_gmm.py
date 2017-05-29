import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

d = 10
k = 8
n = 1000
tot = k * n
s = 2
dist = 20
spher = True
cov_range = 2

def generate_data():
    A = -dist+(dist+dist)*np.random.rand(d, k)
    X = np.zeros((tot, d))

    plt.axis('equal')

    for i in range(k):
        mean = np.transpose(A[:, i])
        if spher:
            covariance = s * np.identity(d)
        else:
            a = -cov_range + (cov_range + cov_range) * np.random.rand(d, d)
            covariance = np.matmul(np.transpose(a), a)
        mvn = np.random.multivariate_normal(mean, covariance, n)
        X[i*n:(i+1)*n, :] = mvn

    return (X, A)

def calculate_first_moment(X):
    mu = np.zeros((d, 1))
    for t in range(tot):
        for i in range(d):
            mu[i] += + X[t, i]
    mu /= tot
    return mu

def calculate_second_moment(X):
    Sigma = np.zeros((d, d))
    for t in range(tot):
        for i in range(d):
            for j in range(d):
                Sigma[i, j] += np.dot(X[t, i],X[t, j])
    Sigma /= tot
    return Sigma

def extract_information_from_second_moment(Sigma, X):
    U, S, _ = np.linalg.svd(Sigma)
    s_est = S[-1]
    W, X_whit = perform_whitening(X, U, S, s_est)
    return (s_est, W, X_whit)

def perform_whitening(X, U, S, s_est):
    W = np.matmul(U[:, 0:k], np.sqrt(np.linalg.pinv(np.diag(S[0:k]) - s_est * np.eye(k))))
    X_whit = np.matmul(X, W)
    return (W, X_whit)

def perform_tensor_power_method(X_whit, W, s_est, mu):
    TOL = 1e-8
    maxiter = 100
    V_est = np.zeros((k, k))
    lamb = np.zeros((k, 1))

    for i in range(k):
        v_old = np.random.rand(k, 1)
        v_old = np.divide(v_old, np.linalg.norm(v_old))
        for iter in range(maxiter):
            v_new = (np.matmul(np.transpose(X_whit), (np.matmul(X_whit, v_old) * np.matmul(X_whit, v_old)))) / tot
            #v_new = v_new - s_est * (W' * mu * dot((W*v_old),(W*v_old)));
            #v_new = v_new - s_est * (2 * W' * W * v_old * ((W'*mu)' * (v_old)));
            v_new -= s_est * (np.matmul(np.matmul(W.T, mu), np.dot(np.matmul(W, v_old).T,np.matmul(W, v_old))))
            v_new -= s_est * (2 * np.matmul(W.T, np.matmul(W, np.matmul(v_old, np.matmul(np.matmul(W.T, mu).T, v_old)))))
            if i > 0:
                for j in range(i):
                    v_new -= np.reshape(V_est[:, j] * np.power(np.matmul(np.transpose(v_old), V_est[:, j]), 2) * lamb[j], (k, 1))
            l = np.linalg.norm(v_new)
            v_new = np.divide(v_new, np.linalg.norm(v_new))
            if np.linalg.norm(v_old - v_new) < TOL:
                V_est[:, i] = np.reshape(v_new, k)
                lamb[i] = l
                break
            v_old = v_new

    return (V_est, lamb)

def perform_backwards_transformation(V_est, lamb):
    return np.matmul(np.matmul(np.linalg.pinv(np.transpose(W)), V_est), np.diag(np.reshape(lamb.T, k)))

def plot_results(X, A, A_est, s_est):
    plt.axis('equal')

    ax = plt.subplot(aspect='equal')

    plt.plot(X[:,0], X[:,1], '.', zorder=-3)

    for i in range(k):
        mean = A[:, i].T
        mean_est = A_est[:, i].T

        plt.plot(mean[0], mean[1], 'x', color='y', zorder=-2)
        plt.plot(mean_est[0], mean_est[1], '+', color='r', zorder=-1)

        ell = Ellipse(xy=(mean_est[0], mean_est[1]),
                      width=s_est, height=s_est,
                      angle=0, color='red')
        ell.set_facecolor('none')
        ax.add_artist(ell)

    plt.show()

if __name__ == "__main__":
    X, A = generate_data()

    mu = calculate_first_moment(X)
    Sigma = calculate_second_moment(X)

    s_est, W, X_whit = extract_information_from_second_moment(Sigma, X)

    V_est, lamb = perform_tensor_power_method(X_whit, W, s_est, mu)

    A_est = perform_backwards_transformation(V_est, lamb)

    plot_results(X, A, A_est, s_est)
