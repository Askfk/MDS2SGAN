# -*- coding: utf-8 -*-
"""
Variational mode decomposition.
"""

import numpy as np
import matplotlib.pyplot as plt


def VMD(f, alpha, tau, K, DC, init, tol):
    """
    u,u_hat,omega = VMD(f, alpha, tau, K, DC, init, tol)
    Variational mode decomposition
    Python implementation by Vinícius Rezende Carvalho - vrcarva@gmail.com
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
    Original paper:
    Dragomiretskiy, K. and Zosso, D. (2014) ‘Variational Mode Decomposition’,
    IEEE Transactions on Signal Processing, 62(3), pp. 531–544. doi: 10.1109/TSP.2013.2288675.


    Input and Parameters:
    ---------------------
    f       - the time domain signal (1D) to be decomposed
    alpha   - the balancing parameter of the data-fidelity constraint
    tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    K       - the number of modes to be recovered
    DC      - true if the first mode is put and kept at DC (0-freq)
    init    - 0 = all omegas start at 0
              1 = all omegas start uniformly distributed
              2 = all omegas initialized randomly
    tol     - tolerance of convergence criterion; typically around 1e-6
    Output:
    -------
    u       - the collection of decomposed modes
    u_hat   - spectra of the modes
    omega   - estimated mode center-frequencies
    """

    if len(f) % 2:
        f = f[:-1]

    # Period and sampling frequency of input signal
    fs = 1. / len(f)

    ltemp = len(f) // 2
    fMirr = np.append(np.flip(f[:ltemp], axis=0), f)
    fMirr = np.append(fMirr, np.flip(f[-ltemp:], axis=0))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(fMirr)))
    f_hat_plus = np.copy(f_hat)  # copy f_hat
    f_hat_plus[:T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K)))
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

    # other inits
    uDiff = tol + np.spacing(1)  # update step
    n = 0  # loop counter
    sum_uk = 0  # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

    # *** Main loop for iterative updates***

    while uDiff > tol and n < Niter - 1:  # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                    1. + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)

        # update first omega if not held at 0
        if not (DC):
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # update of any other mode
        for k in np.arange(1, K):
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum
            u_hat_plus[n + 1, :, k] = (f_hat_plus - sum_uk - lambda_hat[n, :] / 2) / (
                        1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            # center frequencies
            omega_plus[n + 1, k] = np.dot(freqs[T // 2:T], (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)) / np.sum(
                abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)

        # Dual ascent
        lambda_hat[n + 1, :] = lambda_hat[n, :] + tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + (1 / T) * np.dot((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]),
                                             np.conj((u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])))

        uDiff = np.abs(uDiff)

        # Postprocessing and cleanup

    # discard empty space if converged early
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]

    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)
    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2:T, :] = u_hat_plus[Niter - 1, T // 2:T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2:T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4:3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u, u_hat, omega


def autoVMD(data, alpha=500, tau=0., K=3, DC=0, init=1, tol=1e-7):
    # Time Domain 0 to T
    m = len(data)
    for j in range(m):
        signal = data[j]

        f_hat = np.fft.fftshift((np.fft.fft(signal)))
        u, u_hat, omega = VMD(signal, alpha, tau, K, DC, init, tol)
        t = np.arange(0, 8000)

        plt.figure(figsize=(16, 16))

        n = len(u)
        plt.subplot(n + 2, 1, 1)
        plt.plot(signal)
        plt.title('original')

        combined = 0

        for i in range(n):
            plt.subplot(n + 2, 1, i + 3)
            plt.plot(u[i])
            combined += u[i]

            plt.title('Decomposed modes_{}'.format(i + 1))

        plt.subplot(n + 2, 1, 2)
        plt.plot(combined)
        plt.plot(combined - np.array(signal))
        plt.title('combined')
        plt.show()


if __name__ == '__main__':
    import scipy.io as scio
    import os
    from EDA import WaveletDenoising

    ROOT_DIR = '/Users/liyiming/Desktop/研究生毕设/lamb wave dataset/wield/lym'

    file_name = '165-24-10000-400-6500-20-n.mat'
    fn = os.path.join(ROOT_DIR, file_name)

    data = scio.loadmat(fn)
    t = np.arange(0, 10000 / 24000, 1 / 24000)
    f = np.arange(0, 10000) * 24000 / 10000

    s0 = data['s0'][:, 0]
    s1 = data['s1'][:, 0]
    s2 = data['s2'][:, 0]
    s3 = data['s3'][:, 0]
    s4 = data['s4'][:, 0]
    s5 = data['s5'][:, 0]
    d = WaveletDenoising(s1).out

    alpha = 500
    tau = 0.
    K = 6
    DC = 0
    init = 1
    tol = 1e-7
    u, u_hat, omega = VMD(d, alpha, tau, K, DC, init, tol)

    plt.figure(figsize=(16, 16))
    title = 'VMD'

    n = len(u)
    fig = plt.figure(figsize=(12, 16))
    ax_main = fig.add_subplot(n + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(t, d)


    rec_b = []
    for i in u:
        rec_b.append(d - i)
        d -= i

    for i, y in enumerate(rec_b):
        ax = fig.add_subplot(len(rec_b) + 1, 2, 3 + i * 2)
        ax.plot(t, y, 'r')

        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(u):
        ax = fig.add_subplot(len(u) + 1, 2, 4 + i * 2)
        ax.plot(t, y, 'g')

        ax.set_ylabel("D%d" % (i + 1))

    from PyEMD import Visualisation
    vis = Visualisation()
    vis.plot_instant_freq(f, imfs=u)
    vis.show()