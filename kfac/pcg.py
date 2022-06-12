#from jax.config import config
#config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.scipy.sparse.linalg import cg
import time
from matplotlib import pyplot as plt

def pcg(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None, verbose=False, has_aux=False):
    if M is None:
        M = lambda x: x
    info = -1
    normb = np.linalg.norm(b)
    tolb = tol*normb
    if x0 is None:
        x = np.zeros((b.shape[0],))
        r = b
    else:
        x = x0
        r = b - A(x)
    z = M(r)
    p = z
    gamma = r.T @ z
    if has_aux:
        _val = np.zeros((maxiter+1,))
        _relres = np.zeros((maxiter+1,))
        _val = _val.at[0].set(-0.5*(x.T @ (r + b)))
        _relres = _relres.at[0].set(np.linalg.norm(r)/normb)
    for i in range(1,maxiter+1):
        Ap = A(p)
        alpha = gamma / (p.T @ Ap)
        x += alpha*p
        r -= alpha*Ap
        normr = np.linalg.norm(r)
        if verbose or has_aux:
            #val = 0.5*x.T@A(x) - x.T@b
            val = -0.5*(x.T @ (r + b))
        if verbose:
            print(f'{i}: relres = {normr/normb}, val = {val}')
        if has_aux:
            _val = _val.at[i].set(val)
            _relres = _relres.at[i].set(normr/normb)
        if normr < tolb:
            info = i
            break
        z = M(r)
        gamma_old = gamma
        gamma = r.T @ z
        beta = gamma / gamma_old
        p = z + beta*p
    if has_aux:
        return x, info, _val, _relres
    else:
        return x, info

if __name__ == '__main__':
    n = 32
    tol=1e-05
    maxiter=100
    has_aux=False

    I = np.identity(n)
    T = -np.tri(n,n,1) + np.tri(n,n,-2) + 3.0*I
    A = np.kron(I,T) + np.kron(T,I)
    #b = np.ones((n**2,))
    y = np.arange(1.,n**2+1.,1.).reshape((-1,)) / n**2
    b = A @ y

    relres = lambda x: np.linalg.norm(b-A@x) / np.linalg.norm(b)
    relerr = lambda x: np.linalg.norm(y-x) / np.linalg.norm(y)

    # warmup
    x, info = cg(lambda x: A@x, b, tol=tol, atol=0.0, maxiter=maxiter)
    # benchmark
    t0 = time.time()
    x, info = cg(lambda x: A@x, b, tol=tol, atol=0.0, maxiter=maxiter)
    t1 = time.time()
    print(f'niters = {info}')
    print(f'relres = {relres(x)}')
    print(f'relerr = {relerr(x)}')
    print(f'time   = {t1 - t0}')

    # warmup
    x, info = pcg(lambda x: A@x, b, tol=tol, atol=0.0, maxiter=maxiter)
    # benchmark
    t0 = time.time()
    x, info, *_ = pcg(lambda x: A@x, b, tol=tol, atol=0.0, maxiter=maxiter, has_aux=has_aux)
    t1 = time.time()
    print(f'niters = {info}')
    print(f'relres = {relres(x)}')
    print(f'relerr = {relerr(x)}')
    print(f'time   = {t1 - t0}')

    if has_aux:
        # unpack
        _val, _relres = _

        fig, axs = plt.subplots(1, 2, figsize=(2*6.4, 4.8))
        fig.suptitle('PCG convergence plots')

        axs[0].plot(_val)
        axs[0].set_title('val')
        axs[0].grid()

        axs[1].plot(_relres)
        axs[1].set_title('relres')
        axs[1].set_yscale('log')
        axs[1].grid()

        plt.savefig('pcg_convergence.png')
