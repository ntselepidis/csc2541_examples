#from jax.config import config
#config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.scipy.sparse.linalg import cg
import time

def pcg(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None, verbose=False):
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
    for i in range(1,maxiter+1):
        Ap = A(p)
        alpha = gamma / (p.T @ Ap)
        x += alpha*p
        r -= alpha*Ap
        normr = np.linalg.norm(r)
        if verbose:
            val = 0.5*x.T@A(x) - x.T@b
            print(f'{i}: relres = {normr/normb}, val = {val}')
        if normr < tolb:
            info = i
            break
        z = M(r)
        gamma_old = gamma
        gamma = r.T @ z
        beta = gamma / gamma_old
        p = z + beta*p
    return x, info

if __name__ == '__main__':
    n = 32
    tol=1e-05
    maxiter=100

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
    x, info = pcg(lambda x: A@x, b, tol=tol, atol=0.0, maxiter=maxiter)
    t1 = time.time()
    print(f'niters = {info}')
    print(f'relres = {relres(x)}')
    print(f'relerr = {relerr(x)}')
    print(f'time   = {t1 - t0}')
