from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
from jax.scipy.sparse.linalg import cg

def pcg(A, b, x0=None, *, tol=1e-05, atol=0.0, maxiter=None, M=None, verbose=False):
    if M is None:
        M = lambda x: x
    info = -1
    normb = np.linalg.norm(b)
    tolb = tol*normb
    if x0 is None:
        x = np.zeros((b.shape[0],))
    else:
        x = x0
    r = b - A(x)
    z = M(r)
    p = z
    gamma = r.T @ z
    for i in range(1,maxiter+1):
        Ap = A(p)
        alpha = gamma / (p.T @ Ap)
        x = x + alpha*p
        r = r - alpha*Ap
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
    I = np.identity(n)
    T = -np.tri(n,n,1) + np.tri(n,n,-2) + 3.0*I
    A = np.kron(I,T) + np.kron(T,I)
    b = np.ones((n**2,))

    x, info = cg(lambda x: A@x, b, tol=1e-05, atol=0.0, maxiter=100)
    relres = np.linalg.norm(b-A@x) / np.linalg.norm(b)
    print(f'relres = {relres}, niters = {info}')

    x, info = pcg(lambda x: A@x, b, tol=1e-05, atol=0.0, maxiter=100)
    relres = np.linalg.norm(b-A@x) / np.linalg.norm(b)
    print(f'relres = {relres}, niters = {info}')
