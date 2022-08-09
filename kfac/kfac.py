from jax import grad, jit, numpy as np, random, vjp, jvp
from jax.scipy.linalg import eigh
from jax.lax import batch_matmul, dynamic_update_slice
from pcg import pcg as cg
from copy import copy
import numpy as onp


import kfac_util


scale_fn_dict = {
        'sum'  : lambda n: 1,
        'mean' : lambda n: n,
        'norm' : lambda n: onp.sqrt(n)
        }

scale_fn = scale_fn_dict['norm']


def L2_penalty(arch, w):
    # FIXME: don't regularize the biases
    return 0.5 * np.sum(w**2)

def get_batch_size(step, ndata, config):
    """Exponentially increasing batch size schedule."""
    step = np.floor(step/config['batch_size_granularity']) * config['batch_size_granularity']
    pwr = onp.minimum(step / (config['final_batch_size_iter']-1), 1.)
    return onp.floor(config['initial_batch_size'] * (ndata / config['initial_batch_size'])**pwr).astype(onp.uint32)

def get_sample_batch_size(batch_size, config):
    """Batch size to use for sampling the activation statistics."""
    return onp.ceil(config['cov_batch_ratio'] * batch_size).astype(onp.uint32)

def get_chunks(batch_size, chunk_size):
    """Iterator that breaks a range into smaller chunks. Useful for simulating
    larger batches than can fit on the GPU."""
    start = 0
    
    while start < batch_size:
        end = min(start+chunk_size, batch_size)
        yield slice(start, end)
        start = end


def gnhvp_helper(apply_fn, unflatten_fn, nll_fn, w, X_chunk, T_chunk, vin, chunk_size):

    mvp = lambda v: kfac_util.gnhvp_chunk(lambda w: apply_fn(unflatten_fn(w), X_chunk),
                                          lambda y: nll_fn(y, T_chunk), w, v)
    return mvp(vin)

gnhvp_helper = jit(gnhvp_helper, static_argnums=(0,1,2))


def gnhvp(arch, output_model, w, X, T, vin, chunk_size):
    batch_size = X.shape[0]
    vout = 0

    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]

        mvp = lambda v: gnhvp_helper(arch.net_apply, arch.unflatten,
                output_model.nll_fn, w, X_chunk, T_chunk, v, chunk_size)
        vout += mvp(vin)

    vout /= batch_size

    return vout


def make_instrumented_vjp(apply_fn, params, inputs):
    """Returns a function which takes in the output layer gradients and returns a dict
    containing the gradients for all the intermediate layers."""
    dummy_input = np.zeros((2,) + inputs.shape[1:])
    _, dummy_activations = apply_fn(params, dummy_input, ret_all=True)
    
    batch_size = inputs.shape[0]
    add_to = {name: np.zeros((batch_size,) + dummy_activations[name].shape[1:])
              for name in dummy_activations}
    apply_wrap = lambda a: apply_fn(params, inputs, a, ret_all=True)
    primals_out, vjp_fn, activations = vjp(apply_wrap, add_to, has_aux=True)
    return primals_out, vjp_fn, activations


def estimate_covariances_chunk(apply_fn, param_info, output_model, net_params, X_chunk, rng, has_aux=False):
    """Compute the empirical covariances on a chunk of data."""
    logits, vjp_fn, activations = make_instrumented_vjp(apply_fn, net_params, X_chunk)
    key, rng = random.split(rng)
    output_grads = output_model.sample_grads_fn(logits, key)
    act_grads = vjp_fn(output_grads)[0]

    A = {}
    G = {}
    a_hom_mean, ds_mean = {}, {}
    a_hom_storage, ds_storage = {}, {}
    for in_name, out_name in param_info:
        a = activations[in_name]
        a_hom = np.hstack([a, np.ones((a.shape[0], 1))])
        A[in_name] = a_hom.T @ a_hom

        ds = act_grads[out_name]
        G[out_name] = ds.T @ ds

        a_hom_mean[in_name] = np.sum(a_hom, axis=1) / scale_fn(a_hom.shape[1])
        ds_mean[out_name] = np.sum(ds, axis=1) / scale_fn(ds.shape[1])

        if has_aux:
            a_hom_storage[in_name] = a_hom
            ds_storage[out_name] = ds

    a_hom_mean_stacked = np.vstack([a_hom_mean[in_name] for in_name in a_hom_mean])
    ds_mean_stacked = np.vstack([ds_mean[out_name] for out_name in ds_mean])

    # compute kronecker-factored coarse Fisher
    A_coarse = (a_hom_mean_stacked @ a_hom_mean_stacked.T)
    G_coarse = (   ds_mean_stacked @    ds_mean_stacked.T)

    # compute actual coarse Fisher
    a_dot_g = (a_hom_mean_stacked * ds_mean_stacked)
    F_coarse = a_dot_g @ a_dot_g.T

    if has_aux:
        return A, G, A_coarse, G_coarse, F_coarse, a_hom_storage, ds_storage
    else:
        return A, G, A_coarse, G_coarse, F_coarse

estimate_covariances_chunk = jit(estimate_covariances_chunk, static_argnums=(0,1,2,6))


def estimate_covariances(arch, output_model, w, X, rng, chunk_size, has_aux=False):
    """Compute the empirical covariances on a batch of data."""
    batch_size = X.shape[0]
    net_params = arch.unflatten(w)
    A_sum = {in_name: 0. for in_name, out_name in arch.param_info}
    G_sum = {out_name: 0. for in_name, out_name in arch.param_info}
    nlayers = len(A_sum)
    A_coarse_sum = np.zeros((nlayers, nlayers))
    G_coarse_sum = np.zeros((nlayers, nlayers))
    F_coarse_sum = np.zeros((nlayers, nlayers))

    if has_aux:
        A_sum_, G_sum_ = {}, {}
        for i in range(nlayers):
            for j in range(nlayers):
                A_sum_[(i, j)] = 0.
                G_sum_[(i, j)] = 0.

    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk = X[chunk_idxs,:]
        key, rng = random.split(rng)
        
        A_curr, G_curr, A_coarse_curr, G_coarse_curr, F_coarse_curr, *_ = estimate_covariances_chunk(
            arch.net_apply, arch.param_info, output_model, net_params, X_chunk, key, has_aux)
        A_sum = {name: A_sum[name] + A_curr[name] for name in A_sum}
        G_sum = {name: G_sum[name] + G_curr[name] for name in G_sum}
        A_coarse_sum = A_coarse_sum + A_coarse_curr
        G_coarse_sum = G_coarse_sum + G_coarse_curr
        F_coarse_sum = F_coarse_sum + F_coarse_curr

        if has_aux:
            # unpack
            a_hom_storage, ds_storage = _
            # compute all As on current chunk
            for i, (in_name_i, _) in enumerate(arch.param_info):
                for j, (in_name_j, _) in enumerate(arch.param_info):
                    A_sum_[(i, j)] = A_sum_[(i, j)] + a_hom_storage[in_name_i].T @ a_hom_storage[in_name_j]
            # compute all Gs on current chunk
            for i, (_, out_name_i) in enumerate(arch.param_info):
                for j, (_, out_name_j) in enumerate(arch.param_info):
                    G_sum_[(i, j)] = G_sum_[(i, j)] + ds_storage[out_name_i].T @ ds_storage[out_name_j]

    A_mean = {name: A_sum[name] / batch_size for name in A_sum}
    G_mean = {name: G_sum[name] / batch_size for name in G_sum}
    A_coarse_mean = A_coarse_sum / batch_size
    G_coarse_mean = G_coarse_sum / batch_size
    F_coarse_mean = F_coarse_sum / batch_size

    if has_aux:
        A_mean_, G_mean_ = {}, {}
        for i in range(nlayers):
            for j in range(nlayers):
                A_mean_[(i, j)] = A_sum_[(i, j)] / batch_size
                G_mean_[(i, j)] = G_sum_[(i, j)] / batch_size
    
        return A_mean, G_mean, A_coarse_mean, G_coarse_mean, F_coarse_mean, A_mean_, G_mean_
    else:
        return A_mean, G_mean, A_coarse_mean, G_coarse_mean, F_coarse_mean

def update_covariances(A, G, A_coarse, G_coarse, F_coarse, arch, output_model, w, X, rng, cov_timescale, chunk_size, A_, G_, has_aux=False):
    """Exponential moving average of the covariances."""
    A, G = dict(A), dict(G)
    curr_A, curr_G, curr_A_coarse, curr_G_coarse, curr_F_coarse, *_ = estimate_covariances(arch, output_model, w, X, rng, chunk_size, has_aux)
    ema_param = kfac_util.get_ema_param(cov_timescale)
    for k in A.keys():
        A[k] = ema_param * A[k] + (1-ema_param) * curr_A[k]
    for k in G.keys():
        G[k] = ema_param * G[k] + (1-ema_param) * curr_G[k]

    A_coarse = ema_param * A_coarse + (1-ema_param) * curr_A_coarse
    G_coarse = ema_param * G_coarse + (1-ema_param) * curr_G_coarse
    F_coarse = ema_param * F_coarse + (1-ema_param) * curr_F_coarse

    if has_aux:
        curr_A_, curr_G_ = _
        nlayers = len(A)
        for i in range(nlayers):
            for j in range(nlayers):
                A_[(i, j)] = ema_param * A_[(i, j)] + (1-ema_param) * curr_A_[(i, j)]
                G_[(i, j)] = ema_param * G_[(i, j)] + (1-ema_param) * curr_G_[(i, j)]

        return A, G, A_coarse, G_coarse, F_coarse, A_, G_
    else:
        return A, G, A_coarse, G_coarse, F_coarse

def compute_pi(A, G):
    return np.sqrt((np.trace(A) * G.shape[0]) / (A.shape[0] * np.trace(G)))
    
def compute_inverses(arch, A, G, gamma):
    A_inv, G_inv = {}, {}
    for in_name, out_name in arch.param_info:
        pi = compute_pi(A[in_name], G[out_name])
        
        A_damp = gamma * pi
        A_inv[in_name] = np.linalg.inv(A[in_name] + A_damp * np.eye(A[in_name].shape[0]))
        
        G_damp = gamma / pi
        G_inv[out_name] = np.linalg.inv(G[out_name] + G_damp * np.eye(G[out_name].shape[0]))
        
    return A_inv, G_inv

def compute_eigs(arch, A, G):
    A_eig, G_eig, pi = {}, {}, {}
    for in_name, out_name in arch.param_info:
        A_eig[in_name] = eigh(A[in_name])
        G_eig[out_name] = eigh(G[out_name])
        pi[out_name] = compute_pi(A[in_name], G[out_name])
    return A_eig, G_eig, pi

## ------------------------------------
## enriched coarse-space util functions
## ------------------------------------

def compute_eigv_importance(arch, A_eig, G_eig, grad_w):
    param_grad = arch.unflatten(grad_w)
    result = {}
    for in_name, out_name in arch.param_info:
        grad_W, grad_b = param_grad[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])

        # efficient kronecker-factored matrix-transpose times vector
        # to compute projection of layer eigenvectors on gradient, i.e.:
        # np.kron(A_eig[in_name][1], G_eig[out_name][1]).T @ grad_Wb.reshape((1, -1))
        result_Wb = A_eig[in_name][1].T @ grad_Wb @ G_eig[out_name][1]

        result_W, result_b = result_Wb[:-1, :], result_Wb[-1, :]
        result[out_name] = (result_W, result_b)
    return result

compute_eigv_importance = jit(compute_eigv_importance, static_argnums=(0,))

def compute_important_eigv_inds(importance, neigv):
    important_eigv_inds = {}
    for key in importance.keys():
        importance_W, importance_b = importance[key]
        importance_Wb = np.vstack([importance_W, importance_b.reshape((1, -1))]).reshape((-1,))
        important_eigv_inds[key] = np.argsort( np.abs( importance_Wb ) )[-neigv:]
    return important_eigv_inds

compute_important_eigv_inds = jit(compute_important_eigv_inds, static_argnums=(1,))

def compute_kron_prod_col(x, y, index):
    xcol = index // y.shape[1]
    ycol = index % y.shape[1]
    return np.kron( x[:, xcol].reshape((-1, 1)), y[:, ycol].reshape((-1, 1)) )

compute_kron_prod_col = jit(compute_kron_prod_col)

def compute_selected_kron_prod_cols(arch, A_eig, G_eig, important_eigv_inds):
    eigvs_stacked = {}
    for in_name, out_name in arch.param_info:
        eigvs = []
        for eigv_ind in important_eigv_inds[out_name]:
            eigv = compute_kron_prod_col(A_eig[in_name][1], G_eig[out_name][1], eigv_ind)
            eigvs.append( eigv )
        eigvs_stacked[out_name] = np.vstack([eigv.T[0] for eigv in eigvs])
    return eigvs_stacked

compute_selected_kron_prod_cols = jit(compute_selected_kron_prod_cols, static_argnums=(0,))

def concat_const_and_eigv_basis(blk, importance, eigv_basis, orthonormalize_basis=True):
    const_basis = {}
    basis = {}
    for index, key in enumerate(sorted(importance.keys())):
        npk = blk[index+1] - blk[index]
        const_basis[key] = np.ones((1, npk)) / scale_fn(npk)
        if orthonormalize_basis:
            u = const_basis[key].reshape((-1,))
            for v in eigv_basis[key]:
                u -= np.dot(v, u) * v
            u /= np.linalg.norm(u)
            const_basis[key] = u.reshape((1, -1))
        basis[key] = np.vstack([const_basis[key], eigv_basis[key]])
    return basis

def compute_restriction_matrix(arch, state, importance, nbasis, basis):
    nlayers = len(arch.param_info)
    nparams = len(state['w'])
    perm = state['perm']
    blk = state['blk']
    Z = np.zeros((nbasis*nlayers, nparams))
    for index, key in enumerate(sorted(importance.keys())):
        row = perm[key]
        offset = nbasis*row
        Z = Z.at[offset:offset+nbasis, blk[index]:blk[index+1]].set(basis[key])
        #Z = dynamic_update_slice(Z, basis[key], (np.uint32(offset), blk[index]))
    return Z

def recompute_coarse_fisher(arch, A_, G_, nbasis, basis):
    nlayers = len(arch.param_info)
    F_hat_coarse = np.zeros((nbasis*nlayers, nbasis*nlayers))
    for i, (_, out_name_i) in enumerate(arch.param_info):
        for j, (_, out_name_j) in enumerate(arch.param_info):
            Aij = A_[(i, j)]
            Gij = G_[(i, j)]

            Aij_batched = np.stack([Aij for _ in range(nbasis)])
            GijT_batched = np.stack([Gij.T for _ in range(nbasis)])

            ZjT = np.transpose( basis[out_name_j].T.reshape((Aij.shape[1], Gij.shape[1], -1)), axes=[2, 0, 1] )

            A_kron_G_times_ZjT = batch_matmul( Aij_batched, batch_matmul(ZjT, GijT_batched ) )

            Zi_times_A_kron_G_times_ZjT = basis[out_name_i] @ np.transpose( A_kron_G_times_ZjT, axes=[1, 2, 0] ).reshape((-1, nbasis))

            F_hat_coarse = F_hat_coarse.at[i*nbasis:(i+1)*nbasis, j*nbasis:(j+1)*nbasis].set(Zi_times_A_kron_G_times_ZjT)
    return F_hat_coarse

recompute_coarse_fisher = jit(recompute_coarse_fisher, static_argnums=(0, 3))

def recompute_enriched_coarse_space(arch, state, grad_w, nbasis):
    # compute per-layer eigv importance based on projection on grad_w
    importance = compute_eigv_importance(arch, state['A_eig'], state['G_eig'], grad_w)

    # compute `nbasis-1` important per-layer eigv indices
    important_eigv_inds = compute_important_eigv_inds(importance, nbasis-1)

    # selectively compute eigenvectors based on precomputed important per-layer indices
    eigv_basis = compute_selected_kron_prod_cols(arch, state['A_eig'], state['G_eig'], important_eigv_inds)

    # concat const and eigv basis
    basis = concat_const_and_eigv_basis(state['blk'], importance, eigv_basis)

    # compute Z
    Z = compute_restriction_matrix(arch, state, importance, nbasis, basis)

    # compute Z F_hat Z.T
    F_hat_coarse = recompute_coarse_fisher(arch, state['A_'], state['G_'], nbasis, basis)

    return Z, Z@Z.T, F_hat_coarse, basis

##

def nll_cost(apply_fn, nll_fn, unflatten_fn, w, X, T):
    logits = apply_fn(unflatten_fn(w), X)
    return nll_fn(logits, T)

nll_cost = jit(nll_cost, static_argnums=(0, 1, 2))
grad_nll_cost = jit(grad(nll_cost, 3), static_argnums=(0, 1, 2))
    
def compute_cost(arch, nll_fn, w, X, T, weight_cost, chunk_size):
    batch_size = X.shape[0]
    total = 0

    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]
        total += nll_cost(arch.net_apply, nll_fn, arch.unflatten,
                          w, X_chunk, T_chunk)
        
    return total / batch_size + weight_cost * L2_penalty(arch, w)

def compute_gradient(arch, output_model, w, X, T, weight_cost, chunk_size):
    batch_size = X.shape[0]
    grad_w = 0
    
    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]
        
        grad_w += grad_nll_cost(arch.net_apply, output_model.nll_fn, arch.unflatten,
                                w, X_chunk, T_chunk)
        
    grad_w /= batch_size
    grad_w += weight_cost * grad(L2_penalty, 1)(arch, w)
    return grad_w

def compute_natgrad_from_inverses(arch, grad_w, A_inv, G_inv):
    param_grad = arch.unflatten(grad_w)
    natgrad = {}
    for in_name, out_name in arch.param_info:
        grad_W, grad_b = param_grad[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])
        
        natgrad_Wb = A_inv[in_name] @ grad_Wb @ G_inv[out_name]
        
        natgrad_W, natgrad_b = natgrad_Wb[:-1, :], natgrad_Wb[-1, :]
        natgrad[out_name] = (natgrad_W, natgrad_b)
    return arch.flatten(natgrad)

def compute_natgrad_from_eigs_helper(param_info, param_grad, A_eig, G_eig, pi, gamma):
    natgrad = {}
    for in_name, out_name in param_info:
        grad_W, grad_b = param_grad[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])
        
        A_d, A_Q = A_eig[in_name]
        G_d, G_Q = G_eig[out_name]
        
        # rotate into Kronecker eigenbasis
        grad_rot = A_Q.T @ grad_Wb @ G_Q
        
        # add damping and divide
        denom = np.outer(A_d + gamma * pi[out_name],
                         G_d + gamma / pi[out_name])
        natgrad_rot = grad_rot / denom
        
        # rotate back to the original basis
        natgrad_Wb = A_Q @ natgrad_rot @ G_Q.T
        
        natgrad_W, natgrad_b = natgrad_Wb[:-1, :], natgrad_Wb[-1, :]
        natgrad[out_name] = (natgrad_W, natgrad_b)
    return natgrad

compute_natgrad_from_eigs_helper = jit(compute_natgrad_from_eigs_helper, static_argnums=(0,))

def compute_natgrad_from_eigs(arch, grad_w, A_eig, G_eig, pi, gamma):
    param_grad = arch.unflatten(grad_w)
    natgrad = compute_natgrad_from_eigs_helper(
        arch.param_info, param_grad, A_eig, G_eig, pi, gamma)
    return arch.flatten(natgrad)

def compute_A_chunk(apply_fn, nll_fn, unflatten_fn, w, X, T, dirs, grad_w):
    ndir = len(dirs)
    predict_wrap = lambda w: apply_fn(unflatten_fn(w), X)
    
    RY, RgY = [], []
    for v in dirs:
        Y, RY_ = jvp(predict_wrap, (w,), (v,))
        nll_wrap = lambda Y: nll_fn(Y, T)
        RgY_ = kfac_util.hvp(nll_wrap, Y, RY_)
        RY.append(RY_)
        RgY.append(RgY_)

    A = np.array([[onp.sum(RY[i] * RgY[j])
                   for j in range(ndir)]
                  for i in range(ndir)])

    return A

compute_A_chunk = jit(compute_A_chunk, static_argnums=(0, 1, 2))


def compute_step_coeffs(arch, output_model, w, X, T, dirs, grad_w,
                        weight_cost, lmbda, chunk_size):
    """Compute the coefficients alpha and beta which minimize the quadratic
    approximation to the cost in the update:
    
            new_update = sum of coeffs[i] * dirs[i]

    Note that, unlike the rest of the K-FAC algorithm, this function assumes
    the loss function is negative log-likelihood for an exponential family.
    (This is because it relies on the Fisher information matrix approximating
    the Hessian of the NLL.)
    """
    ndir = len(dirs)

    # First, compute the "function space" portion of the quadratic approximation.
    # This is based on the Gauss-Newton approximation to the NLL, or equivalently,
    # the Fisher information matrix.
    
    A_func = onp.zeros((ndir, ndir))
    batch_size = X.shape[0]
    for chunk_idxs in get_chunks(batch_size, chunk_size):
        X_chunk, T_chunk = X[chunk_idxs, :], T[chunk_idxs, :]

        A_func += compute_A_chunk(arch.net_apply, output_model.nll_fn, arch.unflatten,
                                  w, X_chunk, T_chunk, dirs, grad_w)
    
    A_func /= batch_size
    
    # Now compute the weight space terms, which include both the Hessian of the
    # L2 regularizer and the damping term. This is almost a multiple of the
    # identity matrix, except that the L2 penalty only applies to weights, not
    # biases. Hence, we need to apply a mask to zero out the entries corresponding
    # to biases. This can be done using a Hessian-vector product with the L2
    # regularizer, which has the added benefit that the solution generalizes
    # to non-uniform L2 regularizers as well.
    
    wrap = lambda w: L2_penalty(arch, w)
    Hv = [kfac_util.hvp(wrap, w, v) for v in dirs]
    A_L2 = onp.array([[weight_cost * Hv[i] @ dirs[j]
                       for i in range(ndir)]
                      for j in range(ndir)])
    A_prox = onp.array([[lmbda * dirs[i] @ dirs[j]
                         for i in range(ndir)]
                        for j in range(ndir)])
    A = A_func + A_L2 + A_prox
    
    # The linear term is much simpler: it's just the dot product with the gradient.
    b = onp.array([v @ grad_w for v in dirs])
    
    # Minimize the quadratic approximation by solving the linear system.
    coeffs = onp.linalg.solve(A, -b)
    
    # The decrease in the quadratic objective is used to adapt lambda.
    quad_decrease = -0.5 * coeffs @ A @ coeffs - b @ coeffs
    
    return coeffs, quad_decrease

def compute_update(coeffs, dirs):
    ans = 0
    for coeff, v in zip(coeffs, dirs):
        ans = ans + coeff * v
    return ans

# Two-Level K-FAC (sum of inverses - standard Nicolaides coarse space)
def compute_natgrad_correction_cgc_const_basis(param_info, flatten_fn, unflatten_fn, ZZt, grad_w, F_coarse, gamma):
    # compute coarse grad
    grad_dict = unflatten_fn(grad_w)
    grad_Wb_mean = {out_name: 0.0 for _, out_name in param_info}
    for _, out_name in param_info:
        grad_W, grad_b = grad_dict[out_name]
        grad_Wb = np.vstack([grad_W, grad_b.reshape((1, -1))])
        grad_Wb_mean[out_name] = np.sum(grad_Wb) / scale_fn( onp.prod(grad_Wb.shape) )

    grad_w_coarse = np.vstack([grad_Wb_mean[name] for name in grad_Wb_mean])

    # solve for coarse natgrad
    natgrad_w_coarse = np.linalg.solve(F_coarse + (gamma**2)*ZZt, grad_w_coarse)

    # prolongate
    natgrad_corr_dict = {out_name: 0.0 for _, out_name in param_info}
    for index, (_, out_name) in enumerate(param_info):
        W_shape, b_shape = grad_dict[out_name][0].shape, grad_dict[out_name][1].shape
        val = natgrad_w_coarse[index] / scale_fn( onp.prod(W_shape) + onp.prod(b_shape) )
        natgrad_corr_dict[out_name] = (val*np.ones(W_shape), val*np.ones(b_shape))

    natgrad_corr_w = flatten_fn(natgrad_corr_dict)

    return natgrad_corr_w

compute_natgrad_correction_cgc_const_basis = jit(compute_natgrad_correction_cgc_const_basis, static_argnums=(0,1,2))

# Two-Level K-FAC (sum of inverses - enriched coarse space)
def compute_natgrad_correction_cgc_general(state, arch, grad_w, F_coarse, gamma):
    Z = state['Z']
    ZZt = state['ZZt']

    grad_w_coarse = Z @ grad_w.reshape((-1, 1))

    natgrad_w_coarse = np.linalg.solve(F_coarse + (gamma**2)*ZZt, grad_w_coarse)

    natgrad_corr_w = (Z.T @ natgrad_w_coarse).reshape((-1,))

    return natgrad_corr_w

# Two-Level K-FAC (sum of inverses)
def compute_natgrad_correction_cgc(state, arch, grad_w, F_coarse, gamma):
    # Note: we force the use of F_hat_coarse when using enriched coarse spaces
    use_enriched_coarse_space = (state['F_hat_coarse'].shape[0] > len(state['A'].keys()))
    if use_enriched_coarse_space:
        return compute_natgrad_correction_cgc_general(state, arch, grad_w, state['F_hat_coarse'], gamma)
    else:
        return compute_natgrad_correction_cgc_const_basis(arch.param_info, arch.flatten,
                arch.unflatten, state['ZZt'], grad_w, F_coarse, gamma)

# Computes P*v = (I - F*Q)*v or P.transpose()*v = (I - Q*F)*v
def P(state, arch, output_model, w, X, T, F_coarse, gamma, v, chunk_size, weight_cost, transpose):

    # GGN-vector product
    mvp = lambda v: gnhvp(arch, output_model, w, X, T, v, chunk_size)

    # damped GGN-vector product
    mvp_damp = kfac_util.dampen(mvp, state['lambda'] + weight_cost)

    if not transpose:
        Qv = compute_natgrad_correction_cgc(state, arch, v, F_coarse, gamma)
        result = v - mvp_damp(Qv)
    else:
        QFv = compute_natgrad_correction_cgc(state, arch, mvp_damp(v), F_coarse, gamma)
        result = v - QFv

    return result

# Two-Level K-FAC (inverse of sum, non-singular (v1) / singular (v2) F_coarse)
# TODO(nikolas): Current implementation is hacky. Needs heavy optimization.
def compute_natgrad_correction_woodbury(state, arch, natgrad_w, F_coarse, gamma, variant):

    # Unpack utils
    Z = state['Z']
    perm = state['perm']
    blk = state['blk']

    nlayers = len(blk)-1
    nbasis = Z.shape[0] // nlayers

    # compute invF_dot_Zt = F \ Z.T
    if nbasis > 1:
        # general case of multiple basis vectors per layer
        basis = state['basis']
        basis_hstacked = np.hstack([basis[key] for key in basis.keys()])

        invF_dot_Zt_multicol = np.zeros(basis_hstacked.T.shape)
        for bi in range(nbasis):
            Zt_col = basis_hstacked[bi, :].T
            invF_dot_Zt_col = compute_natgrad_from_eigs(
                arch, Zt_col, state['A_eig'], state['G_eig'], state['pi'], gamma)
            invF_dot_Zt_multicol = invF_dot_Zt_multicol.at[:, bi].set(invF_dot_Zt_col)

        invF_dot_Zt = np.zeros(Z.T.shape)
        index = 0
        for key in sorted(perm.keys()):
            offset = nbasis*perm[key]
            invF_dot_Zt = invF_dot_Zt.at[blk[index]:blk[index+1], offset:offset+nbasis].set(invF_dot_Zt_multicol[blk[index]:blk[index+1], :])
            index = index + 1
    else:
        # special case for single basis vector per layer
        Zt_col = np.sum(Z, axis=0)
        invF_dot_Zt_col = compute_natgrad_from_eigs(
            arch, Zt_col, state['A_eig'], state['G_eig'], state['pi'], gamma)

        invF_dot_Zt = np.zeros(Z.T.shape)
        index = 0
        for key in sorted(perm.keys()):
            invF_dot_Zt = invF_dot_Zt.at[blk[index]:blk[index+1], perm[key]].set(invF_dot_Zt_col[blk[index]:blk[index+1]])
            index = index + 1

    # Zero-out diagonal of F_coarse
    F_coarse = F_coarse - np.diag(np.diag(F_coarse))

    if variant == 1:
        IF = np.linalg.inv(F_coarse) +  Z @ invF_dot_Zt
        natgrad_w_hat = np.linalg.solve(IF, (Z @ natgrad_w))
    else:
        IF = np.eye(F_coarse.shape[0]) + F_coarse @ ( Z @ invF_dot_Zt )
        natgrad_w_hat = np.linalg.solve(IF, F_coarse @ (Z @ natgrad_w))

    natgrad_corr = -invF_dot_Zt @ natgrad_w_hat

    return natgrad_corr

def compute_gammas(curr_gamma, step, config):
    if step % config['gamma_update_interval'] == 0:
        gamma_less = onp.maximum(
            curr_gamma * config['gamma_drop']**config['gamma_update_interval'],
            config['gamma_min'])
        gamma_more = onp.minimum(
            curr_gamma * config['gamma_boost']**config['gamma_update_interval'],
            config['gamma_max'])
        gammas = [gamma_less, curr_gamma, gamma_more]
    else:
        gammas = [curr_gamma]
    return gammas

def update_lambda(arch, output_model, lmbda, old_w, new_w, X, T, quad_dec, config):
    old_cost = compute_cost(
        arch, output_model.nll_fn, old_w, X, T, config['weight_cost'], config['chunk_size'])
    new_cost = compute_cost(
        arch, output_model.nll_fn, new_w, X, T, config['weight_cost'], config['chunk_size'])
    rho = (old_cost - new_cost) / quad_dec
    
    if np.isnan(rho) or rho < 0.25:
        new_lambda = np.minimum(
            lmbda * config['lambda_boost']**config['lambda_update_interval'],
            config['lambda_max'])
    elif rho > 0.75:
        new_lambda = np.maximum(
            lmbda * config['lambda_drop']**config['lambda_update_interval'],
            config['lambda_min'])
    else:
        new_lambda = lmbda
    
    return new_lambda, rho
    
def apply_preconditioner(state, arch, output_model, grad_w, X, T, F_coarse, gamma, config):
    if ('m1' in config['optimizer']) or ('m3' in config['optimizer']):
        P_fn = lambda v: P(state, arch, output_model, state['w'], X, T,
                F_coarse, gamma, v, config['chunk_size'],
                config['weight_cost'], transpose=False)
    else:
        P_fn = lambda v: v

    if ('m2' in config['optimizer']) or ('m3' in config['optimizer']):
        Pt_fn = lambda v: P(state, arch, output_model, state['w'], X, T,
                F_coarse, gamma, v, config['chunk_size'],
                config['weight_cost'], transpose=True)
    else:
        Pt_fn = lambda v: v

    # Compute the approximate natural gradient
    natgrad_w = Pt_fn(compute_natgrad_from_eigs(
        arch, P_fn(grad_w), state['A_eig'], state['G_eig'], state['pi'], gamma))
    #natgrad_w_pre_norm[idx] = np.linalg.norm(natgrad_w)

    if config['has_correction']:
        natgrad_corr = config['natgrad_correction_fn'](
                state, arch, grad_w, natgrad_w, F_coarse, gamma)
        #natgrad_w_corr_norm[idx] = np.linalg.norm(natgrad_corr)
        natgrad_w = natgrad_w + natgrad_corr
    return natgrad_w

def cg_benchmark(state, arch, output_model, X, T, F_coarse, gamma, config, mvp_damp, grad_w, tol, maxiter):
    val = {}
    relres = {}
    x0 = -state['update']

    preconditioners = ['none',
                       'kfac',
                       'kfac-cgc',
                       'kfac-cgc-m1',
                       #'kfac-cgc-m2',
                       'kfac-cgc-m3',
                       #'kfac-m3',
                       #'kfac-m2',
                       #'kfac-cgc-m1-Qb',
                       'kfac-cgc-m2-Qb',
                       #'kfac-cgc-m3-Qb',
                       'kfac-m3-Qb',
                       'kfac-m2-Qb']
                       #'kfac-woodbury-v1',
                       #'kfac-woodbury-v2',
                       #'none-x0',
                       #'kfac-x0',
                       #'kfac-cgc-x0',
                       #'kfac-cgc-m1-x0',
                       #'kfac-cgc-m2-x0',
                       #'kfac-cgc-m3-x0',
                       #'kfac-m3-x0',
                       #'kfac-m2-x0',
                       #'kfac-cgc-m1-Qb-plus-Ptx0',
                       #'kfac-cgc-m2-Qb-plus-Ptx0',
                       #'kfac-cgc-m3-Qb-plus-Ptx0',
                       #'kfac-m3-Qb-plus-Ptx0',
                       #'kfac-m2-Qb-plus-Ptx0']

    # Compute initial estimate for two-level preconditioned CG
    Qb = compute_natgrad_correction_cgc(state, arch, grad_w, F_coarse, gamma)

    Pt_fn = lambda v: P(state, arch, output_model, state['w'], X, T, F_coarse,
            gamma, v, config['chunk_size'], config['weight_cost'],
            transpose=True)

    Qb_plus_Ptx0 = Qb + Pt_fn(x0)

    # Preconditioned CG
    for prec in preconditioners:
        print(f'Running {maxiter} CG iterations with preconditioner M = {prec} ...')

        # Setup preconditioner's config
        _config = copy(config)
        _config['optimizer'] = prec
        _config['has_correction'] = False
        if 'cgc' in prec:
            _config['has_correction'] = True
            _config['natgrad_correction_fn'] = \
                lambda state, arch, grad_w, natgrad_w, F_coarse, gamma: \
                    compute_natgrad_correction_cgc(state, arch, grad_w, F_coarse, gamma)
        if 'woodbury' in prec:
            _config['has_correction'] = True
            variant = int(prec[-1])
            _config['natgrad_correction_fn'] = \
                lambda state, arch, grad_w, natgrad_w, F_coarse, gamma: \
                    compute_natgrad_correction_woodbury(state, arch, natgrad_w, F_coarse, gamma, variant)

        # Setup preconditioner
        if prec != 'none':
            M = lambda grad_w: apply_preconditioner(state, arch, output_model, grad_w, X, T, F_coarse, gamma, _config)
        else:
            M = None

        # Initial estimate for CG
        if 'Qb-plus-Ptx0' in prec:
            _x0 = Qb_plus_Ptx0
        elif 'Qb' in prec:
            _x0 = Qb
        elif 'x0' in prec:
            _x0 = x0
        else:
            _x0 = None

        # Run CG
        _, _, val[prec], relres[prec] = cg(mvp_damp, grad_w, x0=_x0, tol=tol, atol=0.0, maxiter=maxiter, M=M, has_aux=True)

    return val, relres

def kfac_init(arch, output_model, X_train, T_train, config, random_seed=0):
    state = {}
    
    state['step'] = 0
    state['rng'] = random.PRNGKey(random_seed)
    
    state['gamma'] = config['init_gamma']
    state['lambda'] = config['init_lambda']
    
    key, state['rng'] = random.split(state['rng'])
    _, params = arch.net_init(key, X_train.shape)
    state['w'] = arch.flatten(params)
    state['w_avg'] = state['w']
    
    key, state['rng'] = random.split(state['rng'])
    state['A'], state['G'], state['A_coarse'], state['G_coarse'], state['F_coarse'], *_ = estimate_covariances(
        arch, output_model, state['w'], X_train, key, config['chunk_size'], has_aux=(config['nbasis'] > 1))
    if config['nbasis'] > 1:
        state['A_'], state['G_'] = _
    else:
        state['A_'], state['G_'] = {}, {}

    state['A_eig'], state['G_eig'], state['pi'] = compute_eigs(
        arch, state['A'], state['G'])

    config['has_correction'] = False

    if 'cgc' in config['optimizer']:
        config['has_correction'] = True
        config['natgrad_correction_fn'] = \
            lambda state, arch, grad_w, natgrad_w, F_coarse, gamma: \
                compute_natgrad_correction_cgc(state, arch, grad_w, F_coarse, gamma)
    elif 'woodbury' in config['optimizer']:
        config['has_correction'] = True
        variant = int(config['optimizer'][-1])
        config['natgrad_correction_fn'] = \
            lambda state, arch, grad_w, natgrad_w, F_coarse, gamma: \
                compute_natgrad_correction_woodbury(state, arch, natgrad_w, F_coarse, gamma, variant)
    else:
        pass

    #flag = any(tag in config['optimizer'] for tag in ('m1', 'm2', 'm3'))
    flag = True

    if config['has_correction'] or flag:
        nlayers = len(arch.param_info)
        nparams = len(state['w'])
        Z = onp.zeros((nlayers, nparams), dtype=onp.float32)
        blk = onp.zeros((nlayers+1,), dtype=onp.uint32)

        index = 0
        perm = {}
        for key in params.keys():
            if len(params[key]) == 0: continue
            perm[key] = index
            index = index + 1

        index = 0
        for key in sorted(params.keys()):
            pk = params[key]
            if len(pk) == 0: continue
            npk = onp.prod(pk[0].shape) + onp.prod(pk[1].shape)
            blk[index+1] = blk[index] + npk
            Z[perm[key], blk[index]:blk[index+1]] = 1. / scale_fn(npk)
            index = index + 1

        state['Z'] = np.asarray(Z)
        state['ZZt'] = np.asarray(Z@Z.T)
        state['perm'] = perm
        state['blk'] = blk

    return state

        
def kfac_iter(state, arch, output_model, X_train, T_train, config):
    old_state = state
    state = dict(state)  # shallow copy
    
    state['step'] += 1

    ndata = X_train.shape[0]
    batch_size = get_batch_size(state['step'], ndata, config)
    state['batch_size'] = batch_size
    
    # Sample with replacement
    key, state['rng'] = random.split(state['rng'])
    idxs = random.permutation(key, np.arange(ndata))[:batch_size]
    X_batch, T_batch = X_train[idxs, :], T_train[idxs, :]

    # Compute the gradient
    grad_w = compute_gradient(
        arch, output_model, state['w'], X_batch, T_batch,
        config['weight_cost'], config['chunk_size'])

    # Update statistics by running backprop on the sampled targets
    if state['step'] % config['cov_update_interval'] == 0:
        batch_size_samp = get_sample_batch_size(batch_size, config)
        X_samp = X_batch[:batch_size_samp, :]
        state['A'], state['G'], state['A_coarse'], state['G_coarse'], state['F_coarse'], *_ = update_covariances(
            state['A'], state['G'], state['A_coarse'], state['G_coarse'], state['F_coarse'], arch, output_model, state['w'], X_samp, state['rng'],
            config['cov_timescale'], config['chunk_size'], state['A_'], state['G_'], has_aux=(config['nbasis'] > 1))
        if config['nbasis'] > 1:
            state['A_'], state['G_'] = _
        if config['nbasis'] == 1:
            state['F_hat_coarse'] = state['A_coarse'] * state['G_coarse']

    # Update the inverses
    if state['step'] % config['eig_update_interval'] == 0:
        state['A_eig'], state['G_eig'], state['pi'] = compute_eigs(
            arch, state['A'], state['G'])

    # Recompute enriched coarse space utils (Z, ZZt, F_hat_coarse)
    if (config['nbasis'] > 1) and \
            ((state['step'] == 1) or \
            (state['step'] % config['eig_update_interval'] == 0)):
        state['Z'], state['ZZt'], state['F_hat_coarse'], state['basis'] = recompute_enriched_coarse_space(arch, state, grad_w, config['nbasis'])

    # Update gamma
    if config['adapt_gamma']:
        gammas = compute_gammas(state['gamma'], state['step'], config)
    else:
        gammas = [np.sqrt(state['lambda'] + config['weight_cost'])]

    natgrad_w_pre_norm = [0. for _ in range(len(gammas))]
    natgrad_w_corr_norm = [0. for _ in range(len(gammas))]
    coeffs = [[] for _ in range(len(gammas))]
    quad_dec = [[] for _ in range(len(gammas))]
    update = [[] for _ in range(len(gammas))]
    new_w = [[] for _ in range(len(gammas))]
    results = []

    # GGN-vector product
    mvp = lambda v: gnhvp(arch, output_model, state['w'], X_batch, T_batch, v, config['chunk_size'])

    # damped GGN-vector product
    mvp_damp = kfac_util.dampen(mvp, state['lambda'] + config['weight_cost'])

    for idx, gamma in enumerate(gammas):

        # preconditioner
        precon = lambda grad_w: apply_preconditioner(state, arch, output_model,
                grad_w, X_batch, T_batch, state['F_hat_coarse'], gamma, config)

        # initial estimate for current step
        if 'Qb' in config['optimizer']:
            x0 = compute_natgrad_correction_cgc(state, arch, grad_w, state['F_hat_coarse'], gamma)
        else:
            x0 = None

        if 'conjgrad' in config['optimizer']:
            tol = config['conjgrad_tol']
            maxiter = config['conjgrad_maxiter']
            if 'kfac' in config['optimizer']:
                natgrad_w, info = cg(mvp_damp, grad_w, x0=x0, tol=tol, atol=0.0, maxiter=maxiter, M=precon)
            else:
                natgrad_w, info = cg(mvp_damp, grad_w, x0=None, tol=tol, atol=0.0, maxiter=maxiter)
            state['conjgrad_niters'] = info
        else:
            if 'Qb' in config['optimizer']:
                if 'alpha' in config['optimizer']:
                    r = grad_w - mvp_damp(x0)
                    p = precon(r)
                    Ap = mvp_damp(p)
                    alpha = (r.T @ p) / (p.T @ Ap)
                    natgrad_w = x0 + alpha*p
                else:
                    natgrad_w = x0 + precon(grad_w - mvp_damp(x0))
            else:
                natgrad_w = precon(grad_w)

        # Determine the step size parameters using MVPs
        if config['use_momentum'] and 'update' in state:
            prev_update = state['update']
            dirs = [-natgrad_w, prev_update]
        else:
            dirs = [-natgrad_w]

        #quad_dec[idx] = -0.5 * (dirs[0].T @ mvp_damp(dirs[0])) - grad_w.T @ dirs[0]
        #print(f"quad_dec = {quad_dec[idx]}")

        coeffs[idx], quad_dec[idx] = compute_step_coeffs(
            arch, output_model, state['w'], X_batch, T_batch, dirs,
            grad_w, config['weight_cost'], state['lambda'], config['chunk_size'])
        #print(f"quad_dec = {quad_dec[idx]}")

        update[idx] = compute_update(coeffs[idx], dirs)
        new_w[idx] = state['w'] + update[idx]

        results.append(compute_cost(
            arch, output_model.nll_fn, new_w[idx], X_batch, T_batch,
            config['weight_cost'], config['chunk_size']))

    # Store values for best_idx in state
    best_idx = onp.argmin(results)
    state['gamma'] = gammas[best_idx]

    # Run CG benchmark
    if state['step'] % config['conjgrad_benchmark_interval'] == 0:
        state['conjgrad_val'], state['conjgrad_relres'] = \
        cg_benchmark(state, arch, output_model, X_batch, T_batch,
                state['F_hat_coarse'], state['gamma'], config, mvp_damp, grad_w,
                config['conjgrad_tol'], config['conjgrad_maxiter'])

    #state['natgrad_w_pre_norm'] = natgrad_w_pre_norm[best_idx]
    #state['natgrad_w_corr_norm'] = natgrad_w_corr_norm[best_idx]
    state['coeffs'] = coeffs[best_idx]
    state['quad_dec'] = quad_dec[best_idx]
    state['update'] = update[best_idx]
    state['w'] = new_w[best_idx]

    # Update lambda
    if state['step'] % config['lambda_update_interval'] == 0:
        state['lambda'], state['rho'] = update_lambda(
            arch, output_model, state['lambda'], old_state['w'], state['w'], X_batch,
            T_batch, state['quad_dec'], config)
        
    # Iterate averaging
    ema_param = kfac_util.get_ema_param(config['param_timescale'])
    state['w_avg'] = ema_param * state['w_avg'] + (1-ema_param) * state['w']
    
    return state



