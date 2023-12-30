
import jax.numpy as jnp
import jax.numpy.linalg as jla

# from jax.config import config
# config.update("jax_enable_x64", True)


def asym(mat):
    return 0.5*(mat - mat.T)


def sym2(mat):
    return mat + mat.T


def sym(a):
    return 0.5*sym2(a)


def Lyapunov(A, B):
    # solve AU + UA = B
    # A, B, U are symmetric
    yei, yv = jla.eigh(A)
    return yv@((yv.T@B@yv)/(yei[:, None] + yei[None, :]))@yv.T


def vcat(x, y):
    return jnp.concatenate([x, y], axis=0)


def projF(x, omg):
    """
    """
    # return omg - x@jla.solve(x.T@x, asym(x.T@omg))
    return omg - 2*x@Lyapunov(x.T@x, asym(x.T@omg))


def DprojF(x, xi, omg):
    """
    """
    U = Lyapunov(x.T@x, asym(x.T@omg))
    Ap = sym2(xi.T@x)
    return - 2*xi@U - 2*x@Lyapunov(x.T@x, asym(xi.T@omg) - Ap@U - U@Ap)


def find_LWInit0(x, tol=1e-4, maxiter=10):
    err = 10
    if jnp.abs(x - jnp.e) < tol:
        return 1.
    if x >= jnp.e:
        ww = 1.
        for i in range(maxiter):
            wwn = jnp.log(x) - jnp.log(ww)
            err = jnp.abs(wwn - ww)
            if (err < tol):
                return wwn, i
            ww = wwn
    else:
        ww = 1.
        for i in range(maxiter):
            wwn = x/jnp.exp(ww)
            err = jnp.abs(wwn - ww)
            if (err < tol):
                return wwn, i
            ww = wwn
    return ww, maxiter


def find_LWInit1(x, tol=1e-4, maxiter=10):
    """ second branch, x between -1/e and 0
        return w between 0 and -1
    """
    err = 10
    if jnp.abs(x - jnp.e) < tol:
        return 1.
    ww = -1.
    for i in range(maxiter):
        wwn = jnp.log(-x) - jnp.log(-ww)
        err = jnp.abs(wwn - ww)
        if (err < tol):
            return wwn, i
        ww = wwn
    return ww, maxiter


def LambertW(x, branch=0, maxiter=10):
    if branch == 0:
        ww, _ = find_LWInit0(x, maxiter=maxiter)
    else:
        ww, _ = find_LWInit1(x, maxiter=maxiter)
    for i in range(maxiter):
        zn = jnp.log(x/ww) - ww
        qn = 2*(1+ww)*(1+ww+2/3.*zn)
        en = zn/(1+ww)*(qn-zn)/(qn-2*zn)
        wwn = ww*(1+en)
        err = jnp.abs(wwn - ww)
        if (err < 1e-10):
            return wwn, i
        ww = wwn
    return ww, maxiter







