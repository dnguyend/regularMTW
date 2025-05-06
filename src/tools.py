""" various util files
"""
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random


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


def cz(mat):
    return jnp.max(jnp.abs(mat))


def grand(key, dims):
    key, sk = random.split(key)
    return random.normal(sk, dims), key


def splitzero(v):
    n = v.shape[0]//2
    zr = jnp.zeros(n)
    return vcat(v[:n], zr), vcat(zr, v[n:])


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


def bisect_newton(func, x1, x2, maxiter=20, tol=1e-13):
    # float df, dx, dxold, f, fh, fl;
    # float temp, xh, xl, rts;

    fl, _ = func(x1)
    fh, _ = func(x2)
    # print(f"{x1} {fl} {x2} {fh}")
    if ((fl > 0.0) and (fh > 0.0)) or ((fl < 0.0) and (fh < 0.0)):
        raise ValueError("Root must be bracketed in rtsafe")
    if fl == 0.0:
        return x1, 0, None
    if fh == 0.0:
        return x2, 0, None
    if fl < 0.0:
        xl = x1
        xh = x2
    else:
        xh = x1
        xl = x2
    rts = 0.5*(x1 + x2)
    dxold = jnp.abs(x2 - x1)
    dx = dxold
    f, df = func(rts)
    for j in jnp.arange(maxiter):
        if (((rts - xh)*df - f)*((rts - xl)*df - f) > 0) \
           or (jnp.abs(f) > 0.5*jnp.abs(dxold*df)):
            dxold = dx
            dx = 0.5*(xh - xl)
            rts = xl + dx
            # print("in bisect")
            if xl == rts:
                return rts, j+1, None
        else:
            dxold = dx
            dx = f / df
            temp = rts
            rts -= dx
            # print("in newton")
            if (temp == rts):
                return rts, j+1, None

        if jnp.abs(dx) < tol:
            return rts, j+1, None
        f, df = func(rts)
        if (f < 0.0):
            xl = rts
        else:
            xh = rts
        # print(f"{rts} {xl} {xh} {f} {df}")
    return rts, j, "Maximum number of iterations exceeded in rtsafe"
