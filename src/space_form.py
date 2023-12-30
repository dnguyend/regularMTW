import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import random
from .tools import vcat


# jax.config.update("jax_enable_x64", True)


def grand(key, dims):
    key, sk = random.split(key)
    return random.normal(sk, dims), key


def splitzero(v):
    n = v.shape[0]//2
    zr = jnp.zeros(n)
    return vcat(v[:n], zr), vcat(zr, v[n:])


class SimpleKH(object):
    def __init__(self, n, A, eps, der_func):
        self.n = n
        self.A = A
        self.iA = A   # assume this is the case.
        self.eps = eps
        self.der_func = der_func
        self.nm = jnp.where(jnp.diagonal(A) < 0)[0].shape[0]

    @staticmethod
    def flip0(vec):
        if vec[0] < 0:
            return - vec
        else:
            return vec

    def Adot(self, omg1, omg2):
        return jnp.sum(omg1*(self.A@omg2))

    def Adot2(self, Omg1, Omg2):
        A = self.A
        n = self.n
        return jnp.sum(Omg1[:n]*(A@Omg2[:n])) \
            + jnp.sum(Omg1[n:]*(A@Omg2[n:]))

    def KMMetric(self, q, Omg1, Omg2):
        n = self.n
        Adot = self.Adot
        x = q[:n]
        y = q[n:]
        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))

        return -0.5*(-h2/h1**3*(Adot(Omg1[:n], y)*Adot(Omg2[n:], x)
                                + Adot(Omg1[n:], x)*Adot(Omg2[:n], y))
                     + 1/h1*(Adot(Omg1[:n], Omg2[n:])
                             + Adot(Omg1[n:], Omg2[:n])))

    def projSimpleSphere(self, x, omg):
        eps = self.eps
        Adot = self.Adot
        return omg - eps*x*Adot(x, omg)

    def projSphere(self, qs, Omg):
        n = self.n
        Adot = self.Adot
        eps = self.eps

        x = qs[:n]
        y = qs[n:]
        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))

        Omgx = Omg[:n]
        Omgy = Omg[n:]

        return Omg - vcat(
            ((h1**2-h0*h2)*y + eps*h2*x) *
            Adot(x, Omgx)/(h0*(h1**2-h0*h2) + h2),
            ((h1**2-h0*h2)*x + eps*h2*y) *
            Adot(y, Omgy)/(h0*(h1**2-h0*h2) + h2))

    def DprojSphere(self, qs, Xi, Eta):
        n = self.n
        Adot = self.Adot
        eps = self.eps

        x = qs[:n]
        y = qs[n:]
        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))

        Omgx = Eta[:n]
        Omgy = Eta[n:]

        return - vcat(
            ((h1**2-h0*h2)*y + eps*h2*x) *
            Adot(Xi[:n], Omgx)/(h0*(h1**2-h0*h2) + h2),
            ((h1**2-h0*h2)*x + eps*h2*y) *
            Adot(Xi[n:], Omgy)/(h0*(h1**2-h0*h2) + h2))

    def TwoSphereContract(self, qs, Xi):
        n = self.n
        Adot = self.Adot
        eps = self.eps

        x = qs[:n]
        y = qs[n:]

        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))                
        ret = -0.5*(eps*(h2**2 - h3*h1)*Adot(y, Xi[:n])**2
                    - Adot(Xi[:n], Xi[:n])*h1**2*(h1**2-h0*h2)
                    )*(eps*(h2**2 - h3*h1)*Adot(x, Xi[n:])**2
                       - Adot(Xi[n:], Xi[n:])*h1**2*(h1**2-h0*h2)
                       ) / (h0*(h1**2-h0*h2) + h2)/(h1**2-h0*h2) / h1**5

        return ret

    def crossCurvSphere(self, qs, Xi):
        n = self.n
        Adot = self.Adot
        eps = self.eps

        x = qs[:n]
        y = qs[n:]
        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))                

        TxS = Adot(Xi[:n], y)
        TyS = Adot(Xi[n:], x)
        TxyS = Adot(Xi[:n], Xi[n:])

        II = -(- h2*TxS*TyS + h1**2*TxyS)/h1**3
        S2 = TxS*TyS
        SXiperp2 = Adot(Xi[:n], Xi[:n]) - eps*TxS**2/(1-h0**2)
        SbXiperp2 = Adot(Xi[n:], Xi[n:])-eps*TyS**2/(1-h0**2)
        DD = (- h2-h0*h1**2+h2*h0**2)*(h1**2-h2*h0)*h1**5
        nullComp = - 2*(h1*h3 - h2**2)/h1**4*S2*II  + h2/h1*II*II
        R4 = h1**4*(h1**2-h2*h0)**2/DD
        sR23 = ((h3*h1-h2**2)*(1-h0**2) + h1**2*(h1**2-h2*h0))*h1**2*(h1**2-h2*h0)/DD
        sR1 = (((h1**2 - h0*h2)*h4 + h0*h3**2 - 2*h1*h2*h3 + h2**3)*(- h2-h0*h1**2+h2*h0**2)*(1-h0**2)**2 \
               + ((h3*h1-h2**2)*(1-h0**2) + h1**2*(h1**2-h2*h0))**2)/DD

        retnull = 0.5*(sR1*S2**2/(1-h0**2)**2
                       + eps*(sR23*TyS**2*SXiperp2 + sR23*TxS**2*SbXiperp2)/(1-h0**2)
                       + R4*SXiperp2*SbXiperp2)
        ret = retnull + nullComp

        return ret, sR1, sR23, R4

    def GammaAmbient(self, q, Omg1, Omg2):
        n = self.n
        x = q[:n]
        y = q[n:]
        Adot = self.Adot

        u, h0, h1, h2, h3, h4 = self.der_func(Adot(x, y))

        retx = (h2**2 - h3*h1)/h1**2/(h1**2-h0*h2)*Adot(y, Omg1[:n])*Adot(Omg2[:n], y)*x \
            - h2/h1**2*Omg1[:n]*Adot(Omg2[:n], y) \
            - h2/h1**2*Omg2[:n]*Adot(Omg1[:n], y)

        rety = (h2**2 - h3*h1)/h1**2/(h1**2-h0*h2)*Adot(x, Omg1[n:])*Adot(Omg2[n:], x)*y \
            - h2/h1**2*Omg1[n:]*Adot(Omg2[n:], x) \
            - h2/h1**2*Omg2[n:]*Adot(Omg1[n:], x)

        return vcat(retx, rety)

    def Gamma(self, qs, Xi1, Xi2):
        GA = self.GammaAmbient(qs, Xi1, Xi2)
        return - self.DprojSphere(qs, Xi1, Xi2) + self.projSphere(qs, GA)

    def gen_qs(self, key):
        nm = self.nm
        eps = self.eps
        n = self.n
        if ((nm == 0) and eps < 0) or ((nm == n) and eps > 0):
            raise ValueError("definite form with opposite constraint")
        sk, key = random.split(key)
        tmp = random.normal(sk, (2*n,))
        if (nm == 0) or (nm == n):
            return jnp.concatenate(
                [tmp[:n]/jnp.sqrt(jnp.sum(tmp[:n]**2)),
                 tmp[n:]/jnp.sqrt(jnp.sum(tmp[n:]**2))]), key

        np = n - nm
        if eps < 0:
            qm0 = tmp[2*np:2*np+nm]*jnp.sqrt(1. + jnp.sum(tmp[:np]**2)) /\
                jnp.sqrt(jnp.sum(tmp[2*np:2*np+nm]**2))
            qm1 = tmp[2*np+nm:]*jnp.sqrt(1. + jnp.sum(tmp[np:2*np]**2)) /\
                jnp.sqrt(jnp.sum(tmp[2*np+nm:]**2))
            ret0 = SimpleKH.flip0(jnp.concatenate([qm0, tmp[:np]]))
            ret1 = SimpleKH.flip0(jnp.concatenate([qm1, tmp[np:2*np]]))
            return jnp.concatenate([ret0, ret1]), key
        else:
            qp0 = tmp[2*nm:2*nm+np]*jnp.sqrt(1. + jnp.sum(tmp[:nm]**2)) /\
                jnp.sqrt(jnp.sum(tmp[2*nm:2*nm+np]**2))
            qp1 = tmp[2*nm+np:]*jnp.sqrt(1. + jnp.sum(tmp[nm:2*nm]**2)) /\
                jnp.sqrt(jnp.sum(tmp[2*nm+np:]**2))                
            return jnp.concatenate([tmp[:nm], qp0, tmp[nm:2*nm], qp1]), key

    def gen_xs(self, key):
        nm = self.nm
        eps = self.eps
        n = self.n
        
        if ((nm == 0) and eps < 0) or ((nm == n) and eps > 0):
            raise ValueError("definite form with opposite constraint")
        sk, key = random.split(key)
        tmp = random.normal(sk, (n,))
        if (nm == 0) or (nm == n):
            return tmp[:n]/jnp.sqrt(jnp.sum(tmp[:n]**2)), key

        np = n - nm
        if eps < 0:
            qm0 = tmp[np:np+nm]*jnp.sqrt(1. + jnp.sum(tmp[:np]**2)) /\
                jnp.sqrt(jnp.sum(tmp[np:np+nm]**2))
            ret = SimpleKH.flip0(jnp.concatenate([qm0, tmp[:np]]))
            return ret, key
        else:
            qp0 = tmp[nm:nm+np]*jnp.sqrt(1. + jnp.sum(tmp[:nm]**2)) /\
                jnp.sqrt(jnp.sum(tmp[nm:nm+np]**2))
            ret = SimpleKH.flip0(jnp.concatenate([tmp[:nm], qp0]))
            return ret, key

    def gennull_sphere(self, key, q):
        n = self.n
        iA = self.iA
        Adot = self.Adot
        tmp, key = grand(key, (n, 2))
        x = q[:n]
        y = q[n:]
        sig = Adot(x, y)
        tmp0 = tmp[:, 0]
        tmp1 = tmp[:, 1]

        tmp0 = self.projSimpleSphere(x, tmp0)
        TxS = Adot(tmp0, y)

        u, h0, h1, h2, h3, h4 = self.der_func(sig)

        # d1, d2 = calc_derivs(ufunc(sig), 2)
        mat = jnp.concatenate([-h2*TxS/h1**2*x + tmp0[None, :], y[None, :]], axis=0)

        return vcat(tmp0, tmp1 - mat.T@jla.solve(mat@iA@mat.T, mat@iA@tmp1)), key

    def krandvec(self, key, qs):
        """ random tangent vector
        """
        n = self.n

        tmp, key = grand(key, (n, 2))
        return vcat(self.projSimpleSphere(qs[:n], tmp[:, 0]),
                    self.projSimpleSphere(qs[n:], tmp[:, 1])), key
