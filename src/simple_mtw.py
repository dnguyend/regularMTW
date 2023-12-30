import jax.numpy as jnp
import jax.numpy.linalg as jla

from jax import jvp, random
from .tools import vcat, LambertW

# from jax.config import config
# config.update("jax_enable_x64", True)


def grand(key, dims):
    key, sk = random.split(key)
    return random.normal(sk, dims), key


def splitzero(v):
    n = v.shape[0]//2
    zr = jnp.zeros(n)
    return vcat(v[:n], zr), vcat(zr, v[n:])


class baseSimpleMTW(object):
    def __init__(self, n, A, params):
        self.n = n
        self.A = A
        self.iA = jla.inv(A)
        self.p = jnp.array(params)

    def Adot(self, v1, v2):
        return jnp.sum(v1*(self.A@v2))

    def sfunc(self, u):
        raise NotImplementedError

    def dsfunc(self, u, order):
        """order = 1, 2, 3, 4.
        return array of size order + 1
        entry 0 is sfunc, 1 is sfunc' etc
        """
        raise NotImplementedError

    def ufunc(self, sigma):
        """ invert function of sfunc
        """
        raise NotImplementedError

    def c(self, q):
        n = self.n
        x, y = q[:n], q[n:]
        return self.ufunc(self.Adot(x, y))

    def dufunc(self, u, order):
        ha = self.dsfunc(u, order)
        if order == 1:
            return jnp.array([1/ha[1]])
        elif order == 2:
            dd1 = 1/ha[1]
            dd2 = - ha[2]/ha[1]**3
            return jnp.array([dd1, dd2])
        elif order == 3:
            dd1 = 1/ha[1]
            dd2 = - ha[2]/ha[1]**3
            dd3 = - (ha[3]*ha[1]-3*ha[2]**2)/ha[1]**5
            return jnp.array([dd1, dd2, dd3])
        elif order == 4:
            dd1 = 1/ha[1]
            dd2 = - ha[2]/ha[1]**3
            dd3 = - (ha[3]*ha[1]-3*ha[2]**2)/ha[1]**5
            dd4 = - (ha[4]*ha[1]**2 - 10*ha[3]*ha[2]*ha[1] + 15*ha[2]**3)/ha[1]**7
            return jnp.array([dd1, dd2, dd3, dd4])

    def KMMetric(self, q, Omg1, Omg2):
        n = self.n

        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        # d1, d2 = calc_derivs(ufunc(sig), 2)
        return -0.5*(-ha[2]/ha[1]**3*(
            self.Adot(Omg1[:n], y)*self.Adot(Omg2[n:], x)
            + self.Adot(Omg1[n:], x)*self.Adot(Omg2[:n], y))
                     + 1/ha[1]*(self.Adot(Omg1[:n], Omg2[n:])
                                + self.Adot(Omg1[n:], Omg2[:n]))
                     )

    def g(self, q, Omg):
        n = self.n
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        return -0.5*vcat(-ha[2]/ha[1]**3*y*self.Adot(Omg[n:], x) + 1/ha[1]*Omg[n:],
                         -ha[2]/ha[1]**3*x*self.Adot(Omg[:n], y) + 1/ha[1]*Omg[:n])

    def ginv(self, q, Omg):
        n = self.n
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        return -2*vcat(
            ha[1]*Omg[n:] + ha[1]*ha[2]*self.Adot(y, Omg[n:])/(ha[1]**2-ha[0]*ha[2])*x,
            ha[1]*Omg[:n] + ha[1]*ha[2]*self.Adot(x, Omg[:n])/(ha[1]**2-ha[0]*ha[2])*y)

    def Gamma(self, q, Omg1, Omg2):
        n = self.n
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        retx = (ha[2]**2 - ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*self.Adot(y, Omg1[:n])*self.Adot(Omg2[:n], y)*x \
            - ha[2]/ha[1]**2*Omg1[:n]*self.Adot(Omg2[:n], y) \
            - ha[2]/ha[1]**2*Omg2[:n]*self.Adot(Omg1[:n], y)

        rety = (ha[2]**2 - ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*self.Adot(x, Omg1[n:])*self.Adot(Omg2[n:], x)*y \
            - ha[2]/ha[1]**2*Omg1[n:]*self.Adot(Omg2[n:], x) \
            - ha[2]/ha[1]**2*Omg2[n:]*self.Adot(Omg1[n:], x)

        return vcat(retx, rety)

    def Curv3(self, q, Omg1, Omg2, Omg3):
        D1 = jvp(lambda q: self.Gamma(q, Omg2, Omg3), (q,), (Omg1,))[1]
        D2 = jvp(lambda q: self.Gamma(q, Omg1, Omg3), (q,), (Omg2,))[1]
        G1 = self.Gamma(q, Omg1, self.Gamma(q, Omg2, Omg3))
        G2 = self.Gamma(q, Omg2, self.Gamma(q, Omg1, Omg3))
        return D1 - D2 + G1 - G2

    def crossCurv(self, q, Omg):
        n = self.n
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 4)

        TxS = self.Adot(Omg[:n], y)
        TyS = self.Adot(Omg[n:], x)
        TxyS = self.Adot(Omg[:n], Omg[n:])

        II = -(- ha[2]*TxS*TyS + ha[1]**2*TxyS)/ha[1]**3
        S2 = TxS*TyS

        c1 = -((ha[1]**2 - ha[0]*ha[2])*ha[4] + ha[0]*ha[3]**2 - 2*ha[1]*ha[2]*ha[3] + ha[2]**3) \
            / (ha[1]**2-ha[0]*ha[2])/ha[1]**5*S2**2 \
            + 4*(ha[1]*ha[3] - ha[2]**2)/ha[1]**4*S2*II \
            - 2*ha[2]/ha[1]*II*II

        return -0.5*c1

    def grand(self, key):
        key, sk = random.split(key)
        return random.normal(sk, (self.n,)), key

    def gennull(self, key, q):
        n = self.n        
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        tmp, key = grand(key, (n, 2))
        TxS = self.Adot(tmp[:n, 0], y)

        # d1, d2 = calc_derivs(ufunc(sig), 2)

        vv = -ha[2]/ha[1]**2*TxS*x + tmp[:n, 0]

        return vcat(tmp[:, 0],
                    tmp[:, 1] - self.Adot(tmp[:, 1], vv)/self.Adot(vv, vv)*vv), key

    def projSphere(self, qs, Omg):
        n, iA = self.n, self.iA
        x = qs[:n]
        y = qs[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        # sig = self.Adot(x, y)
        # d1, d2 = calc_derivs(ufunc(sig), 2)

        Omgx = Omg[:n]
        Omgy = Omg[n:]

        return Omg - vcat(
            (ha[1]*iA@y + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*x)*
            jnp.sum(x*Omgx)/(ha[1]*jnp.sum(x*(iA@y)) + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])),
            (ha[1]*iA@x + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*y)*
            jnp.sum(y*Omgy)/(ha[1]*jnp.sum(y*(iA@x)) + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])))

    def DprojSphere(self, qs, Xi, Eta):
        """ works for pair of tangent vectors only.
        """
        n, iA = self.n, self.iA
        x = qs[:n]
        y = qs[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        Omgx = Eta[:n]
        Omgy = Eta[n:]
        return -vcat((ha[1]*iA@y + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*x)*
                      jnp.sum(Xi[:n]*Omgx)/(ha[1]*jnp.sum(x*(iA@y)) + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])),
                      (ha[1]*iA@x + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*y)*
                      jnp.sum(Xi[n:]*Omgy)/(ha[1]*jnp.sum(y*(iA@x)) + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])))

    def pSimpleSphere(self, qs, Omg):
        n = self.n
        x = qs[:n]
        y = qs[n:]

        return Omg - vcat(x*jnp.sum(x*Omg[:n]),
                          y*jnp.sum(y*Omg[n:]))

    def gennull_sphere(self, key, q):
        n = self.n

        tmp, key = grand(key, (n, 2))
        x = q[:n]
        y = q[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        tmp0 = tmp[:, 0]
        tmp1 = tmp[:, 1]

        tmp0 = tmp0 - x*jnp.sum(x*tmp0)

        TxS = self.Adot(tmp0, y)

        mat = jnp.concatenate([-ha[2]/ha[1]**2*TxS*x + tmp0[None, :], y[None, :]], axis=0)

        return vcat(tmp0, tmp1 - mat.T@jla.solve(mat@mat.T, mat@tmp1)), key

    def TwoSphere(self, qs, Xi1, Xi2):
        GA = self.Gamma(qs, Xi1, Xi2)
        return self.DprojSphere(qs, Xi1, Xi2) + GA - self.projSphere(qs, GA)

    def GammaSphere(self, qs, Xi1, Xi2):
        GA = self.Gamma(qs, Xi1, Xi2)
        return - self.DprojSphere(qs, Xi1, Xi2) + self.projSphere(qs, GA)

    def TwoSphereContrAI(self, qs, Xi):
        """ this works if A = I, for A != I
        we need to extend this formula a bit
        """
        n = self.n
        x = qs[:n]
        y = qs[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        return 0.5*((ha[3]*ha[1]-ha[2]**2)*self.Adot(y, Xi[:n])**2 \
                    + ha[1]**2*(ha[1]**2-ha[0]*ha[2])*jnp.sum(Xi[:n]*Xi[:n])) * \
            ((ha[3]*ha[1]-ha[2]**2)*self.Adot(x, Xi[n:])**2 \
             + ha[1]**2*(ha[1]**2-ha[0]*ha[2])*jnp.sum(Xi[n:]*Xi[n:])) /\
            (- ha[2]-ha[0]*ha[1]**2+ha[0]**2*ha[2])/(ha[1]**2-ha[0]*ha[2])/ha[1]**5

    def crossCurvSphere_alt(self, qs, Xi):
        c1 = self.crossCurv(qs, Xi) + self.TwoSphereContrAI(qs, Xi)
        return c1

    def crossCurvSphere(self, qs, Xi):
        n = self.n
        x = qs[:n]
        y = qs[n:]
        TxS = self.Adot(Xi[:n], y)
        TyS = self.Adot(Xi[n:], x)
        TxyS = self.Adot(Xi[:n], Xi[n:])
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 4)

        II = -(- ha[2]*TxS*TyS + ha[1]**2*TxyS)/ha[1]**3
        S2 = TxS*TyS

        return 0.5*((ha[1]**2 - ha[0]*ha[2])*ha[4] + ha[0]*ha[3]**2 - 2*ha[1]*ha[2]*ha[3] + ha[2]**3)/(ha[1]**2-ha[0]*ha[2])/ha[1]**5*S2**2 \
            + 0.5*((ha[3]*ha[1]-ha[2]**2)*TxS**2 \
                   + ha[1]**2*(ha[1]**2-ha[2]*ha[0])*jnp.sum(Xi[:n]*Xi[:n])) * \
            ((ha[3]*ha[1]-ha[2]**2)*TyS**2 \
             + ha[1]**2*(ha[1]**2-ha[2]*ha[0])*jnp.sum(Xi[n:]*Xi[n:])) /\
            (- ha[2]-ha[0]*ha[1]**2+ha[2]*ha[0]**2)/(ha[1]**2-ha[2]*ha[0])/ha[1]**5 \
            \
            - 2*(ha[1]*ha[3] - ha[2]**2)/ha[1]**4*S2*II \
            + ha[2]/ha[1]*II*II

    def clog(self, x, y):
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 1)

        return -1/ha[1]*y

    def tsolve(self, x, nug):
        """t solves y = t*nug
           t = -h'(u)
           - h'(u)*x.nug = h(u)
        """
        raise NotImplementedError

    def cexp(self, x, nu):
        # nu in position of $y$, return one in barM
        return self.tsolve(x, nu)*nu

    def PiComp(self, qe, Omg, ptl):
        n = self.n

        return 0.5*vcat(
            Omg[:n] + ptl.iDOptMap(qe[:n], Omg[n:], self),
            ptl.DOptMap(qe[:n], Omg[:n], self) + Omg[n:])

    def DPiComp(self, qe, Xie, Omg, ptl):
        n = self.n

        # u = self.ufunc(self.Adot(x, y))
        # ha = self.dsfunc(u)

        return 0.5*vcat(
            - ptl.iDOptMap(qe[:n],
                           ptl.HessOptMap(
                               qe[:n], Xie[:n],
                               ptl.iDOptMap(qe[:n], Omg[n:], self), self), self),
            ptl.HessOptMap(qe[:n], Xie[:n], Omg[:n], self))

    def GammaAmbTan(self, qe, Xie, Etae):
        """ only works for tangent vectors.
        """
        n = self.n
        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)
        # Sx = self.Adot(Xie[:n], y)
        # Sy = self.Adot(Xie[n:], x)

        GAx = (ha[2]**2 - ha[3]*ha[1])/(ha[1]**2-ha[0]*ha[2])/ha[1]**2*self.Adot(y, Xie[:n])*self.Adot(Etae[:n], y)*x \
            - ha[2]/ha[1]**2*Xie[:n]*self.Adot(Etae[:n], y) \
            - ha[2]/ha[1]**2*Etae[:n]*self.Adot(Xie[:n], y)
        
        GAy = (ha[2]**2 - ha[3]*ha[1])/(ha[1]**2-ha[0]*ha[2])/ha[1]**2*self.Adot(x, Xie[n:])*self.Adot(Etae[n:], x)*y \
            - ha[2]/ha[1]**2*Xie[n:]*self.Adot(Etae[n:], x) \
            - ha[2]/ha[1]**2*Etae[n:]*self.Adot(Xie[n:], x)

        return vcat(GAx, GAy)    
    
    def GammaLeviCivita(self, qe, Xie, Etae, ptl):
        return - self.DPiComp(qe, Xie, Etae, ptl) \
            + self.PiComp(qe, self.GammaAmbTan(qe, Xie, Etae), ptl)

    def left_geodesic(self, qe, Xie, t, ptl):
        # x(t) solves
        # bD c(x(t), y) = bD c(x(0), y) + t D_(Xiex)\bD c(x0, y0)
        # Set right hand side = ppy =  bD c(x(0), y) + t D_(Xiex)\bD c(x0, y0)
        #         = 1/h1*x + t(-h2/h1**3*Adot(Xiex, y)*x + 1/h1*Xiex)
        # left hand side is 1/h1x(t), so 
        # x(t) = h1(u)*ppy
        # h0(u) = Adot(x(t), y) = h1(u)*ppy. y
        # thus, sove for u from the above
        # h'(u) ppy. y = h(u)        
        # set set x = h1(u)*ppy
        n = self.n

        x = qe[:n]
        y = qe[n:]
        Xiex = Xie[:n]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        unew, dunew, ppy = self.solveux(qe, Xie, t, ptl)

        hanew = self.dsfunc(unew, 3)

        xs = hanew[1]*ppy
        dxs = hanew[2]*dunew*ppy \
            + hanew[1]*(-ha[2]/ha[1]**3*self.Adot(Xiex, y)*x + 1/ha[1]*Xiex)
        return vcat(xs, ptl.OptMap(xs, self)), vcat(dxs, ptl.DOptMap(xs, dxs, self))

    def right_geodesic(self, qe, Xie, s, ptl):
        n = self.n

        x = qe[:n]
        y = qe[n:]

        Xiey = Xie[n:]

        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)
        unew, dunew, ppx = self.solveuy(qe, Xie, s, ptl)
        hanew = self.dsfunc(unew, 3)

        ys = hanew[1]*ppx
        dys = hanew[2]*dunew*ppx \
            + hanew[1]*(-ha[2]/ha[1]**3*self.Adot(Xiey, x)*y + 1/ha[1]*Xiey)
        xs = ptl.iOptMap(ys, self)
        return vcat(xs, ys), vcat(ptl.iDOptMap(xs, dys, self), dys)

    def Gamma0(self, qe, Xie, Etae, ptl):
        # Geodesic: given a tangent vector $Xie = (Xiex, Xiey)
        # Then (Xiex, 0)$ then $(0, pXie)$, using the induced pairing.
        # KM geodesic (with potential) at (x, y) with initial velocity (Xiex, 0)
        # So for a vector field S(q) we can do S(x(t)) and compute the derivative
        # what we need: cexp.
        n = self.n

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        Xiex = Xie[:n]
        Etaex = Etae[:n]

        Gammax = - ha[2]/ha[1]**2*self.Adot(Etaex, y)*Xiex \
            - ha[2]/ha[1]**2*self.Adot(Xiex, y)*Etaex \
            + (ha[2]**2-ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*self.Adot(Xiex, y)*self.Adot(Etaex, y)*x

        Gammay = -ptl.HessOptMap(x, Xiex, Etaex, self) \
            + ptl.DOptMap(x, Gammax, self)
        return vcat(Gammax, Gammay)

    def Gamma1(self, qe, Xie, Etae, ptl):
        # Geodesic: given a tangent vector $Xie = (Xiex, Xiey)
        # Then (Xiex, 0)$ then $(0, pXie)$, using the induced pairing.
        # KM geodesic (with potential) at (x, y) with initial velocity (Xiex, 0)
        # So for a vector field S(q) we can do S(x(t)) and compute the derivative
        # what we need: cexp.
        n = self.n

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        Xiey = Xie[n:]
        Etaey = Etae[n:]

        Gammay = - ha[2]/ha[1]**2*self.Adot(Etaey, x)*Xiey \
            - ha[2]/ha[1]**2*self.Adot(Xiey, x)*Etaey \
            + (ha[2]**2-ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*self.Adot(Xiey, x)*self.Adot(Etaey, x)*y

        Gammax = ptl.iDOptMap(
            x,
            Gammay + ptl.HessOptMap(x, Xie[:n], Etae[:n], self), self)
        return vcat(Gammax, Gammay)

    def Pi0(self, qe, Omg, ptl):
        n = self.n

        x = qe[:n]
        y = ptl.OptMap(x, self)
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        # return vcat(Omg[:n], DOMap(qe[:n], Omg[:n]))
        rety = ha[1]*ptl.hessphi(x, Omg[:n]) \
            + ha[2]/(ha[1]**2-ha[0]*ha[2])*self.Adot(Omg[:n], y)*y \
            + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*self.Adot(x, ptl.hessphi(x, Omg[:n]))*y
        return vcat(Omg[:n], rety)

    def DPi0(self, qe, Xie, Omg, ptl):
        n = self.n
        x = qe[:n]
        y = ptl.OptMap(x, self)
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 3)

        Sx = self.Adot(Xie[:n], y)
        Sy = self.Adot(Xie[n:], x)

        d1 = 1/ha[1]
        d2 = - ha[2]/ha[1]**3
        d3 = - (ha[3]*ha[1]-3*ha[2]**2)/ha[1]**5

        rety = ha[2]/ha[1]*(Sx+Sy)*ptl.hessphi(x, Omg[:n]) + ha[1]*ptl.D3phi(x, Xie[:n], Omg[:n]) \
            - (d3*(d1+d2*ha[0])-d2*(d2+d3*ha[0]+d2))/(d1+d2*ha[0])**2*(Sx+Sy)*self.Adot(Omg[:n], y)*y \
            - d2/(d1+d2*ha[0])*(self.Adot(Xie[n:], Omg[:n])*y + self.Adot(Omg[:n], y)*Xie[n:]) \
            - (d3*(d1**2+d1*d2*ha[0]) - (2*d1*d2+d2*d2*ha[0]+d1*d3*ha[0]+d1*d2)*d2)/(d1**2+d1*d2*ha[0])**2*(Sx+Sy)*self.Adot(x, ptl.hessphi(x, Omg[:n]))*y \
            + ha[1]*ha[2]/(ha[1]**2-ha[0]*ha[2])*(
                self.Adot(Xie[:n], ptl.hessphi(x, Omg[:n]))*y
                + self.Adot(x, ptl.D3phi(x, Xie[:n], Omg[:n]))*y
                + self.Adot(x, ptl.hessphi(x, Omg[:n]))*Xie[n:]
            )
        return vcat(jnp.zeros(n), rety)

    def Gamma0a(self, qe, Xie, Etae, ptl):
        return - self.DPi0(qe, Xie, Etae, ptl) + self.Pi0(qe, self.GammaAmbTan(qe, Xie, Etae), ptl)

    def CurvLeviCivita(self, qe, Xie, Etae, Phie, plt):
        D1 = jvp(lambda q: self.GammaLeviCivita(q, Etae, Phie, plt), (qe,), (Xie,))[1]
        D2 = jvp(lambda q: self.GammaLeviCivita(q, Xie, Phie, plt), (qe,), (Etae,))[1]
        G1 = self.GammaLeviCivita(qe, Xie, self.GammaLeviCivita(qe, Etae, Phie, plt), plt)
        G2 = self.GammaLeviCivita(qe, Etae, self.GammaLeviCivita(qe, Xie, Phie, plt), plt)
        return D1 - D2 + G1 - G2

    def CurvAmb(self, qe, Xie, Etae, Phie):
        D1 = jvp(lambda q: self.GammaAmbTan(q, Etae, Phie), (qe,), (Xie,))[1]
        D2 = jvp(lambda q: self.GammaAmbTan(q, Xie, Phie), (qe,), (Etae,))[1]
        G1 = self.GammaAmbTan(qe, Xie, self.GammaAmbTan(qe, Etae, Phie))
        G2 = self.GammaAmbTan(qe, Etae, self.GammaAmbTan(qe, Xie, Phie))
        return D1 - D2 + G1 - G2

    def Two(self, qe, Xie, Etae, ptl):
        GA = self.GammaAmbTan(qe, Xie, Etae)
        return self.DPiComp(qe, Xie, Etae, ptl) + GA - self.PiComp(qe, GA, ptl)

    def KMMetricInduced(self, qe, Xie, Etae, ptl):
        # formula is
        #  Hesscx(x, y, xi, eta) - Hessphi(x, xi, eta)
        # where $y$ solves Dc(x, y) = Dphi(x)
        # not too hard to find curvature.
        # need to make phi c-convex
        n = self.n

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        return ha[2]/ha[1]**3*self.Adot(x, Etae[n:])*self.Adot(Xie[:n], y) \
            - 1/ha[1]*self.Adot(Xie[:n], Etae[n:])


class LambertSimpleMTW(baseSimpleMTW):
    """ Always need branch. Two branches
    -1 for u from -infty to u_c
    1 for u from u_c to infty.
    """
    def __init__(self, n, A, params, branch):
        super().__init__(n, A, params)
        self.branch = branch
        a0, a1, a2 = self.p

        self.uc = -(a0*a2+a1)/(a1*a2)
        self.suc = -(a1/a2)*jnp.exp(-a2*self.uc)
        self.rng = self.set_range()

    def set_range(self):
        # return s(u_left), s(u_right), one end is a critical point and
        # branch is between u_left < u_right.
        # return s(-\infty), s(uc) if -1 and s(uc), s(infty) if 1

        a0, a1 , a2 = self.p

        if self.branch < 0:
            if a2 < 0:
                return jnp.array((-a1*jnp.inf, self.suc))
            else:
                return jnp.array((0, self.suc))
        else:
            if a2 < 0:
                return jnp.array((self.suc, 0))
            else:
                return jnp.array((self.suc, a1*jnp.inf))

    def is_in_range(self, s):
        if (s > jnp.min(self.rng)) and (s < jnp.max(self.rng)):
            return True
        return False

    def sfunc(self, u):
        a0, a1, a2 = self.p
        return (a0+a1*u)*jnp.exp(a2*u)

    def dsfunc(self, u, order):
        """order = 1, 2, 3, 4.
        return array of size order + 1
        entry 0 is sfunc, 1 is sfunc' etc
        """
        a0, a1, a2 = self.p
        ex = jnp.exp(a2*u)
        ret = [(a0 + a1*u)*ex]
        a, b = a0, a1
        for i in range(1, order+1):
            ret.append(ret[-1]*a2 + b*ex)
            a = a*a2 + b
            b = b*a2        
        return ret

    def ufunc(self, s):
        a0, a1, a2 = self.p
        if jnp.isinf(jnp.max(jnp.abs(self.rng))):
            return 1/a2*LambertW(a2*s*jnp.exp(a0*a2/a1)/a1, 0)[0] - a0/a1
        else:
            return 1/a2*LambertW(a2*s*jnp.exp(a0*a2/a1)/a1, 1)[0] - a0/a1

    def tsolve(self, x, nu):
        # t = -h1(u), solve h0(u) = - h1(u)x.T@ nu
        a0, a1, a2 = self.p
        xnu = self.Adot(x, nu)
        u = - ((a1+a0*a2)*xnu + a0)/(a1 + a1*a2*xnu)

        return -self.dsfunc(u, 1)[1]

    def solveux(self, qe, Xie, s, ptl):
        # bD c(x(s), y) = ppy
        # where ppy = bDc(x, y) + s D_Xie[:n] bDc(x, y)
        # (bD c(x(s), y), y) = Adot(ppy, y)
        # h0(unew) = Adot(ppy, y)*h1(unew)
        # set mtpyh1 = Adot(ppy, y)
        # in this case, 
        # (a0 + a1 unew) exp(a2 unew) = mtpyh1((a0a2+a1 + a1a2 unew) exp(a2 unew))
        # (a0 - mtpyh1(a0a2+a1))/(mtpyh1 a1a2 - a1) = unew
        # solve for unew from the last, then for x

        n = self.n
        a0, a1, a2, p3 = self.p
        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        Xiex = Xie[:n]

        ppy = 1/ha[1]*x + s*(-ha[2]/ha[1]**3*self.Adot(Xiex, y)*x + 1/ha[1]*Xiex)
        mtpyh1 = self.Adot(ppy, y)
        dmtpyh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiex, y) + 1/ha[1]*self.Adot(Xiex, y)

        unew = (a0 - mtpyh1*(a0*a2+a1))/(mtpyh1*a1*a2 - a1)
        dunew = dmtpyh1*((a0*a2+a1)*a1 - a0*a1*a2)/(mtpyh1*a1*a2 - a1)**2
        return unew, dunew, ppy

    def solveuy(self, qe, Xie, s, ptl):
        n = self.n
        a0, a1, a2, p3 = self.p

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)
        Xiey = Xie[n:]
        ppx = 1/ha[1]*y + s*(-ha[2]/ha[1]**3*self.Adot(Xiey, x)*y + 1/ha[1]*Xiey)
        mtpxh1 = jnp.dot(ppx, x)
        dmtpxh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiey, x) + 1/ha[1]*self.Adot(Xiey, x)

        unew = (a0 - mtpxh1*(a0*a2+a1))/(mtpxh1*a1*a2 - a1)
        dunew = dmtpxh1*((a0*a2+a1)*a1 - a0*a1*a2)/(mtpxh1*a1*a2 - a1)**2

        return unew, dunew, ppx


class TrigSimpleMTW(baseSimpleMTW):
    def __init__(self, n, A, params, branch=None):
        # branch is an integer
        super().__init__(n, A, params)
        self.branch = branch
        self.set_range()

    def set_range(self):
        b0, b1, b2, b3 = self.p
        k = self.branch
        self.uc0 = 1/b2*(jnp.arctan(-b2/b1)-b3)
        self.u_rng = jnp.sort(jnp.array((self.uc0 + k*jnp.pi/b2, self.uc0 + (k+1)*jnp.pi/b2)))
        self.rng = jnp.array((self.sfunc(self.u_rng[0]), self.sfunc(self.u_rng[1])))

    def is_in_range(self, s):
        if (s > jnp.min(self.rng)) and (s < jnp.max(self.rng)):
            return True
        return False
        
    def sfunc(self, u):
        b0, b1, b2, b3 = self.p
        return b0*jnp.exp(b1*u)*jnp.sin(b2*u+b3)

    def _ufunc_notused(self, s):
        """ do bisect
        """
        b0, b1 , b2, b3 = self.p

        # umin = uc + self.branch*jnp.pi
        # umax = umin + jnp.pi/p2
        umin = self.u_rng[0]
        umax = self.u_rng[1]
        u0 = (umin + umax)/2
        tol = 1e-7

        if not self.is_in_range(s):
            return None

        for i in range(40):
            scl = 1.
            ha = self.dsfunc(u0, 1)
            val, grd = ha[0], ha[1]
            # if grd is small, do bisect
            #  if grd is large, do newton
            if s > val:
                if grd > 0:
                    newu = 0.5*(umax + u0)
                else:
                    newu = 0.5*(umin + u0)                    
                newval = self.sfunc(newu)                
            elif s < val:
                if grd > 0:
                    newu = 0.5*(umin + u0)
                else:
                    newu = 0.5*(umax + u0)
                newval = self.sfunc(newu)                
            u0 = newu
            val = newval
            if jnp.abs(val - s) <= tol:
                # print("FOUND", i)
                return u0
        print("NOTFOUND", i)
        return u0
    
    def ufunc(self, s):
        b0, b1 , b2, b3 = self.p

        # umin = uc + self.branch*jnp.pi
        # umax = umin + jnp.pi/p2
        umin = self.u_rng[0]
        umax = self.u_rng[1]
        u0 = (umin + umax)/2
        tol = 1e-7

        for i in range(40):
            scl = 1.
            ha = self.dsfunc(u0, 1)
            val, grd = ha[0], ha[1]
            # if grd is small, do bisect
            #  if grd is large, do newton
            newu = u0-scl*(val - s)/grd
            if newu < umin:
                newu = 0.5*(umin + u0)
                newval = self.sfunc(newu)                
            elif newu > umax:
                newu = 0.5*(umax + u0)
                newval = self.sfunc(newu)                
            else:
                newval, grd = self.dsfunc(newu, 1)
                while jnp.abs(newval-s) > jnp.abs(val-s):
                    scl = scl*.8
                    newu = u0-scl*(val - s)/grd
                    newval = self.sfunc(newu)
                # print(newval, scl)
            u0 = newu
            val = newval
            if jnp.abs(val - s) <= tol:
                # print("FOUND", i)
                return u0
        print("NOTFOUND", i)
        return u0
    
    def _bad_ufunc(self, s):
        p0, p1 , p2, p3 = self.p        
        # u0 = (-p0/p2)**(p1/(p1-p3))
        umin = -0.1
        umax = umin + jnp.pi/p2
        u0 = (umin + umax)/2
        tol = 1e-10

        for i in range(20):
            scl = 1.
            ha = self.dsfunc(u0, 1)
            val, grd = ha[0], ha[1]
            newu = u0-scl*(val - s)/grd
            if newu < umin:
                newu = umin
            if newu > umax:
                newu = umax
            newval, grd = self.dsfunc(newu, 1)
            while jnp.abs(newval-s) > jnp.abs(val-s):
                scl = scl*.8
                newu = u0-scl*(val - s)/grd
                newval = self.sfunc(newu)
            # print(newval, scl)
            u0 = newu
            val = newval
            if jnp.abs(val - s) <= tol:
                # print("FOUND", i)
                return u0
        print("NOTFOUND", i)
        return u0
    
    def dsfunc(self, u, order):
        """order = 1, 2, 3, 4.
        return array of size order + 1
        entry 0 is sfunc, 1 is sfunc' etc
        """
        b0, b1, b2, b3 = self.p
        ex1 = jnp.exp(b1*u)
        esin2 = ex1*jnp.sin(b2*u+b3)
        ecos2 = ex1*jnp.cos(b2*u+b3)        
        ret = [b0*esin2]
        vec = jnp.array([b0, 0])

        mat = jnp.array([[b1, -b2], [b2, b1]])
        # tlist = [sin2, cos2, - sin2, -cos2]
        
        for i in range(1, order+1):
            # ret.append(b1*ret[-1] + b0*ex1*b2**i*tlist[i % 4])
            vec = mat@vec
            ret.append(vec[0]*esin2 + vec[1]*ecos2)
        return ret

    def tsolve(self, x, nu):
        """ t is defined as y = tp, or t = -h'(u)
               h(u) = x.Ty = x(-h'(u)p) = - h'(u)x.p
        u us unknown. Then y = t nu 
        equation is 
        - h'(u) x.nu = h(u)
        """
        b0, b1 , b2, b3 = self.p

        xnu = jnp.sum(x*nu)
        
        u = 1/b2*(jnp.arctan(-xnu*b2/(1 + b1*xnu)) - b3)
        krng = jnp.sort(jnp.array([(self.u_rng[0] - u)*b2/jnp.pi,
                                   (self.u_rng[1] - u)*b2/jnp.pi]))
        u += jnp.floor(krng[1])*jnp.pi/b2
        return - self.dsfunc(u, 1)[1]

    def solveux(self, qe, Xie, s, ptl):
        # bD c(x(s), y) = ppy
        # where ppy = bDc(x, y) + s D_Xie[:n] bDc(x, y)
        # (bD c(x(s), y), y) = Adot(ppy, y)
        # h0(unew) = Adot(ppy, y)*h1(unew)
        # set mtpyh1 = Adot(ppy, y)
        # in this case, 
        # unew  = 1/p2*arctan(mtpyh1*p0*p2/(p0  - mtpyh1*p0*p1)) - p3
        # solve for unew from the last, then for x

        n = self.n
        p0, p1, p2, p3 = self.p
        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        Xiex = Xie[:n]

        ppy = 1/ha[1]*x + s*(-ha[2]/ha[1]**3*self.Adot(Xiex, y)*x + 1/ha[1]*Xiex)
        mtpyh1 = self.Adot(ppy, y)
        dmtpyh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiex, y) + 1/ha[1]*self.Adot(Xiex, y)

        znew = mtpyh1*p2/(1. - mtpyh1*p1)
        dznew = dmtpyh1*(p2 + p1*p2)/(1. - mtpyh1*p1)**2
        unew = 1/p2*(jnp.arctan(znew) - p3)
        dunew = 1/p2*dznew/(1+znew**2)
        return unew, dunew, ppy

    def solveuy(self, qe, Xie, s, ptl):
        n = self.n
        p0, p1, p2, p3 = self.p

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)
        Xiey = Xie[n:]

        ppx = 1/ha[1]*y + s*(-ha[2]/ha[1]**3*self.Adot(Xiey, x)*y + 1/ha[1]*Xiey)
        mtpxh1 = jnp.dot(ppx, x)
        dmtpxh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiey, x) + 1/ha[1]*self.Adot(Xiey, x)

        znew = mtpxh1*p2/(1. - mtpxh1*p1)
        dznew = dmtpxh1*(p2 + p1*p2)/(1. - mtpxh1*p1)**2
        unew = 1/p2*(jnp.arctan(znew) - p3)
        dunew = 1/p2*dznew/(1+znew**2)
        return unew, dunew, ppx


class GenHyperbolicSimpleMTW(baseSimpleMTW):
    def __init__(self, n, A, params, branch=None):
        super().__init__(n, A, params)
        self.branch = branch
        p0, p1, p2, p3 = self.p
        if p1 >= p3:
            raise ValueError("p1=%f >= p3=%f. Need p3 > p1" % (p1, p3))
        if jnp.prod(self.p) >= 0:
            self.uc = None
            self.suc = None
            self.has_critical = False
            self.branch = None
        else:
            lg = jnp.log(-p2*p3/(p0*p1))
            self.uc = 1/(p1 - p3)*lg
            self.suc = p2*(p1-p3)/p1*jnp.exp(lg*p3/(p1-p3))
            self.has_critical = True
            if branch is None:
                raise ValueError("Has critical point, branch must be -1 or 1")
        # if has critical, then branch -1 is negative, branch 1 is positive part.
        self.rng = self.set_range()

    def set_range(self):
        # return s(-infty, s(infty) if no critical point
        # return s(u_left), s(u_right) if there is a critical point and
        # branch is between u_left < u_right.
        p0, p1, p2, p3 = self.p

        if self.branch is None:
            # in this case p0*p1*p2*p3 >= 0
            if p3 == 0:
                return jnp.array((p0*jnp.inf, p2))
            elif p1 == 0:
                return jnp.array((p0, p2*jnp.inf))
            elif (p1*p3 > 0):
                # then p0 p2 > 0
                if (p1 > 0):
                    return jnp.array((0., p0*jnp.inf))
                else:
                    return jnp.array((p0*jnp.inf, 0))
            else:
                # p1p3 < 0, thus p1 < 0 and p3 > 0
                return jnp.array((p0*jnp.inf, p2*jnp.inf))
        elif self.branch < 0:
            # in this case p0*p1*p2*p3 < 0
            # return s(-\infty), s(uc) if -1 and s(uc), s(infty) if 1
            if p1 < 0:     
                # infty on two sides.
                return jnp.array((p0*jnp.inf, self.suc))
            else:
                # both p3 and p1 > 0
                return jnp.array((0, self.suc))
        else:
            if p3 > 0:
                # infty on two sides.
                return jnp.array((self.suc, p2*jnp.inf))
            else:
                # then both p3 and p1 < 0
                return jnp.array((self.suc, 0))

    def is_in_range(self, s):
        if (s > jnp.min(self.rng)) and (s < jnp.max(self.rng)):
            return True
        return False

    def sfunc(self, u):
        """ s = p0 exp(p1u) + p2 exp3(u)
        condition p0p2 !=0 and p1 != p3
        Four scenarios dividing to several groups.
        """
        return self.p[0]*jnp.exp(self.p[1]*u) + self.p[2]*jnp.exp(self.p[3]*u)

    def ufunc(self, s):
        """ truncated Newton with line search
        two cases, if there is a critical point then
        either left (branch = -1) and right (branch = 1)
        """
        p0, p1, p2, p3 = self.p
        if not self.is_in_range(s):
            raise ValueError("s is out of range")

        # u0 = p1/(p1-p3)*jnp.log(jnp.abs((p0*p1)/(p2*p3))
        # umin, umax = jnp.sort(self.rng)
        
        if self.branch is None:
            u0 = 0
        elif self.branch < 0:
            u0 = self.uc - 1
        elif self.branch > 0:
            u0 = self.uc + 1

        tol = 1e-7
        val, grd = self.dsfunc(u0, 1)
        for i in range(20):
            scl = 1.
            newu = u0-scl*(val - s)/grd
            newval, newgrd = self.dsfunc(newu, 1)
            while jnp.isnan(newval) or (jnp.abs(newval-s) > jnp.abs(val-s)):
                scl = scl*.8
                newu = u0-scl*(val - s)/grd
                newval, newgrd = self.dsfunc(newu, 1)
            u0 = newu
            val = newval
            grd = newgrd
            if jnp.abs(val - s) <= tol:
                # print("FOUND", i)
                return u0
        print("NOTFOUND", i, jnp.abs(val - s))
        return u0
    

    def _ufunc_bad(self, s):
        """ truncated Newton with line search
        two cases, if there is a critical point then
        either left (branch = -1) and right (branch = 1)
        """
        p0, p1, p2, p3 = self.p
        if not self.is_in_range(s):
            raise ValueError("s is out of range")

        # u0 = p1/(p1-p3)*jnp.log(jnp.abs((p0*p1)/(p2*p3))
        if self.branch is None:
            u0 = 0
        elif self.branch < 0:
            u0 = self.uc - 1
        elif self.branch > 0:
            u0 = self.uc + 1

        tol = 1e-10
        val, grd = self.dsfunc(u0, 1)
        for i in range(20):
            scl = 1.
            newu = u0-scl*(val - s)/grd
            newval, newgrd = self.dsfunc(newu, 1)
            while jnp.abs(newval-s) > jnp.abs(val-s):
                scl = scl*.8
                newu = u0-scl*(val - s)/grd
                newval, newgrd = self.dsfunc(newu, 1)
            u0 = newu
            val = newval
            grd = newgrd
            if jnp.abs(val - s) <= tol:
                # print("FOUND", i)
                return u0
        print("NOTFOUND", i)
        return u0

    def dsfunc(self, u, order):
        """order = 1, 2, 3, 4.
        return array of size order + 1
        entry 0 is sfunc, 1 is sfunc' etc
        """
        p0, p1, p2, p3 = self.p
        ex1 = jnp.exp(p1*u)
        ex3 = jnp.exp(p3*u)
        ret = [p0*ex1 + p2*ex3]

        for i in range(1, order+1):
            ret.append(p0*p1**i*ex1 + p2*p3**i*ex3)
        return ret

    def tsolve(self, x, nug):
        """ t solves 
               t = -h'(u)
               - h'(u). x.nug = h(u)
        u us unknown. Then y = t (nu + gradc1(x))
        equation is
        h'(u) x.(nu + gradc1(x)) = h(u)
        """
        p0, p1 , p2, p3 = self.p

        xnug = jnp.sum(x*nug)
        u = 1/(p3-p1)*jnp.log((-p0*p1*xnug - p0)/(p2+p2*p3*xnug))
        return - self.dsfunc(u, 1)[1]

    def solveux(self, qe, Xie, s, ptl):
        # bD c(x(s), y) = ppy
        # where ppy = bDc(x, y) + s D_Xie[:n] bDc(x, y)
        # (bD c(x(s), y), y) = Adot(ppy, y)
        # h0(unew) = Adot(ppy, y)*h1(unew)
        # set mtpyh1 = Adot(ppy, y)
        # in this case, 
        # p0 exp(p1 unew) + p2 exp(p3 unew) = mtpyh1(p0p1 exp(p1 unew) + p2p3 exp(p3 unew))
        # solve for unew from the last, then for x

        n = self.n
        p0, p1, p2, p3 = self.p
        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)

        Xiex = Xie[:n]

        ppy = 1/ha[1]*x + s*(-ha[2]/ha[1]**3*self.Adot(Xiex, y)*x + 1/ha[1]*Xiex)
        mtpyh1 = self.Adot(ppy, y)
        dmtpyh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiex, y) + 1/ha[1]*self.Adot(Xiex, y)

        unew = 1/(p3-p1)*jnp.log((p0-mtpyh1*p0*p1)/(mtpyh1*p2*p3 - p2))
        dunew = 1/(p3-p1)*(
            - dmtpyh1*p0*p1/(p0-mtpyh1*p0*p1)
            - dmtpyh1*p2*p3/(mtpyh1*p2*p3 - p2))
        return unew, dunew, ppy

    def solveuy(self, qe, Xie, s, ptl):
        n = self.n
        p0, p1, p2, p3 = self.p

        x = qe[:n]
        y = qe[n:]
        u = self.ufunc(self.Adot(x, y))
        ha = self.dsfunc(u, 2)
        Xiey = Xie[n:]

        ppx = 1/ha[1]*y + s*(-ha[2]/ha[1]**3*self.Adot(Xiey, x)*y + 1/ha[1]*Xiey)
        mtpxh1 = jnp.dot(ppx, x)
        dmtpxh1 = -ha[0]*ha[2]/ha[1]**3*self.Adot(Xiey, x) + 1/ha[1]*self.Adot(Xiey, x)

        unew = 1/(p3-p1)*jnp.log((p0-mtpxh1*p0*p1)/(mtpxh1*p2*p3 - p2))
        dunew = 1/(p3-p1)*(
            - dmtpxh1*p0*p1/(p0-mtpxh1*p0*p1)
            - dmtpxh1*p2*p3/(mtpxh1*p2*p3 - p2))
        return unew, dunew, ppx


class basePotential(object):
    """ base class for a convex potential
    """
    def fphi(self, x):
        raise NotImplementedError

    def gradphi(self, x):
        raise NotImplementedError

    def hessphi(self, x, omg):
        raise NotImplementedError

    def ihessphi(x, bomg):
        raise NotImplementedError

    def D3phi(self, x, omg1, omg2):
        raise NotImplementedError

    def Divc(self, x, xp, mtw):
        """ Divergence.
        ptl: a potential object
        """
        gp = self.gradphi(xp)
        yp = mtw.tsolve(xp, gp)*gp
        return mtw.ufunc(self.Adot(x, yp)) - mtw.ufunc(mtw.Adot(xp, yp)) \
            + self.fphi(x) - self.fphi(xp)

    def OptMap(self, x, mtw):
        """ solve Dc(x, y) = -gradphi(x)
        """
        return mtw.cexp(x, self.gradphi(x))

    def iOptMap(self, y, mtw):
        raise NotImplementedError

    def DOptMap(self, x, omg, mtw):
        # return jvp(lambda x: OMap(x), (x,), (omg,))[1]
        y = self.OptMap(x, mtw)
        # sig = mtw.Adot(x, y)
        u = mtw.ufunc(mtw.Adot(x, y))
        ha = mtw.dsfunc(u, 2)
        return -ha[1]*self.hessphi(x, omg) \
            - ha[2]/(ha[1]**2-ha[2]*ha[0])*ha[1]*mtw.Adot(x, self.hessphi(x, omg))*y \
            + ha[2]/(ha[1]**2-ha[2]*ha[0])*mtw.Adot(omg, y)*y

    def iDOptMap(self, x, B, mtw):
        y = self.OptMap(x, mtw)
        u = mtw.ufunc(mtw.Adot(x, y))
        ha = mtw.dsfunc(u, 2)

        return - 1/ha[1]*self.ihessphi(x, B) \
            + ha[2]/ha[1]*(ha[1]*mtw.Adot(x, B) - mtw.Adot(self.ihessphi(x, B), y)) /\
            (ha[1]**3 - ha[2]*mtw.Adot(y, self.ihessphi(x, y)))*self.ihessphi(x, y)

    def HessOptMap(self, x, xi1, xi2, mtw):
        # return jvp(lambda x: OMap(x), (x,), (omg,))[1]
        y = self.OptMap(x, mtw)
        u = mtw.ufunc(mtw.Adot(x, y))
        ha = mtw.dsfunc(u, 3)

        Duomg = (mtw.Adot(xi1, y) + mtw.Adot(x, self.DOptMap(x, xi1, mtw)))/ha[1]
        return - ha[2]*Duomg*self.hessphi(x, xi2) \
            - ha[1]*self.D3phi(x, xi1, xi2) \
            + (ha[3]/(ha[1]**2-ha[2]*ha[0])
               - ha[2]*(2*ha[1]*ha[2]-ha[1]*ha[2]-ha[0]*ha[3])/(ha[1]**2-ha[2]*ha[0])**2
               )*Duomg*(
                   - ha[1]*mtw.Adot(x, self.hessphi(x, xi2))*y + mtw.Adot(xi2, y)*y) \
            + ha[2]/(ha[1]**2-ha[2]*ha[0])*(
                - ha[2]*Duomg*mtw.Adot(x, self.hessphi(x, xi2))*y
                - ha[1]*mtw.Adot(xi1, self.hessphi(x, xi2))*y
                - ha[1]*mtw.Adot(x, self.D3phi(x, xi1, xi2))*y
                - ha[1]*mtw.Adot(x, self.hessphi(x, xi2))*self.DOptMap(x, xi1, mtw)
                + mtw.Adot(xi2, self.DOptMap(x, xi1, mtw))*y
                + mtw.Adot(xi2, y)*self.DOptMap(x, xi1, mtw)
            )


class GHPatterns(object):
    patterns = [[1, 1, 1, 1, 0],
                [-1, 1, -1, 1, 0],
                [1, -1, -1, 1, 0],
                [-1, -1, 1, 1, 0],
                [1, -1, 1, -1, 0],
                [-1, -1, -1, -1, 0],
                [-1, 1, 1, 1, -1],
                [1, 1, -1, 1, -1],
                [-1, -1, -1, 1, -1],
                [1, -1, 1, 1, -1],
                [-1, -1, 1, -1, -1],
                [1, -1, -1, -1, -1],
                [-1, 1, 1, 1, 1],
                [1, 1, -1, 1, 1],
                [-1, -1, -1, 1, 1],
                [1, -1, 1, 1, 1],
                [-1, -1, 1, -1, 1],
                [1, -1, -1, -1, 1],
                [1, 0, 1, 1, 0],
                [-1, 0, 1, 1, 0],
                [1, 0, -1, 1, 0],
                [-1, 0, -1, 1, 0],
                [1, -1, 1, 0, 0],
                [-1, -1, 1, 0, 0],
                [1, -1, -1, 0, 0],
                [-1, -1, -1, 0, 0]]

    @classmethod
    def match(cls, p, branch):
        # return the pattern id
        if branch is None:
            br = 0
        else:
            br = branch
        newrow = jnp.sign(p).tolist() + [br]
        try:
            ret = cls.patterns.index(newrow)
            return ret
        except Exception:
            return None

    @classmethod
    def rand_params(cls, key, pid):
        pp = []
        aline = cls.patterns[pid]
        sk, key = random.split(key)
        tmp = (jnp.abs(random.normal(sk, (4,))) + .01).tolist()
        if aline[3] > 0 and aline[1] > 0:
            tmp2 = sorted([tmp[1], tmp[3]])
        elif aline[3] < 0 and aline[1] < 0:
            tmp2 = sorted([tmp[1], tmp[3]])[::-1]
        else:
            tmp2 = [tmp[1], tmp[3]]

        tmp[1] = tmp2[0]
        tmp[3] = tmp2[1]

        for i in range(4):
            if aline[i] == 0:
                pp.append(0.)
            else:
                pp.append(abs(tmp[i])*aline[i])
        return pp, aline[-1], key


class LambertPatterns(object):
    """ Excluding the Euclidean case a2 = 0
    pattern of a1, a2, branch.

    """
    patterns = [[-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                \
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1]]

    @classmethod
    def match(cls, p, branch):
        # return the pattern id
        newrow = [jnp.sign(p[1]), jnp.sign(p[2])] + [branch]
        try:
            ret = cls.patterns.index(newrow)
            return ret
        except Exception:
            return None

    @classmethod
    def rand_params(cls, key, pid):
        aline = cls.patterns[pid]
        sk, key = random.split(key)
        tmp0 = random.normal(sk, (3,))
        tmp = (jnp.abs(tmp0[1:]) + .01).tolist()
        pp = [tmp0[0], tmp[0]*aline[0], tmp[0]*aline[1]]

        return pp, aline[-1], key
