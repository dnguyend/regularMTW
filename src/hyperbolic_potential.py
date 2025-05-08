import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from regularMTW.src.tools import vcat

from regularMTW.src.simple_mtw import (basePotential)


def splitq(q):
    n = q.shape[0] // 2
    return q[:n], q[n:]


def al_func(t, al, a, b, c):
    return a*t**(2*al) + b*t**(2*al-2) + c


def al_solve(al, a, b, c):
    # solve for the unique positive root of a*t**(al) + b*t**(al-1) + c = 0
    # with al > 1, a >=0, b > 0, c <0
    # change variable a*z**al + b*z**(al-1) + c = 0
    # newton step with some scaling to be in positive region
    # if al = 1 need b+c < 0 and a !=0
    z = 1.
    err = a*z**al + b*z**(al-1) + c
    for i in range(10):
        step = - err/(al*a*z**(al-1) + (al-1)*b*z**(al-2))
        if z+step < 0:
            z = 1e-3
        else:
            z = z + step
        err = a*z**al + b*z**(al-1) + c
        # print(err)
        if abs(err) < 1e-10:
            break
    return jnp.sqrt(z)


def d_a_al_solve(al, a, b, c, t):
    return - t**(2*al)/(2*al*a*t**(2*al-1)+(2*al-2)*b*t**(2*al-3))


def hyper_ufunc(mtw, s):
    """ truncated Newton with line search
    two cases, if there is a critical point then
    either left (branch = -1) and right (branch = 1)
    """
    p0, _, p2, r = mtw.p
    return -1/r*jnp.log((s+jnp.sqrt(s**2-4*p0*p2))/(2*p0))


class PtHyperbolic(basePotential):
    """ potential for hyperbolic cost
    """
    def __init__(self, mtw):
        self.mtw = mtw

    def i_diag_metric(self, x, xiy):
        r = self.mtw.p[-1]
        g = self.gradphi(x)
        ihg = self.ihessphi(x, g)
        return self.ihessphi(x, xiy) - ihg*ihg.dot(xiy)/(
            1/(r**2*x.dot(g)) + g.dot(ihg))

    def OMap(self, x):
        p0, _, p2, r = self.mtw.p
        g = self.gradphi(x)
        return 2*r*(-p0*p2)**.5/(1-r**2*x.dot(g)**2)**.5*g

    def DOptMapHyp(self, x, omg):
        p0, _, p2, r = self.mtw.p
        g = self.gradphi(x)
        xg = x.dot(g)
        return 2*r*jnp.sign(p0)*(-p0*p2)**.5/(1-r**2*xg**2)**.5*self.hessphi(x, omg) \
            + 2*r**3*jnp.sign(p0)*(-p0*p2)**.5/(1-r**2*xg**2)**1.5*xg*x.dot(self.hessphi(x, omg))*g \
            + 2*r**3*(-p0*p2)**.5/(1-r**2*xg**2)**1.5*omg.dot(g)*xg*g

    def DOptMapHyp_r(self, x, omg, r):
        p0 = .5/r
        p2 = -.5/r
        g = self.gradphi(x)
        xg = x.dot(g)
        # dd = self.hessphi(x, omg) + r**2*g*omg.dot(g)*xg
        
        def iDDc(dd):
            return -1/(1-r**2*xg**2)**.5*dd - (1-r**2*xg**2)**(-1.5)*r**2*xg*x.dot(dd)*g

        def DDc(dd):
            return -jnp.sign(p0)*(1-r**2*x.dot(g)**2)**.5/(-4*r**2*p0*p2)**.5*(dd - r**2*xg*x.dot(dd)*g)
                    
        # return -iDDc(dd)
        return 1/(1-r**2*xg**2)**.5*self.hessphi(x, omg) \
            + (1-r**2*xg**2)**(-1.5)*r**2*xg*(omg.dot(g) \
                                              + x.dot(self.hessphi(x, omg)))*g
    
    def DOptMapHypAlt(self, x, omg):
        p0, _, p2, r = self.mtw.p
        g = self.gradphi(x)
        xg = x.dot(g)
        dd = self.hessphi(x, omg) + r**2*g*omg.dot(g)*xg

        def iDDc(dd):
            return -jnp.sign(p0)*(-4*r**2*p0*p2/(1-r**2*x.dot(g)**2))**.5*dd \
                - jnp.sign(p0)*(-p0*p2)**.5*(1-r**2*xg**2)**(-1.5)*2*r**3*xg*x.dot(dd)*g

        def DDc(dd):
            a = -jnp.sign(p0)*(-4*r**2*p0*p2/(1-r**2*x.dot(g)**2))**.5
            b = - jnp.sign(p0)*(-p0*p2)**.5*(1-r**2*xg**2)**(-1.5)*2*r**3*xg
            return 1/a*dd - b/a/(a + xg*b)*x.dot(dd)*g

        return -iDDc(dd)

    def iDOptMapHypAlt(self, x, omg):
        p0, _, p2, r = self.mtw.p

        g = self.gradphi(x)
        xg = x.dot(g)

        def DDc(dd):
            return -jnp.sign(p0)*(1-r**2*x.dot(g)**2)**.5 \
                / (-4*r**2*p0*p2)**.5*(dd - r**2*xg*x.dot(dd)*g)
        return -self.i_diag_metric(x, DDc(omg))

    def DOptMapT(self, x, omg):
        mtw = self.mtw
        y = self.OptMap(x, mtw)
        u = mtw.ufunc(mtw.Adot(x, y))
        ha = mtw.dsfunc(u, 2)
        return -ha[1]*self.hessphi(x, omg) \
            - ha[2]/(ha[1]**2-ha[2]*ha[0])*ha[1]*y.dot(omg)*self.hessphi(x, x) \
            + ha[2]/(ha[1]**2-ha[2]*ha[0])*mtw.Adot(omg, y)*y

    def DOptMapHypT(self, x, omg):
        mtw = self.mtw
        p0, _, p2, r = mtw.p

        g = self.gradphi(x)
        xg = x.dot(g)
        return 2*r*jnp.sign(p0)*(-p0*p2)**.5/(1-r**2*xg**2)**.5*self.hessphi(x, omg) \
            + 2*r**3*jnp.sign(p0)*(-p0*p2)**.5/(1-r**2*xg**2)**1.5*xg*g.dot(omg)*self.hessphi(x, x) \
            + 2*r**3*(-p0*p2)**.5/(1-r**2*xg**2)**1.5*omg.dot(g)*xg*g

    def hatT(self, q):
        x, bx = splitq(q)
        return vcat(x, self.OptMap(bx, self.mtw))

    def dhatT(self, q, Omg):
        x, bx = splitq(q)
        omg, bomg = splitq(Omg)

        return vcat(omg, self.DOptMap(bx, bomg, self.mtw))

    def cdiv(self, qd):
        mtw = self.mtw
        n = self.mtw.n
        x, xp = splitq(qd)
        qh = vcat(x, self.OptMap(xp, mtw))
        return mtw.c(qh) - mtw.c(vcat(xp, qh[n:])) + self.fphi(x) - self.fphi(xp)

    def gradcdiv(self, qd):
        mtw = self.mtw
        n = self.mtw.n
        x, xp = splitq(qd)
        qh = vcat(x, self.OptMap(xp, mtw))
        u = mtw.ufunc(qh[:n].dot(qh[n:]))
        ha = mtw.dsfunc(u, 2)

        return 1/ha[1]*qh[n:] + self.gradphi(x)

    def diag_metric(self, x, xiy):
        mtw = self.mtw
        r = mtw.p[-1]
        g = self.gradphi(x)
        ret = self.hessphi(x, xiy) + r**2*g*xiy.dot(g)*x.dot(g)

        return ret

    def GammaDivDualistic(self, x, xix1, xix2):
        r = self.mtw.p[-1]
        p0 = .5/r
        p2 = -.5/r
        gphi = self.gradphi
        hphi = self.hessphi
        t3phi = self.D3phi

        g = self.gradphi(x)
        xg = x.dot(g)

        def iDDc(dd):
            return -(1/(1-r**2*x.dot(g)**2))**.5*dd \
                - .5/r*(1-r**2*xg**2)**(-1.5)*2*r**3*xg*x.dot(dd)*g

        def DDc(dd):
            a = -jnp.sign(p0)*(-4*r**2*p0*p2/(1-r**2*x.dot(g)**2))**.5
            b = - jnp.sign(p0)*(-p0*p2)**.5*(1-r**2*xg**2)**(-1.5)*2*r**3*xg
            return 1/a*dd - b/a/(a + xg*b)*x.dot(dd)*g

        def DDcT(dd):
            a = -jnp.sign(p0)*(-4*r**2*p0*p2/(1-r**2*x.dot(g)**2))**.5
            b = - jnp.sign(p0)*(-p0*p2)**.5*(1-r**2*xg**2)**(-1.5)*2*r**3*xg
            return 1/a*dd - b/a/(a + xg*b)*g.dot(dd)*x
        
        dxdxdy = r**2*(
            x.dot(gphi(x))*xix1.dot(gphi(x))*hphi(x, xix2)
            + x.dot(gphi(x))*xix2.dot(gphi(x))*hphi(x, xix1)
            + xix2.dot(gphi(x))*xix1.dot(gphi(x))*hphi(x, x)
            + 3*r**2*x.dot(gphi(x))**2*xix1.dot(gphi(x))*xix2.dot(gphi(x))*gphi(x)
        )
        retx = -self.i_diag_metric(x, dxdxdy)
        
        dydydx = - t3phi(x, xix2, xix1) \
            - r**2*xg*xix1.dot(g)*hphi(x, xix2) \
            - r**2*xix2.dot(g)*xg*hphi(x, xix1) \
            - r**2*g*(
                3*r**2*xg**2*xix1.dot(g)*xix2.dot(g) \
                + 2*xix2.dot(hphi(x, xix1))*xg
                + xix2.dot(g)*xix1.dot(g)
                + xix2.dot(g)*x.dot(hphi(x, xix1))
                + x.dot(hphi(x, xix2))*xix1.dot(g)
            )

        rety = -self.i_diag_metric(x, dydydx)

        return vcat(retx, rety)

    def cexp(self, x, p):
        p0, _, p2, r = self.mtw.p
        return 2*r*(-p0*p2)**.5/(1-r**2*x.dot(p)**2)**.5*p

    def HessOptMapHyprNotwork(self, x, omg1, omg2):
        # still need debug
        r = self.mtw.p[-1]
        g = self.gradphi(x)
        xg = x.dot(g)
        dxg = omg1.dot(g) + x.dot(self.hessphi(x, omg1))
        # ret0 = (1-r**2*xg**2)**(-.5)*self.hessphi(x, omg2) \
        #    + (1-r**2*xg**2)**(-1.5)*r**2*xg*(omg2.dot(g) \
        #                                      + x.dot(self.hessphi(x, omg2)))*g

        ret = r**2*dxg*xg*(1-r**2*xg**2)**(-1.5)*self.hessphi(x, omg2) \
            + (1-r**2*xg**2)**(-.5)*self.D3phi(x, omg1, omg2) \
            - 3*xg*dxg*r**2*(1-r**2*xg**2)**(-2.5)*r**2*xg*(omg2.dot(g) \
                                                           + x.dot(self.hessphi(x, omg2)))*g \
            + (1-r**2*xg**2)**(-1.5)*r**2*dxg*(omg2.dot(g) \
                                               + x.dot(self.hessphi(x, omg2)))*g \
            + (1-r**2*xg**2)**(-1.5)*r**2*xg*(2*omg2.dot(self.hessphi(x, omg1)) \
                                              + x.dot(self.D3phi(x, omg1, omg2))
                                              )*g \
            + (1-r**2*xg**2)**(-1.5)*r**2*xg*(omg2.dot(g) \
                                              + x.dot(self.hessphi(x, omg2)))*self.hessphi(x, omg1)

        return ret

    def div_metric(self, qd, Omg1, Omg2):
        mtw = self.mtw
        x, xp = splitq(qd)
        qh = vcat(x, self.OptMap(xp, mtw))

        return mtw.KMMetric(qh,
                            self.dhatT(qd, Omg1),
                            self.dhatT(qd, Omg2))

    def g_div_metric(self, qd, Omg):
        x, xp = splitq(qd)
        xpt = self.OptMap(xp, self.mtw)
        
        n = self.mtw.n
        u = self.mtw.ufunc(x.dot(xpt))
        ha = self.mtw.dsfunc(u, 2)
        dTOmg = self.DOptMap(xp, Omg[n:], self.mtw)
        
        ret0 = -0.5*vcat(-ha[2]/ha[1]**3*xpt*self.mtw.Adot(dTOmg, x) + 1/ha[1]*dTOmg,
                         -ha[2]/ha[1]**3*x*self.mtw.Adot(Omg[:n], xpt) + 1/ha[1]*Omg[:n])
        return vcat(ret0[:n], self.DOptMapT(xp, ret0[n:]))
    
    def GammaDiv(self, qd, Omg1, Omg2):
        """ Christoffel function of the divergence
        """
        mtw = self.mtw
        n = mtw.n
        x = qd[:n]
        xp = qd[n:]
        xpt = self.OptMap(xp, mtw)
        u = mtw.ufunc(mtw.Adot(x, xpt))
        ha = mtw.dsfunc(u, 3)

        retx = (ha[2]**2 - ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*mtw.Adot(xpt, Omg1[:n])*mtw.Adot(Omg2[:n], xpt)*x \
            - ha[2]/ha[1]**2*Omg1[:n]*mtw.Adot(Omg2[:n], xpt) \
            - ha[2]/ha[1]**2*Omg2[:n]*mtw.Adot(Omg1[:n], xpt)

        dTbomg1 = self.DOptMap(xp, Omg1[n:], mtw)
        dTbomg2 = self.DOptMap(xp, Omg2[n:], mtw)

        gmcy = (ha[2]**2 - ha[3]*ha[1])/ha[1]**2/(ha[1]**2-ha[0]*ha[2])*mtw.Adot(x, dTbomg1)*mtw.Adot(dTbomg2, x)*xpt \
            - ha[2]/ha[1]**2*dTbomg1*mtw.Adot(dTbomg2, x) \
            - ha[2]/ha[1]**2*dTbomg2*mtw.Adot(dTbomg1, x)
        rety = self.iDOptMap(xp, self.HessOptMap(xp, Omg1[n:], Omg2[n:], mtw) \
                             + gmcy, mtw)

        return vcat(retx, rety)

    def iDOptMapHyp_r(self, x, omg):
        r = self.mtw.p[-1]
        g = self.gradphi(x)
        xg = x.dot(g)
        
        def DDc(dd):
            return -(1-r**2*xg**2)**.5*(dd - r**2*xg*x.dot(dd)*g)
                    
        return -self.i_diag_metric(x, DDc(omg))
    
    def GammaDivSINH(self, qe, Omge1, Omge2, mtw):
        r = mtw.p[-1]
        n = mtw.n
        x = qe[:n]

        g = self.gradphi(x)
        # y = 1/(1-r**2*x.dot(g)**2)**.5*g
        xg = x.dot(g)
        HO1 = self.hessphi(x, Omge1[n:])
        Hxo1 = self.hessphi(x, Omge1[n:]).dot(x)
        go1 = Omge1[n:].dot(g)

        HO2 = self.hessphi(x, Omge2[n:])
        Hxo2 = self.hessphi(x, Omge2[n:]).dot(x)
        go2 = Omge2[n:].dot(g)

        retx = -r**2*mtw.Adot(g, Omge1[:n])*mtw.Adot(Omge2[:n], g)*x\
            - r**2*x.dot(g)*mtw.Adot(Omge2[:n], g)*Omge1[:n] \
            - r**2*x.dot(g)*mtw.Adot(Omge1[:n], g)*Omge2[:n]

        gmcy = -((Hxo2 + r**2*xg**2*go2)*(Hxo1 + r**2*xg**2*go1)
                 + (Hxo2 + r**2*xg**2*go2)*r**2*xg**2*Hxo1
                 + (Hxo2 + r**2*xg**2*go2)*r**2*xg**2*go1
                 + (Hxo1 + r**2*xg**2*go1)*r**2*xg**2*Hxo2
                 + (Hxo1 + r**2*xg**2*go1)*r**2*xg**2*go2
                 )*r**2/(1-r**2*xg**2)**2.5*g \
            - r**2*xg/(1-r**2*xg**2)**1.5*(Hxo2 + r**2*xg**2*go2)*HO1 \
            - r**2*xg/(1-r**2*xg**2)**1.5*(Hxo1 + r**2*xg**2*go1)*HO2

        def DDc(dd):
            return -(1-r**2*xg**2)**.5*(dd - r**2*xg*x.dot(dd)*g)
        retya = DDc(self.HessOptMap(x, Omge1[n:], Omge2[n:], mtw) + gmcy)
        rety = -self.i_diag_metric(x, retya)

        return vcat(retx, rety)

    def dx_div(self, x, y):
        r = self.mtw.p[-1]
        gy = self.gradphi(y)
        B = r*x.dot(gy)
        dxB = r*gy
        M = r*y.dot(gy)
        return self.gradphi(x) - 1/r*dxB/jnp.sqrt(B**2 - M**2 + 1)

    def dydx_div(self, x, y, xiy):
        r = self.mtw.p[-1]
        gy = self.gradphi(y)
        hpxi = self.hessphi(y, xiy)
        B = r*x.dot(gy)
        M = r*y.dot(gy)

        ret = - 1/(B**2 - M**2 + 1)**(3/2)*(
            hpxi*(B**2 - M**2 + 1)
            - r**2*gy*(
                x.dot(hpxi)*x.dot(gy)
                - xiy.dot(gy)*y.dot(gy)
                - y.dot(hpxi)*y.dot(gy)
            )
        )
        return ret

    def dxdy_div(self, x, y, xix):
        r = self.mtw.p[-1]
        gy = self.gradphi(y)
        hpxi = self.hessphi(y, xix)
        
        B = r*x.dot(gy)
        M = r*y.dot(gy)

        ret = - 1/(B**2 - M**2 + 1)**(3/2)*(
            hpxi*(B**2 - M**2 + 1)
            - r**2*(
                xix.dot(gy)*self.hessphi(y, x)*x.dot(gy)
                - xix.dot(gy)*gy*y.dot(gy)
                - xix.dot(gy)*self.hessphi(y, y)*y.dot(gy)
            )
        )

        return ret
    
    def dydydx_div_eq(self, x, xiy1, xiy2):
        r = self.mtw.p[-1]
        # y = x
        gphi = self.gradphi
        hphi = self.hessphi
        t3phi = self.D3phi
        gx = gphi(x)
        xgx = x.dot(gx)
        # hxxi1 = hphi(x, xiy1)
        # hxxi2 = hphi(x, xiy2)
        
        # B = r*xgx
        # dyB1 = r*x.dot(hxxi1)
        # dyxB1 = r*hphi(y, xiy1)
        # M = r*x.dot(gx)
        # dMy1 = r*(xiy1.dot(gx) + x.dot(hphi(x, xiy1)))
        # dxB/jnp.sqrt(B**2 - M**2 + 1)
        
        ret = -3*r**2*xgx*xiy1.dot(gx)*(hphi(x, xiy2) + r**2*gx*xiy2.dot(gx)*xgx) \
            - (
            t3phi(x, xiy2, xiy1)
            - 2*r*hphi(x, xiy2)*xgx*r*xiy1.dot(gx)
            + r**2*hphi(x, xiy1)*(xiy2.dot(gx)*xgx)
            + r**2*gx*(
                - x.dot(t3phi(x, xiy2, xiy1))*xgx
                - x.dot(hphi(x, xiy2))*x.dot(hphi(x, xiy1))
                \
                + xiy2.dot(hphi(x, xiy1))*xgx
                + xiy2.dot(gx)*xiy1.dot(gx)
                + xiy2.dot(gx)*x.dot(hphi(x, xiy1))
                \
                + xiy1.dot(hphi(x, xiy2))*xgx
                + x.dot(t3phi(x, xiy2, xiy1))*xgx
                + x.dot(hphi(x, xiy2))*xiy1.dot(gx)
                + x.dot(hphi(x, xiy2))*x.dot(hphi(x, xiy1))
            )
        )

        ret = - t3phi(x, xiy2, xiy1) \
            - r**2*xgx*xiy1.dot(gx)*hphi(x, xiy2) \
            - r**2*xiy2.dot(gx)*xgx*hphi(x, xiy1) \
            - r**2*gx*(
                3*r**2*xgx**2*xiy1.dot(gx)*xiy2.dot(gx) \
                + 2*xiy2.dot(hphi(x, xiy1))*xgx
                + xiy2.dot(gx)*xiy1.dot(gx)
                + xiy2.dot(gx)*x.dot(hphi(x, xiy1))
                + 0*xiy1.dot(hphi(x, xiy2))*xgx
                + x.dot(hphi(x, xiy2))*xiy1.dot(gx)
            )
        
        return ret

    def dydydx_div(self, x, y, xiy1, xiy2):
        r = self.mtw.p[-1]
        gphi = self.gradphi
        hphi = self.hessphi
        t3phi = self.D3phi
        
        B = r*x.dot(gphi(y))
        dyB1 = r*x.dot(hphi(y, xiy1))
        # dyxB1 = r*hphi(y, xiy1)
        M = r*y.dot(gphi(y))
        dMy1 = r*(xiy1.dot(gphi(y)) + y.dot(hphi(y, xiy1)))
        # dxB/jnp.sqrt(B**2 - M**2 + 1)
        
        ret = 3*(B*dyB1 - M*dMy1)/(B**2 - M**2 + 1)**(5/2)*(
            hphi(y, xiy2)*(B**2 - M**2 + 1)
            - r**2*gphi(y)*(
                x.dot(hphi(y, xiy2))*x.dot(gphi(y))
                - xiy2.dot(gphi(y))*y.dot(gphi(y))
                - y.dot(hphi(y, xiy2))*y.dot(gphi(y))
            )
        )
        ret += - 1/(B**2 - M**2 + 1)**(3/2)*(
            t3phi(y, xiy2, xiy1)*(B**2 - M**2 + 1)
            + 2*hphi(y, xiy2)*(B*dyB1 - M*dMy1)
            - r**2*hphi(y, xiy1)*(
                x.dot(hphi(y, xiy2))*x.dot(gphi(y))
                - xiy2.dot(gphi(y))*y.dot(gphi(y))
                - y.dot(hphi(y, xiy2))*y.dot(gphi(y))
            )
            - r**2*gphi(y)*(
                x.dot(t3phi(y, xiy2, xiy1))*x.dot(gphi(y))
                + x.dot(hphi(y, xiy2))*x.dot(hphi(y, xiy1))
                \
                - xiy2.dot(hphi(y, xiy1))*y.dot(gphi(y))
                - xiy2.dot(gphi(y))*xiy1.dot(gphi(y))
                - xiy2.dot(gphi(y))*y.dot(hphi(y, xiy1))
                \
                - xiy1.dot(hphi(y, xiy2))*y.dot(gphi(y))
                - y.dot(t3phi(y, xiy2, xiy1))*y.dot(gphi(y))
                - y.dot(hphi(y, xiy2))*xiy1.dot(gphi(y))
                - y.dot(hphi(y, xiy2))*y.dot(hphi(y, xiy1))
            )
        )
        return ret

    def dydx_div_eq(self, x, xiy):
        r = self.mtw.p[-1]
        g = self.gradphi(x)
        ret = - (self.hessphi(x, xiy) + r**2*x.dot(g)*g*xiy.dot(g))
        return ret

    def AmariChentsov(self, x, xix1, xix2, xix3):
        r = self.mtw.p[-1]
        hphi = self.hessphi
        t3phi = self.D3phi

        g = self.gradphi(x)
        xg = x.dot(g)

        return t3phi(x, xix2, xix1).dot(xix3) \
            + r**2*(
                + 2*xg*g.dot(xix1)*hphi(x, xix2).dot(xix3)
                + 2*xg*g.dot(xix2)*hphi(x, xix1).dot(xix3)
                + 2*g.dot(xix3)*xix2.dot(hphi(x, xix1))*xg
                + (1+6*r**2*xg**2)*g.dot(xix3)*xix1.dot(g)*xix2.dot(g)
                + g.dot(xix3)*xix2.dot(g)*x.dot(hphi(x, xix1))
                + g.dot(xix1)*g.dot(xix2)*hphi(x, x).dot(xix3)
                + g.dot(xix3)*xix1.dot(g)*x.dot(hphi(x, xix2))
            )

    def GammaDiv1(self, x, xi1, xi2):
        """ Primal Gamma of the divergence
        """
        r = self.mtw.p[-1]
        g = self.gradphi(x)
        return - r**2*g.dot(xi1)*g.dot(xi2)*x\
            - r**2*x.dot(g)*g.dot(xi2)*xi1 \
            - r**2*x.dot(g)*g.dot(xi1)*xi2

    def Curv13(self, x, xi1, xi2, xi3):
        """ Curvature of the primal connection
        """
        r = self.mtw.p[-1]
        
        g = self.gradphi(x)
        hh = self.hessphi
        xg = x.dot(g)
        ret = - r**2*g.dot(xi2)*hh(x, xi1).dot(xi3)*x\
            + r**2*g.dot(xi1)*hh(x, xi2).dot(xi3)*x\
            \
            + r**2*x.dot(hh(x, xi2))*g.dot(xi3)*xi1 \
            + r**2*x.dot(g)*hh(x, xi2).dot(xi3)*xi1 \
            + 2*r**4*xg**2*g.dot(xi2)*g.dot(xi3)*xi1 \
            \
            - r**2*x.dot(hh(x, xi1))*g.dot(xi3)*xi2 \
            - r**2*x.dot(g)*hh(x, xi1).dot(xi3)*xi2 \
            - 2*r**4*xg**2*g.dot(xi1)*g.dot(xi3)*xi2 \
            \
            - r**2*x.dot(hh(x, xi1))*g.dot(xi2)*xi3 \
            + r**2*x.dot(hh(x, xi2))*g.dot(xi1)*xi3 \

        return ret
    
    


class PtPow(PtHyperbolic):
    def __init__(self, pw, mtw):
        self.pw = pw
        self.mtw = mtw

    def fphi(self, x):
        return jnp.sum(jnp.abs(x)**self.pw)

    def gradphi(self, x):
        pw = self.pw
        return pw*jnp.abs(x)**pw/x

    def igradphi(self, g):
        pw = self.pw
        return (jnp.abs(g)/pw)**(1/(pw-1))*jnp.sign(g)

    def hessphi(self, x, omg):
        pw = self.pw
        return pw*(pw-1)*jnp.abs(x)**(pw-2)*omg

    def ihessphi(self, x, bomg):
        pw = self.pw
        return bomg/(pw*(pw-1)*jnp.abs(x)**(pw-2))

    def D3phi(self, x, omg1, omg2):
        pw = self.pw

        return pw*(pw-1)*(pw-2)*jnp.abs(x)**(pw-2)/x*omg1*omg2

    def iOptMap(self, y, mtw):
        p0, _, p2, r = mtw.p
        xdir = self.igradphi(y)
        s = jnp.sum(y*xdir)
        # w = 0.5*jnp.sum(y*xdir)
        t = al_solve(self.pw, s**2, -4*p0*p2, -1/r**2)
        # t = 1/(-2*p0*p2*r**2 + 2*((p0*p2*r**2)**2+r**2*w**2)**.5)**.5

        return t*xdir

    def OptMapSimp(self, x):
        p0, _, p2, r = self.mtw.p
        pw = self.pw
        px = ((jnp.abs(x)**pw)).sum()
        return (1-r**2*pw**2*px**2)**(-.5)*pw*jnp.abs(x)**pw/x

    def DOptMapSimp(self, x, omg):
        # g = self.gradphi(x)
        # g = pw*jnp.abs(x)**(pw-1)*jnp.sign(x)
        # xg = x.dot(pw*jnp.abs(x)**pw/x)
        pw = self.pw
        px = ((jnp.abs(x)**pw)).sum()
        r = self.mtw.p[-1]
        # hpo = pw*(pw-1)*jnp.abs(x)**(pw-2)*omg
        hpx = pw*(pw-1)*jnp.abs(x)**(pw-2)*x
        mret = 1/(1-r**2*pw**2*px**2)**.5*pw*(pw-1)*jnp.diag(jnp.abs(x)**(pw-2)) \
            + (1-r**2*pw**2*px**2)**(-1.5)*r**2*pw*px * \
            (pw*(jnp.abs(x)**(pw-1)*jnp.sign(x))[:, None]@(pw*jnp.abs(x)**(pw-1)*jnp.sign(x)+hpx)[None, :])

        return mret@omg

    def DOptMapSimpMat(self, x):
        # g = self.gradphi(x)
        # g = pw*jnp.abs(x)**(pw-1)*jnp.sign(x)
        # xg = x.dot(pw*jnp.abs(x)**pw/x)
        pw = self.pw
        px = ((jnp.abs(x)**pw)).sum()
        r = self.mtw.p[-1]
        # hpo = pw*(pw-1)*jnp.abs(x)**(pw-2)
        hpx = pw*(pw-1)*jnp.abs(x)**(pw-2)*x
        mret = 1/(1-r**2*pw**2*px**2)**.5*pw*(pw-1)*jnp.diag(jnp.abs(x)**(pw-2)) \
            + (1-r**2*pw**2*px**2)**(-1.5)*r**2*pw*px * \
            (pw*(jnp.abs(x)**(pw-1)*jnp.sign(x))[:, None]@(pw*jnp.abs(x)**(pw-1)*jnp.sign(x)+hpx)[None, :])
        
        return mret

    def detDOptMapSimpMat(self, x):
        # g = self.gradphi(x)
        # g = pw*jnp.abs(x)**(pw-1)*jnp.sign(x)
        # xg = x.dot(pw*jnp.abs(x)**pw/x)
        pw = self.pw
        n = self.mtw.n
        px = ((jnp.abs(x)**pw)).sum()
        r = self.mtw.p[-1]
        # hpo = pw*(pw-1)*jnp.abs(x)**(pw-2)
        # hpx =
        return (1-r**2*pw**2*px**2)**(-n/2-1)*(1+r**2*pw**2/(pw-1)*px**2) \
            * (pw*(pw-1))**n*jnp.prod(jnp.abs(x)**(pw-2))


class PtPowerDual():
    def __init__(self, primal):
        self.primal = primal
        self.mtw = primal.mtw

    def _Lfunc(self, x, bx):
        return self.mtw.ufunc(jnp.sum(x*bx)) + self.primal.fphi(x)

    def fphi(self, bx):
        x = self.primal.iOptMap(bx, self.mtw)
        return - self._Lfunc(x, bx)

    def gradphi(self, bx):
        pr = self.primal
        x = pr.iOptMap(bx, self.mtw)

        u = self.mtw.ufunc(jnp.sum(x*bx))
        ha = self.mtw.dsfunc(u, 1)
        return -1/ha[1]*(x + pr.iDOptMapHypAlt(x, bx)) - pr.iDOptMapHypAlt(x, pr.gradphi(x))

    def OptMap(self, bx, mtw):
        p0, _, p2, r = mtw.p
        pr = self.primal
        xdir = pr.igradphi(bx)
        s = jnp.sum(bx*xdir)
        t = al_solve(pr.pw, s**2, -4*p0*p2, -1/r**2)
        return t*xdir

    def iOptMap(self, x, mtw):
        """optimal map
        defined if tPhi(x) < 1/r
        range is F(gradFInv(y)) >= (2*r*(-p0*p2)**.5/pw)**(pw/(pw-1))
        """
        p0, _, p2, r = mtw.p
        pw = self.primal.pw
        fphi = self.primal.fphi
        gradphi = self.primal.gradphi
        return 2*r*jnp.sqrt(-p0*p2)/(1-r**2*pw**2*fphi(x)**2)**0.5*gradphi(x)

    def DOptMap(self, bx, eta, _):
        """inverse optimal map
        range is x gradphi(x) < 1/r
        """
        p0, _, p2, r = self.mtw.p
        
        pr = self.primal
        pw = pr.pw
        xdir, dxdir = jax.jvp(pr.igradphi, (bx,), (eta,))
        if jnp.allclose(bx, 0):
            dxdir = jnp.zeros(pr.mtw.n)
        s = jnp.sum(bx*xdir)
        t = al_solve(pw, s**2, -4*p0*p2, -1/r**2)
        dt = d_a_al_solve(pw, s**2, -4*p0*p2, -1/r**2, t)*2*s*(jnp.sum(eta*xdir)+jnp.sum(bx*dxdir))
        return dt*xdir + t*dxdir


class PtQuadratic(PtHyperbolic):
    def __init__(self, A, mtw):
        self.pw = 2
        self.mtw = mtw
        self.A = A

    def fphi(self, x):
        return 0.5*x.dot(self.A@x)

    def gradphi(self, x):
        return self.A@x

    def igradphi(self, g):
        return jla.solve(self.A, g)

    def hessphi(self, x, omg):
        return self.A@omg

    def ihessphi(self, x, bomg):
        return jla.solve(self.A, bomg)

    def D3phi(self, x, omg1, omg2):
        return jnp.zeros(self.GH.n)

    def iOptMap(self, y, mtw):
        p0, _, p2, r = mtw.p
        xdir = self.igradphi(y)
        # s = jnp.sum(y*xdir)
        # t = al_solve(self.pw, s**2, -4*p0*p2, -1/r**2)
        w = 0.5*jnp.sum(y*xdir)
        t = 1/(-2*p0*p2*r**2 + 2*((p0*p2*r**2)**2+r**2*w**2)**.5)**.5

        return t*xdir

    def OptMapSimp(self, x):
        p0, _, p2, r = self.mtw.p
        g = self.A@x
        return (1-r**2*x.dot(g)**2)**(-.5)*g

    def DOptMapSimp(self, x, omg):
        # g = self.gradphi(x)
        # g = pw*jnp.abs(x)**(pw-1)*jnp.sign(x)
        # xg = x.dot(pw*jnp.abs(x)**pw/x)
        A = self.A
        px2 = x.dot(self.A@x)
        r = self.mtw.p[-1]
        return 1/(1-r**2*px2**2)**.5*A@omg \
            + 2*(1-r**2*px2**2)**(-1.5)*r**2*px2*(A@x).dot(omg)*A@x

    def DOptMapSimpMat(self, x):
        r = self.mtw.p[-1]
        A = self.A
        px2 = x.dot(self.A@x)

        return 1/(1-r**2*px2**2)**.5*A \
            + 2*(1-r**2*px2**2)**(-1.5)*r**2*px2*A@x[:, None]@x[None, :]@A

    def detDOptMapSimpMat(self, x):
        r = self.mtw.p[-1]
        A = self.A
        n = self.mtw.n
        px2 = x.dot(self.A@x)

        return (1-r**2*px2**2)**(-n/2-1)*jla.det(A)*(1+r**2*px2**2)


class PtQuadraticDual():
    def __init__(self, primal):
        self.primal = primal
        self.mtw = primal.mtw

    def _Lfunc(self, x, bx):
        return self.mtw.ufunc(jnp.sum(x*bx)) + self.primal.fphi(x)

    def fphi(self, bx):
        x = self.primal.iOptMap(bx, self.mtw)
        return - self._Lfunc(x, bx)

    def gradphi(self, bx):
        pr = self.primal
        x = pr.iOptMap(bx, self.mtw)

        u = self.mtw.ufunc(jnp.sum(x*bx))
        ha = self.mtw.dsfunc(u, 1)
        return -1/ha[1]*(x + pr.iDOptMapHypAlt(x, bx)) - pr.iDOptMapHypAlt(x, pr.gradphi(x))

    def OptMap(self, bx, mtw):
        p0, _, p2, r = mtw.p
        pr = self.primal
        xdir = pr.igradphi(bx)
        s = jnp.sum(bx*xdir)
        t = al_solve(pr.pw, s**2, -4*p0*p2, -1/r**2)
        return t*xdir

    def iOptMap(self, x, mtw):
        """optimal map
        defined if tPhi(x) < 1/r
        range is F(gradFInv(y)) >= (2*r*(-p0*p2)**.5/pw)**(pw/(pw-1))
        """
        p0, _, p2, r = mtw.p
        pw = self.primal.pw
        fphi = self.primal.fphi
        gradphi = self.primal.gradphi
        return 2*r*jnp.sqrt(-p0*p2)/(1-r**2*pw**2*fphi(x)**2)**0.5*gradphi(x)

    def DOptMap(self, bx, eta, _):
        """inverse optimal map
        range is x gradphi(x) < 1/r
        """
        p0, _, p2, r = self.mtw.p

        pr = self.primal
        pw = pr.pw
        xdir, dxdir = jax.jvp(pr.igradphi, (bx,), (eta,))
        if jnp.allclose(bx, 0):
            dxdir = jnp.zeros(pr.mtw.n)
        s = jnp.sum(bx*xdir)
        t = al_solve(pw, s**2, -4*p0*p2, -1/r**2)
        dt = d_a_al_solve(pw, s**2, -4*p0*p2, -1/r**2, t)*2*s*(jnp.sum(eta*xdir)+jnp.sum(bx*dxdir))
        return dt*xdir + t*dxdir


def GammaHyperbolic(mtw, q, Omg1, Omg2):
    p0, _, p2, r = mtw.p
    n = mtw.n
    x = q[:n]
    y = q[n:]
    ha0 = mtw.Adot(x, y)

    # u = mtw.ufunc(ha0)
    # ha = mtw.dsfunc(u, 3)

    retx = -1/(ha0**2 - 4*p0*p2)*mtw.Adot(y, Omg1[:n])*mtw.Adot(Omg2[:n], y)*x \
        - ha0/(ha0**2 - 4*p0*p2)*Omg1[:n]*mtw.Adot(Omg2[:n], y) \
        - ha0/(ha0**2 - 4*p0*p2)*Omg2[:n]*mtw.Adot(Omg1[:n], y)

    rety = -1/(ha0**2 - 4*p0*p2)*mtw.Adot(x, Omg1[n:])*mtw.Adot(Omg2[n:], x)*y \
        - ha0/(ha0**2 - 4*p0*p2)*Omg1[n:]*mtw.Adot(Omg2[n:], x) \
        - ha0/(ha0**2 - 4*p0*p2)*Omg2[n:]*mtw.Adot(Omg1[n:], x)

    return vcat(retx, rety)
