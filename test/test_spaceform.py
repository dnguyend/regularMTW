import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax.scipy.optimize import minimize
from jax import jvp, jacfwd, jacrev, random, grad
from regularMTW.src.tools import asym, sym2, sym, Lyapunov, vcat, LambertW
from regularMTW.src.space_form import SimpleKH, grand, splitzero
from regularMTW.src.simple_mtw import (GenHyperbolicSimpleMTW,
                                       LambertSimpleMTW)


from jax.config import config
config.update("jax_enable_x64", True)


def deriv_antenna(s, pp):
    """ function of form s = p0exp(p1u) + p2
    """
    p0, p1, p2 = pp
    u = 1/p1*jnp.log((s-p2)/p0)
    h0 = p0*jnp.exp(p1*u) + p2
    h1 = p0*p1*jnp.exp(p1*u)
    h2 = p0*p1**2*jnp.exp(p1*u)
    h3 = p0*p1**3*jnp.exp(p1*u)
    h4 = p0*p1**4*jnp.exp(p1*u)

    return u, h0, h1, h2, h3, h4


def deriv_hyperb(s, pp, branch=None):
    """ function of form s = p0exp(-p3u) + p2exp(p3u)
    thus, p1 = -p3
    if p0p2 < 0 then always has root, one branch, so ignore branch
    if p0p2 > 0 then need s > 4p0p2, two branches
    """
    p0, p1, p2 = pp
    p3 = - p1
    if p0*p2 < 0:
        rt = (s + jnp.sign(p2)*jnp.sqrt(s**2-4*p2*p0))/(2*p2)
        u = 1/p3*jnp.log(rt)
    else:
        u = 1/p3*jnp.log((s + branch*jnp.sign(p2)*jnp.sqrt(s**2-4*p2*p0))/(2*p2))
    h0 = p0*jnp.exp(p1*u) + p2*jnp.exp(p3*u)
    h1 = p0*p1*jnp.exp(p1*u) + p2*p3*jnp.exp(p3*u)
    h2 = p0*p1**2*jnp.exp(p1*u) + p2*p3**2*jnp.exp(p3*u)
    h3 = p0*p1**3*jnp.exp(p1*u) + p2*p3**3*jnp.exp(p3*u)
    h4 = p0*p1**4*jnp.exp(p1*u) + p2*p3**4*jnp.exp(p3*u)

    return u, h0, h1, h2, h3, h4


def deriv_quad_hyperb(s, pp, branch):
    """ function of form s = p0exp(p1u) + p2exp(2p1u)
    thus, p3 = 2p1

    range is between 0 and p2infty ( at u = p1 infty). So s needs to be
    of same sign as p2.

    if p0p2 > 0 then one branch, so ignore branch
    if p0p2 < 0 then need s > 4p0p2, two branches
    """
    p0, p1, p2 = pp
    if jnp.sign(p2) != jnp.sign(s):
        raise ValueError("s and p2 should have same sign")
    p3 = 2*p1
    if p0*p2 < 0:
        u = 1/p1*jnp.log((jnp.sign(p2)*jnp.sqrt(4*s*p2+p0**2)-p0)/(2*p2))
    else:
        u = 1/p1*jnp.log((branch*jnp.sign(p2)*jnp.sqrt(4*s*p2+p0**2)-p0)/(2*p2))
    h0 = p0*jnp.exp(p1*u) + p2*jnp.exp(p3*u)
    h1 = p0*p1*jnp.exp(p1*u) + p2*p3*jnp.exp(p3*u)
    h2 = p0*p1**2*jnp.exp(p1*u) + p2*p3**2*jnp.exp(p3*u)
    h3 = p0*p1**3*jnp.exp(p1*u) + p2*p3**3*jnp.exp(p3*u)
    h4 = p0*p1**4*jnp.exp(p1*u) + p2*p3**4*jnp.exp(p3*u)

    return u, h0, h1, h2, h3, h4


def deriv_gh(s, GH):
    u = GH.ufunc(s)
    h0, h1, h2, h3, h4 = GH.dsfunc(u, 4)
    return u, h0, h1, h2, h3, h4


def deriv_lambertA(s, LB):
    u = LB.ufunc(s)
    h0, h1, h2, h3, h4 = LB.dsfunc(u, 4)
    return u, h0, h1, h2, h3, h4


def deriv_lambertB(s, aa, branch):
    # function is (a0 + a1u)*exp(a2u) = s
    a0, a1, a2 = aa
    u = 1/a2*LambertW(a2*s*jnp.exp(a0*a2/a1)/a1, 0)[0] - a0/a1
    ex = jnp.exp(a2*u)
    h = [(a0 + a1*u)*ex]
    a, b = a0, a1
    for i in range(1, 5):
        h.append(h[-1]*a2 + b*ex)
        a = a*a2 + b
        b = b*a2              

    return u, h[0], h[1], h[2], h[3], h[4]


def deriv_power_al(s, eps, al, al0):
    u = - jnp.abs(eps*s-eps*al0)**(1/al)

    h0 = eps*(-u)**al + al0
    h1 = -al*eps*(-u)**(al-1)
    h1 = al*(h0-al0)/u
    if al == 2:
        h2 = al*(al-1)*eps
        h3 = 0
        h4 = 0
    else: 
        h2 = (al-1)*al*(h0-al0)/u**2
        if al == 3:
            h3 = - al*(al-1)*(al-2)*eps
            h4 = 0.
        else:
            h3 = al*(al-1)*(al-2)*(h0-al0)/u**3
            if al == 4:
                h4 = al*(al-1)*(al-2)*(al-3)*eps
            else:            
                h4 = al*(al-1)*(al-2)*(al-3)*(h0-al0)/u**4
    return u, h0, h1, h2, h3, h4


def deriv_power(s, al, p0=1):
    u = - 1/p0*jnp.abs(s+1)**(1/al)

    h0 = jnp.abs(p0*u)**al - 1
    h1 = - al*p0*jnp.abs(p0*u)**(al-1)
    if al == 2:
        h2 = al*(al-1)*p0**2
        h3 = 0
        h4 = 0
    else: 
        h2 = (al-1)*al*p0**2*jnp.abs(p0*u)**(al-2)
        if al == 3:
            h3 = - al*(al-1)*(al-2)*p0**3
            h4 = 0.
        else:
            h3 = - al*(al-1)*(al-2)*p0**3*jnp.abs(p0*u)**(al-3)
            if al == 4:
                h4 = al*(al-1)*(al-2)*(al-3)*p0**4
            else:            
                h4 = al*(al-1)*(al-2)*(al-3)*p0**4*jnp.abs(p0*u)**(al-4)
    return u, h0, h1, h2, h3, h4


def check_gauss_codazzi():
    n = 5

    # power case
    key = random.PRNGKey(0)

    eps = 1
    nm = 1

    # need nm = 0 then eps != -1
    # nm = n then eps != 1
    A = jnp.diag(jnp.array(nm*[-1.] + (n-nm)*[1.]))

    ppt, key = grand(key, (4,))
    ppt = jnp.abs(ppt)
    pex = sorted([ppt[1], ppt[3]], reverse=True)
    pp = [-ppt[0], -pex[0], ppt[2], -pex[1]]

    GH = GenHyperbolicSimpleMTW(n, A, pp, branch=-1)
    SKH = SimpleKH(n, A, eps, lambda s: deriv_gh(s, GH))
    qs, key = SKH.gen_qs(key)

    Omg1, key = grand(key, (2*n,))
    Omg2, key = grand(key, (2*n,))
    Omg3, key = grand(key, (2*n,))
    Omg4, key = grand(key, (2*n,))
    
    print(GH.KMMetric(qs, Omg1, Omg2))
    print(SKH.KMMetric(qs, Omg1, Omg2))

    print(jvp(lambda qq: GH.KMMetric(qq, Omg2, Omg2), (qs,), (Omg1,))[1])
    print(2*GH.KMMetric(qs, Omg2, GH.Gamma(qs, Omg1, Omg2)))

    Xi1 = SKH.projSphere(qs, Omg1)
    Xi2 = SKH.projSphere(qs, Omg2)
    # Xi3 = SKH.projSphere(qs, Omg3)
    # Xi4 = SKH.projSphere(qs, Omg4)

    print(jvp(lambda qq: SKH.KMMetric(qq, Xi2, Xi2), (qs,), (Xi1,))[1])
    print(2*GH.KMMetric(qs, Xi2, GH.Gamma(qs, Xi1, Xi2)))
    print(2*SKH.KMMetric(qs, Xi2, SKH.GammaAmbient(qs, Xi1, Xi2)))
    print(2*SKH.KMMetric(qs, Xi2, SKH.Gamma(qs, Xi1, Xi2)))

    Xi1x, Xi1y = splitzero(Xi1)

    def Curv3(self, q, Omg1, Omg2, Omg3):
        D1 = jvp(lambda q: self.Gamma(q, Omg2, Omg3), (q,), (Omg1,))[1]
        D2 = jvp(lambda q: self.Gamma(q, Omg1, Omg3), (q,), (Omg2,))[1]
        G1 = self.Gamma(q, Omg1, self.Gamma(q, Omg2, Omg3))
        G2 = self.Gamma(q, Omg2, self.Gamma(q, Omg1, Omg3))
        return D1 - D2 + G1 - G2
        
    print(SKH.KMMetric(qs, Xi1x, Curv3(SKH, qs, Xi1x, Xi1y, Xi1y)))
    print(SKH.crossCurvSphere(qs, Xi1)[0])

    # second fundamental form
    def TwoSphere(self, qs, Xi1, Xi2):
        GA = self.GammaAmbient(qs, Xi1, Xi2)
        return self.DprojSphere(qs, Xi1, Xi2) + GA - self.projSphere(qs, GA)

    # Gauss Codazzi
    print(GH.crossCurv(qs, Xi1) + GH.KMMetric(qs,
                                              TwoSphere(SKH, qs, Xi1x, Xi1x),
                                              TwoSphere(SKH, qs, Xi1y, Xi1y)))

    
    
def testSphere():
    n = 5

    # power case
    key = random.PRNGKey(0)

    eps = 1
    nm = 0

    # need nm = 0 then eps != -1
    # nm = n then eps != 1
    A = jnp.diag(jnp.array(nm*[-1.] + (n-nm)*[1.]))

    # Antenna case,
    # need p0 <0, p1 < 0, p2 >0

    bad = False
    for i in range(100):
        sk, key = random.split(key)
        al,  p0 = random.uniform(sk, (2,), minval= 2, maxval=10)
        p0 = p0/5

        skp = SimpleKH(n, A, eps, lambda q: deriv_power(q, al, p0))
        qs, key = skp.gen_qs(key)
        Xinull, key = skp.gennull_sphere(key, qs)
        # print(ska.KMMetric(qs, Xinull, Xinull))
        curv, sR1, sR23, R4 = skp.crossCurvSphere(qs, Xinull)
        if not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD")
            bad = True
            break

    if bad:
        print("BAD")
    else:
        print("GOOD")


def testHyperboloidPower():
    n = 5

    # power case
    key = random.PRNGKey(0)

    eps = -1
    nm = 1
    A = jnp.diag(jnp.array(nm*[-1.] + (n-nm)*[1.]))

    bad = False
    for i in range(1000):
        sk, key = random.split(key)
        # al,  al0 = random.uniform(sk, (2,), minval=2, maxval=10)
        al, al0 = random.uniform(sk, (2,), minval=0.5, maxval=1)
        al0 = 0.
        skp = SimpleKH(n, A, eps, lambda q: deriv_power_al(q, eps, al, al0))

        qs, key = skp.gen_qs(key)
        Xinull, key = skp.gennull_sphere(key, qs)
        # print(ska.KMMetric(qs, Xinull, Xinull))
        curv, sR1, sR23, R4 = skp.crossCurvSphere(qs, Xinull)
        # print(curv, sR1, sR23, R4)
        if not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD")
            bad = True
            break

    if bad:
        print("BAD")
    else:
        print("GOOD")


def test_hyperboloid_GH_Lambert():
    n = 5
    key = random.PRNGKey(0)

    # nm is number of negative eigenvalues
    # nm = 0 is the sphere, nm = 1 is the hyperboloid model
    # constraint is - sum x_{negative) + sum x_pos = eps
    eps = -1
    nm = 1

    # need nm = 0 then eps != -1
    # nm = n then eps != 1
    A = jnp.diag(jnp.array(nm*[-1.] + (n-nm)*[1.]))

    # Antenna case,
    # need p0 <0, p1 < 0, p2 >0

    bad = False
    for i in range(1000):
        ppt, key = grand(key, (3,))
        ppt = jnp.abs(ppt)
        pp = [-ppt[0], -ppt[1], ppt[2]]
        ska = SimpleKH(n, A, eps, lambda q: deriv_antenna(q, pp[:3]))
        qs, key = ska.gen_qs(key)
        Xinull, key = ska.gennull_sphere(key, qs)
        # print(ska.KMMetric(qs, Xinull, Xinull))
        curv, sR1, sR23, R4 = ska.crossCurvSphere(qs, Xinull)
        if not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD")
            bad = True
            break

    if bad:
        print("BAD")
    else:
        print("GOOD")

    # sksinh = SimpleKH(n, A, eps, lambda q: deriv_hyperb(q, pp))        
    # generalized sinh case
    # need p0 < 0, p1 < 0, p2 > 0, -p1 >= p3 > 0
    # for example pp = [-.26, -.3, 1., .31]
    bad = False
    for i in range(1000):
        ppt, key = grand(key, (4,))
        ppt = jnp.abs(ppt)
        pex = sorted([ppt[1], ppt[3]], reverse=True)
        pp = [-ppt[0], -pex[0], ppt[2], pex[1]]
        GH = GenHyperbolicSimpleMTW(n, A, pp)

        sksinh = SimpleKH(n, A, eps, lambda s: deriv_gh(s, GH))
        qs, key = sksinh.gen_qs(key)
        Xinull, key = sksinh.gennull_sphere(key, qs)
        curv, sR1, sR23, R4 = sksinh.crossCurvSphere(qs, Xinull)
        if not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD", curv, sR1, sR23, R4)
            bad = True
            break

    if bad:
        print("BAD")
    else:
        print("GOOD")

    # branch case
    # skhypbranch. Need p0 < 0, p_1 < 0, p2 > 0, p1 < p3 <=0
    # like [-1, -1, 2, -.5]
    bad = False
    for i in range(1000):
        ppt, key = grand(key, (4,))
        ppt = jnp.abs(ppt)
        pex = sorted([ppt[1], ppt[3]], reverse=True)
        pp = [-ppt[0], -pex[0], ppt[2], -pex[1]]

        GH = GenHyperbolicSimpleMTW(n, A, pp, branch=-1)
        if GH.rng[1] > 1e8:
            continue
        skhypbranch = SimpleKH(n, A, eps, lambda s: deriv_gh(s, GH))
        qs, key = skhypbranch.gen_qs(key)
        Xinull, key = skhypbranch.gennull_sphere(key, qs)
        # print(skhypbranch.KMMetric(qs, Xinull, Xinull))
        curv, sR1, sR23, R4 = skhypbranch.crossCurvSphere(qs, Xinull)
        if jnp.isnan(curv):
            print("NAN", i, curv, sR1, sR23, R4)
        elif not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD", i, curv, sR1, sR23, R4)
            bad = True
            break

    if bad:
        print("BAD")
    else:
        print("GOOD")
    
    # Lambert case
    # a0, a1, a2. No constraint on a0, a1 > 0, a2 <0
    # example [-1, 1, -2]
    
    bad = False
    for i in range(1000):
        ppt, key = grand(key, (3,))
        aa = [ppt[0], jnp.abs(ppt[1]), - jnp.abs(ppt[2])]
        skl = SimpleKH(n, A, eps, lambda s: deriv_lambertB(s, aa, 0))
        qs, key = skl.gen_qs(key)
        Xinull, key = skl.gennull_sphere(key, qs)
        # print(skhypbranch.KMMetric(qs, Xinull, Xinull))
        curv, sR1, sR23, R4 = skl.crossCurvSphere(qs, Xinull)
        if jnp.isnan(curv):
            print("NAN", i, curv, sR1, sR23, R4)
        elif not jnp.all(jnp.array([curv, sR1, sR23, R4]) > 0):
            print("BAD", i, curv, sR1, sR23, R4)
            bad = True
            break
    if bad:
        print("BAD")
    else:
        print("GOOD")
