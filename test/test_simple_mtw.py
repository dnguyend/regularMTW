import jax.numpy as jnp
import jax.numpy.linalg as jla
from jax import jvp, jacfwd, jacrev, random, grad
from regularMTW.src.tools import asym, sym2, sym, Lyapunov, vcat
from regularMTW.src.simple_mtw import (
    GenHyperbolicSimpleMTW,
    LambertSimpleMTW,
    TrigSimpleMTW,
    GHPatterns, LambertPatterns,
    basePotential, grand, splitzero)

from jax.config import config
config.update("jax_enable_x64", True)


def test_u_p():
    """ test the expression of the optimal map
    there is a value u_p corresponding  to  the critical
    point of u(x.y) + phi(x)
    """
    n = 4
    key = random.PRNGKey(0)

    def UphiGH(GH, x, pvec):
        p0, p1, p2, p3 = GH.p
        return 1/(p3-p1)*jnp.log(p0*(1+p1*GH.Adot(x, pvec)) /
                                 (-p2*(1+p3*GH.Adot(x, pvec))))

    A = jnp.eye(n)
    kid = 1
    for jj in range(5):
        pp, br, key = GHPatterns.rand_params(key, kid)
        A = jnp.eye(n)
        GH = GenHyperbolicSimpleMTW(n, A, pp, br)
        pvec, key = grand(key, (n,))
        x, key = GH.grand(key)
        pvec = -1/GH.Adot(pvec, x)*0.5*(1/GH.p[1] + 1/GH.p[3])*pvec

        ts = GH.tsolve(x, pvec)

        ts1 = - GH.dsfunc(UphiGH(GH, x, pvec), 1)[1]
        print(ts1 - ts)

        yx = ts*pvec
        print(grad(lambda x: GH.c(vcat(x, yx)))(x) + pvec)

    kid = 2
    for jj in range(5):
        pp, br, key = GHPatterns.rand_params(key, kid)
        A = jnp.eye(n)
        GH = GenHyperbolicSimpleMTW(n, A, pp, br)
        pvec, key = grand(key, (n,))
        x, key = GH.grand(key)
        pvec = -1/GH.Adot(pvec, x)*0.5*(1/GH.p[1] + 1/GH.p[3])*pvec

        ts = GH.tsolve(x, pvec)

        ts1 = - GH.dsfunc(UphiGH(GH, x, pvec), 1)[1]
        print(ts1 - ts)

        yx = ts*pvec
        print(grad(lambda x: GH.c(vcat(x, yx)))(x) + pvec)

    # now test Lambert

    def UphiLB(LB, x, pvec):
        a0, a1, a2 = LB.p        
        return (-(a1+a0*a2)*LB.Adot(x, pvec) - a0) / (a1*(1+a2*LB.Adot(x, pvec)))

    kid = 0
    pp, br, key = LambertPatterns.rand_params(key, kid)
    LB = LambertSimpleMTW(n, A, pp, br)

    pvec, key = grand(key, (n,))
    x, key = LB.grand(key)
    ts = LB.tsolve(x, pvec)

    uphi = UphiLB(LB, x, pvec)
    ts1 = - LB.dsfunc(uphi, 1)[1]

    yx = ts*pvec

    print(grad(lambda x: LB.c(vcat(x, yx)))(x) + pvec)

    # Trigonometric. Many branches
    # make sure the range not too small
    kid = 2
    while True:
        pp, key = grand(key, (4,))
        A = jnp.eye(n)
        TG = TrigSimpleMTW(n, A, pp, branch=kid)
        if (jnp.abs(TG.rng[0]-TG.rng[1]) > 1e-1) \
           and (jnp.abs(TG.rng[0]-TG.rng[1]) < 1e3):
            break

    x, key = TG.grand(key)
    y, key = gen_in_range(key, TG, x)
    uy = TG.ufunc(TG.Adot(x, y))

    def UphiTrig(TG, x, pvec):
        b0, b1, b2, b3 = TG.p
        w = TG.Adot(x, pvec)
        u = 1/b2*(jnp.arctan(-b2*w/(1+b1*w)) - b3)
        krng = jnp.sort(jnp.array([(TG.u_rng[0] - u)*b2/jnp.pi,
                                   (TG.u_rng[1] - u)*b2/jnp.pi]))
        u += jnp.floor(krng[1])*jnp.pi/b2
        return u

    pvec = - 1/TG.dsfunc(uy, 1)[1]*y

    ts = TG.tsolve(x, pvec)

    uphi = UphiTrig(TG, x, pvec)
    ts1 = - TG.dsfunc(uphi, 1)[1]
    print(ts - ts1)

    yx = ts*pvec

    print(grad(lambda x: TG.c(vcat(x, yx)))(x) + pvec)


def checkall(key):
    allret = []
    for i in range(1000):
        sk, key = random.split(key)
        prm = random.randint(sk, (5,), -30, 30)
        pp = (prm[:4]*.5).tolist()
        if pp[0] == 0:
            pp[0] = 0.1
        if pp[2] == 0:
            pp[2] = 0.1

        if pp[1] == pp[3]:
            pp[3] = pp[3] + .2

        if pp[1] > pp[3]:
            tmp = pp[3]
            pp[3] = pp[1]
            pp[1] = tmp

        br = prm[5] % 3 - 1        
        if pp[0]*pp[1]*pp[2]*pp[3] >= 0:
            br = None
        elif br == 0:
            br = jnp.sign(pp[0]*pp[2])

        pp = jnp.array(pp)

        ret = GHPatterns.match(pp, br)
        if ret is None:
            print("BAD")
            print(pp, br)
            break
        else:
            allret.append(ret)

    n_rpt = 100

    for i in range(len(GHPatterns.patterns)):
        for jj in range(n_rpt):
            pp, br, key = GHPatterns.rand_params(key, i)
            i2 = GHPatterns.match(jnp.array(pp), br)
            if pp[1] >= pp[3]:
                print("BAD")
                print(pp, br)
                break
            if i2 != i:
                print("UNMATCH")
                print(pp, br)
                break


def gen_in_range(key, GH, x):
    # y such that x.y is in range
    n = GH.n
    yt, key = grand(key, (GH.n+1,))

    s = GH.Adot(x, yt[:n])
    if GH.is_in_range(s):
        return yt[:n], key
    else:
        big = max(GH.rng)
        small = min(GH.rng)
        if jnp.isinf(big):
            big = small + 100
        elif jnp.isinf(small):
            small = big - 100
        return (small + (big-small)/(jnp.abs(yt[-1]) + 1))/s*yt[:n], key


def test_gen_in_range(key):
    n = 3
    for kid in range(len(GHPatterns.patterns)):
        for jj in range(5):
            # bad = test_one(kid, n, jj, key)
            pp, br, key = GHPatterns.rand_params(
                key, kid)
            A = jnp.eye(n)
            GH = GenHyperbolicSimpleMTW(n, A, pp, br)

            x, key = GH.grand(key)
            y, key = gen_in_range(key, GH, x)
            s = GH.Adot(x, y)
            bad = not GH.is_in_range(s)

            if bad:
                print("BAD", kid)
                break
        if bad:
            break


def test_all_hyperbolic():
    key = random.PRNGKey(0)

    # test all patterns:

    def test_one(kid, n, jj, key):
        pp, br, key = GHPatterns.rand_params(
            key, kid)
        A = jnp.eye(n)
        GH = GenHyperbolicSimpleMTW(n, A, pp, br)

        x, key = GH.grand(key)
        y, key = gen_in_range(key, GH, x)
        q = vcat(x, y)
        u = GH.ufunc(GH.Adot(x, y))
        diff0 = GH.Adot(x, y) - GH.dsfunc(u, 1)[0]

        Omg0, key = grand(key, (2*n,))
        Omg1, key = grand(key, (2*n,))

        # Levi Civita connection
        j1 = jvp(lambda q: GH.KMMetric(q, Omg1, Omg1), (q,), (Omg0,))[1]
        j2 = 2*GH.KMMetric(q, Omg1, GH.Gamma(q, Omg0, Omg1))
        diff1 = j1 - j2

        # curvature
        # OmgNull, key = GH.gennull(key, q)
        Omgx, Omgy = splitzero(Omg0)
        cc = GH.KMMetric(q, Omgx, GH.Curv3(q, Omgx, Omgy, Omgy))
        diff2 = cc - GH.crossCurv(q, Omg0)
        # print(diff2)

        bad = jnp.max(jnp.array([diff0, diff1, diff2])) > 1e-5

        if bad:
            return bad, x, y, GH, key
        return bad, None, None, GH, key

    n = 3
    bad = False
    for kid in range(len(GHPatterns.patterns)):
        for jj in range(5):
            bad = test_one(kid, n, jj, key)            
            if bad[0]:
                print("BAD", kid, jj, bad)
                break
            else:
                key = bad[-1]
        if bad:
            break

        
def checkAllLambert(key):
    key = random.PRNGKey(0)
    
    allret = []
    for i in range(10000):
        sk, key = random.split(key)
        prm = random.randint(sk, (3,), -30, 30)
        pp = (prm[:3]*.5).tolist()
        if pp[1] == 0:
            pp[1] = 0.1
        if pp[2] == 0:
            pp[2] = 0.1

        br = 2*(prm[5] % 2) - 1        

        pp = jnp.array(pp)

        ret = LambertPatterns.match(pp, br)
        if ret is None:
            print("BAD")
            print(pp, br)
            break
        else:
            allret.append(ret)

    n_rpt = 100

    for i in range(len(LambertPatterns.patterns)):
        for jj in range(n_rpt):
            pp, br, key = LambertPatterns.rand_params(key, i)
            i2 = LambertPatterns.match(jnp.array(pp), br)
            if i2 != i:
                print("UNMATCH")
                print(pp, br)
                break


def test_all_Lambert():
    key = random.PRNGKey(0)

    # test all patterns:

    def test_one(kid, n, jj, key):
        pp, br, key = LambertPatterns.rand_params(
            key, kid)
        A = jnp.eye(n)
        LB = LambertSimpleMTW(n, A, pp, br)

        x, key = LB.grand(key)
        y, key = gen_in_range(key, LB, x)
        q = vcat(x, y)
        u = LB.ufunc(LB.Adot(x, y))
        diff0 = LB.Adot(x, y) - LB.dsfunc(u, 1)[0]

        Omg0, key = grand(key, (2*n,))
        Omg1, key = grand(key, (2*n,))

        # Levi Civita connection
        j1 = jvp(lambda q: LB.KMMetric(q, Omg1, Omg1), (q,), (Omg0,))[1]
        j2 = 2*LB.KMMetric(q, Omg1, LB.Gamma(q, Omg0, Omg1))
        diff1 = j1 - j2

        # curvature
        # OmgNull, key = LB.gennull(key, q)
        Omgx, Omgy = splitzero(Omg0)
        cc = LB.KMMetric(q, Omgx, LB.Curv3(q, Omgx, Omgy, Omgy))
        diff2 = cc - LB.crossCurv(q, Omg0)

        bad = jnp.max(jnp.array([diff0, diff1, diff2])) > 1e-5

        if bad:
            return bad, x, y, LB, key
        return bad, None, None, LB, key

    n = 3
    bad = False
    for kid in range(len(LambertPatterns.patterns)):
        for jj in range(5):
            bad = test_one(kid, n, jj, key)
            if bad[0]:
                print("BAD", kid, jj, bad)
                break
            else:
                key = bad[-1]
        if bad:
            break


def test_all_trig():
    key = random.PRNGKey(0)

    # test all patterns:

    def test_one(kid, n, jj, key):
        # k branch with kid
        while True:
            pp, key = grand(key, (4,))
            A = jnp.eye(n)
            TG = TrigSimpleMTW(n, A, pp, branch=kid)
            if (jnp.abs(TG.rng[0]-TG.rng[1]) > 1e-1) \
               and (jnp.abs(TG.rng[0]-TG.rng[1]) < 1e3):
                break

        x, key = TG.grand(key)
        y, key = gen_in_range(key, TG, x)
        q = vcat(x, y)
        u = TG.ufunc(TG.Adot(x, y))
        diff0 = TG.Adot(x, y) - TG.dsfunc(u, 1)[0]

        Omg0, key = grand(key, (2*n,))
        Omg1, key = grand(key, (2*n,))

        # Levi Civita connection
        j1 = jvp(lambda q: TG.KMMetric(q, Omg1, Omg1), (q,), (Omg0,))[1]
        j2 = 2*TG.KMMetric(q, Omg1, TG.Gamma(q, Omg0, Omg1))
        diff1 = 2*(j1 - j2)/(jnp.abs(j1) + jnp.abs(j2))

        # curvature
        # OmgNull, key = TG.gennull(key, q)
        Omgx, Omgy = splitzero(Omg0)
        cc = TG.KMMetric(q, Omgx, TG.Curv3(q, Omgx, Omgy, Omgy))
        cc1 = TG.crossCurv(q, Omg0)
        diff2 = 2*(cc - cc1)/(jnp.abs(cc) + jnp.abs(cc1))

        bad = jnp.max(jnp.array([diff0, diff1, diff2])) > 1e-4

        if bad:
            print("DIFFS", diff0, diff1, diff2)
            return bad, x, y, TG, key
        return bad, None, None, TG, key

    n = 4
    for kid in range(-3, 4):
        print("DOING %i" % kid)
        for jj in range(10):
            bad = test_one(kid, n, jj, key)
            if bad[0]:
                print("BAD", kid, jj, bad)
                key = bad[-1]
                break
            else:
                key = bad[-1]
        if bad[0]:
            # pass
            break
