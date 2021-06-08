import random
import pickle
import multiprocessing as mp
import numpy as np
import astropy.coordinates as acoo
import astropy.table as atpy
import astropy.units as auni
import astropy.time as ati
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.optimize
import lc_model_opt
import scipy.stats

plt.ioff()


def betw(x, x1, x2):
    return (x > x1) & (x < x2)


BADV = -1e20  # logl bad value


class FFCache:
    # caching heavy arguments
    xdat = None


class UniformPrior:
    """
    Uniform prior
    """
    def __init__(self, x1, x2):
        """
        Args: Edges of the prior
        """
        self.x1 = x1
        self.x2 = x2
        self.ledge = x1
        self.redge = x2
        assert (x2 > x1)

    def __call__(self, x):
        """
        Return a inverse CDF transform, map [0,1] range into the distribution
        """
        return self.x1 + (self.x2 - self.x1) * x

    def logpdf(self, x):
        inside = (self.x1 < x < self.x2)
        return np.log(1 / (self.x2 - self.x1)) * inside + BADV * (~inside)

    def rvs(self):
        return self.x1 + np.random.uniform() * (self.x2 - self.x1)


class LogUniformPrior:
    """
    Uniform in log-space prior
    """
    def __init__(self, x1, x2):
        """
        Args are edges of the prior
        """
        self.x1 = x1
        self.x2 = x2
        self.ledge = x1
        self.redge = x2
        assert (x2 > x1)

    def __call__(self, x):
        return self.x1 * np.exp(np.log(self.x2 / self.x1) * x)

    def logpdf(self, x):
        inside = (self.x1 < x < self.x2)
        return (-np.log(np.abs(x)) -
                np.log(np.log(self.x2 / self.x1))) * inside + BADV * (~inside)

    def rvs(self):
        return np.exp(np.random.uniform() * np.log(self.x2 / self.x1) +
                      np.log(self.x1))


class BetaPrior:
    """
    Beta(a,b) prior definition
    """
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.B = scipy.stats.beta(self.a, self.b)
        self.legde = 0
        self.redge = 1

    def __call__(self, x):
        return self.B.ppf(x)

    def logpdf(self, x):
        return self.B.logpdf(x)

    def rvs(self):
        return self.B.rvs()


# This is list of priors for each parameter
priorList = [
    UniformPrior(-0.5, 2),  # log10 major axis
    UniformPrior(-0.5, 2),  # log10 minor axis
    UniformPrior(0, np.pi),  # PA with respect to the velocity
    UniformPrior(-3, 2),  # log10 rimpact
    UniformPrior(0, 2 * np.pi),  #  angle at minimum impact distance
    #    UniformPrior(-0.2, 0.2),  # vphi
    UniformPrior(0, 0.2),  # vphi
    UniformPrior(0.5, 0.7),  # acoeff
    UniformPrior(0.05, 0.25),  # bcoeff
    UniformPrior(56000 - 200, 56000 + 200),  # t0
    UniformPrior(-6, 1),  # lplx
    UniformPrior(-5, 0),  # transparency
    UniformPrior(0, 1) # direction of motion parameter 
]


def plx_mod(x, t):
    # parallax model as sum of two sines
    return (x[0] * np.sin((t - x[1]) * 2 * np.pi / 365.2422) + x[2] * np.sin(
        (t - x[3]) * 4 * np.pi / 365.2422) + x[4])


def getplx(ra0, dec0):
    """
    GET plx motion coefficients A1,...A5
    in the form of (ra-ra0)*cosdec=(A1*sin((MJD-A2)*2*pi*t/365.25)+
    (A3*sin((MJD-A4)*4*pi*t/365.25) + A5
    and the same for (dec-dec0)
    """

    t0 = 56000
    N = 1000
    t = np.linspace(t0, t0 + 365.2422, N)

    C = acoo.SkyCoord(ra=ra0 * auni.degree,
                      dec=dec0 * auni.degree,
                      distance=1 * auni.kpc,
                      pm_ra_cosdec=0 * auni.mas / auni.year,
                      pm_dec=0 * auni.mas / auni.year,
                      obstime=ati.Time(t0, format='mjd')).apply_space_motion(
                          ati.Time(t, format='mjd')).transform_to(acoo.GCRS)
    C0 = acoo.SkyCoord(ra=ra0 * auni.degree,
                       dec=dec0 * auni.degree,
                       distance=1e3 * auni.kpc,
                       pm_ra_cosdec=0 * auni.mas / auni.year,
                       pm_dec=0 * auni.mas / auni.year,
                       obstime=ati.Time(t0, format='mjd')).apply_space_motion(
                           ati.Time(t, format='mjd')).transform_to(acoo.GCRS)

    # since GCRS includes aberration + plx motion
    # here I'm subtracting the position placed at infinity

    def RESID(x, t, y):
        return np.sum((plx_mod(x, t) - y)**2)

    dra = ((C.ra - C0.ra).to_value(auni.mas) * np.cos(np.deg2rad(dec0)))
    ddec = (C.dec - C0.dec).to_value(auni.mas)

    guess0 = t[np.argmin(np.abs(dra))]
    p0 = [1, guess0, 0, guess0, 0.1]
    for i in range(3):
        R = scipy.optimize.minimize(RESID,
                                    p0,
                                    args=(t, dra),
                                    method='Nelder-Mead')
        if i == 2:
            break
        R = scipy.optimize.minimize(RESID, R['x'], args=(t, dra))
        p0 = R['x']
    p0 = [1, guess0, 0, guess0, .1]
    for i in range(3):
        R1 = scipy.optimize.minimize(RESID,
                                     p0,
                                     args=(t, ddec),
                                     method='Nelder-Mead')
        if i == 2:
            break
        R1 = scipy.optimize.minimize(RESID, R1['x'], args=(t, ddec))
        p0 = R1['x']
    ra_param = R['x']
    ra_param[-1] = 0
    dec_param = R1['x']
    dec_param[-1] = 0
    assert R['success']
    assert R1['success']
    print(R, R1)
    assert R['fun'] < 1e-4 * N * np.median(np.abs(dra))
    assert R1['fun'] < 1e-4 * N * np.median(np.abs(ddec))
    return ra_param, dec_param


def getlc_my(r0=1,
             r11=1,
             r12=1,
             rpa=0,
             vx=1,
             vy=0.1,
             x0=-2,
             y0=-2,
             transp=0.4,
             acoeff=0.55,
             bcoeff=0.15,
             xgrid=None,
             ygrid=None,
             mugrid=None,
             stepgrid=None,
             plxs=None,
             times=None,
             getMotion=False,
             t0=None):
    """
    This is mostly the wrapper around C++ code now evaluating the model

    Parameters are
    r0: float
        size of the giant
    r11: float
        major axis of the ellipse
    r12: float
        minor axis of the ellipse
    rpa: float
        angle of the ellipse
    vx: float
        velocity of the occultor
    vy: float
        velocity of the occultor
    x0: float
        position of the occultor at time t0
    y0: float
        position of the occultor at time t0
    t0: float
        reference time (in mjd scale)
    transp: float
        transparency of occultor
    acoeff: float
        the a coefficient in the quadratic limb darkening parameterization
    bcoeff: float
        the b coefficient in the quadratic limb darkening parameterization
    xgrid: numpy
        The array of grid positions on the face of the occulted star
    ygrid: numpy
        The array of grid positions on the face of the occulted star
    mugrid: numpy
        The array of mu=cos(phi) at grid positions on the face of the
        occulted star
    times: numpy
        The array of observation times (in mjd scale)
    velocities wrt giant, starting point, npts in the LC,
    limb darkening
    plx prop

    """
    curt = times - t0
    curx = x0 + vx * curt + plxs[0]
    cury = y0 + vy * curt + plxs[1]
    # penalty because of degeneracy between x0,y0,vx,vy  and time
    # so we softly enforce  the perpendicularity of X0,Y0 and VX, VY at t0
    penalty = 1e4 * (x0 * vx + y0 * vy)**2 / (vx**2 + vy**2) / (x0**2 + y0**2)

    if getMotion:
        return times, curx, cury

    res = lc_model_opt.getlc(curx, cury, transp, r0, r11, r12, rpa, acoeff,
                             bcoeff, xgrid, ygrid, mugrid, stepgrid)
    return res, penalty


def like(p, dat, mult=1, getModel=False):
    """ Log-likelihood function
    Parameter vector components
    0) log10 major axis of occultor
    1) log10 minor axis 
    2) PA of occultor
    3) log10 of the impact parameter 
    4) angle of the closest approach position
    5) velocity in vphi (tangential)
    6) acoeff of limb dark
    7) bcoeff of limb dark
    8) time at starting point (closest approach)
    9) log10 parallax
    10) log10 transparency
    11) Direction of motion parameter >.5 clockwise < .5 anticlockwise

    Arguments:
    p: numpy
        Parameter vector (below)
    dat: tuple
        tuple with various arrays for likelihood eval
        (mjd, flux, errors, xgrid, ygrid, mugrid, plxs)
    mult: float
        set to 1 for chi-square scale and to -0.5 for log-likelihood scale
    getModel: bool
        set to true to get prediction
    """
    (lr11, lr12, rpa0, lrimpact, phi0, vphi0, acoeff, bcoeff, t0, lplx,
     ltransp, direction) = p

    extra_err = 0.012  # systematic magnitude error
    plx = 10**lplx
    r11 = 10**lr11
    r12 = 10**lr12
    rimpact = 10**lrimpact
    transp = 10**ltransp
    x0 = rimpact * np.cos(phi0)
    y0 = rimpact * np.sin(phi0)
    # phi0 = np.arctan2(y0, x0)
    vphi = vphi0 * np.sign(direction - 0.5)
    vx = -vphi * np.sin(phi0)
    vy = vphi * np.cos(phi0)
    if vphi > 0:
        # rpa0 is angle towards the giant
        rpa = np.arctan2(vy, vx) + rpa0
    else:
        rpa = np.arctan2(vy, vx) - rpa0

    (times, val, err, xgrid, ygrid, mugrid, plxs,
     stepgrid) = (dat['mjd'], dat['flux'], dat['eflux'], dat['xgrid'],
                  dat['ygrid'], dat['mugrid'], dat['plxs'], dat['stepgrid'])
    if (acoeff > 1 or acoeff < 0 or bcoeff < 0 or (acoeff + bcoeff) > 1
            or  # unphysical limb dark 
            r11 < 0 or r12 > r11 or  # rmaj > rmin
            plx < 0 or transp < 0 or transp > 1):
        # print('oops a,b,plx,r11,r12,transp', acoeff, bcoeff, plx,
        # r11, r12rat, transp)
        return mult * (-2 * BADV)
    err1 = np.sqrt(err**2 + (np.log(10) / 2.5 * val * extra_err)**2)

    pred, penalty = getlc_my(r0=1,
                             r11=r11,
                             r12=r12,
                             rpa=rpa,
                             vx=vx,
                             vy=vy,
                             x0=x0,
                             y0=y0,
                             acoeff=acoeff,
                             bcoeff=bcoeff,
                             t0=t0,
                             transp=transp,
                             times=times,
                             xgrid=xgrid,
                             ygrid=ygrid,
                             mugrid=mugrid,
                             stepgrid=stepgrid,
                             plxs=(plxs[0] * plx, plxs[1] * plx))

    if getModel:
        return pred

    print_prob = 0.001  # 0.001 reporting prob
    plot_prob = 0
    rand = random.random()
    if rand < plot_prob:
        plt.clf()
        plt.plot(times, val, '.')
        plt.plot(times, pred)
        plt.xlim(55600, 56400)
        plt.ylim(-0.05, 1.1)
        plt.pause(0.001)
    chisq = (((pred - val) / err1)**2).sum()
    if dat['fake']:
        if pred[0] < 1:
            chisq = 0
        else:
            chisq = -2 * BADV
    logl = chisq + 2 * np.sum(np.log(err1))  # + penalty
    if rand < print_prob:
        print(logl, chisq, penalty, p)
    dlogjac = scipy.stats.beta(1, 0.5).logpdf(r12 / r11) - (-np.log(r12))
    dlogjac += -2 * np.log(r11)
    # dlogjac += np.log(rimpact)
    logl = logl - 2 * dlogjac

    return mult * logl


def logprior(p):
    """Logprior function
    Return very negative number if outside support
    """
    res = [_.logpdf(__) for _, __ in zip(priorList, p)]
    res = sum(res)
    if not np.isfinite(res):
        res = BADV
    return res


def rvsprior():
    """
    get one sample from the prior
    """
    res = [_.rvs() for _ in priorList]
    return res


def prior_transf(x):
    """
    Apply a CDF transform into a Unit cube to the parameter
    """
    res = [_(__) for _, __ in zip(priorList, x)]
    return res


def edge_wrap(f):
    """
    Protect against going outside the edges
    Parameters:
    f: function
        function that needs to be penalized by going outside the bounds
    Returns:
    wrapped function
    """
    def func(x, *args):
        res = []
        penalty = 0  # 1000
        for curp, curprior in zip(x, priorList):
            ledge, redge = curprior.ledge, redge = curprior.redge
            if curp <= ledge:
                curp = ledge + 1e-6 * (redge - ledge)
                penalty += 1000 + 1000 * ((curp - ledge) / (redge - ledge))**2
            if curp >= redge:
                curp = redge - 1e-6 * (redge - ledge)
                penalty += 1000 + 1000 * ((curp - redge) / (redge - ledge))**2
            res.append(curp)
        return f(res, *args) + penalty

    return func


def wrap_like(x):
    """
    Wrapper for loglikelihood function that uses cached additional likelihood
    arguments
    and return the actual log-likelihood (as opposed to chisqs)
    """
    return like(x, FFCache.xdat, -0.5)


def wrap_like_polycord(x):
    """
    Wrapper for loglikelihood function that uses cached additional likelihood
    arguments
    and return the actual log-likelihood (as opposed to chisqs)
    """
    return (like(x, FFCache.xdat, -0.5), [])


def wrap_post(x):
    """
    Wrapper for logposterior function that used cached additional likelihood
    arguments
    and return the actual log-posterior
    """
    return like(x, FFCache.xdat, -0.5) + logprior(x)


def wrap_post1(x):
    """
    Wrapper for logposterior returning chi-squares (that needs to be minimized)
    """
    ret = like(x, FFCache.xdat, 1) - 2 * logprior(x)
    if not np.isfinite(ret):
        return 1e30
    return ret


def getdata():
    """
    Retrieve OGLE data
    """
    tabi = atpy.Table().read('data/OGLE.I.dat', format='ascii')
    xind = betw(tabi['col1'], 0, 64000)
    tabi = tabi[xind]
    mjd = tabi['col1'] + 50000

    # use mags outside the event to establish a baseline
    maxmag = np.median(tabi['col2'][np.abs(mjd - 56000) > 500])

    # convert mag into relative flux
    flux = 10**((maxmag - tabi['col2']) / 2.5)

    # this is the excess scatter I see in the LC outside the event
    # I do the inflation inside the likelihood function
    inflate_err = 0  # NOTICE IT is zero because I apply it inside
    # likelihood function
    magerr = np.sqrt(inflate_err**2 + tabi['col3']**2)

    # error propagation
    eflux = magerr * flux * np.log(10) / 2.5

    # ensure that everything can be passed safely to c++ code
    def CONV(x):
        return np.ascontiguousarray(x, dtype=np.float64)

    return dict(mjd=CONV(mjd), flux=CONV(flux), eflux=CONV(eflux))


def get_xdat(ngrid=300, fake=False, times=None):
    if not fake:
        data = getdata()
    else:
        if times is None:
            data = dict(mjd=np.ascontiguousarray([56000]),
                        flux=np.ascontiguousarray([0]),
                        eflux=np.ascontiguousarray([1]))
        else:
            data = dict(mjd=np.ascontiguousarray(times),
                        flux=np.ascontiguousarray(times * 0),
                        eflux=np.ascontiguousarray(times * 0))

    xgrid, ygrid, mugrid, step = lc_model_opt.make_mugrid(ngrid)
    ra0, dec0 = 270.197564183775, -30.60257  # coord of the star
    ra_plx_param, dec_plx_param = getplx(ra0, dec0)
    plxs = plx_mod(ra_plx_param, data['mjd']), plx_mod(dec_plx_param,
                                                       data['mjd'])
    xdat = dict(mjd=data['mjd'],
                flux=data['flux'],
                eflux=data['eflux'],
                xgrid=xgrid,
                ygrid=ygrid,
                mugrid=mugrid,
                plxs=plxs,
                stepgrid=step,
                fake=fake)
    return xdat


def sampler_dynesty(pool=None, ngrid=300):
    """
    Sample using dynesty nested sampler
    """
    FFCache.xdat = get_xdat(ngrid=ngrid)
    if pool is None:
        pool = mp.Pool(36)
    import dynesty
    dsampler = dynesty.DynamicNestedSampler(
        wrap_like,
        prior_transf,
        periodic=[2, 4],
        ndim=12,
        pool=pool,
        queue_size=36,
        sample='rslice',
        #bootstrap=5,
        #vol_dec=0.2
        #sample='rslice',
        # use_pool={
        #    'loglikelihood': True,
        #    'evolve': False
        # }
    )
    dsampler.run_nested(dlogz_init=0.01,
                        nlive_init=5000,
                        nlive_batch=5000,
                        wt_kwargs={'pfrac': .9},
                        stop_kwargs={'pfrac': .9})
    return dsampler


def sampler_dynesty_schwim(ngrid=300):
    """
    Parallel tempering sampling with schwimmbad pool
    """
    import schwimmbad as S
    from idlsave import idlsave
    FFCache.xdat = get_xdat(ngrid=ngrid)

    pool = S.MPIPool()
    # pool = None
    R = sampler_dynesty(pool=pool, ngrid=ngrid)
    # Extract sampling results.
    samples = R.results.samples  # samples
    weights = np.exp(R.results.logwt -
                     R.results.logz[-1])  # normalized weights
    from dynesty import utils as dyfunc
    nsamp = len(samples)
    xids = np.arange(nsamp)
    xids_sub = dyfunc.resample_equal(xids, weights)
    chain = samples[xids_sub]
    logp = R.results.logl[xids_sub]
    idlsave.save('dynesty_result.psav', 'sampler,chain,logp', R, chain, logp)


def sampler_multinest(ofname='xx.psav',
                      fake=False,
                      ngrid=300,
                      nlive=5000,
                      resume=False):
    """
    Sampling using pymultinest sampler
    """
    FFCache.xdat = get_xdat(fake=fake, ngrid=ngrid)  #False)  #True)
    from pymultinest.solve import solve
    from pymultinest.analyse import Analyzer
    ndim = 11
    wraps = [0] * ndim
    wraps[2] = 1
    wraps[4] = 1
    prefix = 'chains/multinlc'
    res = solve(LogLikelihood=wrap_like,
                Prior=prior_transf,
                verbose=True,
                outputfiles_basename=prefix,
                n_dims=ndim,
                n_live_points=nlive,
                resume=resume,
                importance_nested_sampling=False,
                multimodal=True,
                log_zero=-1e9,
                sampling_efficiency=0.8,
                wrapped_params=wraps)
    from idlsave import idlsave
    analyzer = Analyzer(ndim, outputfiles_basename=prefix)
    logp = analyzer.get_equal_weighted_posterior()[:, -1]
    idlsave.save(ofname, 'chain,logp', res['samples'], logp)


def poly_dumper(live, dead, logweights, logZ, logZerr):
    print("Last dead point:", dead[-1])


def sampler_polychord():
    # Initialise the settings
    nDims = 11
    nDerived = 0
    FFCache.xdat = get_xdat()
    import pypolychord
    from idlsave import idlsave
    from pypolychord.settings import PolyChordSettings
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    settings = PolyChordSettings(nDims, nDerived, logzero=-1e10)
    settings.file_root = 'polyclc'
    settings.nlive = 10000
    settings.do_clustering = True
    settings.read_resume = True  # False
    names = [
        'logrmaj', 'axrat', 'PA', 'r0_2', 'phi0', 'vphi', 'acoeff', 'bcoeff',
        't0', 'lplx', 'ltransp'
    ]
    xnames = [(_, _) for _ in names]

    output = pypolychord.run_polychord(wrap_like_polycord, nDims, nDerived,
                                       settings, prior_transf, poly_dumper)
    output.make_paramnames_files(xnames)
    if rank == 0:
        idlsave.save('poly.psav', 'post,samples,logl', output.posterior,
                     output.samples, output.loglikes)


def sampler_ultranest():
    """
    Sampling using ultranest sampler
    """
    FFCache.xdat = get_xdat(1000)
    import ultranest
    import ultranest.stepsampler
    ndim = 11
    wraps = [0] * ndim
    wraps[2] = 1
    wraps[4] = 1
    names = [
        'logrmaj', 'logrmin', 'PA', 'r0_2', 'phi0', 'vphi', 'acoeff', 'bcoeff',
        't0', 'lplx', 'ltransp'
    ]

    res = ultranest.ReactiveNestedSampler(
        names,
        # res = ultranest.stepsampler.RegionSliceSampler(names,
        wrap_like,
        prior_transf,
        wrapped_params=wraps,
        log_dir='ultra',
        storage_backend='hdf5')
    result = res.run()
    res.print_results()
    from idlsave import idlsave
    idlsave.save('xx1.psav', 'res', result)


def sampler_pt_schwim():
    """
    Parallel tempering sampling with schwimmbad pool
    """
    import schwimmbad as S
    from idlsave import idlsave
    FFCache.xdat = get_xdat()

    pool = S.MPIPool()
    # pool = None
    R = sampler_pt(pool, nit=40000, fresh_start=False)
    logp = R.logprobability
    chain = R.chain
    idlsave.save('pt_result.psav', 'chain,logp', chain, logp)


def sampler_pt(pool=None,
               nit=2000,
               nthreads=12,
               nw=500,
               fresh_start=True,
               ngrid=100,
               repeats=1):
    """
    Parallel tempering sampling
    """
    import ptemcee

    FFCache.xdat = get_xdat(ngrid=ngrid)
    if pool is None:
        pool = mp.Pool(nthreads)

    ntemp = 5
    ndim = 12
    if not fresh_start:
        with open('pars_pt.pkl', 'rb') as fp:
            p0 = pickle.load(fp)
        # p0 = idlsave.restore('pars.psav', 'pars')[0][:nw * ntemp]
    else:
        p0 = np.array([rvsprior() for _ in range(nw * ntemp)])
        p0 = p0.reshape(ntemp, nw, -1)
    # print(p0)
    # print(p0.shape)
    # [wrap_like(_) for _ in p0]
    samp = ptemcee.Sampler(nw,
                           ndim,
                           wrap_like,
                           logprior,
                           ntemps=ntemp,
                           pool=pool)
    for i in range(repeats):
        if i > 0:
            p0 = samp.chain[:, :, -1, :]
            samp.reset()
        samp.run_mcmc(p0, nit)
    return samp


def sampler_prior(pool=None):
    # run sampling
    FFCache.xdat = get_xdat()
    if pool is None:
        pool = mp.Pool(12)
    import ptemcee
    nw = 200  # 500
    ntemp = 5
    nit = 10000
    with open('pars_pt.pkl', 'rb') as fp:
        p0 = pickle.load(fp)
    # p0 = idlsave.restore('pars.psav', 'pars')[0][:nw * ntemp]
    p0 = np.array([rvsprior() for _ in range(nw * ntemp)])
    # print(p0)
    # print(p0.shape)
    # [wrap_like(_) for _ in p0]
    p0 = p0.reshape((ntemp, nw, -1))
    samp = ptemcee.Sampler(nw, 12, logprior, logprior, ntemps=ntemp, pool=pool)
    samp.run_mcmc(p0, nit)
    return samp


def sampler_zeus_schwim():
    """
    Sample with zeus using schwimmbad pool
    """
    import schwimmbad as S
    from idlsave import idlsave

    FFCache.xdat = get_xdat()

    pool = S.MPIPool()
    R = sampler_zeus(pool)
    logp = R.get_log_prob()
    chain = R.get_chain()
    idlsave.save('zeus.psav', 'chain,logp', chain, logp)


def sampler_zeus(pool=None, nit=1000, fresh_start=True, nw=500, ngrid=300):
    """
    Sample using zeus sampler
    """
    import zeus

    FFCache.xdat = get_xdat(ngrid=ngrid)
    if pool is None:
        pool = mp.Pool(36)
    ndim = 12
    if fresh_start:
        p0 = np.array([rvsprior() for _ in range(nw)])
    else:
        with open('pars_zeus.pkl', 'rb') as fp:
            p0 = pickle.load(fp)
    import zeus.moves
    samp = zeus.EnsembleSampler(
        nw,
        ndim,
        wrap_post,
        pool=pool,
        maxiter=int(1e7),
        # moves=[
        #                                    zeus.moves.DifferentialMove(),
        # zeus.moves.KDEMove(),
        #                                    zeus.moves.GlobalMove(),
        #                                    zeus.moves.GaussianMove()
        # ]
    )
    samp.run_mcmc(p0, nit)
    return samp


def plotter_saver():
    """
    Evaluate the model and save it
    """
    xdat = get_xdat()
    FFCache.xdat = xdat

    p0 = [
        2.14928399e+00, 6.30221940e-01, 2.75166059e+00, -1.96259549e+00,
        7.03808465e-01, 5.26047000e-03, 1.49757697e-02, 5.81073410e-01,
        1.59915284e-01, 5.60375907e+04, 9.44655438e-01, 5.89994829e-03
    ]
    # low plx solution
    p0 = [
        1.96702792e+00, 7.26357060e-01, 2.51915752e+00, -8.37697825e-01,
        6.87868987e-01, 1.18811095e-02, 1.46097125e-02, 6.77618428e-01,
        2.33218856e-01, 5.60220478e+04, 1.46077603e-04, 1.94347621e-05
    ]
    if True:
        M = like(p0, xdat, -0.5, getModel=True)
        print(wrap_post(p0))
        tab = atpy.Table()
        tab['times'] = xdat['mjd']
        tab['flux'] = xdat['flux']
        tab['eflux'] = xdat['eflux']
        tab['xplx'] = xdat['plxs'][0]
        tab['yplx'] = xdat['plxs'][1]
        tab['mod'] = M
        tab.write('xx.fits', overwrite=True)
        # write the data


def optimizer():
    """
    Do a single optimization run
    """
    xdat = get_xdat()
    FFCache.xdat = xdat

    p0 = [10, .5, 0.5 * np.pi, -10, 1, -0.01, .01, 0.55, 0.15, 56000, 0.1, 0.1]
    p0 = [
        18.01, 0.22, 0.1, -17.18, 1.14, 0.0013, 0.019, 0.55, 0.15,
        56075.83169385737, 0.17, 0.01, 0.01
    ]

    p0 = [
        18.400779846117267, 0.22971787544099842, 0.10155253120188,
        -17.419509553338237, 1.10935718731397, 0.001337832080287821,
        0.02101724622679121, 0.5999999, 0.17088848513491867, 56152.7311204481,
        0.19911041030960425, 0.004994250357441403, 0.012459448104491906
    ]
    p0 = [
        1.87415482e+01,
        2.10336313e-01,
        2.88965103e-02,
        -1.80146378e+01,
        1.46305331e+00,
        1.49149247e-03,
        1.83705929e-02,
        5.93203255e-01,
        1.95010188e-01,
        5.61257260e+04,
        3.63695628e-01,
        1.63649945e-05,
    ]  # 1.23094872e-02]
    for i in range(10):

        ret = scipy.optimize.minimize(edge_wrap(like),
                                      p0,
                                      args=(xdat, 1),
                                      method='Nelder-Mead')
        p0 = ret['x']
        ret = scipy.optimize.minimize(
            edge_wrap(like),
            p0,
            args=(xdat, 1),
        )
        p0 = ret['x']


def rand_optimizer(niter=1000):
    """
    Start from a random point in the prior and start optimizing. Do it niter
    times.
    Returns:
    List of best-logl and best points

    """
    xdat = get_xdat()
    FFCache.xdat = xdat
    goods = []
    np.random.seed()
    for j in range(niter):
        p0 = rvsprior()
        bestfun = 1e9
        niter0 = 10
        for i in range(niter0):
            ret = scipy.optimize.minimize(wrap_post1, p0, method='Nelder-Mead')
            p0 = ret['x']
            fun = ret['fun']
            if (fun > bestfun - 1):
                break
            if fun < bestfun:
                bestfun = fun
        # ret = scipy.optimize.minimize(
        #    wrap_post1,
        #    p0
        # )
        # p0 = ret['x']
        goods.append((fun, p0))
    return goods
