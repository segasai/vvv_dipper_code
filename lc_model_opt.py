import numpy as np
from cffi import FFI

ffi = FFI()
ffi.cdef("""
void cgetlc(const double *xc, const double *yc, const int N, const double transp,
           const double r0, const double rmaj, const double rmin,
           const double pa, const double alimb,
           const double blimb, const double *xgrid,
	   const double *ygrid,
	   const double *mugrid, 
	   const int ngrid, const double step, double *res);
  void cmake_mugrid(const int ngrid,
                  const int overbin, double *x, double *y,
                  double *mu, int *npix, double *step);

""")
lib = ffi.dlopen("./liblc_model_opt.so")


def make_mugrid(ngrid, overgrid=10):
    N = ngrid * ngrid
    x = np.zeros(N, dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    mu = np.zeros(N, dtype=np.float64)
    nout = np.zeros(1, dtype=int)
    step = np.zeros(1, dtype=np.float64)
    x_ffi = ffi.cast('double *', ffi.from_buffer(x))
    y_ffi = ffi.cast('double *', ffi.from_buffer(y))
    mu_ffi = ffi.cast('double *', ffi.from_buffer(mu))
    nout_ffi = ffi.cast('int *', ffi.from_buffer(nout))
    step_ffi = ffi.cast('double *', ffi.from_buffer(step))
    lib.cmake_mugrid(ngrid, overgrid, x_ffi, y_ffi, mu_ffi, nout_ffi, step_ffi)
    nout = nout[0]
    return np.ascontiguousarray(x[:nout]), np.ascontiguousarray(
        y[:nout]), np.ascontiguousarray(mu[:nout]), step


def getlc(xc, yc, transp, r0, rmaj, rmin, pa, alimb, blimb, xgrid, ygrid,
          mugrid, stepgrid):
    """

    """
    N = len(xc)
    assert (len(yc) == N)

    xc, yc = [np.ascontiguousarray(_, dtype=np.float64) for _ in [xc, yc]]
    xgrid, ygrid, mugrid = [
        np.ascontiguousarray(_, dtype=np.float64)
        for _ in [xgrid, ygrid, mugrid]
    ]

    lc = np.zeros(N, dtype=np.float64)
    xc_ffi = ffi.cast('double *', ffi.from_buffer(xc))
    yc_ffi = ffi.cast('double *', ffi.from_buffer(yc))
    lc_ffi = ffi.cast('double *', ffi.from_buffer(lc))
    ngrid = len(xgrid)

    xgrid_ffi = ffi.cast('double *', ffi.from_buffer(xgrid))
    ygrid_ffi = ffi.cast('double *', ffi.from_buffer(ygrid))
    mugrid_ffi = ffi.cast('double *', ffi.from_buffer(mugrid))
    lib.cgetlc(xc_ffi, yc_ffi, N, transp, r0, rmaj, rmin, pa, alimb, blimb,
               xgrid_ffi, ygrid_ffi, mugrid_ffi, ngrid, stepgrid, lc_ffi)
    return lc
