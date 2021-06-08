#include <cmath>
#include <cstdlib>
#include <vector>
#include <array>
#include <tuple>
#include <iostream>
#include <cassert>

using namespace std;

double getfrac(const double val1, const double val2, const double val3,
               const double val4)
{
    // if all of them are smaller than 1 return 1
    // if some are are bigger than 1 return
    // (1-mind)/(max-mind)
    // if all are bigger than 1 return 0
    double maxv = -1e30;
    double minv = 1e30;
    if (val1 < minv)
    {
        minv = val1;
    }
    if (val2 < minv)
    {
        minv = val2;
    }
    if (val3 < minv)
    {
        minv = val3;
    }
    if (val4 < minv)
    {
        minv = val4;
    }
    if (minv > 1)
        return 0;
    if (val1 > maxv)
    {
        maxv = val1;
    }
    if (val2 > maxv)
    {
        maxv = val2;
    }
    if (val3 > maxv)
    {
        maxv = val3;
    }
    if (val4 > maxv)
    {
        maxv = val4;
    }
    if (maxv < 1)
        return 1;
    return (1 - minv) / (maxv - minv);
}

bool circle_inside_ellipse(double r0, double x, double y, double rmaj, double rmin)
{
  // return true if circle with radius r0
  // is fully inside an ellipse centered at x,y
  // with semimajor/minor axis rmaj/rmin aligned along x,y
  
  //const double x1 =x/rmaj, y1=y/rmin, x_r0=r0/rmaj, r0/rmin;
  const double val1 = pow((x-r0)/rmaj,2)+pow(y/rmin,2);
  const double val2 = pow((x+r0)/rmaj,2)+pow(y/rmin,2);
  const double val3 = pow(x/rmaj,2)+pow((y-r0)/rmin,2);
  const double val4 = pow(x/rmaj,2)+pow((y+r0)/rmin,2);
  // the check finds if 4 points on the circle are inside the ellipse
  // then the whole thing is
  if ((val1<1) && (val2<1) && (val3<1) && (val4<1))
    {
      return true;
    }
  return false;
}

void make_mugrid(const int ngrid, const int overbin, double *x, double *y,
                 double *mu, int *npix, double *ret_step)
{
    /* make a grid of x,y mu=cos(phi) sampling the star
       The arguments:
       ngrid -- number of gridpts along one axis
       overbin -- the oversampling factor for a pixel
       x -- array of output x's (must be preallocated)
       y -- array of output y's (must be allocated)
       mu -- array of cos(phi)=mu (must be preallocated)
       npix -- output number of nonzero pixels
       Importantly at the edges I'm uniformly spreading the flux
       across the pixel
     */
    const int N = ngrid * overbin;
    double grid[N];
    vector<double> res0(N * N);
    vector<double> x0(N * N);
    vector<double> y0(N * N);
    const double step = 2. / N;
    *ret_step = step;
    // create 1d grid from -1 to 1
    for (int i = 0; i < N; i++)
    {
        grid[i] = -1 + 0.5 * step + i * step;
    }
    *npix = 0;
    //#pragma omp parallel for default(none) shared(grid, res0, x0, y0)
    // The reason for commenting the OMP stuff
    /// is https://bisqwit.iki.fi/story/howto/openmp/#OpenmpAndFork
    // you can't fork after openmp

    for (int i = 0; i < N; i++)
    {
        const double curx = grid[i];
        for (int j = 0; j < N; j++)
        {
            const double cury = grid[j];
            const double curd2 = (curx * curx + cury * cury);
            // This is square distance from the center of the disk
            // This is also sin^2(phi)
            const int i1 = N * i + j;
            x0[i1] = curx;
            y0[i1] = cury;
            double curres;
            if (curd2 >= 1)
            {
                curres = 0;
            }
            else
            {
                curres = sqrt(1 - curd2); // cos(phi)
            }
            res0[i1] = curres;
        }
    }

    for (int i = 0; i < ngrid; i++)
    {
        for (int j = 0; j < ngrid; j++)
        {
            double cursum = 0;
            double curxsum = 0;
            double curysum = 0;
            // accumulators  for x,y and flux
            int curcnt = 0;
            // here I average of grid pts
            for (int k1 = 0; k1 < overbin; k1++)
            {
                for (int k2 = 0; k2 < overbin; k2++)
                {
                    const int curi = i * overbin + k1, curj = j * overbin + k2;
                    const int curi1 = N * curi + curj;

                    // Importantly I average over empty pixels too
                    curcnt += 1;
                    cursum += res0[curi1];
                    curxsum += x0[curi1];
                    curysum += y0[curi1];
                }
            }

            if (cursum == 0)
            {
                continue;
            }
            x[*npix] = curxsum / curcnt;
            y[*npix] = curysum / curcnt;
            mu[*npix] = cursum / curcnt;
            *npix += 1;
        }
    }
}

void getlc(const double *xc0, const double *yc0, const int N,
           const double transp, const double r0, const double rmaj,
           const double rmin, const double pa, const double alimb,
           const double blimb, const double *xgrid0, const double *ygrid0,
           const double *mugrid, const int ngrid, const double step,
           double *res)
{
    /*
       Obtain a LC
       Arguments:
       xc -- array of x coordinates of obscurer for which the LC is computed
       yc -- array of y coordinates of obscurer
       N -- number of coordinates
       transp -- transparency (between 0 and 1)
       r0 -- size of the star
       rmaj -- major axis of the obscuring ellipse
       rmin -- minor axis of the ellipse
       pa -- positional angle of the ellipse (radians)
       alimb -- 1st limb darkening coeff for quadratic ld
       blimb -- 2nd limb darkening coeff for quadratic ld
       xgrid0 -- the grid of x's of pts sampling the star
       ygrid0 -- the grid of ys of pts sampling the star
       mugrid -- the grid of mu=cos(phi) sampling the star
       ngrid -- the number of gridpoints
       step -- step size  of the grid
       res -- the output array (must be preallocated)
     */
    const double transp1 = 1 - transp;
    const double r0_pad = r0 * (1 + step);

    const double minsep2 = pow(fmax(rmin - r0_pad, 0), 2);
    // the minimum separation^2, below which the star will be
    // fully covered, only works if rmin>r0
    // I padded it by the pixel step size

    const double cpa = cos(pa);
    const double spa = sin(pa);

    // This a quick look to quickly stop if ther is no overlap
    int outside_counter = 0;
    for (int i = 0; i < N; i++)
    {
        // iterating over the light curve points

        // projecting coordinates of the obscurer
        // to elliptical coord system
        const double curxc = xc0[i] * cpa + yc0[i] * spa;
        const double curyc = yc0[i] * cpa - xc0[i] * spa;

        // bounding box check
        if ((fabs(curxc) > (r0_pad + rmaj)) || (fabs(curyc) > (r0_pad + rmin)))
        {
            res[i] = 1;
            outside_counter += 1;
            continue;
        }
    }

    // if all the pts are outside we quickly leave
    if (outside_counter == N)
    {
        return;
    }

    vector<double> fgrid(ngrid); // The brightness of the star gridpoints
    // the coordinates of the obscurer in the major axis aligned coord sys

    // the coordinates of the star gridpoints in the maj ax aligned
    // and scaled coord system
    vector<double> xgrid_scale(ngrid);
    vector<double> ygrid_scale(ngrid);
    // these are the corners
    vector<double> xgrid1_scale(ngrid);
    vector<double> ygrid1_scale(ngrid);
    vector<double> xgrid2_scale(ngrid);
    vector<double> ygrid2_scale(ngrid);
    vector<double> xgrid3_scale(ngrid);
    vector<double> ygrid3_scale(ngrid);
    vector<double> xgrid4_scale(ngrid);
    vector<double> ygrid4_scale(ngrid);

    assert(rmaj >= rmin);
    assert(transp <= 1);
    assert(transp >= 0);
    assert((alimb + blimb) >= 0);
    assert((alimb + blimb) <= 1);

    double totbri = 0;
// note constants have to be declared according to gcc10
// and for sharing constants it is better to use firstprivate instead
// of shared (see
// https://docs.oracle.com/cd/E19059-01/stud.10/819-0501/7_tuning.html)
#pragma omp parallel for default(none) shared(mugrid, fgrid)                   \
    firstprivate(ngrid, alimb, blimb) reduction(+ : totbri)
    for (int i = 0; i < ngrid; i++)
    {
        const double mu1 = 1 - mugrid[i];
        // Use quadratic limb darkening 1 - a * (1-mu) - b (1-mu)^2
        double curb = 1 - alimb * mu1 - blimb * mu1 * mu1;
        fgrid[i] = curb;
        totbri += curb; // total brightness  accumulator
    }

    const double r0_rmaj = r0 / rmaj;
    const double r0_rmin = r0 / rmin;
    // threshold for the (x/a)^2+(y/b)^2<1 condition when
    // checking if the pixel is inside or not
    // since I'm transforming the grid into 1/rmaj 1/rmin scale
    // the pixel size is r0*step/rmin or r0*step/rmax
    const double minthresh = pow(fmax(1 - step * r0 / rmin, 0.), 2);
    const double maxthresh = pow(1 + step * r0 / rmin, 2);
#pragma omp parallel for shared(xgrid1_scale, ygrid1_scale, xgrid2_scale,      \
                                ygrid2_scale, xgrid3_scale, ygrid3_scale,      \
                                xgrid4_scale, ygrid4_scale, xgrid_scale,       \
                                ygrid_scale, fgrid, xgrid0, ygrid0,            \
                                totbri) firstprivate(ngrid, r0_rmaj, r0_rmin)
    for (int i = 0; i < ngrid; i++)
    {
        fgrid[i] = fgrid[i] / totbri; // fraction of flux in the given gridpt
        // take into accout the r0, as the original
        // grid is -1...1
        // scaling coord sys, no need for rotation
        xgrid_scale[i] = xgrid0[i] * r0_rmaj;
        ygrid_scale[i] = ygrid0[i] * r0_rmin;
        xgrid1_scale[i] = (xgrid0[i] + .5 * step) * r0_rmaj;
        ygrid1_scale[i] = (ygrid0[i] + .5 * step) * r0_rmin;
        xgrid2_scale[i] = (xgrid0[i] + .5 * step) * r0_rmaj;
        ygrid2_scale[i] = (ygrid0[i] - .5 * step) * r0_rmin;
        xgrid3_scale[i] = (xgrid0[i] - .5 * step) * r0_rmaj;
        ygrid3_scale[i] = (ygrid0[i] + .5 * step) * r0_rmin;
        xgrid4_scale[i] = (xgrid0[i] - .5 * step) * r0_rmaj;
        ygrid4_scale[i] = (ygrid0[i] - .5 * step) * r0_rmin;
    }

#pragma omp parallel for default(none)                                         \
    shared(xgrid_scale, ygrid_scale, xgrid1_scale, ygrid1_scale, xgrid2_scale, \
           ygrid2_scale, xgrid3_scale, ygrid3_scale, xgrid4_scale,             \
           ygrid4_scale, fgrid, res, xc0, yc0)                                 \
    firstprivate(N, spa, cpa, minthresh, maxthresh, minsep2, transp1, ngrid,   \
                 transp, rmaj, rmin, r0) schedule(static, 8)
    for (int i = 0; i < N; i++)
    {
        // iterating over the light curve points

        // projecting coordinates of the obscurer
        // to elliptical coord system
        const double curxc = xc0[i] * cpa + yc0[i] * spa;
        const double curyc = yc0[i] * cpa - xc0[i] * spa;

        const double d2 = curxc * curxc + curyc * curyc;
        if (d2 < minsep2)
        {
            // this is full coverage
            res[i] = transp;
            continue;
        }
        // bounding box check
        if ((fabs(curxc) > (r0 + rmaj)) || (fabs(curyc) > (r0 + rmin)))
        {
            res[i] = 1;
            continue;
        }
        const double curxc_scale = curxc / rmaj;
        const double curyc_scale = curyc / rmin;

        double curres = 0;
	// accumulator for  flux that went through
	// the obscurer
        for (int j = 0; j < ngrid; j++)
        {
            const double curdx = xgrid_scale[j] - curxc_scale,
                         curdy = ygrid_scale[j] - curyc_scale;
            const double rat = curdx * curdx + curdy * curdy;
            if (rat > maxthresh)
            {
                continue;
            } // far outside
            if (rat < minthresh)
            {
                curres += fgrid[j];
                continue;
            } // far inside
            const double curdx1 = xgrid1_scale[j] - curxc_scale,
                         curdy1 = ygrid1_scale[j] - curyc_scale;
            const double curdx2 = xgrid2_scale[j] - curxc_scale,
                         curdy2 = ygrid2_scale[j] - curyc_scale;
            const double curdx3 = xgrid3_scale[j] - curxc_scale,
                         curdy3 = ygrid3_scale[j] - curyc_scale;
            const double curdx4 = xgrid4_scale[j] - curxc_scale,
                         curdy4 = ygrid4_scale[j] - curyc_scale;
            const double rat1 = curdx1 * curdx1 + curdy1 * curdy1;
            const double rat2 = curdx2 * curdx2 + curdy2 * curdy2;
            const double rat3 = curdx3 * curdx3 + curdy3 * curdy3;
            const double rat4 = curdx4 * curdx4 + curdy4 * curdy4;
            const double frac = getfrac(rat1, rat2, rat3, rat4);
            curres += frac * fgrid[j];
        }
        res[i] = 1 - curres * transp1;
    }
}

extern "C" {
void cgetlc(const double *xc, const double *yc, const int N,
            const double transp, const double r0, const double rmaj,
            const double rmin, const double pa, const double alimb,
            const double blimb, const double *xgrid, const double *ygrid,
            const double *mugrid, const int ngrid, const double step,
            double *res)
{
    getlc(xc, yc, N, transp, r0, rmaj, rmin, pa, alimb, blimb, xgrid, ygrid,
          mugrid, ngrid, step, res);
}

void cmake_mugrid(const int ngrid, const int overbin, double *x, double *y,
                  double *mu, int *npix, double *step)
{
    make_mugrid(ngrid, overbin, x, y, mu, npix, step);
}
}
