#include <iostream>
#include <cstdlib>
#include "lc_model_opt.hpp"
#include <cmath>
using namespace std;

void test()
{
    int N = 3000;
    double xc[N];
    double yc[N];
    double res[N];

    for (int i = 0; i < N; i++)
    {
        xc[i] = 200 * i * 1. / N - 100;
        yc[i] = 0;
        //      fprintf(stderr, "Q %lf %lf %lf\n", fluxes[i], times[i],
        // errs[i]);
    }
    double transp = 0.5;
    double r0 = 10;
    double rmaj = 1;
    double rmin = 1;
    double pa = 0;
    int ngrid = 300;
    double alimb = .4;
    double blimb = .4;
    int nit = 1000;
    int nover = 10;
    double *xgrid, *ygrid, *mugrid;
    int ngrid2;
    double step;
    xgrid = (double *)malloc(ngrid * ngrid * sizeof(double));
    ygrid = (double *)malloc(ngrid * ngrid * sizeof(double));
    mugrid = (double *)malloc(ngrid * ngrid * sizeof(double));

    make_mugrid(ngrid, nover, xgrid, ygrid, mugrid, &ngrid2, &step);

    for (int i = 0; i < nit; i++)
    {

        getlc(xc, yc, N, transp, r0, rmaj, rmin, pa, alimb, blimb, xgrid, ygrid,
              mugrid, ngrid2, step, res);
        // getlc(xc,yc, N, transp, r0, rmaj, rmin, pa, ngrid, alimb, blimb,
        // res);
        // cout<<res[i]<<endl;
    }
}

int main()
{
    test();
}
