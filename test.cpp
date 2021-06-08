#include <iostream>
#include <cstdlib>
#include "lc_model_opt.hpp"
#include <cmath>
using namespace std;

void test()
{
    int N = 30;
    double xc[N];
    double yc[N];
    double res[N];

    for (int i = 0; i < N; i++)
    {
        xc[i] = 20 * i * 1. / N - 10;
        yc[i] = 0;
        //      fprintf(stderr, "Q %lf %lf %lf\n", fluxes[i], times[i],
        // errs[i]);
    }
    double transp = 0.5;
    double r0 = 5;
    double rmaj = 1;
    double rmin = 1;
    double pa = 0;
    int ngrid = 100;
    double alimb = .4;
    double blimb = .4;
    int nover = 10;
    double xgrid[ngrid * ngrid];
    double ygrid[ngrid * ngrid];
    double mugrid[ngrid * ngrid];
    double step;
    int ngrid2;
    make_mugrid(ngrid, nover, xgrid, ygrid, mugrid, &ngrid2, &step);
    getlc(xc, yc, N, transp, r0, rmaj, rmin, pa, alimb, blimb, xgrid, ygrid,
          mugrid, ngrid2, step, res);
    for (int i = 0; i < N; i++)
    {
        cout << res[i] << endl;
    }
}

int main()
{
    test();
}
