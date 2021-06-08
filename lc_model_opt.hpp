void make_mugrid(const int ngrid, const int overbin, double *x, double *y,
                 double *mu, int *npix, double *step);
void getlc(const double *xc, const double *yc, const int N, const double transp,
           const double r0, const double rmaj, const double rmin,
           const double pa, const double alimb, const double blimb,
           const double *xgrid, const double *ygrid, const double *mugrid,
           const int ngrid, const double step, double *res);
