#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "fld.h"
#include "eos.h"
#include "inc.h"
#include "rmn.h"
#include "icTrento.h"
#include "eoChiral_s.h"

using namespace std;

ICTrento::ICTrento(char* fileName, EoS *_eos, int ic_nxy, int ic_neta, double ic_dxy, double ic_deta)
:   IC_NX_(ic_nxy),
    IC_NY_(ic_nxy),
    IC_Neta_(ic_neta),
    IC_dx_(ic_dxy),
    IC_dy_(ic_dxy),
    IC_deta_(ic_deta),
    IC_xmin_(-0.5*(ic_nxy-1)*ic_dxy),
    IC_ymin_(-0.5*(ic_nxy-1)*ic_dxy),
    IC_zmin_(-0.5*(ic_neta -1)*ic_deta)
{
    eos = _eos;
    readFile(fileName, eos);
}


int ICTrento::findIndex(int ix, int iy, int ieta)
{
    return IC_Neta_ * IC_NY_ * ix + IC_Neta_ * iy + ieta;
//      return (IC_NX_ * IC_NY_) * ieta + ix*IC_NY_ + iy;
}


double ICTrento::interpolate(double x, double y, double eta, std::vector<double> *tvec)
// a 1-d interpolate (though I am quite curious, it seems like vhlle is dealing with 1d vectors instead of 3d)
{
    int ix = std::floor((x - IC_xmin_)/IC_dx_);
    int iy = std::floor((y - IC_ymin_)/IC_dy_);
    int ieta = std::floor((eta - IC_zmin_)/IC_deta_);

    if (ix < 0 || ix >= (IC_NX_-1) || iy<0 || iy>= (IC_NY_-1) || ieta<0 || ieta>= (IC_Neta_-1))
        return 0.0;

    double sx = x - IC_xmin_ - ix * IC_dx_;
    double sy = y - IC_ymin_ - iy * IC_dy_;
    double sz = eta - IC_zmin_ - ieta * IC_deta_;

    double wx[2] = {1 - sx/IC_dx_, sx/IC_dx_};
    double wy[2] = {1 - sy/IC_dy_, sy/IC_dy_};
    double wz[2] = {1 - sz/IC_deta_, sz/IC_deta_};

    double result = 0.0;
    for (int jx=0; jx<2; jx++){
      for (int jy=0; jy<2; jy++){
        for (int jz=0; jz<2; jz++){
            int i = findIndex(ix+jx, iy+jy, ieta+jz);
                result += wx[jx] * wy[jy] * wz[jz] * (*tvec)[i];
        }

      }
    }
    return result;
}


void ICTrento::setQ(double tau, double x, double y, double eta, double *Q)
{
    double t00, t01, t02, t03;
    t00 = interpolate(x, y, eta, &t00vec_);
    t01 = interpolate(x, y, eta, &t01vec_);
    t02 = interpolate(x, y, eta, &t02vec_);
    t03 = interpolate(x, y, eta, &t03vec_);

    Q[0] = t00;
    Q[1] = t01;
    Q[2] = t02;
    Q[3] = t03 * tau;
    Q[4] = 0.0;
    Q[5] = 0.0;
    Q[6] = 0.0;
}



void ICTrento::setIC(Fluid *f, double tau)
{
    double e = 0., p=0., nb=0., nq=0., ns = 0.;
    double vx = 0., vy = 0., vz = 0.;
    double Q[7];

    Cell *c;
    double avv_num = 0., avv_den = 0.;
    double Etotal = 0.;
    for (int ix=0; ix < f->getNX(); ix++){
      for (int iy=0; iy < f->getNY(); iy++){
        for (int iz=0; iz < f->getNZ(); iz++){
            c = f->getCell(ix, iy, iz);
            double x = f->getX(ix);
            double y = f->getY(iy);
            double eta = f->getZ(iz);
            setQ(tau, x, y, eta, Q);
            transformPV(eos, Q, e, p, nb, nq, ns, vx, vy, vz);
            avv_num += sqrt(vx*vx + vy*vy) * e;
            avv_den += e;
            c->setPrimVar(eos, tau, e, nb, nq, 0., vx, vy, vz);

            double _p = eos->p(e, nb, nq, 0.);
            const double gamma2 = 1./ (1. - vx*vx - vy*vy - vz*vz);
            Etotal += ((e+_p)*gamma2 * (cosh(eta) + vz*sinh(eta)) - _p*cosh(eta));

            c->saveQprev();
            if (e>0.) c->setAllM(1.);
        }
      }
    }

    std::cout << "Initial total energy: " << Etotal << ", avv_num: " << avv_num << ", avv_den: " << avv_den << std::endl;
    std::cout << "Initial condition set up successfully :)" << std::endl;
}




void ICTrento::readFile(char *fileName, EoS *eos)
{
    std::ifstream fin(fileName);
    if (!fin)
    {
        std::cout << "ERROR! File " << fileName << " not found." << std::endl;
        exit(1);
    }

    int ix, iy, ieta;
    double  sd, ed, nb, t00, t01, t02, t03, check;
    int line = 0;

    for (int ix=0; ix < IC_NX_; ix++){
      for (int iy = 0; iy < IC_NY_; iy++){
        for (int ieta = 0; ieta < IC_Neta_; ieta++){
            fin >> sd;
            nb = 0.;

            if(sd < 1.0e-8) {ed = 0.;}
            else eos->gete(sd, ed, nb);

            t00 = ed;
            t01 = 0.0;
            t02 = 0.0;
            t03 = 0.0;

            t00vec_.push_back(t00);
            t01vec_.push_back(t01);
            t02vec_.push_back(t02);
            t03vec_.push_back(t03);
        }
        line ++;
      }
    }
}


