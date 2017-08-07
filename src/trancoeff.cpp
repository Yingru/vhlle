#include <iostream>
#include <iomanip>
#include "eos.h"
#include "trancoeff.h"
#include "inc.h"

double Tc = 0.154;
double A1 = -13.77, A2 = 27.55, A3 = -13.45;
double C1=0.03, C2 = 0.001;
double lambda1 = 0.9, lambda2 = 0.25, lambda3 = 0.9, lambda4 = 0.22;
double sigma1 = 0.025, sigma2 = 0.13, sigma3 = 0.0025, sigma4 = 0.022;


TransportCoeff::TransportCoeff(double _etaS, double _etaS_slope, double _zetaS, EoS *_eos) {
 etaS_min = _etaS;
 etaS_slope = _etaS_slope;
 zetaS0 = _zetaS;

 eos = _eos;
}

void TransportCoeff::printZetaT()
{
 std::cout << "------zeta/s(T):\n";
 for(double e=0.1; e<3.0; e+=0.1){
  double T, mub, muq, mus, p;
  eos->eos(e, 0., 0., 0., T, mub, muq, mus, p);
  std::cout << std::setw(14) << T << std::setw(14) << zetaS(e, T) << std::endl;
 }
 std::cout << "---------------:\n";
}

double TransportCoeff::zetaS(double e, double T)
{
   double x = T/0.18;
   double bulkVisc;
   if (x < 0.995) 
       bulkVisc = lambda3*exp((x-1.0)/sigma3) + lambda4*exp((x-1.0)/sigma4) + C1;
   else if (x > 1.05)
       bulkVisc = lambda1*exp((1.0-x)/sigma1) + lambda2*exp((1.0-x)/sigma2) + C2;
   else
       bulkVisc = A1*x*x + A2*x + A3;
   
   return zetaS0 * bulkVisc;

  //return zetaS0 * (1. / 3. - eos->cs2(e)) / (exp((0.16 - T) / 0.001) + 1.);
}

double TransportCoeff::etaS(double T)
{
    return etaS_min + (T-Tc) * etaS_slope;
}


void TransportCoeff::getEta(double e, double T, double &_etaS, double &_zetaS) {
   _etaS = etaS(T);
   _zetaS = zetaS(e,T);
}

void TransportCoeff::getTau(double e, double T, double &_taupi, double &_tauPi) {
 if (T > 0.)
  _taupi = std::max(5. / 5.068 * etaS(T) / T, 0.003);
 else
  _taupi = 0.1;
 if (T > 0.)
  _tauPi = std::max(1. / 5.068 * zetaS(e,T) / (15. * pow(0.33333-eos->cs2(e),2) * T), 0.005);
 else
  _tauPi = 0.1;
}
