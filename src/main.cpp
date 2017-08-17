/******************************************************************************
*                                                                             *
*            vHLLE : a 3D viscous hydrodynamic code                           *
*            version 1.1,            October 2014                            *
*            by Iurii Karpenko                                                *
*  contact:  yu.karpenko@gmail.com                                            *
*  For the detailed description please refer to:                              *
*  http://arxiv.org/abs/1312.4160                                             *
*                                                                             *
*  This code can be freely used and redistributed, provided that this         *
*  copyright appear in all the copies. If you decide to make modifications    *
*  to the code, please contact the authors, especially if you plan to publish *
* the results obtained with such modified code. Any publication of results    *
* obtained using this code must include the reference to                      *
* arXiv:1312.4160 [nucl-th] or the published version of it, when available.   *
*                                                                             *
*******************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <sstream>
#include "fld.h"
#include "hdo.h"
#include "eos.h"
#include "eoChiral.h"
#include "eoHadron.h"
#include "trancoeff.h"

#include "icTrento.h"
#include "eos_hotqcd.h"

using namespace std;

// program parameters, to be read from file
int nx, ny, nz, eosType;
double xmin, xmax, ymin, ymax, etamin, etamax, tau0, tauMax, dtau;

int ic_nxy, ic_neta;
double ic_dxy, ic_deta;

char outputDir[255];
char icInputFile[255];
char eosFile[255];
double etaS, etaS_slope, zetaS, eCrit;

int icModel,
    glauberVariable =
        1;  // icModel=1 for pure Glauber, 2 for table input (Glissando etc)
double epsilon0, Rgt, Rgz, impactPar, s0ScaleFactor;

void readParameters(char *parFile) {
 char parName[255], parValue[255];
 ifstream fin(parFile);
 if (!fin.is_open()) {
  cout << "cannot open parameters file " << parFile << endl;
  exit(1);
 }
 while (fin.good()) {
  string line;
  getline(fin, line);
  istringstream sline(line);
  sline >> parName >> parValue;
  if (strcmp(parName, "outputDir") == 0)
   strcpy(outputDir, parValue);
  else if (strcmp(parName, "eosType") == 0)
   eosType = atoi(parValue);
  else if (strcmp(parName, "icInputFile") == 0)
   strcpy(icInputFile, parValue);
  else if (strcmp(parName, "eosFile") == 0)
   strcpy(eosFile, parValue);
  else if (strcmp(parName, "nx") == 0)
   nx = atoi(parValue);
  else if (strcmp(parName, "ny") == 0)
   ny = atoi(parValue);
  else if (strcmp(parName, "nz") == 0)
   nz = atoi(parValue);
  else if (strcmp(parName, "icModel") == 0)
   icModel = atoi(parValue);
  else if (strcmp(parName, "glauberVar") == 0)
   glauberVariable = atoi(parValue);
  else if (strcmp(parName, "xmin") == 0)
   xmin = atof(parValue);
  else if (strcmp(parName, "xmax") == 0)
   xmax = atof(parValue);
  else if (strcmp(parName, "ymin") == 0)
   ymin = atof(parValue);
  else if (strcmp(parName, "ymax") == 0)
   ymax = atof(parValue);
  else if (strcmp(parName, "etamin") == 0)
   etamin = atof(parValue);
  else if (strcmp(parName, "etamax") == 0)
   etamax = atof(parValue);
  else if (strcmp(parName, "tau0") == 0)
   tau0 = atof(parValue);
  else if (strcmp(parName, "tauMax") == 0)
   tauMax = atof(parValue);
  else if (strcmp(parName, "dtau") == 0)
   dtau = atof(parValue);
  else if (strcmp(parName, "e_crit") == 0)
   eCrit = atof(parValue);
  else if (strcmp(parName, "etaS") == 0)
   etaS = atof(parValue);
  else if (strcmp(parName, "etaS_slope") == 0)
   etaS_slope = atof(parValue);
  else if (strcmp(parName, "zetaS") == 0)
   zetaS = atof(parValue);
  else if (strcmp(parName, "epsilon0") == 0)
   epsilon0 = atof(parValue);
  else if (strcmp(parName, "Rg") == 0)
   Rgt = atof(parValue);
  else if (strcmp(parName, "Rgz") == 0)
   Rgz = atof(parValue);
  else if (strcmp(parName, "impactPar") == 0)
   impactPar = atof(parValue);
  else if (strcmp(parName, "s0ScaleFactor") == 0)
   s0ScaleFactor = atof(parValue);
  else if (strcmp(parName, "ic_nxy") == 0)
   ic_nxy = atoi(parValue);
  else if (strcmp(parName, "ic_neta") == 0)
   ic_neta = atoi(parValue);
  else if (strcmp(parName, "ic_dxy") == 0)
   ic_dxy = atof(parValue);
  else if (strcmp(parName, "ic_deta") == 0)
   ic_deta = atof(parValue);

  else if (parName[0] == '!')
   cout << "CCC " << sline.str() << endl;
  else
   cout << "UUU " << sline.str() << endl;
 }
}

void printParameters() {
 cout << "\n====== parameters ======\n";
 cout << "outputDir = " << outputDir << endl;
 cout << "icFile = " << icInputFile << endl;
 cout << "eosFile = " << eosFile << endl;
 cout << "icModel = " << icModel << endl;
 cout << "[ic_nxy, ic_neta, ic_dxy, ic_deta] = [" << ic_nxy << ", " << ic_neta << ", " << ic_dxy << ", " << ic_deta << "]" << endl;
 cout << "glauberVar = " << glauberVariable << "   ! 0=epsilon,1=entropy" << endl;
 cout << "[nx, ny, nz] = [" << nx<< ", " << ny << ", " << nz  <<"]" << endl;
 cout << "[(xmin, xmax)*(ymin, ymax)*(etamin, etamax)] = [(" << xmin << ", " << xmax <<")*(" << ymin << ", " << ymax << ")*(" << etamin << ", " << etamax << ")]" << endl;
 cout << "[tau0, tauMax, dtau] = [" << tau0 << ", " << tauMax << ", " << dtau << "]" << endl;
 cout << "eosType = " << eosType << endl;
 cout << "e_crit = " << eCrit << endl;
 cout << "eta/s = " << etaS  <<", eta/s_slope = " << etaS_slope <<  endl;
 cout << "zeta/s = " << zetaS << endl;
 //cout << "epsilon0 = " << epsilon0 << endl;
 //cout << "Rgt = " << Rgt << "  Rgz = " << Rgz << endl;
 //cout << "impactPar = " << impactPar << endl;
 //cout << "s0ScaleFactor = " << s0ScaleFactor << endl;
 cout << "======= end parameters =======\n\n";
}

// program parameters, to be read from file
// int nx, ny, nz, eosType ;
// double xmin, xmax, ymin, ymax, zmin, zmax, tau0, tauMax, dtau ;
// char outputDir[255], eosFile[255], chiBfile[255], chiSfile[255] ;
// char icInputFile [255] ;
// double T_ch, mu_b, mu_q, mu_s, gammaS, gammaFactor, exclVolume ;
// int icModel, NPART, glauberVariable=1 ;
// double epsilon0, alpha, impactPar, s0ScaleFactor ;

int main(int argc, char **argv) {
 // pointers to all the main objects
 EoS *eos;
 TransportCoeff *trcoeff;
 Fluid *f;
 Hydro *h;
 time_t start = 0, end;

 time(&start);

 // read parameters from file
 char *parFile;
 if (argc == 1) {
  cout << "NO PARAMETERS, exiting\n";
  exit(1);
 } else {
  parFile = argv[1];
 }
 readParameters(parFile);
 printParameters();

 // EoS
 if (eosType == 1)
  eos = new EoSChiral();
 else if (eosType == 3)
  eos = new EoS_hotqcd(eosFile);
 else {
  cout << "eosType != 1,3\n";
  return 0;
 }
 EoS *eosH = new EoSHadron("eos/eosHadronLog.dat");

 // transport coefficients
 trcoeff = new TransportCoeff(etaS, etaS_slope, zetaS, eos);

 //fluid 
 f = new Fluid(eos, eosH, trcoeff, nx, ny, nz, xmin, xmax, ymin, ymax, etamin,
               etamax, dtau, eCrit);
 cout << "fluid allocation done\n";

 // initilal conditions-- Yingru: currently only implement the readin IC type
 if (icModel == 5) {
  ICTrento *ic = new ICTrento(icInputFile, eos, ic_nxy, ic_neta, ic_dxy, ic_deta);
  ic->setIC(f, tau0);
  delete ic;
 } else {
  cout << "icModel = " << icModel << " not implemented\n";
 }
 cout << "IC done\n";

 time_t tinit = 0;
 time(&tinit);
 float diff = difftime(tinit, start);
 cout << "Init time = " << diff << " [sec]" << endl;

 // hydro init
 h = new Hydro(f, eos, trcoeff, tau0, dtau);
 int maxstep = ceil((tauMax - tau0) / dtau);
 start = 0;
 time(&start);
 // h->setNSvalues() ; // initialize viscous terms

 f->initOutput(outputDir, maxstep, tau0, 2);
 f->outputCorona(tau0);

  for (int istep = 0; istep < maxstep; istep++) {
  // output the medium information
      f->outputMedium_h5(h->getTau(), istep);

  // decrease timestep automatically, but use fixed dtau for output
      int nSubSteps = 1;
      while (dtau / nSubSteps > 1.0 * (tau0 + dtau * istep) * (etamax - etamin) / (nz - 1))
          nSubSteps *= 2;  // 0.02 in "old" coordinates
      h->setDtau(dtau / nSubSteps);
      // cout<<"dtau = "<<dtau / nSubSteps<<endl;
      for (int j = 0; j < nSubSteps; j++) 
           h->performStep();

      // f->outputGnuplot(h->getTau()); ignore Gnuplot ignore Gnuplot
      f->outputSurface(h->getTau());
      //f->outputMedium_h5(h->getTau(), istep);
  }

 end = 0;
 time(&end);
 float diff2 = difftime(end, start);
 cout << "Execution time = " << diff2 << " [sec]" << endl;

 delete f;
 delete h;
 delete eos;
 delete eosH;
}
