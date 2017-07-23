#ifndef ICTRENTO
#define ICTRENTO
#include <vector>

class ICTrento{
  public:
    ICTrento(char* fileName, EoS *_eos, int IC_Nxy, int IC_Neta, double ic_dxy, double ic_deta);
    void setIC(Fluid *f, double tau);

  private:
    EoS *eos;
    const int IC_NX_, IC_NY_, IC_Neta_;  // initial condition grid size
    const double IC_dx_, IC_dy_, IC_deta_;  // initial condition grid step
    const double IC_xmin_, IC_ymin_, IC_zmin_;
    std::vector<double> t00vec_;
    std::vector<double> t01vec_;
    std::vector<double> t02vec_;
    std::vector<double> t03vec_;

    void readFile(char* fileName, EoS *eos);
    int findIndex(int ix, int iy, int ieta);
    double interpolate(double x, double y, double eta, std::vector<double> *tvec);
    void setQ(double tau, double x, double y, double eta, double* Q);
};

#endif
