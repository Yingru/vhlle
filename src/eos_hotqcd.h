#include <vector>
#include <string>

class EoS;

class EoS_hotqcd: public EoS
{
  private:
    std::vector<double> etab, stab, ptab, ttab;
    double e_min, devene;

  public:
    EoS_hotqcd(char* fileName);
    ~EoS_hotqcd(void);

    virtual void eos(double e, double nb, double nq, double ns,
        double &_T, double &_mub, double &_muq, double &_mus, double &_p);
    virtual double p(double e, double nb, double nq, double ns);
    virtual void gete(double s, double& e, double nb);
    virtual double cs2(double e);
};
