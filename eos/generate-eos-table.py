#!/usr/bin/env python3

import argparse
import subprocess

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


__doc__ = """
Generate a hydodynamic equation of state (EOS) table for VISHNew by blending
the UrQMD hadron resonance gas EOS at low temperature into the HotQCD lattice
EOS at high temperature.  Prints a table with columns (e, p, s, T) as required
by VISHNew.  All units are in GeV and fm, as appropriate.
"""

# http://physics.nist.gov/cgi-bin/cuu/Value?hbcmevf
HBARC = 0.1973269718  # GeV fm


def hotqcd_p_T4(T, Tc=0.154, pid=95*np.pi**2/180,
                ct=3.8706, an=-8.7704, bn=3.9200, cn=0, dn=0.3419,
                t0=0.9761, ad=-1.2600, bd=0.8425, cd=0, dd=-0.0475):
    """
    Evaluate p/T^4 for the HotQCD EOS.
    See Eq. (16) and Table II in http://inspirehep.net/record/1307761.

    """
    t = T/Tc
    return .5 * (1 + np.tanh(ct*(t-t0))) * \
        (pid + an/t + bn/t**2 + cn/t**3 + dn/t**4) / \
        (1   + ad/t + bd/t**2 + cd/t**3 + dd/t**4)


def hotqcd_e3p_T4(T, *args, **kwargs):
    """
    Evaluate the trace anomaly (e-3p)/T^4 for the HotQCD EOS.
    See Eq. (5) in http://inspirehep.net/record/1307761:

        (e-3p)/T^4 = T * d/dT(p/T^4)

    """
    # numerically differentiate via interpolating spline
    spl = Spline(T, hotqcd_p_T4(T, *args, **kwargs))
    return T*spl(T, nu=1)


def hrg(Tstep=0.01, Tmax=0.2):
    """
    Return a HRG EOS table as generated by hrg.pl, with columns:

        T   (e-3p)/T^4   e/T^4   p/T^4   c_s^2

    """
    cmd = './hrg.pl --gev --Tstep={} --Tmax={}'.format(Tstep*1000, Tmax*1000)
    with subprocess.Popen(cmd.split(), stdout=subprocess.PIPE) as proc:
        x = np.array(
            [l.split() for l in proc.stdout if not l.startswith(b'#')],
            dtype=float
        )
    return x


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--min', type=float, default=0.050,
                        help='minimum temperature')
    parser.add_argument('--blend-low', type=float, default=0.110,
                        help='lower bound temperature for blending')
    parser.add_argument('--blend-high', type=float, default=0.130,
                        help='upper bound temperature for blending')
    parser.add_argument('--max', type=float, default=0.600,
                        help='maximum temperature')
    parser.add_argument('--nsteps', type=int, default=155500,
                        help='number of energy density steps')
    parser.add_argument('--plot', action='store_true',
                        help='plot thermodynamic quantities')
    args = parser.parse_args()

    # read temperature points from arguments
    T_min = args.min
    T_blend_low = args.blend_low
    T_blend_high = args.blend_high
    T_max = args.max

    # get HRG EOS data
    hrg_T, hrg_e3p_T4 = hrg(Tstep=0.001, Tmax=T_blend_high).T[:2]

    # divide HRG data into parts below and within the overlap region
    hrg_low_cut = hrg_T <= T_blend_low
    hrg_mid_cut = (T_blend_low < hrg_T) & (hrg_T < T_blend_high)

    # construct table of (T, (e-3p)/T^4) points

    # use pure HRG for T below overlap region
    T_low = hrg_T[hrg_low_cut]
    e3p_T4_low = hrg_e3p_T4[hrg_low_cut]

    # blend HRG and HotQCD for T within overlap region
    T_mid = hrg_T[hrg_mid_cut]
    hrg_e3p_T4_mid = hrg_e3p_T4[hrg_mid_cut]
    hotqcd_e3p_T4_mid = hotqcd_e3p_T4(T_mid)
    # join the two curves using the "smoothstep" function
    # https://en.wikipedia.org/wiki/Smoothstep
    w = np.linspace(0, 1, T_mid.size)
    ss = w*w*(3 - 2*w)
    e3p_T4_mid = hrg_e3p_T4_mid*(1-ss) + hotqcd_e3p_T4_mid*ss

    # use pure HotQCD for T above overlap region
    T_high = np.linspace(T_blend_high, T_max, 10000)
    e3p_T4_high = hotqcd_e3p_T4(T_high)

    # concatenate the three segments together
    blend_T = np.concatenate([T_low, T_mid, T_high])
    blend_e3p_T4 = np.concatenate([e3p_T4_low, e3p_T4_mid, e3p_T4_high])

    # create interpolating functions for (e-3p)/T^4 and p/T^4
    e3p_T4_interp = Spline(blend_T, blend_e3p_T4)
    p_T4_interp = Spline(blend_T, blend_e3p_T4/blend_T).antiderivative()

    if args.plot:
        import matplotlib.pyplot as plt

        plt.rc('lines', linewidth=1.5)

        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(6, 15))

        T_plot = np.linspace(T_blend_low - 0.005, T_blend_high + 0.005, 1000)

        hrg_T_plot, hrg_e3p_T4_plot = hrg(Tstep=0.001, Tmax=T_plot.max()).T[:2]

        ax0.plot(hrg_T_plot, hrg_e3p_T4_plot,
                 label='HRG', color=plt.cm.coolwarm(.1))

        ax0.plot(T_plot, hotqcd_e3p_T4(T_plot), label='HotQCD',
                 color=plt.cm.coolwarm(.9))

        e3p_T4_plot = e3p_T4_interp(T_plot)
        ax0.plot(T_plot, e3p_T4_plot, label='blend',
                 color='.3', dashes=[5, 2])

        ax0.set_xlim(T_plot.min(), T_plot.max())
        ax0.set_ylim(e3p_T4_plot.min(), e3p_T4_plot.max())
        ax0.set_xlabel(r'$T$ [GeV]')
        ax0.set_title(r'blending trace anomaly $(\epsilon-3p)/T^4$')
        ax0.legend(loc='best')

        T_plot = np.linspace(0, T_max, 1000)
        e3p_T4 = e3p_T4_interp(T_plot)
        p_T4 = p_T4_interp(T_plot)
        e_T4 = e3p_T4 + 3*p_T4

        for y, label, cmap in [
                (e3p_T4, r'$(\epsilon-3p)/T^4$', plt.cm.Blues),
                (e_T4, r'$\epsilon/T^4$', plt.cm.Greens),
                (p_T4, r'$p/T^4$', plt.cm.Oranges),
        ]:
            ax1.plot(T_plot, y, color=cmap(0.8), label=label)

        ax1.set_xlabel(r'$T$ [GeV]')
        ax1.set_ylim(ymin=0)
        ax1.set_title('blended EOS thermodynamic quantities')
        ax1.legend(loc='best')

        e = e_T4*T_plot**4
        p = p_T4*T_plot**4
        cs2 = Spline(e, p)(e, nu=1)
        ax2.plot(T_plot, cs2, color=plt.cm.Blues(0.8))

        ax2.set_xlabel(r'$T$ [GeV]')
        ax2.set_ylabel(r'$c_s^2$')
        ax2.set_ylim(0, 1/3)
        ax2.set_title('blended EOS speed of sound')

        fig.tight_layout(pad=.2, h_pad=1.)
        plt.show()

        return

    # convert temperature [GeV] to energy density [GeV/fm^3]
    def e_interp(T):
        return (e3p_T4_interp(T) + 3*p_T4_interp(T))*T**4 / HBARC**3

    # and the inverse
    T_interp = Spline(e_interp(blend_T), blend_T)

    # compute pressure from energy density
    def p_interp(e):
        T = T_interp(e)
        return p_T4_interp(T)*T**4 / HBARC**3

    # compute entropy density from energy density
    def s_interp(e):
        T = T_interp(e)
        return (e3p_T4_interp(T) + 4*p_T4_interp(T))*T**3 / HBARC**3

    # generate evenly-spaced energy density values
    e = np.linspace(e_interp(T_min), e_interp(T_max), args.nsteps)

    # compute pressure, entropy density, temperature for each energy density
    p = p_interp(e)
    s = s_interp(e)
    T = T_interp(e)

    # output table
    for row in zip(e, p, s, T):
        print(*('{:E}'.format(i) for i in row))


if __name__ == "__main__":
    main()
