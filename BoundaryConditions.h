#pragma once

#include <map>
#include <cassert>

#include "SimInfo.h"

namespace fv3d {
  namespace {
  /**
   * @brief Absorbing conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillAbsorbing(Array Q, int iref, int jref, int kref) {
    return getStateFromArray(Q, iref, jref, kref);
  };

  /**
   * @brief Reflecting boundary conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillReflecting(Array Q, int i, int j, int k, int iref, int jref, int kref, IDir dir, const Params &params) {
    int isym, jsym, ksym;
    if (dir == IX) {
      int ipiv = (i < iref ? params.ibeg : params.iend);
      isym = 2*ipiv - i - 1;
      jsym = j;
      ksym = k;
    }
    else if (dir == IY) {
      int jpiv = (j < jref ? params.jbeg : params.jend);
      isym = i;
      jsym = 2*jpiv - j - 1;
      ksym = k;
    }
    else {
      int kpiv = (k < kref ? params.kbeg : params.kend);
      isym = i;
      jsym = j;
      ksym = 2*kpiv - k - 1;
    }

    State q = getStateFromArray(Q, isym, jsym, ksym);
  
    if (dir == IX)
      q[IU] *= -1.0;
    else if (dir == IY)
      q[IV] *= -1.0;
    else 
      q[IW] *= -1.0;

    return q;
  }

  /**
   * @brief Periodic boundary conditions
   * 
   */
  KOKKOS_INLINE_FUNCTION
  State fillPeriodic(Array Q, int i, int j, int k, IDir dir, const Params &params) {
    if (dir == IX) {
      if (i < params.ibeg)
        i += params.Nx;
      else
        i -= params.Nx;
    }
    else if (dir == IY) {
      if (j < params.jbeg)
        j += params.Ny;
      else
        j -= params.Ny;
    }
    else {
      if (k < params.kbeg)
        k += params.Nz;
      else
        k -= params.Nz;
    }

    return getStateFromArray(Q, i, j, k);
  }

  /**
   * @brief C91 bounary conditions
   */
  KOKKOS_INLINE_FUNCTION
  State fillC91(Array Q, int i, int j, int k, int kref, IDir dir, const Params &params) {
    if (dir != IZ)
      return {}; // Should not be called on a direction that is not vertical
    
    int kpiv = (k < kref ? params.kbeg : params.kend);
    int ksym = 2*kpiv - k - 1;
    
    Pos pos = getPos(params, i, j, k);
    State qref = getStateFromArray(Q, i, j, kref);
    State qsym = getStateFromArray(Q, i, j, ksym);

    State res;
    if (k < params.kbeg) // top
    {
      real_t dz = pos[IZ] - 0.5*params.dz;
      real_t T  = 1.0 + pos[IZ]*params.theta1;

      res[IP] = qref[IP] + dz * qref[IR] * params.gravity;
      res[IR] = res[IP] / T;

      res[IP] = Kokkos::fmax(res[IP], 1.0e-6);
      res[IR] = Kokkos::fmax(res[IR], 1.0e-6);

      const real_t dratio = qsym[IR] / res[IR];
      res[IU] =  qsym[IU];
      res[IV] =  qsym[IV];
      res[IW] = -qsym[IW] * dratio;
    }

    else if (k >= params.kend) // bottom
    {
      const real_t Tref = qref[IP]/qref[IR];
      const real_t dz = pos[IZ] - (1.0 - 0.5*params.dz);

      const real_t T = Tref + dz * params.theta1;
      const real_t P = qref[IP] + dz * qref[IR]*params.gravity;
      const real_t rho = P/T;

      res[IP] = Kokkos::fmax(P, 1.0e-6);
      res[IR] = Kokkos::fmax(rho, 1.0e-6);
      const real_t dratio = qsym[IR]/rho;
      res[IU] =  qsym[IU]*dratio;
      res[IV] =  qsym[IV]*dratio;
      res[IW] = -qsym[IW]*dratio;
    }

    return res;
  }
} // anonymous namespace


class BoundaryManager {
public:
  Params params;

  BoundaryManager(const Params &params) 
    : params(params) {};
  ~BoundaryManager() = default;

  void fillBoundaries(Array Q) {
    auto bc_x = params.boundary_x;
    auto bc_y = params.boundary_y;
    auto bc_z = params.boundary_z;
    auto params = this->params;

    Kokkos::parallel_for( "Filling X-boundary",
                          params.range_xbound,
                          KOKKOS_LAMBDA(int i, int j, int k) {

                            int ileft     = i;
                            int iright    = params.iend+i;
                            int iref_left = params.ibeg;
                            int iref_right = params.iend-1;

                            auto fill = [&](int i, int iref) {
                              switch (bc_x) {
                                case BC_ABSORBING:  return fillAbsorbing(Q, iref, j, k); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, k, iref, j, k, IX, params); break;
                                default:            return fillPeriodic(Q, i, j, k, IX, params); break;
                              }
                            };

                            setStateInArray(Q, ileft,  j, k, fill(ileft, iref_left));
                            setStateInArray(Q, iright, j, k, fill(iright, iref_right));
                          });

    Kokkos::parallel_for( "Filling Y-boundary",
                          params.range_ybound,
                          KOKKOS_LAMBDA(int i, int j, int k) {

                            int jtop     = j;
                            int jbot     = params.jend+j;
                            int jref_top = params.jbeg;
                            int jref_bot = params.jend-1;

                            auto fill = [&](int j, int jref) {
                              switch (bc_y) {
                                case BC_ABSORBING:  return fillAbsorbing(Q, i, jref, k); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, k, i, jref, k, IY, params); break;
                                default:            return fillPeriodic(Q, i, j, k, IY, params); break;
                              }
                            };

                            setStateInArray(Q, i, jtop, k, fill(jtop, jref_top));
                            setStateInArray(Q, i, jbot, k, fill(jbot, jref_bot));
                          });

    Kokkos::parallel_for( "Filling Z-boundary",
                          params.range_zbound,
                          KOKKOS_LAMBDA(int i, int j, int k) {

                            int kfront     = k;
                            int kback      = params.kend+k;
                            int kref_front = params.kbeg;
                            int kref_back  = params.kend-1;

                            auto fill = [&](int k, int kref) {
                              switch (bc_z) {
                                case BC_ABSORBING:  return fillAbsorbing(Q, i, j, kref); break;
                                case BC_REFLECTING: return fillReflecting(Q, i, j, k, i, j, kref, IZ, params); break;
                                case BC_C91:        return fillC91(Q, i, j, k, kref, IZ, params); break;
                                default:            return fillPeriodic(Q, i, j, k, IZ, params); break;
                              }
                            };

                            setStateInArray(Q, i, j, kfront, fill(kfront, kref_front));
                            setStateInArray(Q, i, j, kback,  fill(kback,  kref_back));
                          });
  }
};

}