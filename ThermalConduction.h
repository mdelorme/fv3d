#pragma once

#include "SimInfo.h"

namespace fv2d {

KOKKOS_INLINE_FUNCTION
real_t computeKappa(real_t x, real_t y, real_t z, const Params &params) {
  real_t res;
  switch (params.thermal_conductivity_mode) {
    case TCM_B02:
    {
      const real_t tr = (tanh((z-params.b02_zmid)/params.b02_thickness) + 1.0) * 0.5;
      res = params.kappa * (params.b02_kappa1 * (1.0-tr) + params.b02_kappa2 * tr);
      break;
    }
    default:
      res = params.kappa;
  }

  return res;
}

class ThermalConductionFunctor {
public:
  Params params;

  ThermalConductionFunctor(const Params &params) 
    : params(params) {};
  ~ThermalConductionFunctor() = default;

  void applyThermalConduction(Array Q, Array Unew, real_t dt) {
    auto params = this->params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    Kokkos::parallel_for(
      "Thermal conduction", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        Pos pos = getPos(params, i, j, k);
        real_t x = pos[IX];
        real_t y = pos[IY];
        real_t z = pos[IZ];

        real_t kappaL = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x-dx, y, z, params));
        real_t kappaR = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x+dx, y, z, params));
        real_t kappaU = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x, y-dy, z, params));
        real_t kappaD = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x, y+dy, z, params));
        real_t kappaF = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x, y, z-dz, params));
        real_t kappaB = 0.5 * (computeKappa(x, y, z, params) + computeKappa(x, y, z+dz, params));

        // Ideal EOS with R = 1 assumed. T = P/rho
        real_t TC = Q(k, j, i,   IP) / Q(k, j, i,   IR);
        real_t TL = Q(k, j, i-1, IP) / Q(k, j, i-1, IR);
        real_t TR = Q(k, j, i+1, IP) / Q(k, j, i+1, IR);
        real_t TU = Q(k, j-1, i, IP) / Q(k, j-1, i, IR);
        real_t TD = Q(k, j+1, i, IP) / Q(k, j+1, i, IR);
        real_t TF = Q(k-1, j, i, IP) / Q(k-1, j, i, IR);
        real_t TB = Q(k+1, j, i, IP) / Q(k+1, j, i, IR);

        // Computing thermal flux
        real_t FL = kappaL * (TC - TL) / dx;
        real_t FR = kappaR * (TR - TC) / dx;
        real_t FU = kappaU * (TC - TU) / dy;
        real_t FD = kappaD * (TD - TC) / dy;
        real_t FF = kappaF * (TC - TF) / dz;
        real_t FB = kappaB * (TB - TC) / dz;

        /** 
         * Boundaries treatment
         * IMPORTANT NOTE :
         * To be accurate, in the case of fixed temperature, since the temperature is taken at the interface
         * the value of kappa should either be averaged between the cell-centered value and the interface
         * or be evaluated at x=0.25dx / x=xmax-0.25dx
         */
        if (k==params.kbeg && params.bctc_zmin != BCTC_NONE) {
          switch (params.bctc_zmin) {
            case BCTC_FIXED_TEMPERATURE: FF = kappaF * 2.0 * (TC-params.bctc_zmin_value) / dz; break;
            case BCTC_FIXED_GRADIENT:    FF = kappaF * params.bctc_zmin_value; break;
            default: break;
          }
        }

        if (k==params.kend-1 && params.bctc_zmax != BCTC_NONE) {
          switch (params.bctc_zmax) {
            case BCTC_FIXED_TEMPERATURE: FB = kappaB * 2.0 * (params.bctc_zmax_value-TC) / dz; break;
            case BCTC_FIXED_GRADIENT:    FB = kappaB * params.bctc_zmax_value; break;       
            default: break;
          }
        }

        // And updating using a Godunov-like scheme
        Unew(k, j, i, IE) += dt/dx * (FR - FL) + dt/dy * (FD - FU) + dt/dz * (FB - FF);
      });
  }
};

}