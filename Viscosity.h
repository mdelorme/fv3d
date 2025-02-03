#pragma once

#include "SimInfo.h"

namespace fv3d {

KOKKOS_INLINE_FUNCTION
real_t computeMu(int i, int j, int k, const Params &params) {
  switch (params.viscosity_mode) {
    default: return params.mu; break;
  }
}

class ViscosityFunctor {
public:
  Params params;

  ViscosityFunctor(const Params &params) 
    : params(params) {};
  ~ViscosityFunctor() = default;

  void applyViscosity(Array Q, Array Unew, real_t dt) {
    auto params = this->params;
    const real_t dx = params.dx;
    const real_t dy = params.dy;
    const real_t dz = params.dz;

    Kokkos::parallel_for(
      "Viscosity",
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        Pos pos = getPos(params, i, j, k);
        real_t x = pos[IX];
        real_t y = pos[IY];
        real_t z = pos[IZ];

        State stencil[3][3][3];

        // Computing viscous fluxes
        constexpr real_t four_thirds = 4.0/3.0;
        constexpr real_t two_thirds  = 2.0/3.0;

        auto fillStencil = [&](int i, int j, int k) -> void {
          for (int di=-1; di < 2; ++di)
            for (int dj=-1; dj < 2; ++dj)
              for (int dk=-1; dk < 2; ++dk)
                stencil[dk+1][dj+1][di+1] = getStateFromArray(Q, i+di, j+dj, k+dk);
        };

        auto computeViscousFlux = [&](IDir dir) {
          State flux {0.0, 0.0, 0.0, 0.0, 0.0};

          // Here X is the normal component and Y the tangential
          const real_t one_over_dx = 1.0/dx;
          const real_t one_over_dy = 1.0/dy;
          const real_t one_over_dz = 1.0/dz;

          real_t mu = computeMu(i, j, k, params);
          fillStencil(i, j, k);

          for (int side=1; side < 3; ++side) {
            real_t sign = (side == 1 ? -1.0 : 1.0);
            
            if (dir == IX) {
              State qi = 0.5 * (stencil[1][1][side] + stencil[1][1][side-1]);

              real_t dudx = one_over_dx * (stencil[1][1][side][IU] - stencil[1][1][side-1][IU]);
              real_t dvdx = one_over_dx * (stencil[1][1][side][IV] - stencil[1][1][side-1][IV]);
              real_t dwdx = one_over_dx * (stencil[1][1][side][IW] - stencil[1][1][side-1][IW]);

              real_t dudy = 0.25 * one_over_dy * (stencil[1][2][side  ][IU] - stencil[1][0][side  ][IU]
                                                + stencil[1][2][side-1][IU] - stencil[1][0][side-1][IU]);
              real_t dvdy = 0.25 * one_over_dy * (stencil[1][2][side  ][IV] - stencil[1][0][side  ][IV]
                                                + stencil[1][2][side-1][IV] - stencil[1][0][side-1][IV]);
              real_t dudz = 0.25 * one_over_dz * (stencil[2][1][side  ][IU] - stencil[0][1][side  ][IU]
                                                + stencil[2][1][side-1][IU] - stencil[0][1][side-1][IU]);
              real_t dwdz = 0.25 * one_over_dz * (stencil[2][1][side  ][IW] - stencil[0][1][side  ][IW]
                                                + stencil[2][1][side-1][IW] - stencil[0][1][side-1][IW]);

              const real_t tau_xx = four_thirds * dudx - two_thirds * (dvdy + dwdz);
              const real_t tau_xy = dvdx + dudy;
              const real_t tau_xz = dwdx + dudz;

              flux[IU] += sign * mu * tau_xx;
              flux[IV] += sign * mu * tau_xy;
              flux[IW] += sign * mu * tau_xz;
              flux[IE] += sign * mu * (tau_xx*qi[IU] + tau_xy*qi[IV] + tau_xz*qi[IW]);
            }
            else if (dir == IY) {
              State qi = 0.5 * (stencil[1][side][1] + stencil[1][side-1][1]);

              real_t dudy = one_over_dy * (stencil[1][side][1][IU] - stencil[1][side-1][1][IU]);
              real_t dvdy = one_over_dy * (stencil[1][side][1][IV] - stencil[1][side-1][1][IV]);
              real_t dwdy = one_over_dy * (stencil[1][side][1][IW] - stencil[1][side-1][1][IW]);

              real_t dudx = 0.25 * one_over_dx * (stencil[1][side  ][2][IU] - stencil[1][side  ][0][IU]
                                               +  stencil[1][side-1][2][IU] - stencil[1][side-1][0][IU]);
              real_t dvdx = 0.25 * one_over_dx * (stencil[1][side  ][2][IV] - stencil[1][side  ][0][IV]
                                               +  stencil[1][side-1][2][IV] - stencil[1][side-1][0][IV]);
              real_t dwdz = 0.25 * one_over_dz * (stencil[2][side  ][1][IW] - stencil[0][side  ][1][IW]
                                               +  stencil[2][side-1][1][IW] - stencil[0][side-1][1][IW]);
              real_t dvdz = 0.25 * one_over_dz * (stencil[2][side  ][1][IV] - stencil[0][side  ][1][IV]
                                               +  stencil[2][side-1][1][IV] - stencil[0][side-1][1][IV]);
                                              
              const real_t tau_yy = four_thirds * dvdy - two_thirds * (dudx + dwdz);
              const real_t tau_xy = dvdx + dudy;
              const real_t tau_yz = dvdz + dwdy;


              flux[IU] += sign * mu * tau_xy;
              flux[IV] += sign * mu * tau_yy;
              flux[IW] += sign * mu * tau_yz;
              flux[IE] += sign * mu * (tau_xy*qi[IU] + tau_yy*qi[IV] + tau_yz*qi[IW]);
            } 
            else {
              State qi = 0.5 * (stencil[side][1][1] + stencil[side-1][1][1]);

              real_t dudz = one_over_dz * (stencil[side][1][1][IU] - stencil[side-1][1][1][IU]);
              real_t dvdz = one_over_dz * (stencil[side][1][1][IV] - stencil[side-1][1][1][IV]);
              real_t dwdz = one_over_dz * (stencil[side][1][1][IW] - stencil[side-1][1][1][IW]);

              real_t dudx = 0.25 * one_over_dx * (stencil[side  ][1][2][IU] - stencil[1][side  ][0][IU]
                                               +  stencil[side-1][1][2][IU] - stencil[1][side-1][0][IU]);
              real_t dwdx = 0.25 * one_over_dx * (stencil[side  ][1][2][IV] - stencil[1][side  ][0][IW]
                                               +  stencil[side-1][1][2][IV] - stencil[1][side-1][0][IW]);
              real_t dvdy = 0.25 * one_over_dz * (stencil[side  ][2][1][IW] - stencil[side  ][0][1][IV]
                                               +  stencil[side-1][2][1][IW] - stencil[side-1][0][1][IV]);
              real_t dwdy = 0.25 * one_over_dz * (stencil[side  ][2][1][IV] - stencil[side  ][0][1][IW]
                                               +  stencil[side-1][2][1][IV] - stencil[side-1][0][1][IW]);
                                              
              const real_t tau_zz = four_thirds * dwdz - two_thirds * (dudx + dvdy);
              const real_t tau_xz = dudz + dwdx;
              const real_t tau_yz = dvdz + dwdy;


              flux[IU] += sign * mu * tau_xz;
              flux[IV] += sign * mu * tau_yz;
              flux[IW] += sign * mu * tau_zz;
              flux[IE] += sign * mu * (tau_xz*qi[IU] + tau_yz*qi[IV] + tau_zz*qi[IW]);
            } 
          }

          return flux;
        };

        State vf_x = computeViscousFlux(IX);
        State vf_y = computeViscousFlux(IY);
        State vf_z = computeViscousFlux(IZ);

        State un_loc = getStateFromArray(Unew, i, j, k);
        un_loc += dt * (vf_x + vf_y + vf_z);
        setStateInArray(Unew, i, j, k, un_loc);

      });
  }
};


}