#pragma once 

#include <Kokkos_MathematicalFunctions.hpp>

#include "SimInfo.h"
#include "RiemannSolvers.h"
#include "BoundaryConditions.h"
#include "ThermalConduction.h"
#include "Viscosity.h"

namespace fv2d {

namespace {
  KOKKOS_INLINE_FUNCTION
  State reconstruct(Array Q, Array slopes, int i, int j, int k, real_t sign, IDir dir, const Params &params) {
    State q     = getStateFromArray(Q, i, j, k);
    State slope = getStateFromArray(slopes, i, j, k);
    
    State res;
    switch (params.reconstruction) {
      case PLM: res = q + slope * sign * 0.5; break; // Piecewise Linear
      case PCM_WB: // Piecewise constant + Well-balancing
        res[IR] = q[IR];
        res[IU] = q[IU];
        res[IV] = q[IV];
        res[IW] = q[IW];
        res[IP] = (dir == IX || dir == IY ? q[IP] : q[IP] + sign * q[IR] * params.g * params.dz * 0.5);
        break;
      default:  res = q; // Piecewise Constant
    }

    return swap_component(res, dir);
  }
}

class UpdateFunctor {
public:
  Params params;
  BoundaryManager bc_manager;
  ThermalConductionFunctor tc_functor;
  ViscosityFunctor visc_functor;

  Array slopesX, slopesY, slopesZ;

  UpdateFunctor(const Params &params)
    : params(params), bc_manager(params),
      tc_functor(params), visc_functor(params) {
      
      slopesX = Array("SlopesX", params.Ntz, params.Nty, params.Ntx, Nfields);
      slopesY = Array("SlopesY", params.Ntz, params.Nty, params.Ntx, Nfields);
      slopesZ = Array("SlopesZ", params.Ntz, params.Nty, params.Ntx, Nfields);
    };
  ~UpdateFunctor() = default;

  void computeSlopes(const Array &Q) const {
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto slopesZ = this->slopesZ;
    auto params  = this->params;

    Kokkos::parallel_for(
      "Slopes",
      params.range_slopes,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        for (int ivar=0; ivar < Nfields; ++ivar) {
          real_t dL = Q(k, j, i, ivar)   - Q(k, j, i-1, ivar);
          real_t dR = Q(k, j, i+1, ivar) - Q(k, j, i, ivar);
          real_t dU = Q(k, j, i, ivar)   - Q(k, j-1, i, ivar); 
          real_t dD = Q(k, j+1, i, ivar) - Q(k, j, i, ivar); 
          real_t dF = Q(k+1, j, i, ivar) - Q(k, j, i, ivar);
          real_t dB = Q(k, j, i, ivar) - Q(k-1, j, i, ivar);

          auto minmod = [](real_t dL, real_t dR) -> real_t {
            if (dL*dR < 0.0)
              return 0.0;
            else if (Kokkos::fabs(dL) < Kokkos::fabs(dR))
              return dL;
            else
              return dR;
          };

          slopesX(k, j, i, ivar) = minmod(dL, dR);
          slopesY(k, j, i, ivar) = minmod(dU, dD);
          slopesZ(k, j, i, ivar) = minmod(dF, dB);
        }
      });

  }

  void computeFluxesAndUpdate(Array Q, Array Unew, real_t dt) const {
    auto params = this->params;
    auto slopesX = this->slopesX;
    auto slopesY = this->slopesY;
    auto slopesZ = this->slopesZ;
    using offset_t = Kokkos::Array<real_t, 3>;

    Kokkos::parallel_for(
      "Update", 
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        // Lambda to update the cell along a direction
        auto updateAlongDir = [&](int i, int j, int k, IDir dir) {
          auto& slopes = (dir == IX ? slopesX : (dir == IY ? slopesY : slopesZ));

          offset_t dm{}, dp{};
          dm[dir] = -1;
          dp[dir] =  1;
          State qCL = reconstruct(Q, slopes, i, j, k, -1.0, dir, params);
          State qCR = reconstruct(Q, slopes, i, j, k,  1.0, dir, params);
          State qL  = reconstruct(Q, slopes, i+dm[IX], j+dm[IY], k+dm[IZ],  1.0, dir, params);
          State qR  = reconstruct(Q, slopes, i+dp[IX], j+dp[IY], k+dp[IZ], -1.0, dir, params);

          // Calling the right Riemann solver
          auto riemann = [&](State qL, State qR, State &flux, real_t &pout) {
            switch (params.riemann_solver) {
              case HLL: hll(qL, qR, flux, pout, params); break;
              default: hllc(qL, qR, flux, pout, params); break;
            }
          };

          // Calculating flux left and right of the cell
          State fluxL, fluxR;
          real_t poutL, poutR;

          riemann(qL, qCL, fluxL, poutL);
          riemann(qCR, qR, fluxR, poutR);

          fluxL = swap_component(fluxL, dir);
          fluxR = swap_component(fluxR, dir);

          // Remove mechanical flux in a well-balanced fashion
          if (params.well_balanced_flux_at_z_bc && (k==params.kbeg || k==params.kend-1) && dir == IZ) {
            if (k==params.kbeg)
              fluxL = State{0.0, 0.0, 0.0, poutR - Q(k, j, i, IR)*params.g*params.dz, 0.0};
            else 
              fluxR = State{0.0, 0.0, 0.0, poutL + Q(k, j, i, IR)*params.g*params.dz, 0.0};
          }
          else if (dir == IZ && (k==params.kbeg || k==params.kend-1) && params.boundary_z == BC_C91) {
            if (k == params.kbeg)
              fluxL = State{0.0, 0.0, 0.0, poutL, 0.0};
            else
              fluxR = State{0.0, 0.0, 0.0, poutR, 0.0};
          }

          auto un_loc = getStateFromArray(Unew, i, j, k);
          un_loc += dt*(fluxL - fluxR)/(dir == IX ? params.dx : (dir == IY ? params.dy : params.dz));
        
          if (dir == IZ && params.gravity) {
            un_loc[IW] += dt * Q(k, j, i, IR) * params.g;
            un_loc[IE] += dt * 0.5 * (fluxL[IR] + fluxR[IR]) * params.g;
          }

          setStateInArray(Unew, i, j, k, un_loc);
        };

        updateAlongDir(i, j, k, IX);
        updateAlongDir(i, j, k, IY);
        updateAlongDir(i, j, k, IZ);

        Unew(k, j, i, IR) = fmax(1.0e-6, Unew(k, j, i, IR));
      });
  }

  void euler_step(Array Q, Array Unew, real_t dt) {
    // First filling up boundaries for ghosts terms
    bc_manager.fillBoundaries(Q);

    // Hypperbolic udpate
    if (params.reconstruction == PLM)
      computeSlopes(Q);
    computeFluxesAndUpdate(Q, Unew, dt);

    // Splitted terms
    if (params.thermal_conductivity_active)
      tc_functor.applyThermalConduction(Q, Unew, dt);
    if (params.viscosity_active)
      visc_functor.applyViscosity(Q, Unew, dt);
  }

  void update(Array Q, Array Unew, real_t dt) {
    if (params.time_stepping == TS_EULER)
      euler_step(Q, Unew, dt);
    else if (params.time_stepping == TS_RK2) {
      Array U0    = Array("U0", params.Nty, params.Ntx, Nfields);
      Array Ustar = Array("Ustar", params.Nty, params.Ntx, Nfields);
      
      // Step 1
      Kokkos::deep_copy(U0, Unew);
      Kokkos::deep_copy(Ustar, Unew);
      euler_step(Q, Ustar, dt);
      
      // Step 2
      Kokkos::deep_copy(Unew, Ustar);
      consToPrim(Ustar, Q, params);
      euler_step(Q, Unew, dt);

      // SSP-RK2
      Kokkos::parallel_for(
        "RK2 Correct", 
        params.range_dom,
        KOKKOS_LAMBDA(const int i, const int j, const int k) {
          for (int ivar=0; ivar < Nfields; ++ivar)
            Unew(k, j, i, ivar) = 0.5 * (U0(k, j, i, ivar) + Unew(k, j, i, ivar));
        });
    }
  }
};

}