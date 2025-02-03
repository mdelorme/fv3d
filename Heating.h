#pragma once

#include "SimInfo.h"

namespace fv3d {

namespace {

KOKKOS_INLINE_FUNCTION
real_t cooling_layer(Array Q, const int i, const int j, const int k, const Params &params) {
  Pos pos = getPos(params, i, j, k);
  real_t z = pos[IZ];
  real_t kappa = params.kappa*params.iso3_k2; //*params.gamma0/(params.gamma0-1.0);
  real_t F = params.iso3_theta2*kappa/params.dz;

  real_t qc = 0.0;
  real_t zdiff = z-params.iso3_dz0;
  if (fabs(zdiff) <= params.dz && zdiff < 0.0)
    qc = -F;

  return qc;
}
} // namespace

class HeatingFunctor {
public:
  Params params;

  HeatingFunctor(const Params &params)
    : params(params) {};
  ~HeatingFunctor() = default;

  void applyHeating(Array Q, Array Unew, real_t dt) {
    auto params = this->params;

    Kokkos::parallel_for(
      "Heating",
      params.range_dom,
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        real_t q;

        switch(params.heating_mode) {
          case HM_COOLING_ISO: q = cooling_layer(Q, i, j, k, params); break;
        }

        // Explicit update
        Unew(k, j, i, IE) += dt * q;
      });

  }
};


}