#pragma once

#include <fstream>
#include <Kokkos_Random.hpp>

#include "SimInfo.h"
#include "BoundaryConditions.h"

namespace fv3d {

namespace {

  using RandomPool = Kokkos::Random_XorShift64_Pool<>;

  /**
   * @brief Sod Shock tube aligned along the X axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodX(Array Q, int i, int j, int k, const Params &params) {
    if (getPos(params, i, j, k)[IX] <= 0.5) {
      Q(k, j, i, IR) = 1.0;
      Q(k, j, i, IP) = 1.0;
    }
    else {
      Q(k, j, i, IR) = 0.125;
      Q(k, j, i, IP) = 0.1;
    }
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = 0.0;
  }

  /**
   * @brief Sod Shock tube aligned along the Y axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodY(Array Q, int i, int j, int k, const Params &params) {
    if (getPos(params, i, j, k)[IY] <= 0.5) {
      Q(k, j, i, IR) = 1.0;
      Q(k, j, i, IP) = 1.0;
    }
    else {
      Q(k, j, i, IR) = 0.125;
      Q(k, j, i, IP) = 0.1;
    }
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = 0.0;
  }

  /**
   * @brief Sod Shock tube aligned along the Z axis
   */
  KOKKOS_INLINE_FUNCTION
  void initSodZ(Array Q, int i, int j, int k, const Params &params) {
    if (getPos(params, i, j, k)[IZ] <= 0.5) {
      Q(k, j, i, IR) = 1.0;
      Q(k, j, i, IP) = 1.0;
    }
    else {
      Q(k, j, i, IR) = 0.125;
      Q(k, j, i, IP) = 0.1;
    }
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = 0.0;
  }

  /**
   * @brief Sedov blast initial conditions
   */
  KOKKOS_INLINE_FUNCTION
  void initBlast(Array Q, int i, int j, int k, const Params &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);
    real_t zmid = 0.5 * (params.zmin+params.zmax);

    Pos pos = getPos(params, i, j, k);
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t z = pos[IZ];

    real_t xr = xmid - x;
    real_t yr = ymid - y;
    real_t zr = zmid - z;
    real_t r = sqrt(xr*xr+yr*yr+zr*zr);

    if (r < 0.2) {
      Q(k, j, i, IR) = 1.0;
      Q(k, j, i, IP) = 10.0;
    }
    else {
      Q(k, j, i, IR) = 1.2;
      Q(k, j, i, IP) = 0.1;
    }
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = 0.0;
  }

  /**
   * @brief Stratified convection based on Hurlburt et al 1984
   */
  KOKKOS_INLINE_FUNCTION
  void initH84(Array Q, int i, int j, int k, const Params &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j, k);
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t z = pos[IZ];

    real_t rho = pow(z, params.m1);
    real_t prs = pow(z, params.m1+1.0); 

    auto generator = random_pool.get_state();
    real_t pert = params.h84_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    Q(k, j, i, IR) = rho;
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = pert;
    Q(k, j, i, IP) = prs;
  }

  /**
   * @brief Stratified convection based on Cattaneo et al. 1991
   */
  KOKKOS_INLINE_FUNCTION
  void initC91(Array Q, int i, int j, int k, const Params &params, const RandomPool &random_pool) {
    Pos pos = getPos(params, i, j, k);
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t z = pos[IZ];

    real_t T = (1.0 + params.theta1*z);
    real_t rho = pow(T, params.m1);
    real_t prs = pow(T, params.m1+1.0);

    auto generator = random_pool.get_state();
    real_t pert = params.c91_pert * (generator.drand(-0.5, 0.5));
    random_pool.free_state(generator);

    prs = prs * (1.0 + pert);

    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IR) = rho;
    Q(k, j, i, IV) = 0.0;
    Q(k, j, i, IW) = 0.0;
    Q(k, j, i, IP) = prs;
  }

  /**
   * @brief Simple diffusion test with a structure being advected on the grid
   */
  KOKKOS_INLINE_FUNCTION
  void initDiffusion(Array Q, int i, int j, int k, const Params &params) {
    real_t xmid = 0.5 * (params.xmin+params.xmax);
    real_t ymid = 0.5 * (params.ymin+params.ymax);
    real_t zmid = 0.5 * (params.zmin+params.zmax);

    Pos pos = getPos(params, i, j, k);

    real_t x0 = (pos[IX]-xmid);
    real_t y0 = (pos[IY]-ymid);
    real_t z0 = (pos[IZ]-zmid);

    real_t r = sqrt(x0*x0+y0*y0+z0*z0);

    if (r < 0.2) 
      Q(k, j, i, IR) = 1.0;
    else
      Q(k, j, i, IR) = 0.1;

    Q(k, j, i, IP) = 1.0;
    Q(k, j, i, IU) = 1.0;
    Q(k, j, i, IV) = 1.0;
    Q(k, j, i, IW) = 1.0;
  }

  /**
   * @brief Rayleigh-Taylor instability setup
   */
  KOKKOS_INLINE_FUNCTION
  void initRayleighTaylor(Array Q, int i, int j, int k, const Params &params) {
    real_t zmid = 0.5*(params.zmin + params.zmax);

    Pos pos = getPos(params, i, j, k);
    real_t x = pos[IX];
    real_t y = pos[IY];
    real_t z = pos[IZ];

    const real_t P0 = 2.5;

    if (z < zmid) {
      Q(k, j, i, IR) = 1.0;
      Q(k, j, i, IU) = 0.0;
      Q(k, j, i, IV) = 0.0;
      Q(k, j, i, IP) = P0 + 0.1 * params.g * z;
    }
    else {
      Q(k, j, i, IR) = 2.0;
      Q(k, j, i, IU) = 0.0;
      Q(k, j, i, IV) = 0.0;
      Q(k, j, i, IP) = P0 + 0.1 * params.g * z;
    }

    if (z > -1.0/3.0 && z < 1.0/3.0)
      Q(k, j, i, IW) = 0.01 * (1.0 + cos(4*M_PI*x)) * (1.0 + cos(4.0*M_PI*y)) * (1 + cos(3.0*M_PI*z))/4.0;
  }

  /**
   * @brief Triple layer setup with an iso-thermal layer at the top
   */
  KOKKOS_INLINE_FUNCTION
  void initIsothermalTriple(Array Q, int i, int j, int k, const Params &params, const RandomPool &random_pool) {
    const real_t T1   = params.iso3_T0;
    const real_t rho1   = params.iso3_rho0;
    const real_t p1 = rho1 * T1;

    const real_t T0 = T1;
    const real_t rho0 = rho1 * exp(-params.iso3_dz0 * params.g / T0);
    const real_t p0 = rho0 * T0;

    const real_t T2   = T1 + params.iso3_theta1 * params.iso3_dz1;
    const real_t rho2 = rho1 * pow(T2/T1, params.iso3_m1);
    const real_t p2   = p1 * pow(T2/T1, params.iso3_m1+1.0);

    const real_t z1 = params.iso3_dz0;
    const real_t z2 = params.iso3_dz0+params.iso3_dz1;

    Pos pos = getPos(params, i, j, k);
    const real_t d = pos[IZ];

    // Top layer (iso-thermal)
    real_t rho, p;
    real_t T;
    if (d <= z1) {
      p   = p0 * exp(params.g * d / T0);
      rho = p / T0;
    }
    // Middle layer (convective)
    else if (d <= z2) {
      T = T1 + params.iso3_theta1*(d-z1);

      // We add a pressure perturbation as in C91
      auto generator = random_pool.get_state();
      real_t pert = params.iso3_pert * generator.drand(-0.5, 0.5);
      random_pool.free_state(generator);

      if (d - z1 < params.dz || z2 - d < params.dz)
        pert = 0.0; 
      
      rho = rho1 * pow(T/T1, params.iso3_m1);
      p   = p1 * (1.0 + pert) * pow(T/T1, params.iso3_m1+1.0);
    }
    // Bottom layer (stable)
    else {
      T = T2 + params.iso3_theta2 * (d-z2);
      rho = rho2 * pow(T/T2, params.iso3_m2);
      p   = p2 * pow(T/T2, params.iso3_m2+1.0);
    }

    Q(k, j, i, IR) = rho;
    Q(k, j, i, IU) = 0.0;
    Q(k, j, i, IV) = 0.0;
    Q(k, i, j, IW) = 0.0;
    Q(k, j, i, IP) = p;
  }
}



/**
 * @brief Enum describing the type of initialization possible
 */
enum InitType {
  SOD_X,
  SOD_Y,
  SOD_Z,
  BLAST,
  RAYLEIGH_TAYLOR,
  DIFFUSION,
  H84,
  C91,
  ISOTHERMAL_TRIPLE
};

struct InitFunctor {
private:
  Params params;
  InitType init_type;
public:
  InitFunctor(Params &params)
    : params(params) {
    std::map<std::string, InitType> init_map {
      {"sod_x", SOD_X},
      {"sod_y", SOD_Y},
      {"sod_z", SOD_Z},
      {"blast", BLAST},
      {"rayleigh-taylor", RAYLEIGH_TAYLOR},
      {"diffusion", DIFFUSION},
      {"H84", H84},
      {"C91", C91},
      {"isothermal-triple", ISOTHERMAL_TRIPLE}
    };

    if (init_map.count(params.problem) == 0) {
      std::cerr << "Error ! Unknown problem " << params.problem << std::endl;
      std::cerr << "Available problems ";
      for (auto &p : init_map)
        std::cerr << p.first << " ";
      std::cerr << std::endl;
      throw std::runtime_error("Unknown initial conditions");
    }
    init_type = init_map[params.problem];
  };
  ~InitFunctor() = default;

  void init(Array &Q) {
    auto init_type = this->init_type;
    auto params = this->params;

    RandomPool random_pool(params.seed);

    // Filling active domain ...
    Kokkos::parallel_for( "Initialization", 
                          params.range_dom, 
                          KOKKOS_LAMBDA(const int i, const int j, const int k) {
                            switch(init_type) {
                              case SOD_X:             initSodX(Q, i, j, k, params); break;
                              case SOD_Y:             initSodY(Q, i, j, k, params); break;
                              case SOD_Z:             initSodZ(Q, i, j, k, params); break;
                              case BLAST:             initBlast(Q, i, j, k, params); break;
                              case DIFFUSION:         initDiffusion(Q, i, j, k, params); break;
                              case RAYLEIGH_TAYLOR:   initRayleighTaylor(Q, i, j, k, params); break;
                              case H84:               initH84(Q, i, j, k, params, random_pool); break;
                              case C91:               initC91(Q, i, j, k, params, random_pool); break;
                              case ISOTHERMAL_TRIPLE: initIsothermalTriple(Q, i, j, k, params, random_pool); break;
                            }
                          });
  
    // ... and boundaries
    BoundaryManager bc(params);
    bc.fillBoundaries(Q);
  }
};



}