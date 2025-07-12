#pragma once

#include <cmath>
#include <functional>
#include "INIReader.h"
#include <Kokkos_Core.hpp>
#include <math.h>

namespace fv3d {

using real_t = double;
using IFace = uint8_t;
constexpr int Ngrids  = 6;
constexpr int Nfields = 5;
using Pos   = Kokkos::Array<real_t, 3>;
using State = Kokkos::Array<real_t, Nfields>;
using Array = Kokkos::View<real_t*****>;
using ParallelRange = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;

struct RestartInfo {
  real_t time;
  int iteration;
};

enum IDir : uint8_t {
  IX = 0,
  IY = 1,
  IZ = 2,
};

enum ISide : uint8_t {
  ILEFT  = 0,
  IRIGHT = 1,
};

enum IVar : uint8_t {
  IR = 0,
  IU = 1,
  IV = 2,
  IW = 3,
  IP = 4,
  IE = 4
};

enum IFaceEnum : IFace {
  IXM = 0,
  IXP = 1,
  IYM = 2,
  IYP = 3,
  IZM = 4,
  IZP = 5
};

std::map<IFace, std::string> facename_map{
    {IXM, "x-"},
    {IXP, "x+"},
    {IYM, "y-"},
    {IYP, "y+"},
    {IZM, "z-"},
    {IZP, "z+"}
  };

enum RiemannSolver {
  HLL,
  HLLC
};

enum BoundaryType {
  BC_ABSORBING,
  BC_REFLECTING,
  BC_PERIODIC,
  BC_C91,
  BC_TRILAYER_DAMPING,
  BC_CUBED_SPHERE
};

enum TimeStepping {
  TS_EULER,
  TS_RK2
};

enum ReconstructionType {
  PCM,
  PCM_WB,
  PLM
};

enum ThermalConductivityMode {
  TCM_CONSTANT,
  TCM_B02,
  TCM_ISO3
};
enum HeatingMode {
  HM_COOLING_ISO,
};

// Thermal conduction at boundary
enum BCTC_Mode {
  BCTC_NONE,              // Nothing special done
  BCTC_FIXED_TEMPERATURE, // Lock the temperature at the boundary
  BCTC_FIXED_GRADIENT     // Lock the gradient at the boundary
};

enum ViscosityMode {
  VSC_CONSTANT
};

// Run
struct Params {
  real_t save_freq;
  real_t tend;
  std::string filename_out = "run";
  std::string restart_file = "";
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  BoundaryType boundary_z = BC_REFLECTING;
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  TimeStepping time_stepping = TS_EULER;
  real_t CFL = 0.1;

  bool multiple_outputs = false;

  // Parallel stuff
  ParallelRange range_tot;
  ParallelRange range_dom;
  ParallelRange range_xbound;
  ParallelRange range_ybound;
  ParallelRange range_zbound;
  ParallelRange range_slopes;

  // Mesh
  int Nx;      // Number of domain cells
  int Ny;      
  int Nz; 
  int Ng;      // Number of ghosts
  int Ntx;     // Total number of cells
  int Nty;
  int Ntz;
  int ibeg;    // First cell of the domain
  int iend;    // First cell outside of the domain
  int jbeg;
  int jend;
  int kbeg;
  int kend;
  real_t xmin; // Minimum boundary of the domain
  real_t xmax; // Maximum boundary of the domain
  real_t ymin;
  real_t ymax;
  real_t zmin;
  real_t zmax;
  real_t dx;   // Space step
  real_t dy;
  real_t dz;

  // Run and physics
  real_t epsilon = 1.0e-6;
  real_t gamma0 = 5.0/3.0;
  bool gravity = false;
  real_t g;
  bool well_balanced_flux_at_z_bc = false;
  bool well_balanced = false;
  std::string problem;

  // Thermal conduction
  bool thermal_conductivity_active;
  ThermalConductivityMode thermal_conductivity_mode;
  real_t kappa;

  BCTC_Mode bctc_zmin, bctc_zmax;
  real_t bctc_zmin_value, bctc_zmax_value;

  // Viscosity
  bool viscosity_active;
  ViscosityMode viscosity_mode;
  real_t mu;

  // Heating
  bool heating_active;
  HeatingMode heating_mode;
  bool log_total_heating;

  // Polytropes and such
  real_t m1;
  real_t theta1;
  real_t m2;
  real_t theta2;

  // H84
  real_t h84_pert;

  // C91
  real_t c91_pert;

  // B02
  real_t b02_zmid;
  real_t b02_kappa1;
  real_t b02_kappa2;
  real_t b02_thickness;

  // Isothermal-triple layer
  real_t iso3_dz0, iso3_dz1, iso3_dz2;
  real_t iso3_theta1, iso3_theta2;
  real_t iso3_m1, iso3_m2;
  real_t iso3_pert;
  real_t iso3_k1, iso3_k2;
  real_t iso3_T0, iso3_rho0;

  // Misc 
  int seed;
  int log_frequency;
};

// Helper to get the position in the mesh

KOKKOS_INLINE_FUNCTION
Pos operator*(const real_t a, const Pos &p) {
  return {a*p[IX], a*p[IY], a*p[IZ]};
}
KOKKOS_INLINE_FUNCTION
Pos operator/(const Pos &p, const real_t a) {
  return {p[IX]/a, p[IY]/a, p[IZ]/a};
}

struct GridNeighbourIndex {
  IFace neighbour_face;
  int i, j;
};

GridNeighbourIndex getGridNeighbourIndex(IFace face, IDir dir, ISide side, int i, int j, const Params &params) {
  enum : uint8_t {im = 0, ip = 1, jm = 2, jp = 3, __ = 255};

  GridNeighbourIndex info;
  const int beg = params.ibeg; // ibeg = jbeg
  const int end = params.iend; // iend = jend
  const int Ng  = params.Ng;
  int main_dir = (dir == IX) ? i : j;
  int orth_dir = (dir == IX) ? j : i;
  int ghost_id = (side == ILEFT) ? beg - 1 - main_dir : main_dir - end;

  if (dir == IZ)
    throw std::runtime_error("Grids do not have neighbours on the Z direction.");

  constexpr IFace neighbour_connectivity[][2][2] = {
                 /* IX */    /* IY */
    /* IXM */ { {IYP, IYM}, {IZM, IZP} },
    /* IXP */ { {IYP, IYM}, {IZP, IZM} },
    /* IYM */ { {IXP, IXM}, {IZP, IZM} },
    /* IYP */ { {IXP, IXM}, {IZM, IZP} },
    /* IZM */ { {IXP, IXM}, {IYM, IYP} },
    /* IZP */ { {IXP, IXM}, {IYP, IYM} }
  };
  info.neighbour_face = neighbour_connectivity[face][dir][side];
  
  constexpr uint8_t boundary_neighbour_side[6][6] = {
                    /* neighbour */
    /* face *//*  IX      IY      IZ*/
    /* IXM  */ {__, __, ip, ip, ip, ip},
    /* IXP  */ {__, __, im, im, im, im},
    /* IYM  */ {ip, ip, __, __, jm, jp},
    /* IYP  */ {im, im, __, __, jp, jm},
    /* IZM  */ {jm, jp, jp, jm, __, __},
    /* IZP  */ {jp, jm, jm, jp, __, __}
  };
  constexpr uint8_t invert_orth_orientation[6][6] = {
                    /* neighbour */
    /* face *//*  IX      IY      IZ*/
    /* IXM  */ {__, __,  1,  0,  1,  0},
    /* IXP  */ {__, __,  0,  1,  1,  0},
    /* IYM  */ { 1,  0, __, __,  0,  0},
    /* IYP  */ { 0,  1, __, __,  0,  0},
    /* IZM  */ { 1,  1,  0,  0, __, __},
    /* IZP  */ { 0,  0,  0,  0, __, __}
  };
  const uint8_t boundary_side = boundary_neighbour_side[face][info.neighbour_face];
  const uint8_t invert_orth   = invert_orth_orientation[face][info.neighbour_face];

  if (boundary_side == __)
    throw std::runtime_error("Selected faces are not neighbour to each other.");

  info.i = (boundary_side & 1) ? end - 1 - ghost_id : beg + ghost_id;
  info.j = invert_orth ? end - 1 + Ng - orth_dir : orth_dir;
  if (boundary_side > 1) Kokkos::kokkos_swap(info.i, info.j);

  return info;
}

KOKKOS_INLINE_FUNCTION
Pos mapShell(IFace face, real_t x, real_t y, real_t z) {
  const real_t s = Kokkos::tan(M_PI_4 * x);
  const real_t t = Kokkos::tan(M_PI_4 * y);
  const real_t d = Kokkos::sqrt(1 + s*s + t*t);

  Pos p;
  switch(face) {
    case IXP: p = { 1, -s, -t}; break;
    case IXM: p = {-1, -s,  t}; break;
    case IYP: p = {-s,  1,  t}; break;
    case IYM: p = {-s, -1, -t}; break;
    case IZP: p = {-s, -t,  1}; break;
    case IZM: p = {-s,  t, -1}; break;
  };

  return z * p / d;
} 

KOKKOS_INLINE_FUNCTION
Pos getPos(const Params& params, int i, int j, int k) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy,
          params.zmin + (k-params.kbeg+0.5) * params.dz};
}

Params readInifile(std::string filename) {
  INIReader reader(filename);

  Params res;

  // Mesh
  res.Nx = reader.GetInteger("mesh", "Nx", 32);
  res.Ny = reader.GetInteger("mesh", "Ny", 32);
  res.Nz = reader.GetInteger("mesh", "Nz", 32);
  res.Ng = reader.GetInteger("mesh", "Nghosts", 2);
  res.xmin = reader.GetFloat("mesh", "xmin", 0.0);
  res.xmax = reader.GetFloat("mesh", "xmax", 1.0);
  res.ymin = reader.GetFloat("mesh", "ymin", 0.0);
  res.ymax = reader.GetFloat("mesh", "ymax", 1.0);
  res.zmin = reader.GetFloat("mesh", "zmin", 0.0);
  res.zmax = reader.GetFloat("mesh", "zmax", 1.0);

  res.Ntx  = res.Nx + 2*res.Ng;
  res.Nty  = res.Ny + 2*res.Ng;
  res.Ntz  = res.Nz + 2*res.Ng;
  res.ibeg = res.Ng;
  res.iend = res.Ng+res.Nx;
  res.jbeg = res.Ng;
  res.jend = res.Ng+res.Ny;
  res.kbeg = res.Ng;
  res.kend = res.Ng+res.Nz;

  res.dx = (res.xmax-res.xmin) / res.Nx;
  res.dy = (res.ymax-res.ymin) / res.Ny;
  res.dz = (res.zmax-res.zmin) / res.Nz;

  // Run
  res.tend = reader.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = reader.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = reader.Get("run", "restart_file", "");
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
  res.save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = reader.Get("run", "output_filename", "run");

  std::string tmp;
  tmp = reader.Get("run", "boundaries_x", "cubed_sphere");
  std::map<std::string, BoundaryType> bc_map{
    {"reflecting",           BC_REFLECTING},
    {"absorbing",            BC_ABSORBING},
    {"periodic",             BC_PERIODIC},
    {"C91",                  BC_C91},
    {"triple_layer_damping", BC_TRILAYER_DAMPING},
    {"cubed_sphere",         BC_CUBED_SPHERE}
  };
  res.boundary_x = bc_map[tmp];
  tmp = reader.Get("run", "boundaries_y", "cubed_sphere");
  res.boundary_y = bc_map[tmp];
  tmp = reader.Get("run", "boundaries_z", "reflecting");
  res.boundary_z = bc_map[tmp];

  tmp = reader.Get("solvers", "reconstruction", "pcm");
  std::map<std::string, ReconstructionType> recons_map{
    {"pcm",    PCM},
    {"pcm_wb", PCM_WB},
    {"plm",    PLM}
  };
  res.reconstruction = recons_map[tmp];

  tmp = reader.Get("solvers", "riemann_solver", "hllc");
  std::map<std::string, RiemannSolver> riemann_map{
    {"hll", HLL},
    {"hllc", HLLC}
  };
  res.riemann_solver = riemann_map[tmp];

  tmp = reader.Get("solvers", "time_stepping", "euler");
  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = ts_map[tmp];

  res.CFL = reader.GetFloat("solvers", "CFL", 0.8);

  // Physics
  res.epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
  res.gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
  res.gravity = reader.GetBoolean("physics", "gravity", false);
  res.g       = reader.GetFloat("physics", "g", 0.0);
  res.m1      = reader.GetFloat("polytrope", "m1", 1.0);
  res.theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
  res.m2      = reader.GetFloat("polytrope", "m2", 1.0);
  res.theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
  res.problem = reader.Get("physics", "problem", "blast");
  res.well_balanced_flux_at_z_bc = reader.GetBoolean("physics", "well_balanced_flux_at_z_bc", false);

  // Thermal conductivity
  res.thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
  tmp = reader.Get("thermal_conduction", "conductivity_mode", "constant");
  std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
    {"constant" , TCM_CONSTANT},
    {"B02",       TCM_B02},
    {"iso-three", TCM_ISO3}
  };
  res.thermal_conductivity_mode = thermal_conductivity_map[tmp];
  res.kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

  std::map<std::string, BCTC_Mode> bctc_map{
    {"none",              BCTC_NONE},
    {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
    {"fixed_gradient",    BCTC_FIXED_GRADIENT}
  };
  tmp = reader.Get("thermal_conduction", "bc_zmin", "none");
  res.bctc_zmin = bctc_map[tmp];
  tmp = reader.Get("thermal_conduction", "bc_zmax", "none");
  res.bctc_zmax = bctc_map[tmp];
  res.bctc_zmin_value = reader.GetFloat("thermal_conduction", "bc_zmin_value", 1.0);
  res.bctc_zmax_value = reader.GetFloat("thermal_conduction", "bc_zmax_value", 1.0);

  // Viscosity
  res.viscosity_active = reader.GetBoolean("viscosity", "active", false);
  tmp = reader.Get("viscosity", "viscosity_mode", "constant");
  std::map<std::string, ViscosityMode> viscosity_map{
    {"constant", VSC_CONSTANT},
  };
  res.viscosity_mode = viscosity_map[tmp];
  res.mu = reader.GetFloat("viscosity", "mu", 0.0);

  // Heating function 
  res.heating_active = reader.GetBoolean("heating", "active", false);
  tmp = reader.Get("heating", "mode", "C2020");
  std::map<std::string, HeatingMode> heating_map{
    {"isothermal_cooling", HM_COOLING_ISO}
  };
  res.heating_mode = heating_map[tmp];
  res.log_total_heating = reader.GetBoolean("misc", "log_total_heating", false);

  // H84
  res.h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

  // C91
  res.c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

  // Isothermal triple layer
  res.iso3_dz0    = reader.GetFloat("isothermal_triple", "dz0", 1.0);
  res.iso3_dz1    = reader.GetFloat("isothermal_triple", "dz1", 2.0);
  res.iso3_dz2    = reader.GetFloat("isothermal_triple", "dz2", 2.0);
  res.iso3_theta1 = reader.GetFloat("isothermal_triple", "theta1", 2.0);
  res.iso3_theta2 = reader.GetFloat("isothermal_triple", "theta2", 2.0);
  res.iso3_pert   = reader.GetFloat("isothermal_triple", "perturbation", 1.0e-3);
  res.iso3_k1     = reader.GetFloat("isothermal_triple", "k1", 0.07);
  res.iso3_k2     = reader.GetFloat("isothermal_triple", "k2", 1.5);
  res.iso3_m1     = reader.GetFloat("isothermal_triple", "m1", 1.0);
  res.iso3_m2     = reader.GetFloat("isothermal_triple", "m2", 1.0);
  res.iso3_T0     = reader.GetFloat("isothermal_triple", "T0", 1.0);
  res.iso3_rho0   = reader.GetFloat("isothermal_triple", "rho0", 1.0);

  // Misc
  res.seed = reader.GetInteger("misc", "seed", 12345);
  res.log_frequency = reader.GetInteger("misc", "log_frequency", 10);

  // Parallel ranges
  res.range_tot    = ParallelRange({0, 0,          0,          0},           {Ngrids, res.Ntx,    res.Nty,    res.Ntz});
  res.range_dom    = ParallelRange({0, res.ibeg,   res.jbeg,   res.kbeg},    {Ngrids, res.iend,   res.jend,   res.kend});
  res.range_xbound = ParallelRange({0, 0,          res.jbeg,   res.kbeg},    {Ngrids, res.Ng,     res.jend,   res.kend});
  res.range_ybound = ParallelRange({0, 0,          0,          res.kbeg},    {Ngrids, res.Ntx,    res.Ng,     res.kend});
  // res.range_xbound = ParallelRange({0, 0,          res.jbeg,   res.kbeg},    {Ngrids, res.Ng,     res.jend,   res.kend});
  // res.range_ybound = ParallelRange({0, res.ibeg,   0,          res.kbeg},    {Ngrids, res.iend,   res.Ng,     res.kend});
  res.range_zbound = ParallelRange({0, res.ibeg,   res.jbeg,   0},           {Ngrids, res.iend,   res.jend,   res.Ng});
  res.range_slopes = ParallelRange({0, res.ibeg-1, res.jbeg-1, res.kbeg-1},  {Ngrids, res.iend+1, res.jend+1, res.kend+1});

  return res;
} 
}

// All states operations
#include "States.h"

namespace fv3d {
void consToPrim(Array U, Array Q, const Params &params) {
  Kokkos::parallel_for( "Conservative to Primitive", 
                        params.range_tot,
                        KOKKOS_LAMBDA(const IFace face, const int i, const int j, const int k) {
                          State Uloc = getStateFromArray(U, face, i, j, k);
                          State Qloc = consToPrim(Uloc, params);
                          setStateInArray(Q, face, i, j, k, Qloc);
                        });
}
void primToCons(Array &Q, Array &U, const Params &params) {
  Kokkos::parallel_for( "Primitive to Conservative", 
                        params.range_tot,
                        KOKKOS_LAMBDA(const IFace face, const int i, const int j, const int k) {
                          State Qloc = getStateFromArray(Q, face, i, j, k);
                          State Uloc = primToCons(Qloc, params);
                          setStateInArray(U, face, i, j, k, Uloc);
                        });
}
void checkNegatives(Array &Q, const Params &full_params) {
  uint64_t negative_density  = 0;
  uint64_t negative_pressure = 0;
  uint64_t nan_count = 0;

  Kokkos::parallel_reduce(
    "Check negative density/pressure", 
    full_params.range_dom,
    KOKKOS_LAMBDA(const IFace face, const int i, const int j, const int k, uint64_t& lnegative_density, uint64_t& lnegative_pressure, uint64_t& lnan_count) {
      constexpr real_t eps = 1.0e-6;
      if (Q(face, k, j, i, IR) < 0) {
        Q(face, k, j, i, IR) = eps;
        lnegative_density++;
      }
      if (Q(face, k, j, i, IP) < 0) {
        Q(face, k, j, i, IP) = eps;
        lnegative_pressure++;
      }

      for (int ivar=0; ivar < Nfields; ++ivar)
        if (std::isnan(Q(face, k, j, i, ivar)))
          lnan_count++;

    }, negative_density, negative_pressure, nan_count);

    if (negative_density) 
      std::cout << "--> negative density: " << negative_density << std::endl;
    if (negative_pressure)
      std::cout << "--> negative pressure: " << negative_pressure << std::endl;
    if (nan_count)
      std::cout << "--> NaN detected." << std::endl;
}

}
