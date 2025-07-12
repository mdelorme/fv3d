#pragma once

#include <cmath>
#include <functional>
#include <iomanip>
#include "INIReader.h"
#include <Kokkos_Core.hpp>
#include <math.h>

// Add functions HasSection and HasValue to INIReader, remove this when jtilly/inih.git will be updated
struct IniReader : INIReader {
  using INIReader::INIReader, INIReader::GetBoolean, INIReader::GetInteger, INIReader::GetFloat, INIReader::Get;

  bool HasSection(const std::string& section) const
  {
      const std::string key = MakeKey(section, "");
      std::map<std::string, std::string>::const_iterator pos = _values.lower_bound(key);
      if (pos == _values.end())
          return false;
      // Does the key at the lower_bound pos start with "section"?
      return pos->first.compare(0, key.length(), key) == 0;
  }

  bool HasValue(const std::string& section, const std::string& name) const
  {
      std::string key = MakeKey(section, name);
      return _values.count(key);
  }
};


namespace fv3d {

using real_t = double;
using IFace = uint8_t;
constexpr int Ngrids  = 6;
constexpr int Nfields = 5;
using Pos   = Kokkos::Array<real_t, 3>;
using State = Kokkos::Array<real_t, Nfields>;
using Array = Kokkos::View<real_t*****>;
using ParallelRange = Kokkos::MDRangePolicy<Kokkos::Rank<4>>;

struct Reader {
  Reader() = default;
  Reader(const std::string &filename) 
  : reader(filename) {};
  ~Reader() = default;

  struct value_container {
    std::string value;
    bool from_file = false;
    bool used = false;
    bool is_default_value = true;
  };
  std::map<std::string, std::map<std::string, value_container>> _values;
  IniReader reader;

  template<typename T>
  void registerValue(std::string section, std::string name, const T& value, bool is_default_value) {
    // TODO: revoir la logique car affiche unused et default à chaque paramètre.
    // Les valeurs sont par contre correctes.
    
    auto isAlreadyPresent = [&](const std::string& section, const std::string& name) {
      return (this->_values.count(section) != 0) && (this->_values.at(section).count(name) != 0);
    };
    auto isPresent = [&](const std::string& section, const std::string& name) {
      return (this->reader.HasSection(section) && this->reader.HasValue(section, name));
    };

    bool is_already_present_in_file = isAlreadyPresent(section, name);
    if (is_already_present_in_file) {
      throw std::runtime_error(std::string("parameter already set : ") + name);
    }
    bool is_present_in_file = isPresent(section, name);
    if (is_present_in_file) {
      this->_values[section][name].used = true;
      this->_values[section][name].from_file = true;
      this->_values[section][name].is_default_value = is_default_value;
    }
    
    // this->_values[section][name].is_default_value = is_default_value;

    if constexpr (std::is_same_v<T, std::string>){
      this->_values[section][name].value = value;
    }
    else {
      this->_values[section][name].value = std::to_string(value);
    }
  }
  bool GetBoolean(std::string section, std::string name, bool default_value){
    bool res = this->reader.GetBoolean(section, name, default_value); 
    registerValue(section, name, res, res == default_value);
    return res;
  }
  
  int GetInteger(std::string section, std::string name, int default_value){
    int res = this->reader.GetInteger(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  
  real_t GetFloat(std::string section, std::string name, real_t default_value){
    real_t res = this->reader.GetFloat(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  std::string Get(std::string section, std::string name, std::string default_value){
    std::string res = this->reader.Get(section, name, default_value);
    registerValue(section, name, res, res == default_value);
    return res;
  }
  auto GetMapValue(const auto& map, const std::string& section, const std::string& name, const std::string& default_value){
    std::string tmp;
    tmp = this->Get(section, name, default_value);

    if (map.count(tmp) == 0) {
      tmp = "\nallowed values: ";
      for (auto elem : map) tmp += elem.first + ", ";
      throw std::runtime_error(std::string("bad parameter for ") + name + ": " + tmp);
    }
    return map.at(tmp);
  };

  void outputValues(std::ostream& o){
    constexpr std::string::size_type name_width = 26;
    constexpr std::string::size_type value_width = 20;
    auto initial_format = o.flags();
    std::string problem = this->_values["physics"]["problem"].value;
    o << "Parameters used for the problem: " << problem << std::endl;
    o << std::left;
    for( auto p_section : this->_values )
    {
      const std::string& section_name = p_section.first;
      const std::map<std::string, value_container>& map_section = p_section.second;

      o << "\n[" << section_name << "]" << std::endl;
      for( auto p_var : map_section )
      {
        const std::string& var_name = p_var.first;
        const value_container& val = p_var.second;

        o << std::setw(std::max(var_name.length(),name_width)) << var_name 
          << " = " << std::setw(std::max(val.value.length(), value_width)) << val.value 
          << (val.is_default_value ? " ; default " : "")
          << std::endl;
      }
    }
    o.flags(initial_format);
  }
};

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

// Pos arithmetic

KOKKOS_INLINE_FUNCTION
const Pos operator+(const Pos &p, const Pos &q)
{
  return {p[IX] + q[IX],
          p[IY] + q[IY],
          p[IZ] + q[IZ]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator-(const Pos &p, const Pos &q)
{
  return {p[IX] - q[IX],
          p[IY] - q[IY],
          p[IZ] - q[IZ]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator*(real_t f, const Pos &p)
{
  return {f * p[IX],
          f * p[IY],
          f * p[IZ]};
}
KOKKOS_INLINE_FUNCTION
const Pos operator*(const Pos &p, real_t f)
{
  return f * p;
}
KOKKOS_INLINE_FUNCTION
const Pos operator/(const Pos &p, real_t f)
{
  return {p[IX] / f,
          p[IY] / f,
          p[IZ] / f};
}

// All parameters that should be copied on the device
struct DeviceParams {
  // Thermodynamics
  real_t gamma0 = 5.0/3.0;

  // Run and physics
  bool gravity = false;
  real_t g;
  bool well_balanced_flux_at_z_bc = false;
  bool well_balanced = false;

  // Boundaries
  BoundaryType boundary_x = BC_REFLECTING;
  BoundaryType boundary_y = BC_REFLECTING;
  BoundaryType boundary_z = BC_REFLECTING;

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

  // Godunov
  ReconstructionType reconstruction = PCM; 
  RiemannSolver riemann_solver = HLL;
  real_t CFL = 0.1;

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

  // Misc stuff
  real_t epsilon = 1.0e-6;

  void init_from_inifile(Reader &reader) {
    // Mesh
    Nx = reader.GetInteger("mesh", "Nx", 32);
    Ny = reader.GetInteger("mesh", "Ny", 32);
    Nz = reader.GetInteger("mesh", "Nz", 32);
    Ng = reader.GetInteger("mesh", "Nghosts", 2);
    xmin = reader.GetFloat("mesh", "xmin", 0.0);
    xmax = reader.GetFloat("mesh", "xmax", 1.0);
    ymin = reader.GetFloat("mesh", "ymin", 0.0);
    ymax = reader.GetFloat("mesh", "ymax", 1.0);
    zmin = reader.GetFloat("mesh", "zmin", 0.0);
    zmax = reader.GetFloat("mesh", "zmax", 1.0);

    Ntx  = Nx + 2*Ng;
    Nty  = Ny + 2*Ng;
    Ntz  = Nz + 2*Ng;
    ibeg = Ng;
    iend = Ng+Nx;
    jbeg = Ng;
    jend = Ng+Ny;
    kbeg = Ng;
    kend = Ng+Nz;

    dx = (xmax-xmin) / Nx;
    dy = (ymax-ymin) / Ny;
    dz = (zmax-zmin) / Nz;

    std::map<std::string, BoundaryType> bc_map{
      {"reflecting",           BC_REFLECTING},
      {"absorbing",            BC_ABSORBING},
      {"periodic",             BC_PERIODIC},
      {"C91",                  BC_C91},
      {"triple_layer_damping", BC_TRILAYER_DAMPING},
      {"cubed_sphere",         BC_CUBED_SPHERE}
    };
    boundary_x = reader.GetMapValue(bc_map, "run", "boundaries_x", "cubed_sphere");
    boundary_y = reader.GetMapValue(bc_map, "run", "boundaries_y", "cubed_sphere");
    boundary_z = reader.GetMapValue(bc_map, "run", "boundaries_z", "reflecting");

    std::map<std::string, ReconstructionType> recons_map{
      {"pcm",    PCM},
      {"pcm_wb", PCM_WB},
      {"plm",    PLM}
    };
    reconstruction = reader.GetMapValue(recons_map, "solvers", "reconstruction", "pcm");

    std::map<std::string, RiemannSolver> riemann_map{
      {"hll", HLL},
      {"hllc", HLLC}
    };
    riemann_solver = reader.GetMapValue(riemann_map, "solvers", "riemann_solver", "hllc");
    CFL = reader.GetFloat("solvers", "CFL", 0.8);

    // Physics
    epsilon = reader.GetFloat("misc", "epsilon", 1.0e-6);
    gamma0  = reader.GetFloat("physics", "gamma0", 5.0/3.0);
    gravity = reader.GetBoolean("physics", "gravity", false);
    g       = reader.GetFloat("physics", "g", 0.0);
    m1      = reader.GetFloat("polytrope", "m1", 1.0);
    theta1  = reader.GetFloat("polytrope", "theta1", 10.0);
    m2      = reader.GetFloat("polytrope", "m2", 1.0);
    theta2  = reader.GetFloat("polytrope", "theta2", 10.0);
    well_balanced_flux_at_z_bc = reader.GetBoolean("physics", "well_balanced_flux_at_z_bc", false);

    // Thermal conductivity
    thermal_conductivity_active = reader.GetBoolean("thermal_conduction", "active", false);
    std::map<std::string, ThermalConductivityMode> thermal_conductivity_map{
      {"constant" , TCM_CONSTANT},
      {"B02",       TCM_B02},
      {"iso-three", TCM_ISO3}
    };
    thermal_conductivity_mode = reader.GetMapValue(thermal_conductivity_map, "thermal_conduction", "conductivity_mode", "constant");
    kappa = reader.GetFloat("thermal_conduction", "kappa", 0.0);

    std::map<std::string, BCTC_Mode> bctc_map{
      {"none",              BCTC_NONE},
      {"fixed_temperature", BCTC_FIXED_TEMPERATURE},
      {"fixed_gradient",    BCTC_FIXED_GRADIENT}
    };
    bctc_zmin = reader.GetMapValue(bctc_map, "thermal_conduction", "bc_zmin", "none");
    bctc_zmax = reader.GetMapValue(bctc_map, "thermal_conduction", "bc_zmax", "none");
    bctc_zmin_value = reader.GetFloat("thermal_conduction", "bc_zmin_value", 1.0);
    bctc_zmax_value = reader.GetFloat("thermal_conduction", "bc_zmax_value", 1.0);

    // Viscosity
    viscosity_active = reader.GetBoolean("viscosity", "active", false);
    std::map<std::string, ViscosityMode> viscosity_map{
      {"constant", VSC_CONSTANT},
    };
    viscosity_mode = reader.GetMapValue(viscosity_map, "viscosity", "viscosity_mode", "constant");
    mu = reader.GetFloat("viscosity", "mu", 0.0);

    // Heating function 
    heating_active = reader.GetBoolean("heating", "active", false);
    std::map<std::string, HeatingMode> heating_map{
      {"isothermal_cooling", HM_COOLING_ISO}
    };
    heating_mode = reader.GetMapValue(heating_map, "heating", "mode", "isothermal_cooling");

    // H84
    h84_pert = reader.GetFloat("H84", "perturbation", 1.0e-4);

    // C91
    c91_pert = reader.GetFloat("C91", "perturbation", 1.0e-3);

    // Isothermal triple layer
    iso3_dz0    = reader.GetFloat("isothermal_triple", "dz0", 1.0);
    iso3_dz1    = reader.GetFloat("isothermal_triple", "dz1", 2.0);
    iso3_dz2    = reader.GetFloat("isothermal_triple", "dz2", 2.0);
    iso3_theta1 = reader.GetFloat("isothermal_triple", "theta1", 2.0);
    iso3_theta2 = reader.GetFloat("isothermal_triple", "theta2", 2.0);
    iso3_pert   = reader.GetFloat("isothermal_triple", "perturbation", 1.0e-3);
    iso3_k1     = reader.GetFloat("isothermal_triple", "k1", 0.07);
    iso3_k2     = reader.GetFloat("isothermal_triple", "k2", 1.5);
    iso3_m1     = reader.GetFloat("isothermal_triple", "m1", 1.0);
    iso3_m2     = reader.GetFloat("isothermal_triple", "m2", 1.0);
    iso3_T0     = reader.GetFloat("isothermal_triple", "T0", 1.0);
    iso3_rho0   = reader.GetFloat("isothermal_triple", "rho0", 1.0);
  }
};

struct Params {
  real_t save_freq;
  real_t tend;
  Reader reader;
  
  std::string filename_out = "run";
  std::string restart_file = "";
  TimeStepping time_stepping = TS_EULER;

  bool multiple_outputs = false;

  // Parallel stuff
  ParallelRange range_tot;
  ParallelRange range_dom;
  ParallelRange range_xbound;
  ParallelRange range_ybound;
  ParallelRange range_zbound;
  ParallelRange range_slopes;
  
  // Run
  std::string problem;

  // All the physics
  DeviceParams device_params;

  // Misc 
  int seed;
  int log_frequency;
  bool log_total_heating;
};

// Helper to get the position in the mesh

KOKKOS_INLINE_FUNCTION
Pos getPos(const DeviceParams& params, int i, int j, int k) {
  return {params.xmin + (i-params.ibeg+0.5) * params.dx,
          params.ymin + (j-params.jbeg+0.5) * params.dy,
          params.zmin + (k-params.kbeg+0.5) * params.dz};
}

struct GridNeighbourIndex {
  IFace neighbour_face;
  int i, j;
};

KOKKOS_INLINE_FUNCTION
GridNeighbourIndex getGridNeighbourIndex(IFace face, IDir dir, ISide side, int i, int j, const DeviceParams &params) {
  enum : uint8_t {im = 0, ip = 1, jm = 2, jp = 3, __ = 255};

  GridNeighbourIndex info;
  const int beg = params.ibeg; // ibeg = jbeg
  const int end = params.iend; // iend = jend
  const int Ng  = params.Ng;
  const int main_dir = (dir == IX) ? i : j;
  const int orth_dir = (dir == IX) ? j : i;
  const int ghost_id = (side == ILEFT) ? beg - 1 - main_dir : main_dir - end;

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

Params readInifile(std::string filename) {
  // Params reader(filename);
  Params res;
  res.reader = Reader(filename);
  auto &reader = res.reader;
  
  // Run
  res.tend = reader.GetFloat("run", "tend", 1.0);
  res.multiple_outputs = reader.GetBoolean("run", "multiple_outputs", false);
  res.restart_file = reader.Get("run", "restart_file", "");
  if (res.restart_file != "" && !res.multiple_outputs)
    throw std::runtime_error("Restart one unique files is not implemented yet !");
  
  res.save_freq = reader.GetFloat("run", "save_freq", 1.0e-1);
  res.filename_out = reader.Get("run", "output_filename", "run");

  std::map<std::string, TimeStepping> ts_map{
    {"euler", TS_EULER},
    {"RK2",   TS_RK2}
  };
  res.time_stepping = reader.GetMapValue(ts_map, "solvers", "time_stepping", "euler");
  res.problem = reader.Get("physics", "problem", "blast");

  // Misc
  res.seed = reader.GetInteger("misc", "seed", 12345);
  res.log_frequency = reader.GetInteger("misc", "log_frequency", 10);
  res.log_total_heating = reader.GetBoolean("misc", "log_total_heating", false);

  // All device parameters
  res.device_params.init_from_inifile(reader);

  // Parallel ranges
  auto &dparams = res.device_params;
  res.range_tot    = ParallelRange({0, 0,              0,              0},              {Ngrids, dparams.Ntx,    dparams.Nty,    dparams.Ntz});
  res.range_dom    = ParallelRange({0, dparams.ibeg,   dparams.jbeg,   dparams.kbeg},   {Ngrids, dparams.iend,   dparams.jend,   dparams.kend});
  res.range_xbound = ParallelRange({0, 0,              dparams.jbeg,   dparams.kbeg},   {Ngrids, dparams.Ng,     dparams.jend,   dparams.kend});
  res.range_ybound = ParallelRange({0, 0,              0,              dparams.kbeg},   {Ngrids, dparams.Ntx,    dparams.Ng,     dparams.kend});
  // res.range_xbound = ParallelRange({0, 0,              dparams.jbeg,   dparams.kbeg},   {Ngrids, dparams.Ng,     dparams.jend,   dparams.kend});
  // res.range_ybound = ParallelRange({0, dparams.ibeg,   0,              dparams.kbeg},   {Ngrids, dparams.iend,   dparams.Ng,     dparams.kend});
  res.range_zbound = ParallelRange({0, dparams.ibeg,   dparams.jbeg,   0},              {Ngrids, dparams.iend,   dparams.jend,   dparams.Ng});
  res.range_slopes = ParallelRange({0, dparams.ibeg-1, dparams.jbeg-1, dparams.kbeg-1}, {Ngrids, dparams.iend+1, dparams.jend+1, dparams.kend+1});

  return res;
} 
}

// All states operations
#include "States.h"

namespace fv3d {
void consToPrim(Array U, Array Q, const Params &full_params) {
  auto &params = full_params.device_params;
  Kokkos::parallel_for( "Conservative to Primitive", 
                        full_params.range_tot,
                        KOKKOS_LAMBDA(const IFace face, const int i, const int j, const int k) {
                          State Uloc = getStateFromArray(U, face, i, j, k);
                          State Qloc = consToPrim(Uloc, params);
                          setStateInArray(Q, face, i, j, k, Qloc);
                        });
}
void primToCons(Array &Q, Array &U, const Params &full_params) {
  auto &params = full_params.device_params;
  Kokkos::parallel_for( "Primitive to Conservative", 
                        full_params.range_tot,
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
