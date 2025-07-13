#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>
#include <filesystem>

// https://visit-sphinx-github-user-manual.readthedocs.io/en/3.4rc/data_into_visit/XdmfFormat.html

#include "SimInfo.h"

using namespace H5Easy;

namespace fv3d {

constexpr int ite_nzeros = 4;
constexpr std::string_view ite_prefix = "ite_";

  // xdmf strings
namespace {
  char str_xdmf_header[] = R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" [
<!ENTITY file "%s:">
<!ENTITY fdim "%d %d %d">
<!ENTITY gdim "%d %d %d">
<!ENTITY GridEntity '
<Topology TopologyType="3DSMesh" Dimensions="&gdim;"/>
<Geometry GeometryType="X_Y_Z">
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/x</DataItem>
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/y</DataItem>
  <DataItem Dimensions="&gdim;" NumberType="Float" Precision="8" Format="HDF">&file;/z</DataItem>
</Geometry>'>
]>
<Xdmf Version="3.0">
<Domain>
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    )xml";
  #define format_xdmf_header(params, filename)        \
          (filename + ".h5").c_str(),                 \
          params.Nz,     params.Ny,     params.Nx,    \
          params.Nz + 1, params.Ny + 1, params.Nx + 1
  char str_xdmf_footer[] =
  R"xml(
  </Grid>
</Domain>
</Xdmf>)xml";

  char str_xdmf_ite_header[] =
  R"xml(
    <Grid Name="%s" GridType="Uniform">
      <Time Value="%lf" />
      &GridEntity;)xml";
  #define format_xdmf_ite_header(name, time) \
          name.c_str(), time
  char str_xdmf_scalar_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
      </Attribute>)xml";
  #define format_xdmf_scalar_field(group, field) \
          field, group.c_str(), field
  char str_xdmf_vector_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="&fdim; 3" ItemType="Function" Function="JOIN($0, $1, $2)">
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
          <DataItem Dimensions="&fdim;" NumberType="Float" Precision="8" Format="HDF">&file;/%s%s</DataItem>
        </DataItem>
      </Attribute>)xml";
  #define format_xdmf_vector_field(group, name, field_x, field_y, field_z)             \
          name, group.c_str(), field_x, group.c_str(), field_y, group.c_str(), field_z
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
  )xml";
} // anonymous namespace

class IOManager {
public:
  Params &params;
  DeviceParams &device_params;
  bool force_file_truncation = false;

  IOManager(Params &params)
    : params(params), device_params(params.device_params) {};

  ~IOManager() = default;

  void saveSolution(const Array &Q, int iteration, real_t t) {
    if (params.multiple_outputs)
      saveSolutionMultiple(Q, iteration, t);
    else
      saveSolutionUnique(Q, iteration, t);
  }

  void saveSolutionMultiple(const Array &Q, int iteration, real_t t)
  {
    std::ostringstream oss;
    
    oss << params.filename_out << "_" << std::setw(ite_nzeros) << std::setfill('0') << iteration;
    std::string iteration_str = oss.str();
    std::string h5_filename  = oss.str() + ".h5";
    std::string xmf_filename = oss.str() + ".xmf";

    File file(h5_filename, File::Truncate);
    FILE* xdmf_fd = fopen(xmf_filename.c_str(), "w+");

    file.createAttribute("Ntx", device_params.Ntx);
    file.createAttribute("Nty", device_params.Nty);
    file.createAttribute("Ntz", device_params.Ntz);
    file.createAttribute("Nx", device_params.Nx);
    file.createAttribute("Ny", device_params.Ny);
    file.createAttribute("Nz", device_params.Nz);
    file.createAttribute("ibeg", device_params.ibeg);
    file.createAttribute("iend", device_params.iend);
    file.createAttribute("jbeg", device_params.jbeg);
    file.createAttribute("jend", device_params.jend);
    file.createAttribute("kbeg", device_params.kbeg);
    file.createAttribute("kend", device_params.kend);
    file.createAttribute("problem", params.problem);

    std::vector<real_t> x, y, z;
    // -- vertex pos
    for (int k=device_params.kbeg; k <= device_params.kend; ++k) {
      for (int j=device_params.jbeg; j <= device_params.jend; ++j) {
        for (int i=device_params.ibeg; i <= device_params.iend; ++i) {
          x.push_back((i-device_params.ibeg) * device_params.dx);
          y.push_back((j-device_params.jbeg) * device_params.dy);
          z.push_back((k-device_params.kbeg) * device_params.dz);
        }
      }
    }

    file.createDataSet("x", x);
    file.createDataSet("y", y);
    file.createDataSet("z", z);

    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tw, tprs;
    for (int k=device_params.kbeg; k<device_params.kend; ++k) {
      for (int j=device_params.jbeg; j<device_params.jend; ++j) {
        for (int i=device_params.ibeg; i<device_params.iend; ++i) {
          real_t rho = Qhost(k, j, i, IR);
          real_t u   = Qhost(k, j, i, IU);
          real_t v   = Qhost(k, j, i, IV);
          real_t w   = Qhost(k, j, i, IW);
          real_t p   = Qhost(k, j, i, IP);

          trho.push_back(rho);
          tu.push_back(u);
          tv.push_back(v);
          tw.push_back(w);
          tprs.push_back(p);
        }
      }
    }

    file.createDataSet("rho", trho);
    file.createDataSet("u", tu);
    file.createDataSet("v", tv);
    file.createDataSet("w", tw);
    file.createDataSet("prs", tprs);
    file.createAttribute("time", t);
    file.createAttribute("iteration", iteration);

    std::string group = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, iteration_str));
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t) {
    std::ostringstream oss;
    
    oss << ite_prefix << std::setw(ite_nzeros) << std::setfill('0') << iteration;
    std::string iteration_str = oss.str();

    force_file_truncation = (force_file_truncation || iteration == 0);
      
    auto flag_h5 = (force_file_truncation ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (force_file_truncation ? "w+" : "r+");
    File file(params.filename_out + ".h5", flag_h5);
    FILE* xdmf_fd = fopen((params.filename_out + ".xdmf").c_str(), flag_xdmf);

    if (force_file_truncation) {
      force_file_truncation = false;
      file.createAttribute("Ntx", device_params.Ntx);
      file.createAttribute("Nty", device_params.Nty);
      file.createAttribute("Ntz", device_params.Ntz);
      file.createAttribute("Nx", device_params.Nx);
      file.createAttribute("Ny", device_params.Ny);
      file.createAttribute("Nz", device_params.Nz);
      file.createAttribute("ibeg", device_params.ibeg);
      file.createAttribute("iend", device_params.iend);
      file.createAttribute("jbeg", device_params.jbeg);
      file.createAttribute("jend", device_params.jend);
      file.createAttribute("kbeg", device_params.kbeg);
      file.createAttribute("kend", device_params.kend);
      file.createAttribute("problem", params.problem);

      std::vector<real_t> x, y, z;
      // -- vertex pos
      for (int k=device_params.kbeg; k <= device_params.kend; ++k) {
        for (int j=device_params.jbeg; j <= device_params.jend; ++j) {
          for (int i=device_params.ibeg; i <= device_params.iend; ++i) {
            x.push_back((i-device_params.ibeg) * device_params.dx);
            y.push_back((j-device_params.jbeg) * device_params.dy);
            z.push_back((k-device_params.kbeg) * device_params.dz);
          }
        }
      }

      file.createDataSet("x", x);
      file.createDataSet("y", y);
      file.createDataSet("z", z);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, params.filename_out));
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    using Table = std::vector<real_t>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tw, tprs;
    for (int k=device_params.kbeg; k<device_params.kend; ++k) {
      for (int j=device_params.jbeg; j<device_params.jend; ++j) {
        for (int i=device_params.ibeg; i<device_params.iend; ++i) {
          real_t rho = Qhost(k, j, i, IR);
          real_t u   = Qhost(k, j, i, IU);
          real_t v   = Qhost(k, j, i, IV);
          real_t w   = Qhost(k, j, i, IW);
          real_t p   = Qhost(k, j, i, IP);

          trho.push_back(rho);
          tu.push_back(u);
          tv.push_back(v);
          tw.push_back(w);
          tprs.push_back(p);
        }
      }
    }

    auto ite_group = file.createGroup(iteration_str);
    ite_group.createDataSet("rho", trho);
    ite_group.createDataSet("u", tu);
    ite_group.createDataSet("v", tv);
    ite_group.createDataSet("w", tw);
    ite_group.createDataSet("prs", tprs);
    ite_group.createAttribute("time", t);
    ite_group.createAttribute("iteration", iteration);

    const std::string group = iteration_str + "/";

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, format_xdmf_ite_header(iteration_str, t));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(group, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(group, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  RestartInfo loadSnapshot(Array &Q) {
    // example of unique_output restart_file: 'run.h5:/ite_0005'
    // or just 'run.h5' for the last iteration

    std::string restart_file = params.restart_file;
    std::string group = "";

    const auto delim_multi = restart_file.find(".h5:/");
    if (delim_multi != std::string::npos) {
      group = restart_file.substr(delim_multi + 5);
      restart_file = restart_file.substr(0, delim_multi + 3);
    }

    if ( !params.multiple_outputs && std::filesystem::equivalent(restart_file, params.filename_out + ".h5") ) {
      if (delim_multi != std::string::npos) {
        std::cerr << "Invalid restart file : if your restart file and output file are "
                     "the same, you can only start from the last iteration." << std::endl << std::endl;
        throw std::runtime_error("ERROR : Invalid restart_file.");
      }
    }
    else {
      this->force_file_truncation = true;
    }
    
    File file(restart_file, File::ReadOnly);
    real_t time;
    int iteration;

    if (file.hasAttribute("time")) {
      HighFive::Attribute attr_time = file.getAttribute("time");
      attr_time.read(time);
      HighFive::Attribute attr_ite = file.getAttribute("iteration");
      attr_ite.read(iteration);
    }
    else {
      if (group == "") {
        const size_t last_ite_index = file.getNumberObjects() - 4;
        group = file.getObjectName(last_ite_index);
      }
      HighFive::Group h5_group = file.getGroup(group);
      HighFive::Attribute attr_time = h5_group.getAttribute("time");
      attr_time.read(time);
      HighFive::Attribute attr_ite = h5_group.getAttribute("iteration");
      attr_ite.read(iteration);
      group = group + "/";
    }

    auto Nt = getShape(file, group + "rho")[0];

    if (Nt != device_params.Nx*device_params.Ny*device_params.Nz) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << device_params.Nx << "x" << device_params.Ny << "x" << device_params.Nz << "=" << device_params.Nx*device_params.Ny*device_params.Nz << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);
    using Table = std::vector<real_t>;

    std::cout << "Loading restart data from hdf5" << std::endl;
    
    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, group + var_name);
      // Parallel for here ?
      int lid = 0;
      for (int z=0; z < device_params.Nz; ++z) {
        for (int y=0; y < device_params.Ny; ++y) {
          for (int x=0; x < device_params.Nx; ++x) {
            Qhost(z+device_params.kbeg, y+device_params.jbeg, x+device_params.ibeg, var_id) = table[lid++];
          }
        }
      }
    };
    load_and_copy("rho", IR);
    load_and_copy("u",   IU);
    load_and_copy("v",   IV);
    load_and_copy("w",   IW);
    load_and_copy("prs", IP);

    Kokkos::deep_copy(Q, Qhost);

    BoundaryManager bc(params);
    bc.fillBoundaries(Q);

    if (time + params.device_params.epsilon > params.tend) {
      std::cerr << "Restart time is greater than end time : " << std::endl
                << "  time: " << time << "\ttend: " << params.tend << std::endl << std::endl; 
      throw std::runtime_error("ERROR : restart time is greater than the end time.");
    }

    std::cout << "Restart finished !" << std::endl;

    if (force_file_truncation) {
      file.~File(); // free the h5 before saving
      saveSolution(Q, iteration, time);
    }

    return {time, iteration};
  }
};

}