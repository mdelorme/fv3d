#pragma once

#include <highfive/H5Easy.hpp>
#include <ostream>
#include <iomanip>

// https://visit-sphinx-github-user-manual.readthedocs.io/en/3.4rc/data_into_visit/XdmfFormat.html

#include "SimInfo.h"

using namespace H5Easy;

namespace fv3d {

  // xdmf strings
namespace {
  char str_xdmf_header[] = R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain CollectionType="Temporal">
  <Grid Name="MainTimeSeries" GridType="Collection" CollectionType="Temporal">
    <Topology Name="Main Topology" TopologyType="3DSMesh" NumberOfElements="%d %d %d"/>
    <Geometry Name="Main Geometry" GeometryType="X_Y_Z">
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/x</DataItem>
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/y</DataItem>
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/z</DataItem>
    </Geometry>
    )xml";
  #define format_xdmf_header(params, iteration_path)                                     \
          params.Nz + 1, params.Ny + 1, params.Nx + 1,                                   \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (iteration_path + ".h5").c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (iteration_path + ".h5").c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (iteration_path + ".h5").c_str()
  char str_xdmf_footer[] =
  R"xml(
  </Grid>
</Domain>
</Xdmf>)xml";

  char str_xdmf_ite_header[] =
  R"xml(
    <Grid Name="Cells" GridType="Uniform">
      <Time TimeType="Single" Value="%lf" />
      <Topology Reference="//Topology[@Name='Main Topology']" />
      <Geometry Reference="//Geometry[@Name='Main Geometry']" />)xml";
  char str_xdmf_scalar_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
      </Attribute>)xml";
  #define format_xdmf_scalar_field(params, iteration_path, group, field) \
          field,                                                         \
          params.Nz, params.Ny, params.Nx,                               \
          (iteration_path + ".h5").c_str(), group.c_str(), field
  char str_xdmf_vector_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="%d %d %d 3" ItemType="Function" Function="JOIN($0, $1, $2)">
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
        </DataItem>
      </Attribute>)xml";
  #define format_xdmf_vector_field(params, iteration_path, group, name, field_x, field_y, field_z) \
          name,                                                                                    \
          params.Nz, params.Ny, params.Nx,                                                         \
          params.Nz, params.Ny, params.Nx,                                                         \
          (iteration_path + ".h5").c_str(), group.c_str(), field_x,                                \
          params.Nz, params.Ny, params.Nx,                                                         \
          (iteration_path + ".h5").c_str(), group.c_str(), field_y,                                \
          params.Nz, params.Ny, params.Nx,                                                         \
          (iteration_path + ".h5").c_str(), group.c_str(), field_z
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
  )xml";
} // anonymous namespace

class IOManager {
public:
  Params &params;
  DeviceParams &device_params;

  IOManager(Params &params)
    : params(params), device_params(params.device_params) {};

  ~IOManager() = default;

  void saveSolution(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    std::setw(4);
    std::setfill('0');
    oss << "ite_" << iteration;
    std::string iteration_path = oss.str();
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
    file.createAttribute("iteration", iteration);

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

    std::string empty_string = "";

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, iteration_path));
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, iteration_path, empty_string, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, iteration_path, empty_string, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, iteration_path, empty_string, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  void saveSolutionUnique(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << "ite_" << std::setw(4) << std::setfill('0') << iteration;
    std::string iteration_path = oss.str();
      
    auto flag_h5 = (iteration == 0 ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (iteration == 0 ? "w+" : "r+");
    File file(params.filename_out + ".h5", flag_h5);
    FILE* xdmf_fd = fopen((params.filename_out + ".xdmf").c_str(), flag_xdmf);

    if (iteration == 0) {
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

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, iteration_path));
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    using Table = std::vector<std::vector<std::vector<real_t>>>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tw, tprs;
    for (int k=device_params.kbeg; k<device_params.kend; ++k) {
      std::vector<std::vector<real_t>> rcrho, rcu, rcv, rcw, rcprs;

      for (int j=device_params.jbeg; j<device_params.jend; ++j) {
        std::vector<real_t> rrho, ru, rv, rw, rprs;

        for (int i=device_params.ibeg; i<device_params.iend; ++i) {
          real_t rho = Qhost(k, j, i, IR);
          real_t u   = Qhost(k, j, i, IU);
          real_t v   = Qhost(k, j, i, IV);
          real_t w   = Qhost(k, j, i, IW);
          real_t p   = Qhost(k, j, i, IP);

          rrho.push_back(rho);
          ru.push_back(u);
          rv.push_back(v);
          rw.push_back(w);
          rprs.push_back(p);
        }

        rcrho.push_back(rrho);
        rcu.push_back(ru);
        rcv.push_back(rv);
        rcw.push_back(rw);
        rcprs.push_back(rprs);
      }

      trho.push_back(rcrho);
      tu.push_back(rcu);
      tv.push_back(rcv);
      tw.push_back(rcw);
      tprs.push_back(rcprs);
    }

    auto group = file.createGroup(iteration_path);
    group.createDataSet("rho", trho);
    group.createDataSet("u", tu);
    group.createDataSet("v", tv);
    group.createDataSet("w", tw);
    group.createDataSet("prs", tprs);
    group.createAttribute("time", t);

    const std::string empty_string = "";

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, iteration_path, empty_string, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, iteration_path, empty_string, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, iteration_path, empty_string, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }

  RestartInfo loadSnapshot(Array &Q) {
    File file(params.restart_file, File::ReadOnly);

    auto Nt = getShape(file, "rho")[0];

    if (Nt != device_params.Nx*device_params.Ny*device_params.Nz) {
      std::cerr << "Attempting to restart with a different resolution ! Ncells (restart) = " << Nt << "; Run resolution = " 
                << device_params.Nx << "x" << device_params.Ny << "x" << device_params.Nz << "=" << device_params.Nx*device_params.Ny*device_params.Nz << std::endl;
      throw std::runtime_error("ERROR : Trying to restart from a file with a different resolution !");
    }

    auto Qhost = Kokkos::create_mirror(Q);
    using Table = std::vector<real_t>;

    std::cout << "Loading restart data from hdf5" << std::endl;
    
    throw std::runtime_error("Restart on shell grid is not implemented.");

    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, var_name);
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
    load_and_copy("u", IU);
    load_and_copy("v", IV);
    load_and_copy("w", IW);
    load_and_copy("prs", IP);

    Kokkos::deep_copy(Q, Qhost);

    std::cout << "Restart finished !" << std::endl;

    real_t time = loadAttribute<real_t>(file, "/", "time");
    int iteration = loadAttribute<int>(file, "/", "iteration");

    return {time, iteration};
  }
};

}