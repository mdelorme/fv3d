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
  char str_xdmf_header[] = 
  R"xml(<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0">
<Domain CollectionType="Temporal">
  <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">
    <Topology Name="Main Topology" TopologyType="3DSMesh" NumberOfElements="%d %d %d"/>
    )xml";
  #define format_xdmf_header(params)                                          \
          params.Nz + 1, params.Ny + 1, params.Nx + 1
  char str_xdmf_footer[] =
  R"xml(</Grid>
</Domain>
</Xdmf>
)xml";

  char str_xdmf_geometry[] = 
  R"xml(<Geometry Name="%s" GeometryType="X_Y_Z">
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s/x</DataItem>
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s/y</DataItem>
      <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s/z</DataItem>
    </Geometry>
    )xml";
  #define format_xdmf_geometry(params, path, facename)                                      \
          facename.c_str(),                                                                 \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (path + ".h5:/" + facename).c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (path + ".h5:/" + facename).c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (path + ".h5:/" + facename).c_str()

  char str_xdmf_grid_header[] =
  R"xml(<Grid Name="%s" GridType="Uniform">
        <Topology Reference="//Topology[@Name='Main Topology']" />
        <Geometry Reference="//Geometry[@Name='%s']" />)xml";
  #define format_xdmf_grid_header(facename)                                                 \
          facename.c_str(), facename.c_str()
  char str_xdmf_grid_footer[] = 
  R"xml(
      </Grid>)xml";

  char str_xdmf_ite_header[] =
  R"xml(
    <Grid GridType="Collection" CollectionType="Spatial">
      <Time TimeType="Single" Value="%lf" />
      )xml";
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
  )xml";

  char str_xdmf_scalar_field[] =
  R"xml(
        <Attribute Name="%s" AttributeType="Scalar" Center="Cell">
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s/%s</DataItem>
        </Attribute>)xml";
  #define format_xdmf_scalar_field(params, path, iteration, gridname, field)     \
          field,                                                                 \
          params.Nz, params.Ny, params.Nx,                                       \
          (path + ".h5").c_str(), iteration.c_str(), gridname.c_str(), field
  char str_xdmf_vector_field[] =
  R"xml(
        <Attribute Name="%s" AttributeType="Vector" Center="Cell">
          <DataItem Dimensions="%d %d %d 3" ItemType="Function" Function="JOIN($0, $1, $2)">
            <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s/%s</DataItem>
            <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s/%s</DataItem>
            <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s/%s</DataItem>
          </DataItem>
        </Attribute>)xml";
  #define format_xdmf_vector_field(params, path, iteration, gridname, name, field_x, field_y, field_z)   \
          name,                                                                                          \
          params.Nz, params.Ny, params.Nx,                                                               \
          params.Nz, params.Ny, params.Nx,                                                               \
          (path + ".h5").c_str(), iteration.c_str(), gridname.c_str(), field_x,                          \
          params.Nz, params.Ny, params.Nx,                                                               \
          (path + ".h5").c_str(), iteration.c_str(), gridname.c_str(), field_y,                          \
          params.Nz, params.Ny, params.Nx,                                                               \
          (path + ".h5").c_str(), iteration.c_str(), gridname.c_str(), field_z

} // anonymous namespace

class IOManager {
public:
  Params params;
  DeviceParams &device_params;

  IOManager(Params &params)
    : params(params), device_params(params.device_params) {};

  ~IOManager() = default;

  void saveSolution(const Array &Q, int iteration, real_t t, real_t dt) {
    if (params.multiple_outputs)
      saveSolutionMultiple(Q, iteration, t, dt);
    else
      saveSolutionUnique(Q, iteration, t, dt);
  }

  void saveSolutionMultiple(const Array &Q, int iteration, real_t t, real_t dt)
  {}
/*
  void saveSolutionMultiple(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << params.filename_out << "_" << std::setw(4) << std::setfill('0') << iteration;
    std::string path = oss.str();
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

    fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params, path));
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, path, empty_string, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, path, empty_string, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, path, empty_string, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }
*/
  void saveSolutionUnique(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    oss << "ite_" << std::setw(4) << std::setfill('0') << iteration;
    std::string iteration_str = oss.str();
      
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
      file.createAttribute("iteration", iteration);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(device_params));

      for (auto [face, facename] : facename_map) {
        std::vector<real_t> x, y, z;
        // -- vertex pos
        for (int k=device_params.kbeg; k <= device_params.kend; ++k) {
          for (int j=device_params.jbeg; j <= device_params.jend; ++j) {
            for (int i=device_params.ibeg; i <= device_params.iend; ++i) {
              const Pos p = mapShell(face, 
                device_params.xmin + (i-device_params.ibeg) * device_params.dx,
                device_params.ymin + (j-device_params.jbeg) * device_params.dy,
                device_params.zmin + (k-device_params.kbeg) * device_params.dz
              );
              x.push_back(p[IX]);
              y.push_back(p[IY]);
              z.push_back(p[IZ]);
            }
          }
        }

        file.createDataSet(facename + "/x", x);
        file.createDataSet(facename + "/y", y);
        file.createDataSet(facename + "/z", z);
        fprintf(xdmf_fd, str_xdmf_geometry, format_xdmf_geometry(device_params, params.filename_out, facename));
      }
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    using Table = std::vector<std::vector<std::vector<real_t>>>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    auto iteration_group = file.createGroup(iteration_str);
    iteration_group.createAttribute("time", t);

    for (auto [face, facename] : facename_map) {
      auto grid_group = iteration_group.createGroup(facename);

      Table trho, tu, tv, tw, tprs;
      for (int k=device_params.kbeg; k<device_params.kend; ++k) {
        std::vector<std::vector<real_t>> rcrho, rcu, rcv, rcw, rcprs;

        for (int j=device_params.jbeg; j<device_params.jend; ++j) {
          std::vector<real_t> rrho, ru, rv, rw, rprs;

          for (int i=device_params.ibeg; i<device_params.iend; ++i) {
            real_t rho = Qhost(face, k, j, i, IR);
            real_t u   = Qhost(face, k, j, i, IU);
            real_t v   = Qhost(face, k, j, i, IV);
            real_t w   = Qhost(face, k, j, i, IW);
            real_t p   = Qhost(face, k, j, i, IP);

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

      grid_group.createDataSet("rho", trho);
      grid_group.createDataSet("u", tu);
      grid_group.createDataSet("v", tv);
      grid_group.createDataSet("w", tw);
      grid_group.createDataSet("prs", tprs);
    }

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    for (auto [face, facename] : facename_map) {
      fprintf(xdmf_fd, str_xdmf_grid_header, format_xdmf_grid_header(facename));
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, params.filename_out, iteration_str, facename, "rho"));
      fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(device_params, params.filename_out, iteration_str, facename, "velocity", "u", "v", "w"));
      fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(device_params, params.filename_out, iteration_str, facename, "prs"));
      fprintf(xdmf_fd, "%s", str_xdmf_grid_footer);
    }
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
    int face = 0;

    auto load_and_copy = [&](std::string var_name, IVar var_id) {
      auto table = load<Table>(file, var_name);
      // Parallel for here ?
      int lid = 0;
      for (int z=0; z < device_params.Nz; ++z) {
        for (int y=0; y < device_params.Ny; ++y) {
          for (int x=0; x < device_params.Nx; ++x) {
            Qhost(face, z+device_params.kbeg, y+device_params.jbeg, x+device_params.ibeg, var_id) = table[lid++];
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