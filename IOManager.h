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
  #define format_xdmf_header(params)                                          \
          params.Nz + 1, params.Ny + 1, params.Nx + 1,                                        \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (params.filename_out + ".h5").c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (params.filename_out + ".h5").c_str(), \
          params.Nz + 1, params.Ny + 1, params.Nx + 1, (params.filename_out + ".h5").c_str()
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
  #define format_xdmf_scalar_field(params, group, field)                         \
          field,                                                                 \
          params.Nz, params.Ny, params.Nx,                                       \
          (params.filename_out + ".h5").c_str(), group.c_str(), field
  char str_xdmf_vector_field[] =
  R"xml(
      <Attribute Name="%s" AttributeType="Vector" Center="Cell">
        <DataItem Dimensions="%d %d %d 3" ItemType="Function" Function="JOIN($0, $1, $2)">
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
          <DataItem Dimensions="%d %d %d" NumberType="Float" Precision="8" Format="HDF">%s:/%s/%s</DataItem>
        </DataItem>
      </Attribute>)xml";
  #define format_xdmf_vector_field(params, group, name, field_x, field_y, field_z)   \
          name,                                                                      \
          params.Nz, params.Ny, params.Nx,                                           \
          params.Nz, params.Ny, params.Nx,                                           \
          (params.filename_out + ".h5").c_str(), group.c_str(), field_x,             \
          params.Nz, params.Ny, params.Nx,                                           \
          (params.filename_out + ".h5").c_str(), group.c_str(), field_y,             \
          params.Nz, params.Ny, params.Nx,                                           \
          (params.filename_out + ".h5").c_str(), group.c_str(), field_z
  char str_xdmf_ite_footer[] =
  R"xml(
    </Grid>
  )xml";
} // anonymous namespace

class IOManager {
public:
  Params params;

  IOManager(Params &params)
    : params(params) {};

  ~IOManager() = default;

  void saveSolution(const Array &Q, int iteration, real_t t, real_t dt) {
    std::ostringstream oss;
    
    std::setw(4);
    std::setfill('0');
    oss << "ite_" << iteration;
    std::string path = oss.str();
      
    auto flag_h5 = (iteration == 0 ? File::Truncate : File::ReadWrite);
    auto flag_xdmf = (iteration == 0 ? "w+" : "r+");
    File file(params.filename_out + ".h5", flag_h5);
    FILE* xdmf_fd = fopen((params.filename_out + ".xdmf").c_str(), flag_xdmf);

    if (iteration == 0) {
      file.createAttribute("Ntx", params.Ntx);
      file.createAttribute("Nty", params.Nty);
      file.createAttribute("Ntz", params.Ntz);
      file.createAttribute("Nx", params.Nx);
      file.createAttribute("Ny", params.Ny);
      file.createAttribute("Nz", params.Nz);
      file.createAttribute("ibeg", params.ibeg);
      file.createAttribute("iend", params.iend);
      file.createAttribute("jbeg", params.jbeg);
      file.createAttribute("jend", params.jend);
      file.createAttribute("kbeg", params.kbeg);
      file.createAttribute("kend", params.kend);
      file.createAttribute("problem", params.problem);

      std::vector<real_t> x, y, z;
      // -- vertex pos
      for (int k=params.kbeg; k <= params.kend; ++k) {
        for (int j=params.jbeg; j <= params.jend; ++j) {
          for (int i=params.ibeg; i <= params.iend; ++i) {
            x.push_back((i-params.ibeg) * params.dx);
            y.push_back((j-params.jbeg) * params.dy);
            z.push_back((k-params.kbeg) * params.dz);
          }
        }
      }

      file.createDataSet("x", x);
      file.createDataSet("y", y);
      file.createDataSet("z", z);

      fprintf(xdmf_fd, str_xdmf_header, format_xdmf_header(params));
      fprintf(xdmf_fd, "%s", str_xdmf_footer);
    }
    
    using Table = std::vector<std::vector<std::vector<real_t>>>;

    auto Qhost = Kokkos::create_mirror(Q);
    Kokkos::deep_copy(Qhost, Q);

    Table trho, tu, tv, tw, tprs;
    for (int k=params.kbeg; k<params.kend; ++k) {
      std::vector<std::vector<real_t>> rcrho, rcu, rcv, rcw, rcprs;

      for (int j=params.jbeg; j<params.jend; ++j) {
        std::vector<real_t> rrho, ru, rv, rw, rprs;

        for (int i=params.ibeg; i<params.iend; ++i) {
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

    auto group = file.createGroup(path);
    group.createDataSet("rho", trho);
    group.createDataSet("u", tu);
    group.createDataSet("v", tv);
    group.createDataSet("w", tw);
    group.createDataSet("prs", tprs);
    group.createAttribute("time", t);

    fseek(xdmf_fd, -sizeof(str_xdmf_footer), SEEK_END);
    fprintf(xdmf_fd, str_xdmf_ite_header, t);
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, "rho"));
    fprintf(xdmf_fd, str_xdmf_vector_field, format_xdmf_vector_field(params, path, "velocity", "u", "v", "w"));
    fprintf(xdmf_fd, str_xdmf_scalar_field, format_xdmf_scalar_field(params, path, "prs"));
    fprintf(xdmf_fd, "%s", str_xdmf_ite_footer);
    fprintf(xdmf_fd, "%s", str_xdmf_footer);
    fclose(xdmf_fd);
  }
};

}