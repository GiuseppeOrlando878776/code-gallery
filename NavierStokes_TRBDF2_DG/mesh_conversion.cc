#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

int main() {
  using namespace dealii;

  Triangulation<2> triangulation;

  GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream infile("refcircle_structuredV3.msh");
  grid_in.read_msh(infile);

  GridOut grid_out;

  std::ofstream outfile("mesh.ucd");
  grid_out.write_ucd(triangulation, outfile);

  std::ofstream outfile_vtk("mesh.vtk");
  grid_out.write_vtk(triangulation, outfile_vtk);
}
