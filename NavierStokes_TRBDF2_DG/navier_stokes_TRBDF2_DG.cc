/* Author: Giuseppe Orlando, 2022. */

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/fe/component_mask.h>

#include <deal.II/base/timer.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "navier_stokes_TRBDF2_operator.h"

using namespace NS_TRBDF2;

// @sect{The <code>NavierStokesProjection</code> class}

// Now we are ready for the main class of the program. It implements the calls to the various steps
// of the projection method for Navier-Stokes equations.
//
template<int dim>
class NavierStokesProjection {
public:
  NavierStokesProjection(RunTimeParameters::Data_Storage& data);

  void run(const bool verbose = false, const unsigned int output_interval = 10);

protected:
  const double t_0;
  const double T;
  const double gamma;         //--- TR-BDF2 parameter
  unsigned int TR_BDF2_stage; //--- Flag to check at which current stage of TR-BDF2 are
  const double Re;
  double       dt;

  EquationData::Velocity<dim> vel_init;
  EquationData::Pressure<dim> pres_init; /*--- Instance of 'Velocity' and 'Pressure' classes to initialize. ---*/

  parallel::distributed::Triangulation<dim> triangulation;

  /*--- Finite Element spaces ---*/
  FESystem<dim> fe_velocity;
  FESystem<dim> fe_pressure;

  /*--- Handler for dofs ---*/
  DoFHandler<dim> dof_handler_velocity;
  DoFHandler<dim> dof_handler_pressure;

  /*--- Quadrature formulas for velocity and pressure, respectively ---*/
  QGauss<dim> quadrature_pressure;
  QGauss<dim> quadrature_velocity;

  /*--- Now we define all the vectors for the solution. We start from the pressure
        with p^n, p^(n+gamma) and a vector for rhs ---*/
  LinearAlgebra::distributed::Vector<double> pres_n;
  LinearAlgebra::distributed::Vector<double> pres_n_gamma;
  LinearAlgebra::distributed::Vector<double> rhs_p;

  /*--- Next, we move to the velocity, with u^n, u^(n-1), u^(n+gamma/2),
        u^(n+gamma) and other two auxiliary vectors as well as the rhs ---*/
  LinearAlgebra::distributed::Vector<double> u_n;
  LinearAlgebra::distributed::Vector<double> u_n_minus_1;
  LinearAlgebra::distributed::Vector<double> u_extr;
  LinearAlgebra::distributed::Vector<double> u_n_gamma;
  LinearAlgebra::distributed::Vector<double> u_star;
  LinearAlgebra::distributed::Vector<double> u_tmp;
  LinearAlgebra::distributed::Vector<double> rhs_u;
  LinearAlgebra::distributed::Vector<double> grad_pres_n_gamma;

  /*--- Variables for statistics ---*/
  std::vector<Point<dim>> obstacle_points;
  std::vector<Point<dim>> horizontal_wake_points;
  std::vector<Point<dim>> vertical_profile_points1;
  std::vector<Point<dim>> vertical_profile_points2;
  std::vector<Point<dim>> vertical_profile_points3;

  std::vector<double> avg_pressure;
  std::vector<double> avg_stress;

  std::vector<Vector<double>> avg_horizontal_velocity;
  std::vector<Vector<double>> avg_vertical_velocity1;
  std::vector<Vector<double>> avg_vertical_velocity2;
  std::vector<Vector<double>> avg_vertical_velocity3;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation(const unsigned int n_refines);

  void setup_dofs();

  void initialize();

  void interpolate_velocity();

  void diffusion_step();

  void projection_step();

  void project_grad(const unsigned int flag);

  double get_maximal_velocity();

  double get_maximal_difference_velocity();

  void output_results(const unsigned int step);

private:
  void compute_lift_and_drag();

  void compute_lipschitz_number();

  /*--- Technical member to handle the various steps ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  /*--- Now we need an instance of the class implemented before with the weak form ---*/
  NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                 EquationData::degree_p + 1, EquationData::degree_p + 2,
                                 LinearAlgebra::distributed::Vector<double>> navier_stokes_matrix;

  /*--- This is an instance for geometric multigrid preconditioner ---*/
  MGLevelObject<NavierStokesProjectionOperator<dim, EquationData::degree_p, EquationData::degree_p + 1,
                                               EquationData::degree_p + 1, EquationData::degree_p + 2,
                                               LinearAlgebra::distributed::Vector<float>>> mg_matrices;

  /*--- Here we define two 'AffineConstraints' instance, one for each finite element space.
        This is just a technical issue, due to MatrixFree requirements. In general
        this class is used to impose boundary conditions (or any kind of constraints), but in this case, since
        we are using a weak imposition of bcs, everything is already in the weak forms and so these instances
        will be default constructed ---*/
  AffineConstraints<double> constraints_velocity,
                            constraints_pressure;

  /*--- Now a bunch of variables handled by 'ParamHandler' introduced at the beginning of the code ---*/
  unsigned int max_its;
  double       eps;

  unsigned int n_refines;
  bool         import_mesh;

  std::string  saving_dir;

  bool         restart,
               save_for_restart;
  unsigned int step_restart;
  double       time_restart;
  bool         as_initial_conditions;

  /*--- Finally, some output related streams ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_n_dofs_velocity;
  std::ofstream output_n_dofs_pressure;

  std::ofstream output_lift;
  std::ofstream output_drag;

  std::ofstream output_lipschitz;
};


// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
NavierStokesProjection<dim>::NavierStokesProjection(RunTimeParameters::Data_Storage& data):
  t_0(data.initial_time),
  T(data.final_time),
  gamma(2.0 - std::sqrt(2.0)),  //--- Save also in the NavierStokes class the TR-BDF2 parameter value
  TR_BDF2_stage(1),             //--- Initialize the flag for the TR_BDF2 stage
  Re(data.Reynolds),
  dt(data.dt),
  vel_init(data.initial_time),
  pres_init(data.initial_time),
  triangulation(MPI_COMM_WORLD, parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_velocity(FE_DGQ<dim>(EquationData::degree_p + 1), dim),
  fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
  dof_handler_velocity(triangulation),
  dof_handler_pressure(triangulation),
  quadrature_pressure(EquationData::degree_p + 1),
  quadrature_velocity(EquationData::degree_p + 2),
  navier_stokes_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  n_refines(data.n_refines),
  import_mesh(data.import_mesh),
  saving_dir(data.dir),
  restart(data.restart),
  save_for_restart(data.save_for_restart),
  step_restart(data.step_restart),
  time_restart(data.time_restart),
  as_initial_conditions(data.as_initial_conditions),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_n_dofs_velocity("./" + data.dir + "/n_dofs_velocity.dat", std::ofstream::out),
  output_n_dofs_pressure("./" + data.dir + "/n_dofs_pressure.dat", std::ofstream::out),
  output_lift("./" + data.dir + "/lift.dat", std::ofstream::out),
  output_drag("./" + data.dir + "/drag.dat", std::ofstream::out),
  output_lipschitz("./" + data.dir + "/lipschitz.dat", std::ofstream::out)  {

    if(EquationData::degree_p < 1) {
      pcout
      << " WARNING: The chosen pair of finite element spaces is not stable."
      << std::endl
      << " The obtained results will be nonsense" << std::endl;
    }

    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    create_triangulation(n_refines);
    setup_dofs();
    initialize();
  }


// The method that creates the triangulation and refines it the needed number
// of times.
//
template<int dim>
void NavierStokesProjection<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  triangulation.clear();

  double x_start, y_start;

  if(import_mesh) {
    GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream f("unstr_sqcyl_coarse.msh");
    gridin.read_msh(f);

    x_start = -10.0;
    y_start = -10.0;
  }
  else {
    parallel::distributed::Triangulation<dim> tria1(MPI_COMM_WORLD),
                                              tria2(MPI_COMM_WORLD),
                                              tria3(MPI_COMM_WORLD),
                                              tria4(MPI_COMM_WORLD),
                                              tria5(MPI_COMM_WORLD),
                                              tria6(MPI_COMM_WORLD),
                                              tria7(MPI_COMM_WORLD),
                                              tria8(MPI_COMM_WORLD),
                                              tria9(MPI_COMM_WORLD),
                                              tria10(MPI_COMM_WORLD),
                                              tria11(MPI_COMM_WORLD),
                                              tria12(MPI_COMM_WORLD);

    GridGenerator::subdivided_hyper_rectangle(tria1, {10, 16},
                                              Point<dim>(0.0, 10.7),
                                              Point<dim>(9.3, 20.0));
    GridGenerator::subdivided_hyper_rectangle(tria2, {14, 16},
                                              Point<dim>(9.3, 10.7),
                                              Point<dim>(10.7, 20.0));
    GridGenerator::subdivided_hyper_rectangle(tria3, {20, 16},
                                              Point<dim>(10.7, 10.7),
                                              Point<dim>(30.0, 20.0));
    GridGenerator::subdivided_hyper_rectangle(tria4, {10, 14},
                                              Point<dim>(0.0, 9.3),
                                              Point<dim>(9.3, 10.7));
    GridGenerator::subdivided_hyper_rectangle(tria5, {14, 2},
                                              Point<dim>(9.3, 10.5),
                                              Point<dim>(10.7, 10.7));
    GridGenerator::subdivided_hyper_rectangle(tria6, {2, 10},
                                              Point<dim>(9.3, 9.5),
                                              Point<dim>(9.5, 10.5));
    GridGenerator::subdivided_hyper_rectangle(tria7, {2, 10},
                                              Point<dim>(10.5, 9.5),
                                              Point<dim>(10.7, 10.5));
    GridGenerator::subdivided_hyper_rectangle(tria8, {14, 2},
                                              Point<dim>(9.3, 9.3),
                                              Point<dim>(10.7, 9.5));
    GridGenerator::subdivided_hyper_rectangle(tria9, {20, 14},
                                              Point<dim>(10.7, 9.3),
                                              Point<dim>(30.0, 10.7));
    GridGenerator::subdivided_hyper_rectangle(tria10, {10, 16},
                                              Point<dim>(0.0, 0.0),
                                              Point<dim>(9.3, 9.3));
    GridGenerator::subdivided_hyper_rectangle(tria11, {14, 16},
                                              Point<dim>(9.3, 0.0),
                                              Point<dim>(10.7, 9.3));
    GridGenerator::subdivided_hyper_rectangle(tria12, {20, 16},
                                              Point<dim>(10.7, 0.0),
                                              Point<dim>(30.0, 9.3));
    GridGenerator::merge_triangulations({&tria1, &tria2, &tria3, &tria4, &tria5, &tria6,
                                         &tria7, &tria8, &tria9, &tria10, &tria11, &tria12},
                                         triangulation, 1e-8, true);

    x_start = 0.0;
    y_start = 0.0;
  }

  /*--- Set boundary id for the triangulation ---*/
  for(const auto& face : triangulation.active_face_iterators()) {
    if(face->at_boundary()) {
      const Point<dim> center = face->center();
      // left side
      if(std::abs(center[0] - x_start) < 1e-10) {
        face->set_boundary_id(0);
      }
      // right side
      else if(std::abs(center[0] - (30.0 + x_start)) < 1e-10) {
        face->set_boundary_id(1);
      }
      // cylinder boundary
      else if(center[0] < x_start + 10.5 + 1e-10 && center[0] > x_start + 9.5 - 1e-10 &&
              center[1] < y_start + 10.5 + 1e-10 && center[1] > y_start + 9.5 - 1e-10) {
        face->set_boundary_id(2);
      }
      // sides of channel
      else {
        Assert(std::abs(center[1] - y_start) < 1.0e-10 ||
               std::abs(center[1] - (20.0 + y_start)) < 1.0e-10,
               ExcInternalError());
        face->set_boundary_id(3);
      }
    }
  }

  /*--- We strongly advice to check the documentation to verify the meaning of all input parameters. ---*/
  if(restart) {
    triangulation.load("./" + saving_dir + "/solution_ser-" + Utilities::int_to_string(step_restart, 5));
  }
  else {
    pcout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
  }
}


// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom, and
// initializes the vectors that we will use.
//
template<int dim>
void NavierStokesProjection<dim>::setup_dofs() {
  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Distribute dofs ---*/
  dof_handler_velocity.distribute_dofs(fe_velocity);
  dof_handler_pressure.distribute_dofs(fe_pressure);

  pcout << "dim (X_h) = " << dof_handler_velocity.n_dofs()
        << std::endl
        << "dim (M_h) = " << dof_handler_pressure.n_dofs()
        << std::endl
        << "Re        = " << Re << std::endl
        << std::endl;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_n_dofs_velocity << dof_handler_velocity.n_dofs() << std::endl;
    output_n_dofs_pressure << dof_handler_pressure.n_dofs() << std::endl;
  }

  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_gradients | update_JxW_values |
                                                         update_quadrature_points | update_values);
  additional_data.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors | update_values);
  additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors | update_values);
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Vector of dof_handlers to feed the 'MatrixFree'. Here the order
                                                          counts and enters into the game as parameter of FEEvaluation and
                                                          FEFaceEvaluation in the previous class ---*/
  dof_handlers.push_back(&dof_handler_velocity);
  dof_handlers.push_back(&dof_handler_pressure);

  constraints_velocity.clear();
  constraints_velocity.close();
  constraints_pressure.clear();
  constraints_pressure.close();
  std::vector<const AffineConstraints<double>*> constraints;
  constraints.push_back(&constraints_velocity);
  constraints.push_back(&constraints_pressure);

  std::vector<QGauss<1>> quadratures; /*--- We cannot directly use 'quadrature_velocity' and 'quadrature_pressure',
                                            because the 'MatrixFree' structure wants a quadrature formula for 1D
                                            (this is way the template parameter of the previous class was called 'n_q_points_1d_p'
                                             and 'n_q_points_1d_v' and the reason of '1' as QGauss template parameter). ---*/
  quadratures.push_back(QGauss<1>(EquationData::degree_p + 2));
  quadratures.push_back(QGauss<1>(EquationData::degree_p + 1));

  /*--- Initialize the matrix-free structure and size properly the vectors. Here again the
        second input argument of the 'initialize_dof_vector' method depends on the order of 'dof_handlers' ---*/
  matrix_free_storage->reinit(MappingQ1<dim>(), dof_handlers, constraints, quadratures, additional_data);

  matrix_free_storage->initialize_dof_vector(u_star, 0);
  matrix_free_storage->initialize_dof_vector(rhs_u, 0);
  matrix_free_storage->initialize_dof_vector(u_n, 0);
  matrix_free_storage->initialize_dof_vector(u_extr, 0);
  matrix_free_storage->initialize_dof_vector(u_n_minus_1, 0);
  matrix_free_storage->initialize_dof_vector(u_n_gamma, 0);
  matrix_free_storage->initialize_dof_vector(u_tmp, 0);
  matrix_free_storage->initialize_dof_vector(grad_pres_n_gamma, 0);

  matrix_free_storage->initialize_dof_vector(pres_n_gamma, 1);
  matrix_free_storage->initialize_dof_vector(pres_n, 1);
  matrix_free_storage->initialize_dof_vector(rhs_p, 1);

  /*--- Initialize the multigrid structure. We dedicate ad hoc 'dof_handlers_mg' and 'constraints_mg' because
        we use float as type. Moreover we can initialize already with the index of the finite element of the pressure;
        anyway we need by requirement to declare also structures for the velocity for coherence (basically because
        the index of finite element space has to be the same, so the pressure has to be the second).---*/
  mg_matrices.clear_elements();
  dof_handler_velocity.distribute_mg_dofs();
  dof_handler_pressure.distribute_mg_dofs();

  const unsigned int nlevels = triangulation.n_global_levels();
  mg_matrices.resize(0, nlevels - 1);
  for(unsigned int level = 0; level < nlevels; ++level) {
    typename MatrixFree<dim, float>::AdditionalData additional_data_mg;

    additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, float>::AdditionalData::none;
    additional_data_mg.mapping_update_flags                = (update_gradients | update_JxW_values);
    additional_data_mg.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values);
    additional_data_mg.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values);
    additional_data_mg.mg_level = level;

    std::vector<const DoFHandler<dim>*> dof_handlers_mg;
    dof_handlers_mg.push_back(&dof_handler_velocity);
    dof_handlers_mg.push_back(&dof_handler_pressure);
    std::vector<const AffineConstraints<float>*> constraints_mg;
    AffineConstraints<float> constraints_velocity_mg;
    constraints_velocity_mg.clear();
    constraints_velocity_mg.close();
    constraints_mg.push_back(&constraints_velocity_mg);
    AffineConstraints<float> constraints_pressure_mg;
    constraints_pressure_mg.clear();
    constraints_pressure_mg.close();
    constraints_mg.push_back(&constraints_pressure_mg);

    std::shared_ptr<MatrixFree<dim, float>> mg_mf_storage_level(new MatrixFree<dim, float>());
    mg_mf_storage_level->reinit(MappingQ1<dim>(), dof_handlers_mg, constraints_mg, quadratures, additional_data_mg);
    const std::vector<unsigned int> tmp = {1};
    mg_matrices[level].initialize(mg_mf_storage_level, tmp, tmp);
    mg_matrices[level].set_dt(dt);
    mg_matrices[level].set_NS_stage(2);
  }
}


// This method loads the initial data. It simply uses the class <code>Pressure</code> instance for the pressure
// and the class <code>Velocity</code> instance for the velocity.
//
template<int dim>
void NavierStokesProjection<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize pressure and velocity");

  if(restart) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);

    u_n.zero_out_ghost_values();
    u_n_minus_1.zero_out_ghost_values();
    pres_n.zero_out_ghost_values();

    std::vector<LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);

    solution_transfer_velocity.deserialize(velocities);
    solution_transfer_pressure.deserialize(pres_n);

    if(n_refines - (triangulation.n_global_levels() - 1) > 0) {
      std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities_tmp;
      velocities_tmp.push_back(&u_n);
      velocities_tmp.push_back(&u_n_minus_1);

      solution_transfer_velocity.prepare_for_coarsening_and_refinement(velocities_tmp);
      solution_transfer_pressure.prepare_for_coarsening_and_refinement(pres_n);

      triangulation.refine_global(n_refines - (triangulation.n_global_levels() - 1));

      setup_dofs();

      LinearAlgebra::distributed::Vector<double> transfer_velocity,
                                                 transfer_velocity_minus_1,
                                                 transfer_pressure;
      transfer_velocity.reinit(u_n);
      transfer_velocity.zero_out_ghost_values();
      transfer_velocity_minus_1.reinit(u_n_minus_1);
      transfer_velocity_minus_1.zero_out_ghost_values();
      transfer_pressure.reinit(pres_n);
      transfer_pressure.zero_out_ghost_values();

      std::vector<LinearAlgebra::distributed::Vector<double>*> transfer_velocities;
      transfer_velocities.push_back(&transfer_velocity);
      transfer_velocities.push_back(&transfer_velocity_minus_1);
      solution_transfer_velocity.interpolate(transfer_velocities);
      transfer_velocity.update_ghost_values();
      transfer_velocity_minus_1.update_ghost_values();
      solution_transfer_pressure.interpolate(transfer_pressure);
      transfer_pressure.update_ghost_values();

      u_n         = transfer_velocity;
      u_n_minus_1 = transfer_velocity_minus_1;
      pres_n      = transfer_pressure;
    }
  }
  else {
    VectorTools::interpolate(dof_handler_pressure, pres_init, pres_n);

    VectorTools::interpolate(dof_handler_velocity, vel_init, u_n_minus_1);
    VectorTools::interpolate(dof_handler_velocity, vel_init, u_n);
  }
}


// This function computes the extrapolated velocity to be used in the momentum predictor
//
template<int dim>
void NavierStokesProjection<dim>::interpolate_velocity() {
  TimerOutput::Scope t(time_table, "Interpolate velocity");

  //--- TR-BDF2 first step
  if(TR_BDF2_stage == 1) {
    u_extr.equ(1.0 + gamma/(2.0*(1.0 - gamma)), u_n);
    u_tmp.equ(gamma/(2.0*(1.0 - gamma)), u_n_minus_1);
    u_extr.add(-1.0, u_tmp);
  }
  //--- TR-BDF2 second step
  else {
    u_extr.equ(1.0 + (1.0 - gamma)/gamma, u_n_gamma);
    u_tmp.equ((1.0 - gamma)/gamma, u_n);
    u_extr.add(-1.0, u_tmp);
  }
}


// We are finally ready to solve the diffusion step.
//
template<int dim>
void NavierStokesProjection<dim>::diffusion_step() {
  TimerOutput::Scope t(time_table, "Diffusion step");

  /*--- We first speicify that we want to deal with velocity dof_handler (index 0, since it is the first one
        in the 'dof_handlers' vector) ---*/
  const std::vector<unsigned int> tmp = {0};
  navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

  /*--- Next, we specify at we are at stage 1, namely the diffusion step ---*/
  navier_stokes_matrix.set_NS_stage(1);

  /*--- Now, we compute the right-hand side and we set the convective velocity. The necessity of 'set_u_extr' is
        that this quantity is required in the bilinear forms and we can't use a vector of src like on the right-hand side,
        so it has to be available ---*/
  if(TR_BDF2_stage == 1) {
    navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_extr, pres_n});
    navier_stokes_matrix.set_u_extr(u_extr);
    u_star.equ(1.0, u_extr);
  }
  else {
    navier_stokes_matrix.vmult_rhs_velocity(rhs_u, {u_n, u_n_gamma, pres_n_gamma, u_extr});
    navier_stokes_matrix.set_u_extr(u_extr);
    u_star.equ(1.0, u_extr);
  }

  /*--- Build the linear solver; in this case we specifiy the maximum number of iterations and residual ---*/
  SolverControl solver_control(max_its, eps*rhs_u.l2_norm());
  SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

  /*--- Build a Jacobi preconditioner and solve ---*/
  PreconditionJacobi<NavierStokesProjectionOperator<dim,
                                                    EquationData::degree_p,
                                                    EquationData::degree_p + 1,
                                                    EquationData::degree_p + 1,
                                                    EquationData::degree_p + 2,
                                                    LinearAlgebra::distributed::Vector<double>>> preconditioner;
  navier_stokes_matrix.compute_diagonal();
  preconditioner.initialize(navier_stokes_matrix);

  gmres.solve(navier_stokes_matrix, u_star, rhs_u, preconditioner);
}


// Next, we solve the projection step.
//
template<int dim>
void NavierStokesProjection<dim>::projection_step() {
  TimerOutput::Scope t(time_table, "Projection step pressure");

  /*--- We start in the same way of 'diffusion_step': we first reinitialize with the index of FE space,
        we specify that this is the second stage and we compute the right-hand side ---*/
  const std::vector<unsigned int> tmp = {1};
  navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

  navier_stokes_matrix.set_NS_stage(2);

  if(TR_BDF2_stage == 1) {
    navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_n});
  }
  else {
    navier_stokes_matrix.vmult_rhs_pressure(rhs_p, {u_star, pres_n_gamma});
  }

  /*--- Build the linear solver (Conjugate Gradient in this case) ---*/
  SolverControl solver_control(max_its, eps*rhs_p.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Build the preconditioner (as in step-37) ---*/
  MGTransferMatrixFree<dim, float> mg_transfer;
  mg_transfer.build(dof_handler_pressure);

  using SmootherType = PreconditionChebyshev<NavierStokesProjectionOperator<dim,
                                                                            EquationData::degree_p,
                                                                            EquationData::degree_p + 1,
                                                                            EquationData::degree_p + 1,
                                                                            EquationData::degree_p + 2,
                                                                            LinearAlgebra::distributed::Vector<float>>,
                                             LinearAlgebra::distributed::Vector<float>>;
  mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float>> mg_smoother;
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
  smoother_data.resize(0, triangulation.n_global_levels() - 1);
  for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
    if(level > 0) {
      smoother_data[level].smoothing_range     = 15.0;
      smoother_data[level].degree              = 3;
      smoother_data[level].eig_cg_n_iterations = 10;
    }
    else {
      smoother_data[0].smoothing_range     = 2e-2;
      smoother_data[0].degree              = numbers::invalid_unsigned_int;
      smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
    }
    mg_matrices[level].compute_diagonal();
    smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
  }
  mg_smoother.initialize(mg_matrices, smoother_data);

  PreconditionIdentity                                identity;
  SolverCG<LinearAlgebra::distributed::Vector<float>> cg_mg(solver_control);
  MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<float>,
                              SolverCG<LinearAlgebra::distributed::Vector<float>>,
                              NavierStokesProjectionOperator<dim,
                                                             EquationData::degree_p,
                                                             EquationData::degree_p + 1,
                                                             EquationData::degree_p + 1,
                                                             EquationData::degree_p + 2,
                                                             LinearAlgebra::distributed::Vector<float>>,
                              PreconditionIdentity> mg_coarse(cg_mg, mg_matrices[0], identity);

  mg::Matrix<LinearAlgebra::distributed::Vector<float>> mg_matrix(mg_matrices);

  Multigrid<LinearAlgebra::distributed::Vector<float>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  PreconditionMG<dim,
                 LinearAlgebra::distributed::Vector<float>,
                 MGTransferMatrixFree<dim, float>> preconditioner(dof_handler_pressure, mg, mg_transfer);

  /*--- Solve the linear system ---*/
  if(TR_BDF2_stage == 1) {
    pres_n_gamma.equ(1.0, pres_n);
    cg.solve(navier_stokes_matrix, pres_n_gamma, rhs_p, preconditioner);
  }
  else {
    pres_n.equ(1.0, pres_n_gamma);
    cg.solve(navier_stokes_matrix, pres_n, rhs_p, preconditioner);
  }
}


// This implements the projection step for the gradient of pressure
//
template<int dim>
void NavierStokesProjection<dim>::project_grad(const unsigned int flag) {
  TimerOutput::Scope t(time_table, "Gradient of pressure projection");

  /*--- The input parameter flag is used just to specify where we want to save the result ---*/
  AssertIndexRange(flag, 3);
  Assert(flag > 0, ExcInternalError());

  /*--- We need to select the dof handler related to the velocity since the result lives there ---*/
  const std::vector<unsigned int> tmp = {0};
  navier_stokes_matrix.initialize(matrix_free_storage, tmp, tmp);

  if(flag == 1) {
    navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_n);
  }
  else if(flag == 2) {
    navier_stokes_matrix.vmult_grad_p_projection(rhs_u, pres_n_gamma);
  }

  /*--- We conventionally decide that the this corresponds to third stage ---*/
  navier_stokes_matrix.set_NS_stage(3);

  /*--- Solve the system ---*/
  SolverControl solver_control(max_its, 1e-12*rhs_u.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
  cg.solve(navier_stokes_matrix, u_tmp, rhs_u, PreconditionIdentity());
}


// The following function is used in determining the maximal velocity
// in order to compute the Courant number.
//
template<int dim>
double NavierStokesProjection<dim>::get_maximal_velocity() {
  return u_n.linfty_norm();
}


// The following function is used in determining the maximal nodal difference
// between old and current velocity value in order to see if we have reched steady-state.
//
template<int dim>
double NavierStokesProjection<dim>::get_maximal_difference_velocity() {
  u_tmp.equ(1.0, u_n);
  u_tmp.add(-1.0, u_n_minus_1);

  return u_tmp.linfty_norm();
}


// This method plots the current solution. The main difficulty is that we want
// to create a single output file that contains the data for all velocity
// components and the pressure. On the other hand, velocities and the pressure
// live on separate DoFHandler objects, so we need to pay attention when we use
// 'add_data_vector' to select the proper space.
//
template<int dim>
void NavierStokesProjection<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  DataOut<dim> data_out;

  std::vector<std::string> velocity_names(dim, "v");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  u_n.update_ghost_values();

  data_out.add_data_vector(dof_handler_velocity, u_n, velocity_names, component_interpretation_velocity);
  pres_n.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_n, "p", {DataComponentInterpretation::component_is_scalar});

  std::vector<std::string> velocity_names_old(dim, "v_old");
  u_n_minus_1.update_ghost_values();
  data_out.add_data_vector(dof_handler_velocity, u_n_minus_1, velocity_names_old, component_interpretation_velocity);

  /*--- Here we rely on the postprocessor we have built ---*/
  PostprocessorVorticity<dim> postprocessor;
  data_out.add_data_vector(dof_handler_velocity, u_n, postprocessor);

  data_out.build_patches(MappingQ1<dim>(), EquationData::degree_p + 1, DataOut<dim>::curved_inner_cells);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Serialization ---*/
  if(save_for_restart) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);

    u_n.update_ghost_values();
    u_n_minus_1.update_ghost_values();
    pres_n.update_ghost_values();

    std::vector<const LinearAlgebra::distributed::Vector<double>*> velocities;
    velocities.push_back(&u_n);
    velocities.push_back(&u_n_minus_1);
    solution_transfer_velocity.prepare_for_serialization(velocities);
    solution_transfer_pressure.prepare_for_serialization(pres_n);

    triangulation.save("./" + saving_dir + "/solution_ser-" + Utilities::int_to_string(step, 5));
  }
}


// @sect{<code>NavierStokesProjection::compute_lift_and_drag</code>}

// This routine computes the lift and the drag forces in a non-dimensional framework
// (so basically for the classical coefficients, it is necessary to multiply by a factor 2).
//
template<int dim>
void NavierStokesProjection<dim>::compute_lift_and_drag() {
  QGauss<dim - 1> face_quadrature_formula(EquationData::degree_p + 2);
  const int n_q_points = face_quadrature_formula.size();

  std::vector<double>                      pressure_values(n_q_points);
  std::vector<std::vector<Tensor<1, dim>>> velocity_gradients(n_q_points, std::vector<Tensor<1, dim>>(dim));

  Tensor<1, dim> normal_vector;
  Tensor<2, dim> fluid_stress;
  Tensor<2, dim> fluid_pressure;
  Tensor<1, dim> forces;

  /*--- We need to compute the integral over the cylinder boundary, so we need to use 'FEFaceValues' instances.
        For the velocity we need the gradients, for the pressure the values. ---*/
  FEFaceValues<dim> fe_face_values_velocity(fe_velocity, face_quadrature_formula,
                                            update_quadrature_points | update_gradients |
                                            update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> fe_face_values_pressure(fe_pressure, face_quadrature_formula, update_values);

  double local_drag = 0.0;
  double local_lift = 0.0;

  /*--- We need to perform a unique loop because the whole stress tensor takes into account contributions of
        velocity and pressure obviously. However, the two dof_handlers are different, so we neede to create an ad-hoc
        iterator for the pressure that we update manually. It is guaranteed that the cells are visited in the same order
        (see the documentation) ---*/
  auto tmp_cell = dof_handler_pressure.begin_active();
  for(const auto& cell : dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
        if(cell->face(face)->at_boundary() && cell->face(face)->boundary_id() == 2) {
          fe_face_values_velocity.reinit(cell, face);
          fe_face_values_pressure.reinit(tmp_cell, face);

          fe_face_values_velocity.get_function_gradients(u_n, velocity_gradients); /*--- velocity gradients ---*/
          fe_face_values_pressure.get_function_values(pres_n, pressure_values); /*--- pressure values ---*/

          for(int q = 0; q < n_q_points; q++) {
            normal_vector = -fe_face_values_velocity.normal_vector(q);

            for(unsigned int d = 0; d < dim; ++ d) {
              fluid_pressure[d][d] = pressure_values[q];
              for(unsigned int k = 0; k < dim; ++k)
                fluid_stress[d][k] = 1.0/Re*velocity_gradients[q][d][k];
            }
            fluid_stress = fluid_stress - fluid_pressure;

            forces = fluid_stress*normal_vector*fe_face_values_velocity.JxW(q);

            local_drag += forces[0];
            local_lift += forces[1];
          }
        }
      }
    }
    ++tmp_cell;
  }

  /*--- At the end, each processor has computed the contribution to the boundary cells it owns and, therefore,
        we need to sum up all the contributions. ---*/
  const double lift = Utilities::MPI::sum(local_lift, MPI_COMM_WORLD);
  const double drag = Utilities::MPI::sum(local_drag, MPI_COMM_WORLD);
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_lift << lift << std::endl;
    output_drag << drag << std::endl;
  }
}


// compute maximal local voriticity
//
template<int dim>
void NavierStokesProjection<dim>::compute_lipschitz_number() {
  FEValues<dim> fe_values(fe_velocity, quadrature_velocity, update_gradients);
  std::vector<std::vector<Tensor<1, dim, double>>> solution_gradients_velocity(quadrature_velocity.size(),
                                                                               std::vector<Tensor<1, dim, double>>(dim));

  double max_local_vorticity = std::numeric_limits<double>::min();

  for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_gradients(u_n, solution_gradients_velocity);

      for(unsigned int q = 0; q < quadrature_velocity.size(); ++q) {
        max_local_vorticity = std::max(max_local_vorticity,
                                       std::abs(solution_gradients_velocity[q][1][0] - solution_gradients_velocity[q][0][1])*dt);
      }
    }
  }

  const double lipschitz = Utilities::MPI::max(max_local_vorticity, MPI_COMM_WORLD);
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_lipschitz << lipschitz << std::endl;
  }
}


// @sect{ <code>NavierStokesProjection::run</code> }

// This is the time marching function, which starting at <code>t_0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<int dim>
void NavierStokesProjection<dim>::run(const bool verbose, const unsigned int output_interval) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  /*--- Perform the effective loop ---*/
  double time = t_0 + dt;
  unsigned int n = 1;
  if(restart && !as_initial_conditions) {
    n    = step_restart;
    time = time_restart;
  }
  else {
    output_results(1);
  }
  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- First stage of TR-BDF2 and we start by setting the proper flag ---*/
    TR_BDF2_stage = 1;
    navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);
    }

    verbose_cout << "  Interpolating the velocity stage 1" << std::endl;
    interpolate_velocity();

    verbose_cout << "  Diffusion Step stage 1 " << std::endl;
    diffusion_step();

    verbose_cout << "  Projection Step stage 1" << std::endl;
    project_grad(1);
    u_tmp.equ(gamma*dt, u_tmp);
    u_star += u_tmp; /*--- In the rhs of the projection step we need u_star + gamma*dt*grad(pres_n) and we save it into u_star ---*/
    projection_step();

    verbose_cout << "  Updating the Velocity stage 1" << std::endl;
    u_n_gamma.equ(1.0, u_star);
    project_grad(2);
    grad_pres_n_gamma.equ(1.0, u_tmp); /*--- We save grad(pres_n_gamma), because we will need it soon ---*/
    u_tmp.equ(-gamma*dt, u_tmp);
    u_n_gamma.add(1.0, u_tmp); /*--- u_n_gamma = u_star - gamma*dt*grad(pres_n_gamma) ---*/
    u_n_minus_1.equ(1.0, u_n);

    /*--- Second stage of TR-BDF2 ---*/
    TR_BDF2_stage = 2;
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices[level].set_TR_BDF2_stage(TR_BDF2_stage);
    }
    navier_stokes_matrix.set_TR_BDF2_stage(TR_BDF2_stage);

    verbose_cout << "  Interpolating the velocity stage 2" << std::endl;
    interpolate_velocity();

    verbose_cout << "  Diffusion Step stage 2 " << std::endl;
    diffusion_step();

    verbose_cout << "  Projection Step stage 2" << std::endl;
    u_tmp.equ((1.0 - gamma)*dt, grad_pres_n_gamma);
    u_star.add(1.0, u_tmp);  /*--- In the rhs of the projection step we need u_star + (1 - gamma)*dt*grad(pres_n_gamma) ---*/
    projection_step();

    verbose_cout << "  Updating the Velocity stage 2" << std::endl;
    u_n.equ(1.0, u_star);
    project_grad(1);
    u_tmp.equ((gamma - 1.0)*dt, u_tmp);
    u_n.add(1.0, u_tmp);  /*--- u_n = u_star - (1 - gamma)*dt*grad(pres_n) ---*/

    const double max_vel = get_maximal_velocity();
    pcout<< "Maximal velocity = " << max_vel << std::endl;
    /*--- The Courant number is computed taking into account the polynomial degree for the velocity ---*/
    pcout << "CFL = " << dt*max_vel*(EquationData::degree_p + 1)*
                         std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;

    /*--- Now we focus on the statistics ---*/
    compute_lift_and_drag();
    // compute lipschitz number at every timestep
    compute_lipschitz_number();

    /*--- Save the results ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
    /*--- In case dt is not a multiple of T, we reduce dt in order to end up at T ---*/
    if(T - time < dt && T - time > 1e-10) {
      dt = T - time;
      navier_stokes_matrix.set_dt(dt);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
        mg_matrices[level].set_dt(dt);
      }
    }
  }
  if(n % output_interval != 0) {
    verbose_cout << "Plotting Solution final" << std::endl;
    output_results(n);
  }
}


// @sect{ The main function }

// The main function looks very much like in all the other tutorial programs. We first initialize MPI,
// we initialize the class 'NavierStokesProjection' with the dimension as template parameter and then
// let the method 'run' do the job.
//
int main(int argc, char *argv[]) {
  try {
    using namespace NS_TRBDF2;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    NavierStokesProjection<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception &exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
