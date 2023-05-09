/* Author: Giuseppe Orlando, 2022. */

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

namespace MatrixFreeTools {
  using namespace dealii;

  template<int dim, typename Number, typename VectorizedArrayType>
  void compute_diagonal(const MatrixFree<dim, Number, VectorizedArrayType>&                            matrix_free,
                        LinearAlgebra::distributed::Vector<Number>&                                    diagonal_global,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     cell_operation,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     face_operation,
                        const std::function<void(const MatrixFree<dim, Number, VectorizedArrayType>&,
                                                 LinearAlgebra::distributed::Vector<Number>&,
                                                 const unsigned int&,
                                                 const std::pair<unsigned int, unsigned int>&)>& 	     boundary_operation,
                        const unsigned int                                                             dof_no = 0) {
    // initialize vector
    matrix_free.initialize_dof_vector(diagonal_global, dof_no);

    const unsigned int dummy = 0;

    matrix_free.loop(cell_operation, face_operation, boundary_operation,
                     diagonal_global, dummy, false,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }
}

// We include the code in a suitable namespace:
//
namespace NS_TRBDF2 {
  using namespace dealii;

  // The following class is an auxiliary one for post-processing of the vorticity
  //
  template<int dim>
  class PostprocessorVorticity: public DataPostprocessor<dim> {
  public:
    virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                       std::vector<Vector<double>>&                computed_quantities) const override;

    virtual std::vector<std::string> get_names() const override;

    virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags get_needed_update_flags() const override;
  };

  // This function evaluates the vorticty in both 2D and 3D cases
  //
  template <int dim>
  void PostprocessorVorticity<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
                                                          std::vector<Vector<double>>&                computed_quantities) const {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    /*--- Check the correctness of all data structres ---*/
    Assert(inputs.solution_gradients.size() == n_quadrature_points, ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());

    Assert(inputs.solution_values[0].size() == dim, ExcInternalError());

    if(dim == 2) {
      Assert(computed_quantities[0].size() == 1, ExcInternalError());
    }
    else {
      Assert(computed_quantities[0].size() == dim, ExcInternalError());
    }

    /*--- Compute the vorticty ---*/
    if(dim == 2) {
      for(unsigned int q = 0; q < n_quadrature_points; ++q)
        computed_quantities[q](0) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
    }
    else {
      for(unsigned int q = 0; q < n_quadrature_points; ++q) {
        computed_quantities[q](0) = inputs.solution_gradients[q][2][1] - inputs.solution_gradients[q][1][2];
        computed_quantities[q](1) = inputs.solution_gradients[q][0][2] - inputs.solution_gradients[q][2][0];
        computed_quantities[q](2) = inputs.solution_gradients[q][1][0] - inputs.solution_gradients[q][0][1];
      }
    }
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // sets the name for the output file
  //
  template<int dim>
  std::vector<std::string> PostprocessorVorticity<dim>::get_names() const {
    std::vector<std::string> names;
    names.emplace_back("vorticity");
    if(dim == 3) {
      names.emplace_back("vorticity");
      names.emplace_back("vorticity");
    }

    return names;
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // specifies if the vorticity is a scalar (2D) or a vector (3D)
  //
  template<int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  PostprocessorVorticity<dim>::get_data_component_interpretation() const {
    std::vector<DataComponentInterpretation::DataComponentInterpretation> interpretation;
    if(dim == 2) {
      interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    }
    else {
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
    }

    return interpretation;
  }

  // This auxiliary function is required by the base class DataProcessor and simply
  // sets which variables have to updated (only the gradients)
  //
  template<int dim>
  UpdateFlags PostprocessorVorticity<dim>::get_needed_update_flags() const {
    return update_gradients;
  }


  // The following structs are auxiliary objects for mesh refinement. ScratchData simply sets
  // the FEValues object
  //
  template <int dim>
  struct ScratchData {
    ScratchData(const FiniteElement<dim>& fe,
                const unsigned int        quadrature_degree,
                const UpdateFlags         update_flags): fe_values(fe, QGauss<dim>(quadrature_degree), update_flags) {}

    ScratchData(const ScratchData<dim>& scratch_data): fe_values(scratch_data.fe_values.get_fe(),
                                                                 scratch_data.fe_values.get_quadrature(),
                                                                 scratch_data.fe_values.get_update_flags()) {}
    FEValues<dim> fe_values;
  };


  // CopyData simply sets the cell index
  //
  struct CopyData {
    CopyData() : cell_index(numbers::invalid_unsigned_int), value(0.0) {}

    CopyData(const CopyData &) = default;

    unsigned int cell_index;
    double       value;
  };


  // @sect{ <code>NavierStokesProjectionOperator::NavierStokesProjectionOperator</code> }

  // The following class sets effecively the weak formulation of the problems for the different stages
  // and for both velocity and pressure.
  // The template parameters are the dimnesion of the problem, the polynomial degree for the pressure,
  // the polynomial degree for the velocity, the number of quadrature points for integrals for the pressure step,
  // the number of quadrature points for integrals for the velocity step, the type of vector for storage and the type
  // of floating point data (in general double or float for preconditioners structures if desired).
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  class NavierStokesProjectionOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    NavierStokesProjectionOperator();

    NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_Reynolds(const double reynolds);

    void set_TR_BDF2_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_u_extr(const Vec& src);

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_grad_p_projection(Vec& dst, const Vec& src) const;

    virtual void compute_diagonal() override;

  protected:
    double       Re;
    double       dt;

    /*--- Parameters of time-marching scheme ---*/
    double       gamma;
    double       a31;
    double       a32;
    double       a33;

    unsigned int TR_BDF2_stage; /*--- Flag to denote at which stage of the TR-BDF2 are ---*/
    unsigned int NS_stage;      /*--- Flag to denote at which stage of NS solution inside each TR-BDF2 stage we are
                                      (solution of the velocity or of the pressure)---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    /*--- Auxiliary variable for the TR stage
          (just to avoid to report a lot of 0.5 and for my personal choice to be coherent with the article) ---*/
    const double a21 = 0.5;
    const double a22 = 0.5;

    /*--- Penalty method parameters, theta = 1 means SIP, while C_p and C_u are the penalization coefficients ---*/
    const double theta_v = 1.0;
    const double theta_p = 1.0;
    const double C_p     = 100.0*(fe_degree_p + 1)*(fe_degree_p + 1);
    const double C_u     = 100.0*(fe_degree_v + 1)*(fe_degree_v + 1);

    Vec                          u_extr; /*--- Auxiliary variable to update the extrapolated velocity ---*/

    EquationData::Velocity<dim>  vel_boundary_inflow; /*--- Auxiliary variable to impose velocity boundary conditions ---*/

    /*--- The following functions basically assemble the linear and bilinear forms. Their syntax is due to
          the base class MatrixFreeOperators::Base ---*/
    void assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Now we focus on the pressure ---*/
    void assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const Vec&                                   src,
                                                  const std::pair<unsigned int, unsigned int>& cell_range) const;

    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const unsigned int&                          src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;

    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const unsigned int&                          src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const;
};


  // We start with the default constructor. It is important for MultiGrid, so it is fundamental
  // to properly set the parameters of the time scheme.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  NavierStokesProjectionOperator():
  MatrixFreeOperators::Base<dim, Vec>(), Re(), dt(),  gamma(2.0 - std::sqrt(2.0)), a31((1.0 - gamma)/(2.0*(2.0 - gamma))),
                                         a32(a31), a33(1.0/(2.0 - gamma)), TR_BDF2_stage(1), NS_stage(1), u_extr() {}


  // We focus now on the constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  NavierStokesProjectionOperator(RunTimeParameters::Data_Storage& data):
  MatrixFreeOperators::Base<dim, Vec>(), Re(data.Reynolds), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                         a31((1.0 - gamma)/(2.0*(2.0 - gamma))), a32(a31), a33(1.0/(2.0 - gamma)),
                                         TR_BDF2_stage(1), NS_stage(1), u_extr(),
                                         vel_boundary_inflow(data.initial_time) {}


  // Setter of time-step (called by Multigrid and in case a smaller time-step towards the end is needed)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of TR-BDF2 stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_TR_BDF2_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());

    TR_BDF2_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
  }


  // Setter of extrapolated velocity for different stages
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  set_u_extr(const Vec& src) {
    u_extr = src;
    u_extr.update_ghost_values();
  }


  // We are in a DG-MatrixFree framework, so it is convenient to compute separately cell contribution,
  // internal faces contributions and boundary faces contributions. We start by
  // assembling the rhs cell term for the velocity.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the old velocity, the
      extrapolated velocity and the old pressure. 'phi' will be used only to submit the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      velocity and 1 for pressure). ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n(data, 0),
                                                                   phi_n_gamma_ov_2(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n(data, 1);

      /*--- We loop over the cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). ---*/
        phi_n.reinit(cell);
        phi_n.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
                                                           /*--- The 'gather_evaluate' function reads data from the vector.
                                                           The second and third parameter specifies if you want to read
                                                           values and/or derivative related quantities ---*/
        phi_n_gamma_ov_2.reinit(cell);
        phi_n_gamma_ov_2.gather_evaluate(src[1], EvaluationFlags::values);

        phi_pres_n.reinit(cell);
        phi_pres_n.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                = phi_n.get_value(q);

          const auto& grad_u_n           = phi_n.get_gradient(q);

          const auto& u_n_gamma_ov_2     = phi_n_gamma_ov_2.get_value(q);
          const auto& tensor_product_u_n = outer_product(u_n, u_n_gamma_ov_2);

          const auto& pres_n             = phi_pres_n.get_value(q);
          auto pres_n_times_identity     = tensor_product_u_n;
          pres_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            pres_n_times_identity[d][d] = pres_n;
          }

          phi.submit_value(1.0/(gamma*dt)*u_n, q); /*--- 'submit_value' contains quantites that we want to test against the
                                                          test function ---*/
          phi.submit_gradient(-a21/Re*grad_u_n + a21*tensor_product_u_n + pres_n_times_identity, q);
          /*--- 'submit_gradient' contains quantites that we want to test against the gradient of test function ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n(data, 0),
                                                                   phi_n_gamma(data, 0);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n(data, 1);

      /*--- We loop over the cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_n.reinit(cell);
        phi_n.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_n_gamma.reinit(cell);
        phi_n_gamma.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_pres_n.reinit(cell);
        phi_pres_n.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_n                      = phi_n.get_value(q);
          const auto& grad_u_n                 = phi_n.get_gradient(q);

          const auto& u_n_gamma                = phi_n_gamma.get_value(q);
          const auto& grad_u_n_gamma           = phi_n_gamma.get_gradient(q);

          const auto& tensor_product_u_n       = outer_product(u_n, u_n);
          const auto& tensor_product_u_n_gamma = outer_product(u_n_gamma, u_n_gamma);

          const auto& pres_n                   = phi_pres_n.get_value(q);
          auto pres_n_times_identity           = tensor_product_u_n;
          pres_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            pres_n_times_identity[d][d] = pres_n;
          }

          phi.submit_value(1.0/((1.0 - gamma)*dt)*u_n_gamma, q);
          phi.submit_gradient(a32*tensor_product_u_n_gamma + a31*tensor_product_u_n -
                              a32/Re*grad_u_n_gamma - a31/Re*grad_u_n + pres_n_times_identity, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The following function assembles rhs face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. In this case
      we are at the face between two elements and this is the reason of 'FEFaceEvaluation'. It contains an extra
      input argument, the second one, that specifies if it is from 'interior' or not---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_p(data, true, 0),
                                                                       phi_n_m(data, false, 0),
                                                                       phi_n_gamma_ov_2_p(data, true, 0),
                                                                       phi_n_gamma_ov_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n_p(data, true, 1),
                                                                       phi_pres_n_m(data, false, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n_p.reinit(face);
        phi_n_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_n_m.reinit(face);
        phi_n_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_ov_2_p.reinit(face);
        phi_n_gamma_ov_2_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_n_gamma_ov_2_m.reinit(face);
        phi_n_gamma_ov_2_m.gather_evaluate(src[1], EvaluationFlags::values);

        phi_pres_n_p.reinit(face);
        phi_pres_n_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_n_m.reinit(face);
        phi_pres_n_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q); /*--- The normal vector is the same
                                                                                 for both phi_p and phi_m. If the face is interior,
                                                                                 it correspond to the outer normal ---*/
          const auto& grad_u_n_p             = phi_n_p.get_gradient(q);
          const auto& grad_u_n_m             = phi_n_m.get_gradient(q);
          const auto& avg_grad_u_n           = 0.5*(grad_u_n_p + grad_u_n_m);

          const auto& u_n_p                  = phi_n_p.get_value(q);
          const auto& u_n_m                  = phi_n_m.get_value(q);
          const auto& u_n_gamma_ov_2_p       = phi_n_gamma_ov_2_p.get_value(q);
          const auto& u_n_gamma_ov_2_m       = phi_n_gamma_ov_2_m.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(u_n_p, u_n_gamma_ov_2_p) +
                                                    outer_product(u_n_m, u_n_gamma_ov_2_m));

          const auto& pres_n_p               = phi_pres_n_p.get_value(q);
          const auto& pres_n_m               = phi_pres_n_m.get_value(q);
          const auto& avg_pres_n             = 0.5*(pres_n_p + pres_n_m);

          /*--- Compute data for upwind flux ---*/
          const auto& lambda_n               = std::max(std::abs(scalar_product(u_n_p, n_plus)),
                                                        std::abs(scalar_product(u_n_m, n_plus)));
          const auto& jump_u_n               = u_n_p - u_n_m;

          phi_p.submit_value((a21/Re*avg_grad_u_n - a21*avg_tensor_product_u_n)*n_plus - avg_pres_n*n_plus - a21*0.5*lambda_n*jump_u_n, q);
          phi_m.submit_value(-(a21/Re*avg_grad_u_n - a21*avg_tensor_product_u_n)*n_plus + avg_pres_n*n_plus + a21*0.5*lambda_n*jump_u_n, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_p(data, true, 0),
                                                                       phi_n_m(data, false, 0),
                                                                       phi_n_gamma_p(data, true, 0),
                                                                       phi_n_gamma_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n_p(data, true, 1),
                                                                       phi_pres_n_m(data, false, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_n_p.reinit(face);
        phi_n_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_n_m.reinit(face);
        phi_n_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_p.reinit(face);
        phi_n_gamma_p.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_n_gamma_m.reinit(face);
        phi_n_gamma_m.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_pres_n_p.reinit(face);
        phi_pres_n_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_n_m.reinit(face);
        phi_pres_n_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                       = phi_p.get_normal_vector(q);

          const auto& grad_u_n_p                   = phi_n_p.get_gradient(q);
          const auto& grad_u_n_m                   = phi_n_m.get_gradient(q);
          const auto& avg_grad_u_n                 = 0.5*(grad_u_n_p + grad_u_n_m);

          const auto& grad_u_n_gamma_p             = phi_n_gamma_p.get_gradient(q);
          const auto& grad_u_n_gamma_m             = phi_n_gamma_m.get_gradient(q);
          const auto& avg_grad_u_n_gamma           = 0.5*(grad_u_n_gamma_p + grad_u_n_gamma_m);

          const auto& u_n_p                        = phi_n_p.get_value(q);
          const auto& u_n_m                        = phi_n_m.get_value(q);
          const auto& avg_tensor_product_u_n       = 0.5*(outer_product(u_n_p, u_n_p) +
                                                          outer_product(u_n_m, u_n_m));

          const auto& u_n_gamma_p                  = phi_n_gamma_p.get_value(q);
          const auto& u_n_gamma_m                  = phi_n_gamma_m.get_value(q);
          const auto& avg_tensor_product_u_n_gamma = 0.5*(outer_product(u_n_gamma_p, u_n_gamma_p) +
                                                          outer_product(u_n_gamma_m, u_n_gamma_m));

          const auto& pres_n_p                     = phi_pres_n_p.get_value(q);
          const auto& pres_n_m                     = phi_pres_n_m.get_value(q);
          const auto& avg_pres_n                   = 0.5*(pres_n_p + pres_n_m);

          /*--- Compute data for upwind flux ---*/
          const auto& lambda_n                     = std::max(std::abs(scalar_product(u_n_p, n_plus)),
                                                              std::abs(scalar_product(u_n_m, n_plus)));
          const auto& jump_u_n                     = u_n_p - u_n_m;

          const auto& lambda_n_gamma               = std::max(std::abs(scalar_product(u_n_gamma_p, n_plus)),
                                                              std::abs(scalar_product(u_n_gamma_m, n_plus)));
          const auto& jump_u_n_gamma               = u_n_gamma_p - u_n_gamma_m;

          phi_p.submit_value((a31/Re*avg_grad_u_n + a32/Re*avg_grad_u_n_gamma -
                              a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus - avg_pres_n*n_plus -
                              a31*0.5*lambda_n*jump_u_n - a32*0.5*lambda_n_gamma*jump_u_n_gamma, q);
          phi_m.submit_value(-(a31/Re*avg_grad_u_n + a32/Re*avg_grad_u_n_gamma -
                               a31*avg_tensor_product_u_n - a32*avg_tensor_product_u_n_gamma)*n_plus + avg_pres_n*n_plus +
                               a31*0.5*lambda_n*jump_u_n + a32*0.5*lambda_n_gamma*jump_u_n_gamma, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // The followinf function assembles rhs boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. Clearly on the boundary
      the second argument has to be true. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n(data, true, 0),
                                                                       phi_n_gamma_ov_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n(data, true, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n.reinit(face);
        phi_n.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_ov_2.reinit(face);
        phi_n_gamma_ov_2.gather_evaluate(src[1], EvaluationFlags::values);

        phi_pres_n.reinit(face);
        phi_pres_n.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face); /*--- Get the id in order to impose the proper boundary condition ---*/

        const auto coef_jump   = (boundary_id == 3) ?
                                 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 3) ? 0.0 : 1.0;

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                     = phi.get_normal_vector(q);

          const auto& grad_u_n                   = phi_n.get_gradient(q);

          const auto& u_n                        = phi_n.get_value(q);
          const auto& u_n_gamma_ov_2             = phi_n_gamma_ov_2.get_value(q);
          const auto& tensor_product_u_n         = outer_product(u_n, u_n_gamma_ov_2);

          const auto& pres_n                     = phi_pres_n.get_value(q);

          auto u_n_gamma_m                       = Tensor<1, dim, VectorizedArray<Number>>();
          auto u_n_m                             = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 2) {
            const auto& point_vectorized = phi.quadrature_point(q);
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                      and so it is not ready to be directly used. We need to pay attention to the
                                      vectorization ---*/
              for(unsigned int d = 0; d < dim; ++d) {
                point[d] = point_vectorized[d][v];
              }
              for(unsigned int d = 0; d < dim; ++d) {
                u_n_gamma_m[d][v] = vel_boundary_inflow.value(point, d);
                u_n_m[d][v]       = vel_boundary_inflow.value(point, d);
              }
            }
          }
          const auto& tensor_product_u_n_gamma_m = outer_product(u_n_gamma_m, u_n_gamma_ov_2);

          const auto& lambda_n_gamma_ov_2        = (boundary_id == 3) ?
                                                   0.0 : std::abs(scalar_product(u_n_gamma_ov_2, n_plus));

          const auto& lambda_n                   = (boundary_id == 3) ?
                                                   0.0 : std::abs(scalar_product(u_n, n_plus));
          const auto& jump_u_n                   = u_n - u_n_m;

          phi.submit_value((a21/Re*grad_u_n - a21*tensor_product_u_n)*n_plus - pres_n*n_plus +
                           a22/Re*2.0*coef_jump*u_n_gamma_m -
                           aux_coeff*a22*tensor_product_u_n_gamma_m*n_plus + a22*lambda_n_gamma_ov_2*u_n_gamma_m - a21*lambda_n*jump_u_n, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a22/Re*u_n_gamma_m, q); /*--- This is equivalent to multiply to the gradient
                                                                                        with outer product and use 'submit_gradient' ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n(data, true, 0),
                                                                       phi_n_gamma(data, true, 0),
                                                                       phi_n_3gamma_ov_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres_n(data, true, 1);

      /*--- We loop over the faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++ face) {
        phi_n.reinit(face);
        phi_n.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma.reinit(face);
        phi_n_gamma.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_pres_n.reinit(face);
        phi_pres_n.gather_evaluate(src[2], EvaluationFlags::values);

        phi_n_3gamma_ov_2.reinit(face);
        phi_n_3gamma_ov_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = (boundary_id == 3) ?
                                 0.0 : C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);
        const double aux_coeff = (boundary_id == 3) ? 0.0 : 1.0;

        /*--- Now we loop over all the quadrature points to compute the integrals ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                   = phi.get_normal_vector(q);

          const auto& grad_u_n                 = phi_n.get_gradient(q);
          const auto& grad_u_n_gamma           = phi_n_gamma.get_gradient(q);

          const auto& u_n                      = phi_n.get_value(q);
          const auto& tensor_product_u_n       = outer_product(u_n, u_n);

          const auto& u_n_gamma                = phi_n_gamma.get_value(q);
          const auto& tensor_product_u_n_gamma = outer_product(u_n_gamma, u_n_gamma);

          const auto& pres_n                   = phi_pres_n.get_value(q);

          auto u_n_m                           = Tensor<1, dim, VectorizedArray<Number>>();
          auto u_n_gamma_m                     = Tensor<1, dim, VectorizedArray<Number>>();
          auto u_np1_m                         = Tensor<1, dim, VectorizedArray<Number>>();
          if(boundary_id == 2) {
            const auto& point_vectorized = phi.quadrature_point(q);
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              Point<dim> point;
              for(unsigned int d = 0; d < dim; ++d) {
                point[d] = point_vectorized[d][v];
              }
              for(unsigned int d = 0; d < dim; ++d) {
                u_n_m[d][v]       = vel_boundary_inflow.value(point, d);
                u_n_gamma_m[d][v] = vel_boundary_inflow.value(point, d);
                u_np1_m[d][v]     = vel_boundary_inflow.value(point, d);
              }
            }
          }
          const auto& u_n_3gamma_ov_2          = phi_n_3gamma_ov_2.get_value(q);
          const auto& tensor_product_u_np1_m   = outer_product(u_np1_m, u_n_3gamma_ov_2);

          const auto& lambda_n_3gamma_ov_2     = (boundary_id == 3) ?
                                                 0.0 : std::abs(scalar_product(u_n_3gamma_ov_2, n_plus));

          const auto& lambda_n                 = (boundary_id == 3) ?
                                                 0.0 : std::abs(scalar_product(u_n, n_plus));
          const auto& jump_u_n                 = u_n - u_n_m;

          const auto& lambda_n_gamma           = (boundary_id == 3) ?
                                                 0.0 : std::abs(scalar_product(u_n_gamma, n_plus));
          const auto& jump_u_n_gamma           = u_n_gamma - u_n_gamma_m;

          phi.submit_value((a31/Re*grad_u_n + a32/Re*grad_u_n_gamma -
                           a31*tensor_product_u_n - a32*tensor_product_u_n_gamma)*n_plus - pres_n*n_plus +
                           a33/Re*2.0*coef_jump*u_np1_m -
                           aux_coeff*a33*tensor_product_u_np1_m*n_plus + a33*lambda_n_3gamma_ov_2*u_np1_m -
                           a31*lambda_n*jump_u_n - a32*lambda_n_gamma*jump_u_n_gamma, q);
          phi.submit_normal_derivative(-aux_coeff*theta_v*a33/Re*u_np1_m, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Put together all the previous steps for velocity. This is done automatically by the loop function of 'MatrixFree' class
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    for(auto& vec : src) {
      vec.update_ghost_values();
    }

    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_velocity,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Now we need to build the 'matrices', i.e. the bilinear forms. We start by
  // assembling the cell term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. Moreover 'phi' in
      this case serves for a bilinear form and so it will not used only to submit but also to read the src ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n_gamma_ov_2(data, 0);

      /*--- We loop over all cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_ov_2.reinit(cell);
        phi_n_gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u                = phi.get_value(q);
          const auto& grad_u           = phi.get_gradient(q);

          const auto& u_n_gamma_ov_2   = phi_n_gamma_ov_2.get_value(q);
          const auto& tensor_product_u = outer_product(u, u_n_gamma_ov_2);

          phi.submit_value(1.0/(gamma*dt)*u, q);
          phi.submit_gradient(-a22*tensor_product_u + a22/Re*grad_u, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n_3gamma_ov_2(data, 0);

      /*--- We loop over all cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi.reinit(cell);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_3gamma_ov_2.reinit(cell);
        phi_n_3gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u                = phi.get_value(q);
          const auto& grad_u           = phi.get_gradient(q);

          const auto& u_n_3gamma_ov_2  = phi_n_3gamma_ov_2.get_value(q);
          const auto& tensor_product_u = outer_product(u, u_n_3gamma_ov_2);

          phi.submit_value(1.0/((1.0 - gamma)*dt)*u, q);
          phi.submit_gradient(-a33*tensor_product_u + a33/Re*grad_u, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The following function assembles face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_gamma_ov_2_p(data, true, 0),
                                                                       phi_n_gamma_ov_2_m(data, false, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_ov_2_p.reinit(face);
        phi_n_gamma_ov_2_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_n_gamma_ov_2_m.reinit(face);
        phi_n_gamma_ov_2_m.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus               = phi_p.get_normal_vector(q);

          const auto& grad_u_p             = phi_p.get_gradient(q);
          const auto& grad_u_m             = phi_m.get_gradient(q);
          const auto& avg_grad_u           = 0.5*(grad_u_p + grad_u_m);

          const auto& u_p                  = phi_p.get_value(q);
          const auto& u_m                  = phi_m.get_value(q);
          const auto& u_n_gamma_ov_2_p     = phi_n_gamma_ov_2_p.get_value(q);
          const auto& u_n_gamma_ov_2_m     = phi_n_gamma_ov_2_m.get_value(q);
          const auto& avg_tensor_product_u = 0.5*(outer_product(u_p, u_n_gamma_ov_2_p) +
                                                  outer_product(u_m, u_n_gamma_ov_2_m));

          const auto& lambda_n_gamma_ov_2  = std::max(std::abs(scalar_product(u_n_gamma_ov_2_p, n_plus)),
                                                      std::abs(scalar_product(u_n_gamma_ov_2_m, n_plus)));
          const auto& jump_u               = u_p - u_m;

          phi_p.submit_value(a22/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                             a22*avg_tensor_product_u*n_plus + 0.5*a22*lambda_n_gamma_ov_2*jump_u, q);
          phi_m.submit_value(-a22/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                              a22*avg_tensor_product_u*n_plus - 0.5*a22*lambda_n_gamma_ov_2*jump_u, q);
          phi_p.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u, q);
          phi_m.submit_normal_derivative(-theta_v*a22/Re*0.5*jump_u, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_3gamma_ov_2_p(data, true, 0),
                                                                       phi_n_3gamma_ov_2_m(data, false, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_p.reinit(face);
        phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.reinit(face);
        phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_3gamma_ov_2_p.reinit(face);
        phi_n_3gamma_ov_2_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_n_3gamma_ov_2_m.reinit(face);
        phi_n_3gamma_ov_2_m.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Now we loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus               = phi_p.get_normal_vector(q);

          const auto& grad_u_p             = phi_p.get_gradient(q);
          const auto& grad_u_m             = phi_m.get_gradient(q);
          const auto& avg_grad_u           = 0.5*(grad_u_p + grad_u_m);

          const auto& u_p                  = phi_p.get_value(q);
          const auto& u_m                  = phi_m.get_value(q);
          const auto& u_n_3gamma_ov_2_p    = phi_n_3gamma_ov_2_p.get_value(q);
          const auto& u_n_3gamma_ov_2_m    = phi_n_3gamma_ov_2_m.get_value(q);
          const auto& avg_tensor_product_u = 0.5*(outer_product(u_p, u_n_3gamma_ov_2_p) +
                                                  outer_product(u_m, u_n_3gamma_ov_2_m));

          const auto& lambda_n_3gamma_ov_2 = std::max(std::abs(scalar_product(u_n_3gamma_ov_2_p, n_plus)),
                                                      std::abs(scalar_product(u_n_3gamma_ov_2_m, n_plus)));
          const auto& jump_u               = u_p - u_m;


          phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                             a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda_n_3gamma_ov_2*jump_u, q);
          phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                              a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda_n_3gamma_ov_2*jump_u, q);
          phi_p.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
          phi_m.submit_normal_derivative(-theta_v*a33/Re*0.5*jump_u, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // The following function assembles boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n_gamma_ov_2(data, true, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_gamma_ov_2.reinit(face);
        phi_n_gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        /*--- The application of the mirror principle is not so trivial because we have a Dirichlet condition
              on a single component for the outflow; so we distinguish the two cases ---*/
        if(boundary_id != 3) {
          const double coef_trasp = 0.0;

          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus              = phi.get_normal_vector(q);

            const auto& grad_u              = phi.get_gradient(q);

            const auto& u                   = phi.get_value(q);
            const auto& u_n_gamma_ov_2      = phi_n_gamma_ov_2.get_value(q);
            const auto& tensor_product_u    = outer_product(u, u_n_gamma_ov_2);

            const auto& lambda_n_gamma_ov_2 = std::abs(scalar_product(u_n_gamma_ov_2, n_plus));

            phi.submit_value(a22/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                             a22*coef_trasp*tensor_product_u*n_plus + a22*lambda_n_gamma_ov_2*u, q);
            phi.submit_normal_derivative(-theta_v*a22/Re*u, q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
        else {
          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus    = phi.get_normal_vector(q);

            const auto& grad_u    = phi.get_gradient(q);
            const auto& u         = phi.get_value(q);

            auto u_n_gamma_m      = u;
            auto grad_u_n_gamma_m = grad_u;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              u_n_gamma_m[1][v]         = -u_n_gamma_m[1][v];

              grad_u_n_gamma_m[0][0][v] = -grad_u_n_gamma_m[0][0][v];
              grad_u_n_gamma_m[0][1][v] = -grad_u_n_gamma_m[0][1][v];
            }

            const auto& u_n_gamma_ov_2      = phi_n_gamma_ov_2.get_value(q);
            const auto& lambda_n_gamma_ov_2 = std::abs(scalar_product(u_n_gamma_ov_2, n_plus));

            phi.submit_value(a22/Re*(-(0.5*(grad_u + grad_u_n_gamma_m))*n_plus + coef_jump*(u - u_n_gamma_m)) +
                             a22*outer_product(0.5*(u + u_n_gamma_m), u_n_gamma_ov_2)*n_plus +
                             a22*0.5*lambda_n_gamma_ov_2*(u - u_n_gamma_m), q);
            phi.submit_normal_derivative(-theta_v*a22/Re*(u - u_n_gamma_m), q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n_3gamma_ov_2(data, true, 0);

      /*--- We loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        phi_n_3gamma_ov_2.reinit(face);
        phi_n_3gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 3) {
          const double coef_trasp = 0.0;

          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus               = phi.get_normal_vector(q);

            const auto& grad_u               = phi.get_gradient(q);
            const auto& u                    = phi.get_value(q);

            const auto& u_n_3gamma_ov_2      = phi_n_3gamma_ov_2.get_value(q);
            const auto& tensor_product_u     = outer_product(u, u_n_3gamma_ov_2);

            const auto& lambda_n_3gamma_ov_2 = std::abs(scalar_product(u_n_3gamma_ov_2, n_plus));

            phi.submit_value(a33/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                             a33*coef_trasp*tensor_product_u*n_plus + a33*lambda_n_3gamma_ov_2*u, q);
            phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
        else {
          /*--- Now we loop over all quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus  = phi.get_normal_vector(q);

            const auto& grad_u  = phi.get_gradient(q);
            const auto& u       = phi.get_value(q);

            auto u_m            = u;
            auto grad_u_m       = grad_u;
            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
              u_m[1][v]         = -u_m[1][v];

              grad_u_m[0][0][v] = -grad_u_m[0][0][v];
              grad_u_m[0][1][v] = -grad_u_m[0][1][v];
            }

            const auto& u_n_3gamma_ov_2      = phi_n_3gamma_ov_2.get_value(q);
            const auto& lambda_n_3gamma_ov_2 = std::abs(scalar_product(u_n_3gamma_ov_2, n_plus));

            phi.submit_value(a33/Re*(-(0.5*(grad_u + grad_u_m))*n_plus + coef_jump*(u - u_m)) +
                             a33*outer_product(0.5*(u + u_m), u_n_3gamma_ov_2)*n_plus +
                             a33*0.5*lambda_n_3gamma_ov_2*(u - u_m), q);
            phi.submit_normal_derivative(-theta_v*a33/Re*(u - u_m), q);
          }
          phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
      }
    }
  }


  // Now we focus on computing the rhs for the projection step for the pressure with the same ratio.
  // The following function assembles rhs cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities.
          The third parameter specifies that we want to use the second quadrature formula stored. ---*/
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, 1, 1),
                                                                 phi_old(data, 1, 1);
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, 0, 1);

    const double coeff   = (TR_BDF2_stage == 1) ? 1e3*gamma*dt*gamma*dt : 1e3*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    const double coeff_2 = (TR_BDF2_stage == 1) ? gamma*dt : (1.0 - gamma)*dt;

    /*--- We loop over cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_proj.reinit(cell);
      phi_proj.gather_evaluate(src[0], EvaluationFlags::values);

      phi_old.reinit(cell);
      phi_old.gather_evaluate(src[1], EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u_star_star = phi_proj.get_value(q);

        const auto& pres_old    = phi_old.get_value(q);

        phi.submit_value(1.0/coeff*pres_old, q);
        phi.submit_gradient(1.0/coeff_2*u_star_star, q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles rhs face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi_p(data, true, 1, 1),
                                                                     phi_m(data, false, 1, 1),
                                                                     phi_pres_p(data, true, 1, 1),
                                                                     phi_pres_m(data, false, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj_p(data, true, 0, 1),
                                                                     phi_proj_m(data, false, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    /*--- We loop over faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj_p.reinit(face);
      phi_proj_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_proj_m.reinit(face);
      phi_proj_m.gather_evaluate(src[0], EvaluationFlags::values);

      phi_pres_p.reinit(face);
      phi_pres_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_pres_m.reinit(face);
      phi_pres_m.gather_evaluate(src[1], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus          = phi_p.get_normal_vector(q);

        const auto& avg_u_star_star = 0.5*(phi_proj_p.get_value(q) + phi_proj_m.get_value(q));

        phi_p.submit_value(-coeff*(scalar_product(avg_u_star_star, n_plus)), q);
        phi_m.submit_value(coeff*(scalar_product(avg_u_star_star, n_plus)), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // The following function assembles rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number>   phi(data, true, 1, 1);
    FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_p, dim, Number> phi_proj(data, true, 0, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1.0/(gamma*dt) : 1.0/((1.0 - gamma)*dt);

    /*--- We loop over faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_proj.reinit(face);
      phi_proj.gather_evaluate(src[0], EvaluationFlags::values);

      phi.reinit(face);

      /*--- Now we loop over all the quadrature points to compute the integrals ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus      = phi.get_normal_vector(q);

        const auto& u_star_star = phi_proj.get_value(q);

        phi.submit_value(-coeff*(scalar_product(u_star_star, n_plus)), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(auto& vec : src) {
      vec.update_ghost_values();
    }

    this->data->loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_face_term_pressure,
                     &NavierStokesProjectionOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Next, we focus on 'matrices' to compute the pressure. We first assemble cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    const double coeff = (TR_BDF2_stage == 1) ? 1e3*gamma*dt*gamma*dt : 1e3*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Now we loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_gradient(phi.get_gradient(q), q);
        phi.submit_value(1.0/coeff*phi.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1),
                                                                   phi_m(data, false, 1, 1);

    /*--- Loop over all faces in the range ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus        = phi_p.get_normal_vector(q);

        const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
        const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);

        phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
        phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
        phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      phi_m.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // The following function assembles boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, true, 1, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto boundary_id = data.get_boundary_id(face);

      if(boundary_id == 3) {
        phi.reinit(face);
        phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto coef_jump = C_p*std::abs((phi.get_normal_vector(0)*phi.inverse_jacobian(0))[dim - 1]);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus    = phi.get_normal_vector(q);

          const auto& grad_pres = phi.get_gradient(q);
          const auto& pres      = phi.get_value(q);

          phi.submit_value(-scalar_product(grad_pres, n_plus) + 2.0*coef_jump*pres, q);
          phi.submit_normal_derivative(-theta_p*pres, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Before coding the 'apply_add' function, which is the one that will perform the loop, we focus on
  // the linear system that arises to project the gradient of the pressure into the velocity space.
  // The following function assembles rhs cell term for the projection of gradient of pressure. Since no
  // integration by parts is performed, only a cell term contribution is present.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_rhs_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read already available quantities. ---*/
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_v, 1, Number>   phi_pres(data, 1);

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres.reinit(cell);
      phi_pres.gather_evaluate(src, EvaluationFlags::gradients);

      phi.reinit(cell);

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi_pres.get_gradient(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for porjection of pressure gradient. Here we loop only over cells
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  vmult_grad_p_projection(Vec& dst, const Vec& src) const {
    this->data->cell_loop(&NavierStokesProjectionOperator::assemble_rhs_cell_term_projection_grad_p,
                          this, dst, src, true);
  }


  // Assemble now cell term for the projection of gradient of pressure. This is nothing but a mass matrix
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_cell_term_projection_grad_p(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0);

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps. This is the overriden function that effectively performs the
  // matrix-vector multiplication.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    if(NS_stage == 1) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_velocity,
                       &NavierStokesProjectionOperator::assemble_face_term_velocity,
                       &NavierStokesProjectionOperator::assemble_boundary_term_velocity,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 2) {
      this->data->loop(&NavierStokesProjectionOperator::assemble_cell_term_pressure,
                       &NavierStokesProjectionOperator::assemble_face_term_pressure,
                       &NavierStokesProjectionOperator::assemble_boundary_term_pressure,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 3) {
      this->data->cell_loop(&NavierStokesProjectionOperator::assemble_cell_term_projection_grad_p,
                            this, dst, src, false); /*--- Since we have only a cell term contribution, we use cell_loop ---*/
    }
    else {
      Assert(false, ExcNotImplemented());
    }
  }


  // Finally, we focus on computing the diagonal for preconditioners and we start by assembling
  // the diagonal cell term for the velocity. Since we do not have access to the entries of the matrix,
  // in order to compute the element i, we test the matrix against a vector which is equal to 1 in position i and 0 elsewhere.
  // This is why 'src' will result as unused.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(TR_BDF2_stage == 1) {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n_gamma_ov_2(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      /*--- Build a vector of ones to be tested (here we will see the velocity as a whole vector, since
                                                 dof_handler_velocity is vectorial and so the dof values are vectors). ---*/
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0);
      }

      /*--- Loop over cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_n_gamma_ov_2.reinit(cell);
        phi_n_gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j); /*--- Set all dofs to zero ---*/
          }
          phi.submit_dof_value(tmp, i); /*--- Set dof i equal to one ---*/
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& grad_u           = phi.get_gradient(q);

            const auto& u                = phi.get_value(q);
            const auto& u_n_gamma_ov_2   = phi_n_gamma_ov_2.get_value(q);
            const auto& tensor_product_u = outer_product(u, u_n_gamma_ov_2);

            phi.submit_value(1.0/(gamma*dt)*u, q);
            phi.submit_gradient(-a22*tensor_product_u + a22/Re*grad_u, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          phi.submit_dof_value(diagonal[i], i);
        }
        phi.distribute_local_to_global(dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, 0),
                                                                   phi_n_3gamma_ov_2(data, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0);
      }

      /*--- Loop over cells in the range ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_n_3gamma_ov_2.reinit(cell);
        phi_n_3gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over dofs ---*/
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
            phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi.submit_dof_value(tmp, i);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points ---*/
          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& grad_u           = phi.get_gradient(q);

            const auto& u                = phi.get_value(q);
            const auto& u_n_3gamma_ov_2  = phi_n_3gamma_ov_2.get_value(q);
            const auto& tensor_product_u = outer_product(u, u_n_3gamma_ov_2);

            phi.submit_value(1.0/((1.0 - gamma)*dt)*u, q);
            phi.submit_gradient(-a33*tensor_product_u + a33/Re*grad_u, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          phi.submit_dof_value(diagonal[i], i);
        }
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // The following function assembles diagonal face term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_gamma_ov_2_p(data, true, 0),
                                                                       phi_n_gamma_ov_2_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component); /*--- We just assert for safety that dimension match,
                                                                                in the sense that we have selected the proper
                                                                                space ---*/
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component),
                                                             diagonal_m(phi_m.dofs_per_component);

      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0); /*--- We build the usal vector of ones that we will use as dof value ---*/
      }

      /*--- Now we loop over faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n_gamma_ov_2_p.reinit(face);
        phi_n_gamma_ov_2_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_n_gamma_ov_2_m.reinit(face);
        phi_n_gamma_ov_2_m.gather_evaluate(u_extr, EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over dofs. We will set all equal to zero apart from the current one ---*/
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points to compute the integral ---*/
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus               = phi_p.get_normal_vector(q);

            const auto& grad_u_p             = phi_p.get_gradient(q);
            const auto& grad_u_m             = phi_m.get_gradient(q);
            const auto& avg_grad_u           = 0.5*(grad_u_p + grad_u_m);

            const auto& u_p                  = phi_p.get_value(q);
            const auto& u_m                  = phi_m.get_value(q);
            const auto& u_n_gamma_ov_2_p     = phi_n_gamma_ov_2_p.get_value(q);
            const auto& u_n_gamma_ov_2_m     = phi_n_gamma_ov_2_m.get_value(q);
            const auto& avg_tensor_product_u = 0.5*(outer_product(u_p, u_n_gamma_ov_2_p) +
                                                    outer_product(u_m, u_n_gamma_ov_2_m));

            const auto& lambda_n_gamma_ov_2  = std::max(std::abs(scalar_product(u_n_gamma_ov_2_p, n_plus)),
                                                        std::abs(scalar_product(u_n_gamma_ov_2_m, n_plus)));
            const auto& jump_u               = u_p - u_m;

            phi_p.submit_value(a22/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                               a22*avg_tensor_product_u*n_plus + 0.5*a22*lambda_n_gamma_ov_2*jump_u, q);
            phi_m.submit_value(-a22/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                               a22*avg_tensor_product_u*n_plus - 0.5*a22*lambda_n_gamma_ov_2*jump_u, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a22/Re*jump_u, q);
          }
          phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_n_3gamma_ov_2_p(data, true, 0),
                                                                       phi_n_3gamma_ov_2_m(data, false, 0);

      AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component),
                                                             diagonal_m(phi_m.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0);
      }

      /*--- Now we loop over faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n_3gamma_ov_2_p.reinit(face);
        phi_n_3gamma_ov_2_p.gather_evaluate(u_extr, EvaluationFlags::values);
        phi_n_3gamma_ov_2_m.reinit(face);
        phi_n_3gamma_ov_2_m.gather_evaluate(u_extr, EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over dofs. We will set all equal to zero apart from the current one ---*/
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
            phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          }
          phi_p.submit_dof_value(tmp, i);
          phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          phi_m.submit_dof_value(tmp, i);
          phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          /*--- Loop over quadrature points to compute the integral ---*/
          for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
            const auto& n_plus               = phi_p.get_normal_vector(q);

            const auto& grad_u_p             = phi_p.get_gradient(q);
            const auto& grad_u_m             = phi_m.get_gradient(q);
            const auto& avg_grad_u           = 0.5*(grad_u_p + grad_u_m);

            const auto& u_p                  = phi_p.get_value(q);
            const auto& u_m                  = phi_m.get_value(q);
            const auto& u_n_3gamma_ov_2_p    = phi_n_3gamma_ov_2_p.get_value(q);
            const auto& u_n_3gamma_ov_2_m    = phi_n_3gamma_ov_2_m.get_value(q);
            const auto& avg_tensor_product_u = 0.5*(outer_product(u_p, u_n_3gamma_ov_2_p) +
                                                    outer_product(u_m, u_n_3gamma_ov_2_m));

            const auto& lambda_n_3gamma_ov_2 = std::max(std::abs(scalar_product(u_n_3gamma_ov_2_p, n_plus)),
                                                        std::abs(scalar_product(u_n_3gamma_ov_2_m, n_plus)));
            const auto& jump_u               = u_p - u_m;

            phi_p.submit_value(a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) +
                               a33*avg_tensor_product_u*n_plus + 0.5*a33*lambda_n_3gamma_ov_2*jump_u, q);
            phi_m.submit_value(-a33/Re*(-avg_grad_u*n_plus + coef_jump*jump_u) -
                               a33*avg_tensor_product_u*n_plus - 0.5*a33*lambda_n_3gamma_ov_2*jump_u, q);
            phi_p.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
            phi_m.submit_normal_derivative(-theta_v*0.5*a33/Re*jump_u, q);
          }
          phi_p.integrate(true, true);
          diagonal_p[i] = phi_p.get_dof_value(i);
          phi_m.integrate(true, true);
          diagonal_m[i] = phi_m.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
          phi_p.submit_dof_value(diagonal_p[i], i);
          phi_m.submit_dof_value(diagonal_m[i], i);
        }
        phi_p.distribute_local_to_global(dst);
        phi_m.distribute_local_to_global(dst);
      }
    }
  }


  // The following function assembles boundary term for the velocity
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const unsigned int&                          ,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    if(TR_BDF2_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n_gamma_ov_2(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0);
      }

      /*--- Loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n_gamma_ov_2.reinit(face);
        phi_n_gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 3) {
          const double coef_trasp = 0.0;

          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            }
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus              = phi.get_normal_vector(q);

              const auto& grad_u              = phi.get_gradient(q);

              const auto& u                   = phi.get_value(q);
              const auto& u_n_gamma_ov_2      = phi_n_gamma_ov_2.get_value(q);
              const auto& tensor_product_u    = outer_product(u, u_n_gamma_ov_2);

              const auto& lambda_n_gamma_ov_2 = std::abs(scalar_product(u_n_gamma_ov_2, n_plus));

              phi.submit_value(a22/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                               a22*coef_trasp*tensor_product_u*n_plus + a22*lambda_n_gamma_ov_2*u, q);
              phi.submit_normal_derivative(-theta_v*a22/Re*u, q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            phi.submit_dof_value(diagonal[i], i);
          }
          phi.distribute_local_to_global(dst);
        }
        else {
          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            }
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus    = phi.get_normal_vector(q);

              const auto& grad_u    = phi.get_gradient(q);
              const auto& u         = phi.get_value(q);

              auto u_n_gamma_m      = u;
              auto grad_u_n_gamma_m = grad_u;
              for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                u_n_gamma_m[1][v]         = -u_n_gamma_m[1][v];

                grad_u_n_gamma_m[0][0][v] = -grad_u_n_gamma_m[0][0][v];
                grad_u_n_gamma_m[0][1][v] = -grad_u_n_gamma_m[0][1][v];
              }

              const auto& u_n_gamma_ov_2      = phi_n_gamma_ov_2.get_value(q);
              const auto& lambda_n_gamma_ov_2 = std::abs(scalar_product(u_n_gamma_ov_2, n_plus));

              phi.submit_value(a22/Re*(-(0.5*(grad_u + grad_u_n_gamma_m))*n_plus + coef_jump*(u - u_n_gamma_m)) +
                               a22*outer_product(0.5*(u + u_n_gamma_m), u_n_gamma_ov_2)*n_plus +
                               a22*0.5*lambda_n_gamma_ov_2*(u - u_n_gamma_m), q);
              phi.submit_normal_derivative(-theta_v*a22/Re*(u - u_n_gamma_m), q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            phi.submit_dof_value(diagonal[i], i);
          }
          phi.distribute_local_to_global(dst);
        }
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_v, n_q_points_1d_v, dim, Number> phi(data, true, 0),
                                                                       phi_n_3gamma_ov_2(data, true, 0);

      AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
      Tensor<1, dim, VectorizedArray<Number>> tmp;
      for(unsigned int d = 0; d < dim; ++d) {
        tmp[d] = make_vectorized_array<Number>(1.0);
      }

      /*--- Loop over all faces in the range ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_n_3gamma_ov_2.reinit(face);
        phi_n_3gamma_ov_2.gather_evaluate(u_extr, EvaluationFlags::values);

        phi.reinit(face);

        const auto boundary_id = data.get_boundary_id(face);
        const auto coef_jump   = C_u*std::abs((phi.get_normal_vector(0) * phi.inverse_jacobian(0))[dim - 1]);

        if(boundary_id != 3) {
          const double coef_trasp = 0.0;

          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            }
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus               = phi.get_normal_vector(q);

              const auto& grad_u               = phi.get_gradient(q);

              const auto& u                    = phi.get_value(q);
              const auto& u_n_3gamma_ov_2      = phi_n_3gamma_ov_2.get_value(q);
              const auto& tensor_product_u     = outer_product(u, u_n_3gamma_ov_2);

              const auto& lambda_n_3gamma_ov_2 = std::abs(scalar_product(u_n_3gamma_ov_2, n_plus));

              phi.submit_value(a33/Re*(-grad_u*n_plus + 2.0*coef_jump*u) +
                               a33*coef_trasp*tensor_product_u*n_plus + a33*lambda_n_3gamma_ov_2*u, q);
              phi.submit_normal_derivative(-theta_v*a33/Re*u, q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            phi.submit_dof_value(diagonal[i], i);
          }
          phi.distribute_local_to_global(dst);
        }
        else {
          /*--- Loop over all dofs ---*/
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
              phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
            }
            phi.submit_dof_value(tmp, i);
            phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

            /*--- Loop over quadrature points to compute the integral ---*/
            for(unsigned int q = 0; q < phi.n_q_points; ++q) {
              const auto& n_plus  = phi.get_normal_vector(q);

              const auto& grad_u  = phi.get_gradient(q);
              const auto& u       = phi.get_value(q);

              auto u_np1_m        = u;
              auto grad_u_np1_m   = grad_u;
              for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
                u_np1_m[1][v]         = -u_np1_m[1][v];

                grad_u_np1_m[0][0][v] = -grad_u_np1_m[0][0][v];
                grad_u_np1_m[0][1][v] = -grad_u_np1_m[0][1][v];
              }

              const auto& u_n_3gamma_ov_2      = phi_n_3gamma_ov_2.get_value(q);
              const auto& lambda_n_3gamma_ov_2 = std::abs(scalar_product(u_n_3gamma_ov_2, n_plus));

              phi.submit_value(a33/Re*(-(0.5*(grad_u + grad_u_np1_m))*n_plus + coef_jump*(u - u_np1_m)) +
                               a33*outer_product(0.5*(u + u_np1_m), u_n_3gamma_ov_2)*n_plus +
                               a33*0.5*lambda_n_3gamma_ov_2*(u - u_np1_m), q);
              phi.submit_normal_derivative(-theta_v*a33/Re*(u - u_np1_m), q);
            }
            phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
            diagonal[i] = phi.get_dof_value(i);
          }
          for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
            phi.submit_dof_value(diagonal[i], i);
          }
          phi.distribute_local_to_global(dst);
        }
      }
    }
  }


  // Now we consider the pressure related bilinear forms. We first assemble diagonal cell term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component); /*--- Here we are using dofs_per_component but
                                                                                   it coincides with dofs_per_cell since it is
                                                                                   scalar finite element space ---*/

    const double coeff = (TR_BDF2_stage == 1) ? 1e3*gamma*dt*gamma*dt : 1e3*(1.0 - gamma)*dt*(1.0 - gamma)*dt;

    /*--- Loop over all cells in the range ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j); /*--- We set all dofs to zero ---*/
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i); /*--- Now we set the current one to 1; since it is scalar,
                                                                           we can directly use 'make_vectorized_array' without
                                                                           relying on 'Tensor' ---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1.0/coeff*phi.get_value(q), q);
          phi.submit_gradient(phi.get_gradient(q), q);
        }
        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // The following function assembles diagonal face term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi_p(data, true, 1, 1),
                                                                   phi_m(data, false, 1, 1);

    AssertDimension(phi_p.dofs_per_component, phi_m.dofs_per_component);
    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component); /*--- Again, we just assert for safety that dimension
                                                                                       match, in the sense that we have selected
                                                                                       the proper space ---*/

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_p.reinit(face);
      phi_m.reinit(face);

      const auto coef_jump = C_p*0.5*(std::abs((phi_p.get_normal_vector(0)*phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0)*phi_m.inverse_jacobian(0))[dim - 1]));

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points to compute the integral ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus        = phi_p.get_normal_vector(q);

          const auto& avg_grad_pres = 0.5*(phi_p.get_gradient(q) + phi_m.get_gradient(q));
          const auto& jump_pres     = phi_p.get_value(q) - phi_m.get_value(q);

          phi_p.submit_value(-scalar_product(avg_grad_pres, n_plus) + coef_jump*jump_pres, q);
          phi_m.submit_value(scalar_product(avg_grad_pres, n_plus) - coef_jump*jump_pres, q);
          phi_p.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
          phi_m.submit_gradient(-theta_p*0.5*jump_pres*n_plus, q);
        }
        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Eventually, we assemble diagonal boundary term for the pressure
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  assemble_diagonal_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const unsigned int&                          ,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_p, 1, Number> phi(data, true, 1, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      const auto boundary_id = data.get_boundary_id(face);

      if(boundary_id == 3) {
        phi.reinit(face);

        const auto coef_jump = C_p*std::abs((phi.get_normal_vector(0)*phi.inverse_jacobian(0))[dim - 1]);

        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
            phi.submit_dof_value(VectorizedArray<Number>(), j);
          }
          phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
          phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

          for(unsigned int q = 0; q < phi.n_q_points; ++q) {
            const auto& n_plus    = phi.get_normal_vector(q);

            const auto& grad_pres = phi.get_gradient(q);
            const auto& pres      = phi.get_value(q);

            phi.submit_value(-scalar_product(grad_pres, n_plus) + 2.0*coef_jump*pres , q);
            phi.submit_normal_derivative(-theta_p*pres, q);
          }
          phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
          diagonal[i] = phi.get_dof_value(i);
        }
        for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
          phi.submit_dof_value(diagonal[i], i);
        }
        phi.distribute_local_to_global(dst);
      }
    }
  }


  // Put together all previous steps. We create a dummy auxliary vector that serves for the src input argument in
  // the previous functions that as we have seen before is unused. Then everything is done by the 'loop' function
  // and it is saved in the field 'inverse_diagonal_entries' already present in the base class. Anyway since there is
  // only one field, we need to resize properly depending on whether we are considering the velocity or the pressure.
  //
  template<int dim, int fe_degree_p, int fe_degree_v, int n_q_points_1d_p, int n_q_points_1d_v, typename Vec>
  void NavierStokesProjectionOperator<dim, fe_degree_p, fe_degree_v, n_q_points_1d_p, n_q_points_1d_v, Vec>::
  compute_diagonal() {
    Assert(NS_stage == 1 || NS_stage == 2, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    if(NS_stage == 1) {
      ::MatrixFreeTools::compute_diagonal<dim, Number, VectorizedArray<Number>>
      (*(this->data),
       inverse_diagonal,
       [&](const auto& data, auto& dst, const auto& src, const auto& cell_range) {
         (this->assemble_diagonal_cell_term_velocity)(data, dst, src, cell_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& face_range) {
         (this->assemble_diagonal_face_term_velocity)(data, dst, src, face_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& boundary_range) {
         (this->assemble_diagonal_boundary_term_velocity)(data, dst, src, boundary_range);
       },
       0);
    }
    else if(NS_stage == 2) {
        ::MatrixFreeTools::compute_diagonal<dim, Number, VectorizedArray<Number>>
      (*(this->data),
       inverse_diagonal,
       [&](const auto& data, auto& dst, const auto& src, const auto& cell_range) {
         (this->assemble_diagonal_cell_term_pressure)(data, dst, src, cell_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& face_range) {
         (this->assemble_diagonal_face_term_pressure)(data, dst, src, face_range);
       },
       [&](const auto& data, auto& dst, const auto& src, const auto& boundary_range) {
         (this->assemble_diagonal_boundary_term_pressure)(data, dst, src, boundary_range);
       },
       1);
    }

    for(unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }

} // End of namespace TR-BDF2
