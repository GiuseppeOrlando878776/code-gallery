// We start by including all the necessary deal.II header files.
//
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

// @sect{Equation data}

// In the next namespace, we declare and implement suitable functions that may be used for the initial and boundary conditions
//
namespace EquationData {
  using namespace dealii;

  static const unsigned int degree_p = 1;        /*--- Polynomial degree for the pressure. ---*/
  static const unsigned int degree_v = degree_p; /*--- Polynomial degree for the velocity. ---*/

  // We declare class that describes the boundary conditions and initial one for velocity:
  //
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;
  };


  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}


  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      return 1.0;
    }
    else {
      return 0.0;
    }
  }


  template<int dim>
  void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the pressure
  //
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  template<int dim>
  Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}


  template<int dim>
  double Pressure<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    return 30.0 - p(0);
  }

} // namespace EquationData
