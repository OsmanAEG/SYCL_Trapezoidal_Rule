// Standard C++ include items
#include <iostream>
#include <cmath>

// Boost library
#include <boost/math/special_functions/bessel.hpp>

// Including the SYCL library
#include <CL/sycl.hpp>

// Including local header files
#include "Trapezoidal_Integration_Handler.h"

// defining some constants
#define pi M_PI

/////////////////////////////////////////////////
// function to check accuracy
template<typename Scalar_type>
void check(Scalar_type tolerance, Scalar_type result, Scalar_type answer){
  assert(fabs(result-answer) < tolerance);
}

/////////////////////////////////////////////////
// result calculated via analytical relations
template<typename Scalar_type>
auto analytical_result(Scalar_type B, Scalar_type u){
  // constant coefficient
  const auto A = u*sqrt(B);

  // results
  std::vector<Scalar_type> results(14);

  results[0] = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                + boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  results[1] = 0.0;
  results[2] = 2.0*A/sqrt(pi)*exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                                                  - boost::math::cyl_bessel_i(1, -u*u*B/2.0))
              - A*sqrt(pi)/6.0*exp(-A*A/2.0)*(2.0*boost::math::cyl_bessel_i(0, -0.5*A*A)
                                              - 4.0*boost::math::cyl_bessel_i(1, -0.5*A*A)
                                              + (boost::math::cyl_bessel_i(0, -0.5*A*A)
                                                + boost::math::cyl_bessel_i(2, -0.5*A*A)));

  results[3] = 0.0;
  results[4] = 0.0;
  results[5] = pi;
  results[6] = 2.0*A/sqrt(pi)*exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                                                   - boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  results[7] = 0.0;
  results[8] = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                                    - boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  results[9] = A*sqrt(pi)/6.0*exp(-A*A/2.0)*(2.0*boost::math::cyl_bessel_i(0, -0.5*A*A)
                                              - 4.0*boost::math::cyl_bessel_i(1, -0.5*A*A)
                                              + (boost::math::cyl_bessel_i(0, -0.5*A*A)
                                                  + boost::math::cyl_bessel_i(2, -0.5*A*A)));

  results[10] = 0.0;
  results[11] = 0.0;
  results[12] = 0.0;
  results[13] = 0.0;

  return results;
}

/////////////////////////////////////////////////
// coefficients required for analytical solutions
template<typename Vector_type, typename Scalar_type>
auto coefficient_calculation(Vector_type results, Scalar_type n, Scalar_type m,
                             Scalar_type u, Scalar_type B, Scalar_type B_h,
                             Scalar_type a, Scalar_type R, Scalar_type w){
  // constant coefficient
  Vector_type coefficients(14);
  coefficients[0]  = n*m*u/(sqrt(B*pi))*(1-0.5*a);
  coefficients[1]  = -n*m*a/(4.0*sqrt(B*B_h));
  coefficients[2]  = n*m*u*u*(1.0-a/2.0);
  coefficients[3]  = -n*m*u*u*(1.0-a/2.0);
  coefficients[4]  = -n*m*a*sqrt(pi)*u/(4.0*sqrt(B_h));
  coefficients[5]  = n*m*sqrt(pi)*u*a/(4.0*sqrt(B_h));
  coefficients[6]  = n*m/(2.0*B)*(1.0-a/2.0);
  coefficients[7]  = -n*m/(2.0*B)*(1.0-a/2.0);
  coefficients[8]  = n*m*a*u/(2.0*sqrt(B*pi));
  coefficients[9]  = n*m*u*u*a/2.0;
  coefficients[10] = -n*m*u*u*a/2.0;
  coefficients[11] = -n*m*a*R*w/(2.0*sqrt(B*pi));
  coefficients[12] = -n*m*u*a*R*w/2.0;
  coefficients[13] = n*m*u*a*R*w/2.0;

  auto sum = 0.0;

  for(int i = 0; i < 14; ++i){
    sum += coefficients[i]*results[i];
  }

  return sum;
}

int main(){
  // device queue
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // number of integration segments
  const double N = 5000;

  // properties
  const double R   = 1.0;
  const double rho = 1.7838;
  const double u   = 300.0;
  const double w   = 0.0;
  const double p   = 101325.0;
  const double T_h = 273.15;
  const double m   = 6.6335209e-26;

  const double B   = rho/(2.0*p);
  const double B_h = rho/(2.0*p);

  const double n = rho/m;
  const double a = 1.0;

  const auto analytical = analytical_result(B, u);
  const auto D          = coefficient_calculation(analytical, n, m, u, B, B_h, a, R, w);

  const auto cd = D/(rho*u*u*R);

  std::cout << cd << std::endl;
}