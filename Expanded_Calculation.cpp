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
// result calculated via numerical integration
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
auto numerical_result(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // constant coefficient
  const auto A = u*sqrt(B);

  // first function
  const auto function_1 = [=](const double &theta){
    return pow(cos(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // second function
  const auto function_2 = [=](const double &theta){
    return cos(theta)/exp(u*u*B*pow(cos(theta), 2));
  };

  // third function
  const auto function_3 = [=](const double &theta){
    return pow(cos(theta), 3)*erf(A*cos(theta));
  };

  // fourth function
  const auto function_4 = [=](const double &theta){
    return pow(cos(theta), 3);
  };

  // fifth function
  const auto function_5 = [=](const double &theta){
    return pow(cos(theta), 2)*erf(A*cos(theta));
  };

  // sixth function
  const auto function_6 = [=](const double &theta){
    return pow(cos(theta), 2);
  };

  // seventh function
  const auto function_7 = [=](const double &theta){
    return cos(theta)*erf(A*cos(theta));
  };

  // eigth function
  const auto function_8 = [=](const double &theta){
    return cos(theta);
  };

  // ninth function
  const auto function_9 = [=](const double &theta){
    return pow(sin(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // tenth function
  const auto function_10 = [=](const double &theta){
    return pow(sin(theta), 2)*cos(theta)*erf(A*cos(theta));
  };

  // eleventh function
  const auto function_11 = [=](const double &theta){
    return pow(sin(theta), 2)*cos(theta);
  };

  // twelvth function
  const auto function_12 = [=](const double &theta){
    return sin(theta)/exp(u*u*B*pow(cos(theta), 2));
  };

  // thirteenth function
  const auto function_13 = [=](const double &theta){
    return sin(theta)*cos(theta)*erf(A*cos(theta));
  };

  // fourteenth function
  const auto function_14 = [=](const double &theta){
    return sin(theta)*cos(theta);
  };

  // results
  std::vector<Scalar_type> results(14);

  results[0]  = trapezoidal_integration_handler<1>(Q, function_1,  0.0, 2.0*pi, N);
  results[1]  = trapezoidal_integration_handler<1>(Q, function_2,  0.0, 2.0*pi, N);
  results[2]  = trapezoidal_integration_handler<1>(Q, function_3,  0.0, 2.0*pi, N);
  results[3]  = trapezoidal_integration_handler<1>(Q, function_4,  0.0, 2.0*pi, N);
  results[4]  = trapezoidal_integration_handler<1>(Q, function_5,  0.0, 2.0*pi, N);
  results[5]  = trapezoidal_integration_handler<1>(Q, function_6,  0.0, 2.0*pi, N);
  results[6]  = trapezoidal_integration_handler<1>(Q, function_7,  0.0, 2.0*pi, N);
  results[7]  = trapezoidal_integration_handler<1>(Q, function_8,  0.0, 2.0*pi, N);
  results[8]  = trapezoidal_integration_handler<1>(Q, function_9,  0.0, 2.0*pi, N);
  results[9]  = trapezoidal_integration_handler<1>(Q, function_10, 0.0, 2.0*pi, N);
  results[10] = trapezoidal_integration_handler<1>(Q, function_11, 0.0, 2.0*pi, N);
  results[11] = trapezoidal_integration_handler<1>(Q, function_12, 0.0, 2.0*pi, N);
  results[12] = trapezoidal_integration_handler<1>(Q, function_13, 0.0, 2.0*pi, N);
  results[13] = trapezoidal_integration_handler<1>(Q, function_14, 0.0, 2.0*pi, N);

  return results;
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
// result calculated via analytical relations
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

  // defining some variables
  const double B = 8.78E-6;
  const double B_h = 9.87E-6;
  const double n = 0.178;
  const double u = 300;
  const double w = 320;
  const double R = 1.0;
  const double m = 1.0;
  const double a = 1.0;

  const auto num_results = numerical_result(Q, B, u, N);
  const auto ana_results = analytical_result(B, u);

  const auto num_sum = coefficient_calculation(num_results, n, m, u, B, B_h, a, R, w);
  const auto ana_sum = coefficient_calculation(ana_results, n, m, u, B, B_h, a, R, w);
  check(1.0E-6, num_sum, ana_sum);

  std::cout << "numerical: " << num_sum << " and analytical: " << ana_sum << std::endl;
  return 0;
}

