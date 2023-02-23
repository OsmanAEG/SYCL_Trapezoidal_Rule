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
// first function to evaluate
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
void function_1(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // function
  const auto function = [=](const double &theta){
    return pow(cos(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function, 0.0, 2.0*pi, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                                + boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  // checking the answer
  check(1.0E-6, numerical_result, bessel_result);

  std::cout << "The Results to Function 1 are Correct!" << std::endl;
  std::cout << "Result = " << numerical_result << " and Bessel = "
            << bessel_result << std::endl;
}

/////////////////////////////////////////////////
// second function to evaluate
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
void function_2(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // function
  const auto function = [=](const double &theta){
    return pow(sin(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function, 0.0, 2.0*pi, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                                - boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  // checking the answer
  check(1.0E-6, numerical_result, bessel_result);

  std::cout << "The Results to Function 2 are Correct!" << std::endl;
  std::cout << "Result = " << numerical_result << " and Bessel = "
            << bessel_result << std::endl;
}

/////////////////////////////////////////////////
// third function to evaluate
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
void function_3(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // constant coefficient
  const auto A = u*sqrt(B);

  // function
  const auto function = [=](const double &theta){
    return cos(theta)*pow(sin(theta), 2)*erf(A*cos(theta));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function, 0.0, 2.0*pi, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = A*sqrt(pi)/6.0*exp(-A*A/2.0)*(
    2.0*boost::math::cyl_bessel_i(0, -0.5*A*A)
    - 4.0*boost::math::cyl_bessel_i(1, -0.5*A*A)
    + (boost::math::cyl_bessel_i(0, -0.5*A*A) + boost::math::cyl_bessel_i(2, -0.5*A*A)));

  // checking the answer
  check(1.0E-6, numerical_result, bessel_result);

  std::cout << "The Results to Function 3 are Correct!" << std::endl;
  std::cout << "Result = " << numerical_result << " and Bessel = "
            << bessel_result << std::endl;
}

/////////////////////////////////////////////////
// fourth function to evaluate
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
void function_4(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // constant coefficient
  const auto A = u*sqrt(B);

  // function
  const auto function = [=](const double &theta){
    return pow(cos(theta), 3)*erf(A*cos(theta));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function, 0.0, 2.0*pi, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = 2.0*A/sqrt(pi)*exp(-u*u*B/2.0)*pi*(
    boost::math::cyl_bessel_i(0, -u*u*B/2.0)
    - boost::math::cyl_bessel_i(1, -u*u*B/2.0)) -A*sqrt(pi)/6.0*exp(-A*A/2.0)*(
    2.0*boost::math::cyl_bessel_i(0, -0.5*A*A)
    - 4.0*boost::math::cyl_bessel_i(1, -0.5*A*A)
    + (boost::math::cyl_bessel_i(0, -0.5*A*A) + boost::math::cyl_bessel_i(2, -0.5*A*A)));

  // checking the answer
  check(1.0E-6, numerical_result, bessel_result);

  std::cout << "The Results to Function 4 are Correct!" << std::endl;
  std::cout << "Result = " << numerical_result << " and Bessel = "
            << bessel_result << std::endl;
}

/////////////////////////////////////////////////
// fifth function to evaluate
template<typename Sycl_Queue, typename Scalar_type, typename Int_type>
void function_5(Sycl_Queue Q, Scalar_type B, Scalar_type u, Int_type N){
  // constant coefficient
  const auto A = u*sqrt(B);

  // function
  const auto function = [=](const double &theta){
    return cos(theta)*erf(A*cos(theta));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function, 0.0, 2.0*pi, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = 2.0*A/sqrt(pi)*exp(-u*u*B/2.0)*pi*(
    boost::math::cyl_bessel_i(0, -u*u*B/2.0)
    - boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  // checking the answer
  check(1.0E-6, numerical_result, bessel_result);

  std::cout << "The Results to Function 5 are Correct!" << std::endl;
  std::cout << "Result = " << numerical_result << " and Bessel = "
            << bessel_result << std::endl;
}

int main(){
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // number of integration segments
  const double N = 5000;

  // defining some variables
  const double B = 8.78E-6;
  const double u = 300;

  // testing some functions
  function_1(Q, B, u, N);
  function_2(Q, B, u, N);
  function_3(Q, B, u, N);
  function_4(Q, B, u, N);
  function_5(Q, B, u, N);
}