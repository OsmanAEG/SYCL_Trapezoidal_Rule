// Standard C++ include items
#include <iostream>
#include <cmath>

// Boost library
#include <boost/math/special_functions/bessel.hpp>

// Including the SYCL library
#include <CL/sycl.hpp>

// Including local header files
#include "Trapezoidal_Integration_Handler.h"

// first function to evaluate
template<typename Sycl_Queue, typename Int_type>
void function_1(Sycl_Queue Q, Int_type N){
  // defining some constants
  const double pi = M_PI;

  // defining some variables
  const double theta_1 = 0.0;
  const double theta_2 = 2.0*pi;
  const double B = 8.78E-6;
  const double u = 300;

  // tolerance requirement
  constexpr auto tolerance = 1.0e-3;

  auto check = [&](double result, double answer){
    assert(fabs(result-answer) < tolerance);
  };

  // first function
  const auto function_1 = [=](const double &theta){
    return pow(cos(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // evaluating the integral
  const auto numerical_result = trapezoidal_integration_handler<1>(Q, function_1, theta_1, theta_2, N);

  // eveluating the integral as a bessel function
  const auto bessel_result = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(0, -u*u*B/2.0)
                            + boost::math::cyl_bessel_i(1, -u*u*B/2.0));

  // checking the answer
  check(numerical_result, bessel_result);

  std::cout << "The Results to Function 1 are Correct!" << std::endl;
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

  // testing some functions
  function_1(Q, N);
}