// Standard C++ include items
#include <iostream>
#include <cmath>

// Boost library
#include <boost/math/special_functions/bessel.hpp>

// Including the SYCL library
#include <CL/sycl.hpp>

// Including local header files
#include "Trapezoidal_Integration_Handler.h"

int main(){
  // defining some constants
  const double pi = M_PI;

  // number of points
  size_t N = 5000;

  // defining some variables
  const double theta_1 = 0.0;
  const double theta_2 = 2.0*pi;
  const double B = 1.78/(2*101325.0);
  const double u = 300;

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
    return pow(cos(theta), 3)*erf(sqrt(B)*cos(theta)*u);
  };

  // fourth function
  const auto function_4 = [=](const double &theta){
    return pow(cos(theta), 2)*erf(sqrt(B)*cos(theta)*u);
  };

  // fifth function
  const auto function_5 = [=](const double &theta){
    return cos(theta)*erf(sqrt(B)*cos(theta)*u);
  };

  // sixth function
  const auto function_6 = [=](const double &theta){
    return pow(sin(theta), 2)/exp(u*u*B*pow(cos(theta), 2));
  };

  // seventh function
  const auto function_7 = [=](const double &theta){
    return pow(sin(theta), 2)*erf(sqrt(B)*cos(theta)*u)*cos(theta);
  };

  // eighth function
  const auto function_8 = [=](const double &theta){
    return sin(theta)*erf(sqrt(B)*cos(theta)*u)*cos(theta);
  };

  // setting up the SYCL queue
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // evaluating the integral
  const auto result_1 = trapezoidal_integration_handler<1>(Q, function_1, theta_1, theta_2, N);
  const auto result_2 = trapezoidal_integration_handler<1>(Q, function_2, theta_1, theta_2, N);
  const auto result_3 = trapezoidal_integration_handler<1>(Q, function_3, theta_1, theta_2, N);
  const auto result_4 = trapezoidal_integration_handler<1>(Q, function_4, theta_1, theta_2, N);
  const auto result_5 = trapezoidal_integration_handler<1>(Q, function_5, theta_1, theta_2, N);
  const auto result_6 = trapezoidal_integration_handler<1>(Q, function_6, theta_1, theta_2, N);
  const auto result_7 = trapezoidal_integration_handler<1>(Q, function_7, theta_1, theta_2, N);
  const auto result_8 = trapezoidal_integration_handler<1>(Q, function_8, theta_1, theta_2, N);

  // bessel function
  int order0 = 0;
  int order1 = 1;

  const auto bessel_1 = exp(-u*u*B/2.0)*pi*(boost::math::cyl_bessel_i(order0, -u*u*B/2.0)
                            + boost::math::cyl_bessel_i(order1, -u*u*B/2.0));

  std::cout << "Result 1: " << result_1 << " and Bessel 1: " << bessel_1 << std::endl;
  std::cout << "Result 2: " << result_2 << std::endl;
  std::cout << "Result 3: " << result_3 << std::endl;
  std::cout << "Result 4: " << result_4 << std::endl;
  std::cout << "Result 5: " << result_5 << std::endl;
  std::cout << "Result 6: " << result_6 << std::endl;
  std::cout << "Result 7: " << result_7 << std::endl;
  std::cout << "Result 8: " << result_8 << std::endl;

  return 0;
}