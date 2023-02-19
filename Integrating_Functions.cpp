// Standard C++ include items
#include <iostream>

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
  const double B = 2.3;
  const double u = 3.5;

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
  std::cout << "Result 1: " << result_1 << std::endl;

  return 0;
}