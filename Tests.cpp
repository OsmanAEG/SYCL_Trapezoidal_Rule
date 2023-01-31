// Standard C++ include items
#include <iostream>

// Including the SYCL library
#include <CL/sycl.hpp>

// Including local header files
#include "Trapezoidal_Integration_Handler.h"

auto cpu_tests(){
  constexpr auto tolerance = 1.0e-3;

  auto check = [&](double result, double answer){
    assert(fabs((result-answer)/answer) < tolerance);
  };

  auto test_1D = [&](){
    auto function = [](const double& x){
      return x*x;
    };

    const int N = 1000;
    const auto answer = 1.0/3.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, N);
    check(result, answer);
  };

  auto test_2D = [&](){
    auto function = [](const double& x, const double& y){
      return x*x + y*y;
    };

    const int N = 1000;
    const auto answer = 10.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, N, 0.0, 3.0, N);
    check(result, answer);
  };

  auto test_3D = [&](){
    auto function = [](const double& x, const double& y, const double& z){
      return x*x + y*y + z*z;
    };

    const int N = 1000;
    const auto answer = 620.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, N, 0.0, 3.0, N, -2.0, 8.0, N);
    check(result, answer);
  };

  std::cout << "The CPU Tests Passed!" << std::endl;
}

template<typename Sycl_Queue>
auto gpu_tests(Sycl_Queue Q){
  constexpr auto tolerance = 1.0e-3;

  auto check = [&](double result, double answer){
    assert(fabs((result-answer)/answer) < tolerance);
  };

  auto test_1D = [&](){
    auto function = [](const double& x){
      return x*x;
    };

    const int N = 1000;
    const auto answer = 1.0/3.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<1>(Q, function, 0.0, 1.0, N);
    check(result, answer);
  };

  auto test_2D = [&](){
    auto function = [](const double& x, const double& y){
      return x*x + y*y;
    };

    const int N = 1000;
    const auto answer = 10.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<2>(Q, function, 0.0, 1.0, N, 0.0, 3.0, N);
    check(result, answer);
  };

  auto test_3D = [&](){
    auto function = [](const double& x, const double& y, const double& z){
      return x*x + y*y + z*z;
    };

    const int N = 1000;
    const auto answer = 620.0;
    const auto tolerance = 1.0e-3;
    const auto result = trapezoidal_integration_handler<3>(Q, function, 0.0, 1.0, N, 0.0, 3.0, N, -2.0, 8.0, N);
    check(result, answer);
  };

  std::cout << "The GPU Tests Passed!" << std::endl;
}

int main(){
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  cpu_tests();
  gpu_tests(Q);

  return 0;
}