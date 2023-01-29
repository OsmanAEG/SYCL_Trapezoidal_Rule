// Standard C++ include items
#include <iostream>

// Including the SYCL library
#include <CL/sycl.hpp>

// Including local header files
#include "Trapezoidal_Integration_Handler.h"

// TEST 1 - 1D - CPU ///////////////////////
auto test1(){
  auto function = [](const double& x){
    return x*x;
  };

  auto integral = trapezoidal_integration_1D(function, 0.0, 1.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 2 - 1D - CPU ///////////////////////
auto test2(){
  auto function = [](const double& x, const double& y){
    return x*x + y*y;
  };

  auto integral = trapezoidal_integration(function, 0.0, 1.0, 10000, 0.0, 3.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 3 - 1D - GPU ///////////////////////
template<typename Sycl_Queue>
auto test3(Sycl_Queue Q){
  auto function = [](const double& x){
    return x*x;
  };

  auto integral = sycl_trapezoidal_integration_1D(Q, function, 0.0, 1.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 4 - 2D - GPU ///////////////////////
template<typename Sycl_Queue>
auto test4(Sycl_Queue Q){
  auto function = [](const double& x, const double& y){
    return x*x + y*y;
  };

  auto integral = sycl_trapezoidal_integration_2D(Q, function, 0.0, 1.0, 10000, 0.0, 3.0, 10000);
  std::cout << integral << std::endl;
}

int main(){
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  test1();
  test2();
  test3(Q);
  test4(Q);
  return 0;
}