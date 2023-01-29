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

  auto integral = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 2 - 2D - CPU ///////////////////////
auto test2(){
  auto function = [](const double& x, const double& y){
    return x*x + y*y;
  };

  auto integral = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, 10000, 0.0, 3.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 3 - 3D - CPU ///////////////////////
auto test3(){
  auto function = [](const double& x, const double& y, const double& z){
    return x*x + y*y + z*z;
  };

  auto integral = trapezoidal_integration_handler<0>(0, function, 0.0, 1.0, 100, 0.0, 3.0, 100, -2.0, 8.0, 100);
  std::cout << integral << std::endl;
}

// TEST 4 - 1D - GPU ///////////////////////
template<typename Sycl_Queue>
auto test4(Sycl_Queue Q){
  auto function = [](const double& x){
    return x*x;
  };

  auto integral = trapezoidal_integration_handler<1>(Q, function, 0.0, 1.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 5 - 2D - GPU ///////////////////////
template<typename Sycl_Queue>
auto test5(Sycl_Queue Q){
  auto function = [](const double& x, const double& y){
    return x*x + y*y;
  };

  auto integral = trapezoidal_integration_handler<2>(Q, function, 0.0, 1.0, 10000, 0.0, 3.0, 10000);
  std::cout << integral << std::endl;
}

// TEST 6 - 3D - GPU ///////////////////////
template<typename Sycl_Queue>
auto test6(Sycl_Queue Q){
  auto function = [](const double& x, const double& y, const double& z){
    return x*x + y*y + z*z;
  };

  auto integral = trapezoidal_integration_handler<3>(Q, function, 0.0, 1.0, 100, 0.0, 3.0, 100, -2.0, 8.0, 100);
  std::cout << integral << std::endl;
}

int main(){
  sycl::queue Q{sycl::gpu_selector_v};
  std::cout << "DEVICE: "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";
  test1();
  test2();
  test3();
  test4(Q);
  test5(Q);
  test6(Q);
  return 0;
}