// Standard C++ include items
#include <iostream>
#include <array>
#include <cmath>
#include <utility>

// Including the SYCL library
#include <CL/sycl.hpp>

// Trapezoidal Integration 1 Dimensional
template<typename Function_type, typename Scalar_type>
auto trapezoidal_integration_1D(const Function_type& function,
                                const Scalar_type a,
                                const Scalar_type b,
                                const int& M){

  const auto step = (b-a)/M;
  auto result = function(a) + function(b);

  for(int i = 1; i < M; ++i){
    const auto point = a + i*step;
    result += 2.0*function(point);
  }

  return result*step/2.0;
}

// Trapezoidal Integration Multi-Dimensional
template<typename Function_type, typename Scalar_type, typename... Args>
auto trapezoidal_integration(const Function_type& function,
                             const Scalar_type a,
                             const Scalar_type b,
                             const int& M,
                             const Args&... args){

  if constexpr(sizeof...(args) == 0){
    return trapezoidal_integration_1D(function, a, b, M);
  }
  else{
    const auto integrand = [&](const auto& point){
      const auto adjusted_function = [&](const auto&... more_args){
        return function(point, more_args...);
      };
      return trapezoidal_integration(adjusted_function, args...);
    };
    return trapezoidal_integration_1D(integrand, a, b, M);
  }
}

