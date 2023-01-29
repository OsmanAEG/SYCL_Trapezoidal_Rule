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
template<typename Function_type, typename Scalar_type, typename... Arguments>
auto trapezoidal_integration(const Function_type& function,
                             const Scalar_type a,
                             const Scalar_type b,
                             const int& M,
                             const Arguments&... arguments){

  if constexpr(sizeof...(arguments) == 0){
    return trapezoidal_integration_1D(function, a, b, M);
  }
  else{
    const auto integrand = [&](const auto& point){
      const auto adjusted_function = [&](const auto&... more_arguments){
        return function(point, more_arguments...);
      };
      return trapezoidal_integration(adjusted_function, arguments...);
    };
    return trapezoidal_integration_1D(integrand, a, b, M);
  }
}

// Trapezoidal Integration 1 Dimensional in Sycl
template<typename Sycl_Queue, typename Function_type, typename Scalar_type, typename... Arguments>
auto sycl_trapezoidal_integration_1D(Sycl_Queue Q,
                                     const Function_type& function,
                                     const Scalar_type a_x,
                                     const Scalar_type b_x,
                                     const size_t& M,
                                     const Arguments&... arguments){

  Scalar_type* result_collection = sycl::malloc_device<Scalar_type>(M, Q);
  const auto step = (b_x-a_x)/M;
  const auto base_result = 0.0;

  Q.submit([&](sycl::handler&h){
    h.parallel_for(sycl::range{M}, [=](sycl::id<1> idx){
      const auto i = idx[0];
      const auto a_x_i = a_x + i*step;
      const auto b_x_i = a_x_i + step;
      result_collection[idx] = trapezoidal_integration(function, a_x_i, b_x_i, 1,
                                                                 arguments...);
    });
  }).wait();

  Scalar_type result_host = 0.0;
  sycl::buffer<Scalar_type> result_buf{&result_host, 1};

  Q.submit([&](sycl::handler& h){
    sycl::accessor result_acc{result_buf, h, sycl::read_write};
    auto result_reduction = sycl::reduction(result_buf, h, sycl::plus<Scalar_type>());
    h.parallel_for(sycl::range<1>(M), result_reduction, [=](sycl::id<1> idx, auto& result){
      result += result_collection[idx];
    });
  }).wait();

  sycl::host_accessor get_result_host{result_buf, sycl::read_only};
  return get_result_host[0];
}

// Trapezoidal Integration 2 Dimensional in Sycl
template<typename Sycl_Queue, typename Function_type, typename Scalar_type, typename... Arguments>
auto sycl_trapezoidal_integration_2D(Sycl_Queue Q,
                                     const Function_type& function,
                                     const Scalar_type a_x,
                                     const Scalar_type b_x,
                                     const size_t& M,
                                     const Scalar_type a_y,
                                     const Scalar_type b_y,
                                     const size_t& N,
                                     const Arguments&... arguments){

  Scalar_type* result_collection = sycl::malloc_device<Scalar_type>(M*N, Q);
  const auto step_x = (b_x-a_x)/M;
  const auto step_y = (b_y-a_y)/N;
  const auto base_result = 0.0;

  Q.submit([&](sycl::handler&h){
    h.parallel_for(sycl::range{M, N}, [=](sycl::id<2> idx){
      const auto i = idx[0];
      const auto j = idx[1];
      const auto a_x_i = a_x + i*step_x;
      const auto b_x_i = a_x_i + step_x;
      const auto a_y_j = a_y + j*step_y;
      const auto b_y_j = a_y_j + step_y;
      result_collection[i + M*j] = trapezoidal_integration(function, a_x_i, b_x_i, 1,
                                                                     a_y_j, b_y_j, 1,
                                                                     arguments...);
    });
  }).wait();

  Scalar_type result_host = 0.0;
  sycl::buffer<Scalar_type> result_buf{&result_host, 1};

  Q.submit([&](sycl::handler& h){
    sycl::accessor result_acc{result_buf, h, sycl::read_write};
    auto result_reduction = sycl::reduction(result_buf, h, sycl::plus<Scalar_type>());
    h.parallel_for(sycl::range<1>(M*N), result_reduction, [=](sycl::id<1> idx, auto& result){
      result += result_collection[idx];
    });
  }).wait();

  sycl::host_accessor get_result_host{result_buf, sycl::read_only};
  return get_result_host[0];
}

// Trapezoidal Integration 2 Dimensional in Sycl
template<typename Sycl_Queue, typename Function_type, typename Scalar_type, typename... Arguments>
auto sycl_trapezoidal_integration_3D(Sycl_Queue Q,
                                     const Function_type& function,
                                     const Scalar_type a_x,
                                     const Scalar_type b_x,
                                     const size_t& M,
                                     const Scalar_type a_y,
                                     const Scalar_type b_y,
                                     const size_t& N,
                                     const Scalar_type a_z,
                                     const Scalar_type b_z,
                                     const size_t& P,
                                     const Arguments&... arguments){

  Scalar_type* result_collection = sycl::malloc_device<Scalar_type>(M*N*P, Q);
  const auto step_x = (b_x-a_x)/M;
  const auto step_y = (b_y-a_y)/N;
  const auto step_z = (b_z-a_z)/P;
  const auto base_result = 0.0;

  Q.submit([&](sycl::handler&h){
    h.parallel_for(sycl::range{M, N, P}, [=](sycl::id<3> idx){
      const auto i = idx[0];
      const auto j = idx[1];
      const auto k = idx[2];
      const auto a_x_i = a_x + i*step_x;
      const auto b_x_i = a_x_i + step_x;
      const auto a_y_j = a_y + j*step_y;
      const auto b_y_j = a_y_j + step_y;
      const auto a_z_k = a_z + k*step_z;
      const auto b_z_k = a_z_k + step_z;
      result_collection[i + M*j + M*N*k] = trapezoidal_integration(function,
                                                                   a_x_i, b_x_i, 1,
                                                                   a_y_j, b_y_j, 1,
                                                                   a_z_k, b_z_k, 1,
                                                                   arguments...);
    });
  }).wait();

  Scalar_type result_host = 0.0;
  sycl::buffer<Scalar_type> result_buf{&result_host, 1};

  Q.submit([&](sycl::handler& h){
    sycl::accessor result_acc{result_buf, h, sycl::read_write};
    auto result_reduction = sycl::reduction(result_buf, h, sycl::plus<Scalar_type>());
    h.parallel_for(sycl::range<1>(M*N*P), result_reduction, [=](sycl::id<1> idx, auto& result){
      result += result_collection[idx];
    });
  }).wait();

  sycl::host_accessor get_result_host{result_buf, sycl::read_only};
  return get_result_host[0];
}



