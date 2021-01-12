#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128 

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>
#include <iostream>

#include <chrono>
#include <random>

using namespace std::chrono;

#define SEED 123123
std::default_random_engine generator(SEED);

const int n_bidders = 20000;
const int n_items   = 20000;
const int max_cost  = 10;

void uniform_random_problem(int* cost_matrix) {
  std::uniform_int_distribution<int> distribution(0, max_cost);

  for (long i = 0; i < n_items * n_bidders; i++) {
    cost_matrix[i] = distribution(generator);
  }
}

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start, 0); }
    void Stop()  { cudaEventRecord(stop, 0);  }

    float ElapsedMillis() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// C-style indexing
int ci(int row, int column, int nColumns) {
  return row*nColumns+column;
}

// Convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns
  
  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i) {
    return i / C;
  }
};

typedef thrust::tuple<int,int> OpType;

struct Op : public thrust::binary_function<OpType,OpType,OpType> {
    __host__ __device__
        OpType operator()(const OpType& a, const OpType& b) const {
          if (thrust::get<1>(a) > thrust::get<1>(b)){
            return a;
          } else {
            return b;
          }
        }
};

thrust::device_vector<OpType> compute_mins(thrust::device_vector<int> A, int nRows, int nColumns) {
  // allocate storage for row Ops and indices
  thrust::device_vector<OpType> results(nRows);
  thrust::device_vector<int> indices(nRows);          
      
  // compute row Ops by finding Op values with equal row indices
  thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(nColumns)) + (nRows*nColumns),
     thrust::make_zip_iterator(thrust::make_tuple(thrust::counting_iterator<int>(0),A.begin())),
     indices.begin(),
     results.begin(),
     thrust::equal_to<int>(),
     Op());
    
  return results;
}

int main(int argc, char *argv[]) {
  int* A = (int*)malloc(n_items * n_bidders * sizeof(int));
  uniform_random_problem(A);
  std::cout << "uniform_random_problem: done" << std::endl;
  
  // --
  // CPU
  
  auto t_start = high_resolution_clock::now();
  
  int* acc = (int*)malloc(n_bidders * sizeof(int));
  for(int i = 0; i < n_bidders; i++) {
    int min_val = -1;
    for(int j = 0; j < n_items; j++) {
      int val = A[i * n_items + j];
      if(val > min_val) {
        min_val = val;
      }
    }    
    acc[i] = min_val;
  }
  
  auto t_stop = high_resolution_clock::now();
  auto cpu_elapsed = duration_cast<microseconds>(t_stop - t_start).count();
  
  // Inspect
  int cpu_acc = 0;
  for(int i = 0 ; i < n_bidders ; i++) {
    cpu_acc += acc[i];
  }
  
  // --
  // GPU
    
  thrust::host_vector<int> B(A, A + (n_items * n_bidders));
  thrust::device_vector<int> C = B;
  
  // Warmup
  thrust::device_vector<OpType> dummy_results = compute_mins(C, n_bidders, n_items);
  
  GpuTimer timer = GpuTimer();
  timer.Start();
  
  thrust::device_vector<OpType> results = compute_mins(C, n_bidders, n_items);
  
  timer.Stop();
  float gpu_elapsed = timer.ElapsedMillis();

  // Inspect
  thrust::host_vector<OpType> h_results = results;
  int gpu_acc = 0;
  for(int i = 0; i < n_bidders; i++) {
    // std::cout 
    //   << thrust::get<0>(h_results[i]) / 10 << " " 
    //   << thrust::get<0>(h_results[i]) % 10 << " " 
    //   << thrust::get<1>(h_results[i]) << std::endl;
    gpu_acc += thrust::get<1>(h_results[i]);
  }
    
  printf("gpu_elapsed=%f | cpu_elapsed=%f | gpu_acc=%d | cpu_acc=%d \n", gpu_elapsed, (float)cpu_elapsed / 1000, gpu_acc, cpu_acc);
  
  return 0;
  
  // GPU is ~ 10x faster for sufficiently large matrices.  However, doesn't seem to make a ton of different until > 5K or so.
  // Still ... could be useful
}
