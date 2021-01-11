// py_auction.hpp

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "auction.hpp"

namespace py = pybind11;

template <typename Int, typename Real>
long long py_auction(
  torch::Tensor cost_matrix,
  Int n_bidders,
  Int n_items,
  Real eps,
  torch::Tensor bidder2item
) {
  return auction(
    cost_matrix.data_ptr<Real>(),
    n_bidders,
    n_items,
    eps,
    bidder2item.data_ptr<Int>()
  );
}

PYBIND11_MODULE(auction, m) {
  m.def("auction", py_auction<int, float>);
}
