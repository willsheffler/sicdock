/*cppimport
<%
cfg['include_dirs'] = ['../..', '../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../util/dilated_int.hpp', '../util/numeric.hpp',
'xform_hierarchy.hpp']

cfg['parallel'] = False
setup_pybind11(cfg)
%>
*/

#include "sicdock/sampling/xform_hierarchy.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <iostream>
#include "sicdock/util/pybind_types.hpp"

namespace py = pybind11;
using namespace py::literals;

namespace sicdock {
namespace sampling {

using namespace util;
using namespace Eigen;

using std::cout;
using std::endl;

template <int N, typename F, typename I>
py::tuple get_trans(CartHier<N, F, I> ch, int resl,
                    Ref<Matrix<I, Dynamic, 1>> idx) {
  std::vector<size_t> xshape{idx.size(), N};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> xout(xshape);
  bool* iptr = (bool*)iout.request().ptr;
  Matrix<F, N, 1>* tptr = (Matrix<F, N, 1>*)xout.request().ptr;
  size_t nout = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    iptr[i] = ch.get_value(resl, idx[i], tptr[nout]);
    if (iptr[i]) ++nout;
  }
  py::tuple out(2);
  out[0] = iout;
  out[1] = xout[py::slice(0, nout, 1)];
  return out;
}
template <typename Hier, typename F, typename I, typename X = M3<F>>
py::tuple get_ori(Hier const& h, int resl, Vx<I> idx) {
  std::vector<size_t> orishape{idx.size(), dim_of<X>(), dim_of<X>()};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> oriout(orishape);
  bool* iptr = (bool*)iout.request().ptr;
  X* oriptr = (X*)oriout.request().ptr;
  size_t nout = 0;
  for (size_t i = 0; i < idx.size(); ++i) {
    M3<F> rot;
    iptr[i] = h.get_value(resl, idx[i], rot);
    oriptr[nout] = X::Identity();
    set_rot(oriptr[nout], rot);
    if (iptr[i]) ++nout;
  }
  return py::make_tuple(iout, oriout[py::slice(0, nout, 1)]);
}

template <typename H, typename F, typename I>
py::tuple get_xforms(H xh, int resl, Vx<I> idx) {
  std::vector<size_t> xshape{idx.size(), 4, 4};
  py::array_t<bool> iout(idx.size());
  py::array_t<F> xout(xshape);
  bool* iptr = (bool*)iout.request().ptr;
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  size_t nout = 0;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < idx.size(); ++i) {
      iptr[i] = xh.get_value(resl, idx[i], xptr[nout]);
      if (iptr[i]) ++nout;
    }
  }
  return py::make_tuple(iout, xout[py::slice(0, nout, 1)]);
}

struct ScoreIndex {
  double score;
  uint64_t index;
};

template <typename H, typename F, typename I>
py::tuple expand_top_N_impl(H xh, int N, int resl,
                            std::vector<std::pair<double, I>> si) {
  N = std::min<int>(si.size(), N);
  std::vector<size_t> xshape{N * 64, 4, 4};
  py::array_t<F> xout(xshape);
  X3<F>* xptr = (X3<F>*)xout.request().ptr;
  py::array_t<I> iout(N * 64);
  I* iptr = (I*)iout.request().ptr;
  for (auto [s, i] : si)
    if (i >= xh.size(resl))
      throw std::range_error("expand_top_N_impl: index out of bounds");

  size_t nout = 0;
  {
    py::gil_scoped_release release;

    std::nth_element(si.begin(), si.begin() + N, si.end(),
                     std::greater<std::pair<double, I>>());
    for (size_t i = 0; i < N; ++i) {
      I parent = si[i].second;
      I beg = xh.child_of_begin(parent);
      I end = xh.child_of_end(parent);
      for (I idx = beg; idx < end; ++idx) {
        iptr[nout] = idx;
        bool valid = xh.get_value(resl + 1, iptr[nout], xptr[nout]);
        if (valid) ++nout;
      }
    }
  }

  return py::make_tuple(iout[py::slice(0, nout, 1)],
                        xout[py::slice(0, nout, 1)]);
}

template <typename H, typename F, typename I>
py::tuple expand_top_N_pairs(H xh, int N, int resl,
                             py::array_t<ScoreIndex> score_idx,
                             double null_val) {
  // sketchy... should be checking strides etc
  std::pair<double, I>* siptr = (std::pair<double, I>*)score_idx.request().ptr;
  std::vector<std::pair<double, I>> si;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < score_idx.shape()[0]; ++i)
      if (siptr[i].first != null_val) si.push_back(siptr[i]);
  }
  return expand_top_N_impl<H, F, I>(xh, N, resl, si);
}

template <typename H, typename F, typename I>
py::tuple expand_top_N_separate(H xh, int N, int resl, Vx<double> score,
                                Vx<I> index, double null_val) {
  std::vector<std::pair<double, I>> si;
  {
    py::gil_scoped_release release;
    for (size_t i = 0; i < score.size(); ++i)
      if (score[i] != null_val)
        si.push_back(std::make_pair(score[i], index[i]));
    std::pair<double, I>* siptr = &si[0];
  }
  return expand_top_N_impl<H, F, I>(xh, N, resl, si);
}

template <int N, typename F, typename I>
void bind_CartHier(auto m, std::string name) {
  using Fn = Matrix<F, N, 1>;
  using In = Matrix<I, N, 1>;
  py::class_<CartHier<N, F, I>>(m, name.c_str())
      .def(py::init<Fn, Fn, In>(), "lb"_a, "ub"_a, "bs"_a)
      .def("size", &CartHier<N, F, I>::size)
      .def("get_trans", &get_trans<N, F, I>)
      .def("sanity_check", &CartHier<N, F, I>::sanity_check)
      /**/;
}

template <typename F, typename I>
RotHier<F, I> RotHier_nside(F lb, F ub, I nside, V3<F> axis) {
  return RotHier<F, I>(lb, ub, nside, axis);
}
template <typename F, typename I>
void bind_RotHier(auto m, std::string kind) {
  py::class_<RotHier<F, I>>(m, ("RotHier_" + kind).c_str())
      .def(py::init<F, F, F, V3<F>>(), "lb"_a, "ub"_a, "resl"_a,
           "axis"_a = V3<F>(0, 0, 1))
      .def("size", &RotHier<F, I>::size)
      .def_readonly("lb", &RotHier<F, I>::rot_lb_)
      .def_readonly("ub", &RotHier<F, I>::rot_ub_)
      .def_readonly("rot_cell_width", &RotHier<F, I>::rot_cell_width_)
      .def_readonly("rot_ncell", &RotHier<F, I>::rot_ncell_)
      .def_readonly("axis", &RotHier<F, I>::axis_)
      .def("get_ori", &get_ori<RotHier<F, I>, F, I, M3<F>>, "resl"_a, "idx"_a)
      .def("get_xforms", &get_ori<RotHier<F, I>, F, I, X3<F>>, "resl"_a,
           "idx"_a)
      /**/;
  m.def(("create_RotHier_nside_" + kind).c_str(), &RotHier_nside<F, I>, "lb"_a,
        "ub"_a, "ncell"_a, "axis"_a = V3<F>(0, 0, 1));
}

template <typename F, typename I>
OriHier<F, I> OriHier_nside(I nside) {
  return OriHier<F, I>(nside);
}
template <typename F, typename I>
void bind_OriHier(auto m, std::string kind) {
  py::class_<OriHier<F, I>>(m, ("OriHier_" + kind).c_str())
      .def(py::init<F>(), "ori_resl"_a)
      .def("size", &OriHier<F, I>::size)
      .def_readonly("ori_nside", &OriHier<F, I>::onside_)
      .def("get_ori", &get_ori<OriHier<F, I>, F, I>, "resl"_a, "idx"_a)
      /**/;
  m.def(("create_OriHier_nside_" + kind).c_str(), &OriHier_nside<F, I>,
        "nside"_a);
}

template <typename F, typename I>
XformHier<F, I> XformHier_nside(V3<F> lb, V3<F> ub, V3<I> ncart, I nside) {
  return XformHier<F, I>(lb, ub, ncart, nside);
}
template <typename F, typename I>
void bind_XformHier(auto m, std::string kind) {
  using THIS = XformHier<F, I>;
  py::class_<THIS>(m, ("XformHier_" + kind).c_str())
      .def(py::init<V3<F>, V3<F>, V3<I>, F>(), "lb"_a, "ub"_a, "bs"_a,
           "ori_resl"_a)
      .def("size", &THIS::size)
      .def_readonly("ori_nside", &THIS::onside_)
      .def_readonly("ori_resl", &THIS::ori_resl_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_bs", &THIS::cart_bs_)
      .def_readonly("cart_cell_width", &THIS::cart_cell_width_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("ori_ncell", &THIS::ori_ncell_)
      .def_readonly("ncell", &THIS::ncell_)
      .def("dim", [](THIS const& t) { return THIS::FULL_DIM; })
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)
      /**/;
  m.def(("create_XformHier_nside_" + kind).c_str(), &XformHier_nside<F, I>,
        "lb"_a, "ub"_a, "bs"_a, "nside"_a);
}

template <typename F, typename I>
OriCart1Hier<F, I> OriCart1Hier_nside(V1<F> lb, V1<F> ub, V1<I> ncart,
                                      I nside) {
  return OriCart1Hier<F, I>(lb, ub, ncart, nside);
}
template <typename F, typename I>
void bind_RotCart1Hier(auto m, std::string name) {
  using THIS = RotCart1Hier<F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<F, F, I, F, F, I, V3<F>>(), "cartlb"_a, "cartub"_a,
           "cartnc"_a, "rotlb"_a, "rotub"_a, "rotnc"_a,
           "axis"_a = V3<F>(0, 0, 1))
      .def("size", &THIS::size)
      .def_readonly("rot_ncell", &THIS::rot_ncell_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("cart_bs_", &THIS::cart_bs_)
      .def_readonly("cart_cell_width_", &THIS::cart_cell_width_)
      .def_readonly("cart_bs_pref_prod_", &THIS::cart_bs_pref_prod_)
      .def_readonly("rot_lb", &THIS::rot_lb_)
      .def_readonly("rot_ub", &THIS::rot_ub_)
      .def_readonly("rot_ncell", &THIS::rot_ncell_)
      .def_readonly("rot_cell_width", &THIS::rot_cell_width_)
      .def_readonly("axis", &THIS::axis_)
      .def_readonly("ncell", &THIS::ncell_)
      .def("dim", [](THIS const& t) { return THIS::FULL_DIM; })
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)
      /**/;
}

template <typename F, typename I>
void bind_OriCart1Hier(auto m, std::string name) {
  using THIS = OriCart1Hier<F, I>;
  py::class_<THIS>(m, name.c_str())
      .def(py::init<V1<F>, V1<F>, V1<I>, F>(), "lb"_a, "ub"_a, "bs"_a,
           "ori_resl"_a)
      .def("size", &THIS::size)
      .def_readonly("ori_nside", &THIS::onside_)
      .def_readonly("ori_resl", &THIS::ori_resl_)
      .def_readonly("cart_lb", &THIS::cart_lb_)
      .def_readonly("cart_ub", &THIS::cart_ub_)
      .def_readonly("cart_bs", &THIS::cart_bs_)
      .def_readonly("cart_cell_width", &THIS::cart_cell_width_)
      .def_readonly("cart_ncell", &THIS::cart_ncell_)
      .def_readonly("ori_ncell", &THIS::ori_ncell_)
      .def_readonly("ncell", &THIS::ncell_)
      .def("dim", [](THIS const& t) { return THIS::FULL_DIM; })
      .def("sanity_check", &THIS::sanity_check)
      .def("cell_index_of", py::vectorize(&THIS::cell_index_of))
      .def("hier_index_of", py::vectorize(&THIS::hier_index_of))
      .def("parent_of", py::vectorize(&THIS::parent_of))
      .def("child_of_begin", py::vectorize(&THIS::child_of_begin))
      .def("child_of_end", py::vectorize(&THIS::child_of_end))
      .def("get_xforms", &get_xforms<THIS, F, I>, "iresl"_a, "idx"_a)
      .def("expand_top_N", expand_top_N_pairs<THIS, F, I>, "nkeep"_a, "resl"_a,
           "score_idx"_a, "null_val"_a = 0)
      .def("expand_top_N", expand_top_N_separate<THIS, F, I>, "nkeep"_a,
           "resl"_a, "score"_a, "index"_a, "null_val"_a = 0)
      /**/;
}

template <typename I, int DIM>
Matrix<I, Dynamic, DIM + 1, RowMajor> zorder2coeffs(
    Ref<Matrix<I, Dynamic, 1>> idx, I resl) {
  Matrix<I, Dynamic, DIM + 1, RowMajor> out(idx.size(), DIM + 1);
  for (size_t i = 0; i < idx.size(); ++i) {
    out(i, 0) = idx[i] >> (DIM * resl);
    I hier_index = idx[i] & (((I)1 << (DIM * resl)) - 1);
    for (size_t j = 0; j < DIM; ++j) {
      out(i, j + 1) = util::undilate<DIM>(hier_index >> j);
    }
  }
  return out;
}
template <typename I, int DIM>
Matrix<I, Dynamic, 1> coeffs2zorder(
    Ref<Matrix<I, Dynamic, DIM + 1, RowMajor>> idx, I resl) {
  Matrix<I, Dynamic, 1> out(idx.rows());
  for (size_t i = 0; i < idx.rows(); ++i) {
    I cell_index = idx(i, 0);
    I index = 0;
    for (size_t j = 0; j < DIM; ++j)
      index |= util::dilate<DIM>(idx(i, j + 1)) << j;
    index = index | (cell_index << (DIM * resl));
    out[i] = index;
  }
  return out;
}

PYBIND11_MODULE(xform_hierarchy, m) {
  bind_CartHier<1, double, uint64_t>(m, "CartHier1D_f8");
  bind_CartHier<2, double, uint64_t>(m, "CartHier2D_f8");
  bind_CartHier<3, double, uint64_t>(m, "CartHier3D_f8");
  bind_CartHier<4, double, uint64_t>(m, "CartHier4D_f8");
  bind_CartHier<5, double, uint64_t>(m, "CartHier5D_f8");
  bind_CartHier<6, double, uint64_t>(m, "CartHier6D_f8");
  bind_CartHier<1, float, uint64_t>(m, "CartHier1D_f4");
  bind_CartHier<2, float, uint64_t>(m, "CartHier2D_f4");
  bind_CartHier<3, float, uint64_t>(m, "CartHier3D_f4");
  bind_CartHier<4, float, uint64_t>(m, "CartHier4D_f4");
  bind_CartHier<5, float, uint64_t>(m, "CartHier5D_f4");
  bind_CartHier<6, float, uint64_t>(m, "CartHier6D_f4");

  bind_OriHier<double, uint64_t>(m, "f8");
  bind_OriHier<float, uint64_t>(m, "f4");

  bind_RotHier<double, uint64_t>(m, "f8");
  bind_RotHier<float, uint64_t>(m, "f4");

  bind_RotCart1Hier<double, uint64_t>(m, "RotCart1Hier_f8");
  bind_RotCart1Hier<float, uint64_t>(m, "RotCart1Hier_f4");

  bind_XformHier<double, uint64_t>(m, "f8");
  bind_XformHier<float, uint64_t>(m, "f4");

  m.def("zorder3coeffs", &zorder2coeffs<uint64_t, 3>);
  m.def("coeffs3zorder", &coeffs2zorder<uint64_t, 3>);
  m.def("zorder6coeffs", &zorder2coeffs<uint64_t, 6>);
  m.def("coeffs6zorder", &coeffs2zorder<uint64_t, 6>);

  bind_OriCart1Hier<float, uint64_t>(m, "OriCart1Hier_f4");
  m.def("create_OriCart1Hier_4f_nside", &OriCart1Hier_nside<float, uint64_t>,
        "lb"_a, "ub"_a, "bs"_a, "nside"_a);

  PYBIND11_NUMPY_DTYPE(ScoreIndex, score, index);
}

}  // namespace sampling
}  // namespace sicdock