/*/*cppimport
<%
cfg['include_dirs'] = ['../..','../extern']
cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
cfg['dependencies'] = ['../geom/primitive.hpp','../util/assertions.hpp',
'../util/global_rng.hpp', 'bvh.hpp', 'bvh_algo.hpp', '../util/numeric.hpp',
'../util/pybind_types.hpp']

cfg['parallel'] = False

setup_pybind11(cfg)
%>
*/
/** \file */

#include "rpxdock/bvh/bvh.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "iostream"
#include "rpxdock/util/Timer.hpp"
#include "rpxdock/util/assertions.hpp"
#include "rpxdock/util/global_rng.hpp"
#include "rpxdock/util/numeric.hpp"
#include "rpxdock/util/pybind_types.hpp"
#include "rpxdock/util/types.hpp"

using namespace pybind11::literals;
using namespace Eigen;
using namespace rpxdock;
using namespace util;
using namespace geom;

namespace py = pybind11;

namespace Eigen {

template <class F>
struct PtIdx {
  PtIdx() : pos(0), idx(0) {}
  PtIdx(V3<F> v, int i = 0) : pos(v), idx(i) {}
  V3<F> pos;
  int idx;
};
template <typename F>
auto bounding_vol(V3<F> v) {
  return Sphere<F>(v);
}
template <typename F>
auto bounding_vol(PtIdx<F> v) {
  auto s = Sphere<F>(v.pos);
  s.lb = s.ub = v.idx;
  return s;
}
}  // namespace Eigen

template <typename F>
using BVH = rpxdock::bvh::SphereBVH<F, PtIdx<F>>;
using BVHf = BVH<float>;
using BVHd = BVH<double>;

namespace rpxdock {
namespace bvh {

using namespace rpxdock::util;

template <typename F>
int bvh_max_id(BVH<F> const &bvh) {
  int x = 0;
  for (auto o : bvh.objs) x = std::max(x, o.idx);
  return x;
}

template <typename F>
int bvh_min_lb(BVH<F> const &bvh) {
  int x = 0;
  for (auto v : bvh.vols) x = std::min(x, v.lb);
  return x;
}
template <typename F>
int bvh_max_ub(BVH<F> const &bvh) {
  int x = 0;
  for (auto v : bvh.vols) x = std::max(x, v.ub);
  return x;
}
template <typename F>
Vx<int> bvh_obj_ids(BVH<F> const &bvh) {
  Vx<int> x(bvh.objs.size());
  for (int i = 0; i < x.size(); ++i) x[i] = bvh.objs[i].idx;
  return x;
}
template <typename F>
Vx<int> bvh_vol_lbs(BVH<F> const &bvh) {
  Vx<int> x(bvh.vols.size());
  for (int i = 0; i < x.size(); ++i) x[i] = bvh.vols[i].lb;
  return x;
}
template <typename F>
Vx<int> bvh_vol_ubs(BVH<F> const &bvh) {
  Vx<int> x(bvh.vols.size());
  for (int i = 0; i < x.size(); ++i) x[i] = bvh.vols[i].ub;
  return x;
}

template <typename F>
std::unique_ptr<BVH<F>> bvh_create(Mx<F> coords, Vx<bool> which, Vx<int> ids) {
  if (coords.cols() != 3)
    throw std::runtime_error("argument 'coords' shape must be (N, 3)");
  if (which.size() > 0 && which.size() != coords.rows())
    throw std::runtime_error(
        "argument 'which' shape must be (N,) matching coord shape");
  if (ids.size() > 0 && ids.size() != coords.rows())
    throw std::runtime_error(
        "argument 'idx' shape must be (N,) matching coord shape");

  py::gil_scoped_release release;

  typedef std::vector<PtIdx<F>, aligned_allocator<PtIdx<F>>> Objs;
  Objs holder;

  for (int i = 0; i < coords.rows(); ++i) {
    // std::cout << "mask " << i << " " << ptrw[i] << std::endl;
    if (which.size() > 0 && !which[i]) continue;
    int id = ids.size() == 0 ? i : ids[i];
    holder.push_back(PtIdx<F>(coords.row(i), id));
  }
  auto bvh = std::make_unique<BVH<F>>(holder.begin(), holder.end());
  if (ids.size())
    for (auto &v : bvh->vols) {
      v.lb = ids[v.lb];
      v.ub = ids[v.ub];
    }
  return bvh;
}

template <typename F>
struct BVHMinDistOne {
  using Scalar = F;
  int idx = -1;
  F minval = 9e9;
  V3<F> pt;
  BVHMinDistOne(V3<F> p) : pt(p) {}
  F minimumOnVolume(Sphere<F> r) { return r.signdis(pt); }
  F minimumOnObject(PtIdx<F> a) {
    F v = (a.pos - pt).norm();
    if (v < minval) {
      minval = v;
      idx = a.idx;
    }
    return v;
  }
};
template <typename F>
py::tuple bvh_min_dist_one(BVH<F> &bvh, V3<F> pt) {
  int idx;
  F result;
  {
    py::gil_scoped_release release;
    BVHMinDistOne<F> minimizer(pt);
    result = rpxdock::bvh::BVMinimize(bvh, minimizer);
    idx = minimizer.idx;
  }
  return py::make_tuple(result, idx);
}

template <typename F>
struct BVHMinDistQuery {
  using Scalar = F;
  using Xform = X3<F>;
  int idx1 = -1, idx2 = -1;
  Xform bXa = Xform::Identity();
  F minval = 9e9;
  BVHMinDistQuery(Xform x = Xform::Identity()) : bXa(x) {}
  F minimumOnVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    return vol1.signdis(bXa * vol2);
  }
  F minimumOnVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    return vol1.signdis(bXa * obj2.pos);
  }
  F minimumOnObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    return (bXa * vol2).signdis(obj1.pos);
  }
  F minimumOnObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    F v = (obj1.pos - bXa * obj2.pos).norm();
    if (v < minval) {
      // std::cout << v << obj1.pos.transpose() << " " << obj2.pos.transpose()
      // << std::endl;
      minval = v;
      idx1 = obj1.idx;
      idx2 = obj2.idx;
    }
    return v;
  }
};

template <typename F>
py::tuple bvh_min_dist_fixed(BVH<F> &bvh1, BVH<F> &bvh2) {
  BVHMinDistQuery<F> minimizer;
  auto result = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
  return py::make_tuple(result, minimizer.idx1, minimizer.idx2);
}
template <typename F>
py::tuple bvh_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  int idx1, idx2;
  F result;
  {
    py::gil_scoped_release release;
    X3<F> x1(pos1), x2(pos2);
    BVHMinDistQuery<F> minimizer(x1.inverse() * x2);
    result = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
    idx1 = minimizer.idx1;
    idx2 = minimizer.idx2;
  }
  return py::make_tuple(result, idx1, idx2);
}
template <typename F>
py::tuple bvh_min_dist_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                           py::array_t<F> pos2) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size())
    throw std::runtime_error("pos1 and pos2 must have same length");
  auto mindis = std::make_unique<Vx<F>>();
  auto idx1 = std::make_unique<Vx<int>>();
  auto idx2 = std::make_unique<Vx<int>>();
  {
    py::gil_scoped_release release;
    mindis->resize(x1.size());
    idx1->resize(x1.size());
    idx2->resize(x1.size());
    for (size_t i = 0; i < x1.size(); ++i) {
      BVHMinDistQuery<F> minimizer(x1[i].inverse() * x2[i]);
      (*mindis)[i] = rpxdock::bvh::BVMinimize(bvh1, bvh2, minimizer);
      (*idx1)[i] = minimizer.idx1;
      (*idx2)[i] = minimizer.idx2;
    }
  }
  return py::make_tuple(*mindis, *idx1, *idx2);
}
template <typename F>
F naive_min_dist_fixed(BVH<F> &bvh1, BVH<F> &bvh2) {
  F mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<F>(mind2, (o1.pos - o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}
template <typename F>
F naive_min_dist(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F mind2 = 9e9;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      mind2 = std::min<F>(mind2, (o1.pos - pos * o2.pos).squaredNorm());
    }
  }
  return std::sqrt(mind2);
}

template <typename F>
struct BVHIsectQuery {
  using Scalar = F;
  using Xform = X3<F>;
  BVHIsectQuery(F r, Xform x = Xform::Identity())
      : rad(r), rad2(r * r), bXa(x) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    return vol1.signdis(bXa * vol2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    return vol1.signdis(bXa * obj2.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    return (bXa * vol2).signdis(obj1.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < rad2;
    // bool isect = (obj1.pos - bXa * obj2.pos).norm() < rad;
    result |= isect;
    return isect;
  }
  F rad = 0, rad2 = 0;
  bool result = false;
  Xform bXa = Xform::Identity();
};
template <typename F>
bool bvh_isect_fixed(BVH<F> &bvh1, BVH<F> &bvh2, F thresh) {
  BVHIsectQuery<F> query(thresh);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
bool naive_isect_fixed(BVH<F> &bvh1, BVH<F> &bvh2, F thresh) {
  F dist2 = thresh * thresh;
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}
template <typename F>
bool bvh_isect(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2, F mindist) {
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  BVHIsectQuery<F> query(mindist, x1.inverse() * x2);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.result;
}
template <typename F>
Vx<bool> bvh_isect_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                       py::array_t<F> pos2, F mindist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  py::gil_scoped_release release;
  size_t n = std::max(x1.size(), x2.size());
  Vx<bool> out(n);
  for (size_t i = 0; i < n; ++i) {
    size_t i1 = x1.size() == 1 ? 0 : i;
    size_t i2 = x2.size() == 1 ? 0 : i;
    X3<F> xi1 = x1[i1];
    X3<F> x11inv = xi1.inverse();
    BVHIsectQuery<F> query(mindist, x11inv * x2[i2]);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    out[i] = query.result;
  }
  return out;
}
template <typename F>
bool naive_isect(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                 F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;

  // bounding sphere check
  auto vol1 = bvh1.getVolume(bvh1.getRootIndex());
  auto vol2 = bvh2.getVolume(bvh2.getRootIndex());
  vol1.cen = x1 * vol1.cen;
  vol2.cen = x2 * vol2.cen;
  if (!vol1.contact(vol2, mindist)) return false;

  // all pairs
  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) return true;
    }
  }
  return false;
}

/////////////////////////////////////////////////////////

template <typename F>
struct BVHIsectFixedRangeQuery {
  using Scalar = F;
  using Xform = X3<F>;
  BVHIsectFixedRangeQuery(F r, Xform x, int _lb1, int _ub1, int _lb2, int _ub2)
      : rad(r),
        rad2(r * r),
        bXa(x),
        lb1(_lb1),
        ub1(_ub1),
        lb2(_lb2),
        ub2(_ub2) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    // if (vol1.ub < lb1 || vol1.lb > ub1 || vol2.ub < lb2 || vol2.lb > ub2)
    // return false;
    return vol1.signdis(bXa * vol2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    // if (vol1.ub < lb1 || vol1.lb > ub1 || obj2.idx < lb2 || obj2.idx > ub2)
    // return false;
    return vol1.signdis(bXa * obj2.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    // if (obj1.idx < lb1 || obj1.idx > ub1 || vol2.ub < lb2 || vol2.lb > ub2)
    // return false;
    return (bXa * vol2).signdis(obj1.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    if (obj1.idx < lb1 || obj1.idx > ub1 || obj2.idx < lb2 || obj2.idx > ub2) {
      return false;
    }
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < rad2;
    if (isect) {
      clashidA = obj1.idx;
      clashidB = obj2.idx;
      //   std::cout << lb1 << " " << obj1.idx << " " << ub1 << " " << lb2 << "
      //   "
      //             << obj2.idx << " " << ub2 << std::endl;
    }
    result |= isect;
    return isect;
  }
  F rad = 0, rad2 = 0;
  bool result = false;
  Xform bXa = Xform::Identity();
  int lb1, ub1, lb2, ub2;
  int clashidA = -1, clashidB = -1;
};
template <typename F>
py::tuple bvh_isect_fixed_range_vec(BVH<F> &bvh1, BVH<F> &bvh2,
                                    py::array_t<F> pos1, py::array_t<F> pos2,
                                    F mindist, Vx<int> lb1, Vx<int> ub1,
                                    Vx<int> lb2, Vx<int> ub2) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  size_t n = std::max(x1.size(), x2.size());
  Vx<bool> out(n);
  Mx<int> clashid(n, 2);
  {
    py::gil_scoped_release release;

    if (lb1.size() != 1 && lb1.size() != n)
      throw std::runtime_error("lb1 size best be broadcastable with pos1/2");
    if (ub1.size() != 1 && ub1.size() != n)
      throw std::runtime_error("ub1 size best be broadcastable with pos1/2");
    if (lb2.size() != 1 && lb2.size() != n)
      throw std::runtime_error("lb2 size best be broadcastable with pos1/2");
    if (ub2.size() != 1 && ub2.size() != n)
      throw std::runtime_error("ub2 size best be broadcastable with pos1/2");

    // std::cout << lb1[0] << " " << ub1[0] << " " << lb2[0] << " " << ub2[0]
    // << std::endl;

    for (size_t i = 0; i < n; ++i) {
      size_t i1 = x1.size() == 1 ? 0 : i;
      size_t i2 = x2.size() == 1 ? 0 : i;
      int ilb1 = lb1.size() == 1 ? lb1[0] : lb1[i];
      int iub1 = ub1.size() == 1 ? ub1[0] : ub1[i];
      int ilb2 = lb2.size() == 1 ? lb2[0] : lb2[i];
      int iub2 = ub2.size() == 1 ? ub2[0] : ub2[i];
      BVHIsectFixedRangeQuery<F> query(mindist, x1[i1].inverse() * x2[i2], ilb1,
                                       iub1, ilb2, iub2);
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      out[i] = query.result;
      clashid(i, 0) = query.clashidA;
      clashid(i, 1) = query.clashidB;
    }
  }
  return py::make_tuple(out, clashid);
}

///////////////////////////////////////////////////
/* 
variations of query (you could be operating on an atom or a BV): 
At the highest level of bounding volume, at the lowest resolution
Next, check children, which checks for "touching points" at higher resolution 

*/
template <typename F>
struct BVHIsectRange {
  using Scalar = F;
  using Xform = X3<F>;
  F rad = 0, rad2 = 0;
  int lb = 0, ub, mid, nasym1;
  int minrange = 0, max_lb = -1, min_ub = 0;
  Xform bXa = Xform::Identity();
  BVHIsectRange(F r, Xform x, int _ub, int _mid = -1, int maxtrim = -1,
                int maxtrim_lb = -1, int maxtrim_ub = -1, int _nasym1 = -1)
      : rad(r), rad2(r * r), bXa(x), mid(_mid), ub(_ub), nasym1(_nasym1) {
    if (nasym1 < 0) nasym1 = ub + 1;
    ub = nasym1 - 1;
    max_lb = ub;
    if (mid < 0) mid = ub / 2;

    if (maxtrim_lb >= 0 && maxtrim_ub >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_lb + maxtrim_ub : maxtrim;
      max_lb = maxtrim_lb;
      min_ub = ub - maxtrim_ub;
      mid = (max_lb + min_ub) / 2;
    } else if (maxtrim_ub >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_ub : maxtrim;
      min_ub = ub - maxtrim_ub;
      mid = 0;
    } else if (maxtrim_lb >= 0) {
      maxtrim = maxtrim == -1 ? maxtrim_lb : maxtrim;
      max_lb = maxtrim_lb;
      mid = ub;
    } else if (maxtrim < 0) {
      maxtrim = ub + 1;
    }
    minrange = ub - maxtrim;

    // std::cout << "IsectRange query " << maxtrim << " " << maxtrim_lb << " "
    // << maxtrim_ub << " " << lb << "-" << mid << "-" << ub
    // << " nasym1 " << nasym1 << std::endl;
  }
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    // returns true/false to continue checking children; if volumes have interesting higher resolution thing you want to look at
    //
    //  TODO: figure out why this check doesn't work if using
    //        non-unity IDs
    //
    //
    // std::cout << lb << " " << ub << " " << vol1.lb << " " << vol1.ub
    // << std::endl;
    // if (vol1.lb % nasym1 > ub || vol1.ub % nasym1 < lb) {
    // std::cout << "    range miss" << std::endl;
    // std::cout << lb << " " << ub << " " << vol1.lb << " " << vol1.ub
    // << std::endl;
    // return false;
    // }
    // bXa --> transform b to a and multiply by vol2 
    // signdis --> if spheres don't touch > 0; how much they are touching < 0
    // rad --> intersect dist (radius of spheres)
    return vol1.signdis(bXa * vol2) < rad;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    // check "light red" with "light blue"
    // PtIdx = point index; a sphere + index # 
    // B is side getting trimmed 
    // if lowest res# in BV > current ub or ub res# in BV < current lb; BV is done; don't do anything. basically out of range
    if (vol1.lb % nasym1 > ub || vol1.ub % nasym1 < lb) return false;
    return vol1.signdis(bXa * obj2.pos) < rad;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    // atom and BV; trim obj is obj1 (A side)
    // return False / keep going if already trimmed 
    if (obj1.idx % nasym1 > ub || obj1.idx % nasym1 < lb) return false;
    return (bXa * vol2).signdis(obj1.pos) < rad;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    // figure out if two things intersect; otherwise return False
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < rad2;
    // std::cout << "obj/obj " << obj1.idx << " " << obj2.idx << " isect=" <<
    // isect
    // << std::endl;
    // if intersecting, trim more
    if (isect) {
      if (obj1.idx % nasym1 < mid)
        lb = std::max(obj1.idx % nasym1 + 1, lb);
      else
        ub = std::min(obj1.idx % nasym1 - 1, ub);
      // if trimming too much, give up
      bool ok = (ub >= min_ub) && (lb <= max_lb) && ((ub - lb) >= minrange);
      if (!ok) {
        lb = -1;
        ub = -1;
        return true;
      }
    }
    return false;
  }
};
template <typename F>
/* 
Takes BVH objects, 
pos1, pos2: positions of bodies 
mindist: clash dist
maxtrim: maximum allowed trim (exit early if too much of protein is trimmed off)
maxtrim_lb; maxtrim_ub: Nterm and Cterm
nasym1: if bvh is sym, only trim asu (# asym resis); -1 = all 
query obj takes clash dist, and calls BVIntersect and returns lb and ub arrays of N and C term trim objs
*/
py::tuple isect_range(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                      py::array_t<F> pos2, F mindist, int maxtrim = -1,
                      int maxtrim_lb = -1, int maxtrim_ub = -1,
                      int nasym1 = -1) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  auto lb = std::make_unique<Vx<int>>();
  auto ub = std::make_unique<Vx<int>>();
  {
    py::gil_scoped_release release;
    if (nasym1 < 0) nasym1 = bvh_max_id(bvh1) + 1;
    size_t n = std::max(x1.size(), x2.size());
    lb->resize(n);
    ub->resize(n);
    BVHIsectRange<F> query(mindist, X3<F>::Identity(), bvh_max_id(bvh1), -1,
                           maxtrim, maxtrim_lb, maxtrim_ub, nasym1);

    for (size_t i = 0; i < n; ++i) {
      size_t i1 = x1.size() == 1 ? 0 : i;
      size_t i2 = x2.size() == 1 ? 0 : i;
      query.bXa = x1[i1].inverse() * x2[i2];
      query.lb = 0;
      query.ub = nasym1 < 0 ? bvh_max_id(bvh1) : nasym1 - 1;
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lb)[i] = query.lb;
      (*ub)[i] = query.ub;
    }
  }
  return py::make_tuple(*lb, *ub);
}
template <typename F>
py::tuple isect_range_single(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                             F mindist, int maxtrim = -1, int maxtrim_lb = -1,
                             int maxtrim_ub = -1, int nasym1 = -1) {
  int lb, ub;
  {
    py::gil_scoped_release release;
    if (nasym1 < 0) nasym1 = bvh_max_id(bvh1) + 1;
    X3<F> x1(pos1), x2(pos2);
    BVHIsectRange<F> query(mindist, x1.inverse() * x2, bvh_max_id(bvh1), -1,
                           maxtrim, maxtrim_lb, maxtrim_ub, nasym1);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    lb = query.lb;
    ub = query.ub;
  }
  return py::make_tuple(lb, ub);
}

template <typename F>
py::tuple naive_isect_range(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                            F mindist) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = mindist * mindist;
  int lb = 0, ub = bvh_max_id(bvh1), mid = ub / 2;

  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      // query.intersectObjectObject(o1, o2);

      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) {
        if (o1.idx < mid)
          lb = std::max(o1.idx + 1, lb);
        else
          ub = std::min(o1.idx - 1, ub);
      }
    }
  }
  // return py::make_tuple(query.lb, query.ub);
  return py::make_tuple(lb, ub);
}

template <typename F>
struct BVMinAxis {
  using Scalar = F;
  using Xform = X3<F>;
  Xform bXa = Xform::Identity();
  F minval = 9e9;
  F rad;
  V3<F> dirn;
  BVMinAxis(V3<F> d, Xform x, F r) : dirn(d), bXa(x), rad(r) {
    assert(r > 0);
    if (dirn.norm() < 0.0001)
      throw std::runtime_error("Slide direction must not be 0");
    dirn.normalize();
  }
  F minimumOnVolumeVolume(Sphere<F> r1, Sphere<F> r2) {
    return get_slide(r1.cen, bXa * r2.cen, r1.rad + rad, r2.rad + rad);
  }
  F minimumOnVolumeObject(Sphere<F> r, PtIdx<F> v) {
    return get_slide(r.cen, bXa * v.pos, r.rad + rad, rad);
  }
  F minimumOnObjectVolume(PtIdx<F> v, Sphere<F> r) {
    return get_slide(v.pos, bXa * r.cen, rad, r.rad + rad);
  }
  F minimumOnObjectObject(PtIdx<F> a, PtIdx<F> b) {
    // std::cout << "Obj Obj" << std::endl;
    return get_slide(a.pos, bXa * b.pos, rad, rad);
  }
  F get_slide(V3<F> cen1, V3<F> cen2, F rad1, F rad2) {
    V3<F> hypot_start = cen2 - cen1;
    F d_parallel_start = hypot_start.dot(dirn);
    F d_perpendicular_sq = hypot_start.squaredNorm() - square(d_parallel_start);
    F d_hypot_stop_sq = square(rad1 + rad2);
    if (d_hypot_stop_sq < d_perpendicular_sq) return 9e9;  // miss
    F d_parallel_stop = std::sqrt(d_hypot_stop_sq - d_perpendicular_sq);
    F moveby = d_parallel_start - d_parallel_stop;
    // V3<F> cnew = cen1 + moveby * dirn;
    // F dnew = (cnew - cen2).norm();
    return moveby;
  }
};

template <typename F>
F bvh_slide(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2, F rad,
            V3<F> dirn) {
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  X3<F> x1inv = x1.inverse();
  X3<F> pos = x1inv * x2;
  V3<F> local_dir = x1inv.rotation() * dirn;
  BVMinAxis<F> query(local_dir, pos, rad);
  F result = rpxdock::bvh::BVMinimize(bvh1, bvh2, query);
  return result;
}
template <typename F>
Vx<F> bvh_slide_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                    py::array_t<F> pos2, F rad, V3<F> dirn) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size())
    throw std::runtime_error("pos1 and pos2 must have same len");
  py::gil_scoped_release release;
  Vx<F> slides(x1.size());
  for (size_t i = 0; i < x1.size(); ++i) {
    X3<F> x1inv = x1[i].inverse();
    X3<F> pos = x1inv * x2[i];
    V3<F> local_dir = x1inv.rotation() * dirn;
    BVMinAxis<F> query(local_dir, pos, rad);
    slides[i] = rpxdock::bvh::BVMinimize(bvh1, bvh2, query);
  }
  return slides;
}

template <typename F>
struct BVHCountPairs {
  using Scalar = F;
  using Xform = X3<F>;
  F mindis = 0.0, mindis2 = 0.0;
  Xform bXa = Xform::Identity();
  int nout = 0;
  BVHCountPairs(F mind, Xform x) : mindis(mind), bXa(x), mindis2(mind * mind) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    return vol1.signdis(bXa * vol2) < mindis;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    return vol1.signdis(bXa * obj2.pos) < mindis;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    return (bXa * vol2).signdis(obj1.pos) < mindis;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < mindis2;
    if (isect) nout++;
    return false;
  }
};

template <typename F>
int bvh_count_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                    F maxdist) {
  py::gil_scoped_release release;
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  BVHCountPairs<F> query(maxdist, pos);
  rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
  return query.nout;
}
template <typename F>
Vx<int> bvh_count_pairs_vec(BVH<F> &bvh1, BVH<F> &bvh2, py::array_t<F> pos1,
                            py::array_t<F> pos2, F maxdist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same length");
  py::gil_scoped_release release;
  size_t n = std::max(x1.size(), x2.size());
  Vx<int> npair(n);
  for (size_t i = 0; i < x1.size(); ++i) {
    size_t i1 = x1.size() == 1 ? 0 : i;
    size_t i2 = x2.size() == 1 ? 0 : i;
    X3<F> pos = x1[i1].inverse() * x2[i2];
    BVHCountPairs<F> query(maxdist, pos);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    npair[i] = query.nout;
  }
  return npair;
}
template <typename F>
struct BVHCollectPairs {
  using Scalar = F;
  using Xform = X3<F>;
  F mindis = 0.0, mindis2 = 0.0;
  Xform bXa = Xform::Identity();
  int32_t *out;
  int nout = 0, maxout = -1;
  bool overflow = false;
  BVHCollectPairs(F r, Xform x, int32_t *ptr, int mx)
      : mindis(r), bXa(x), out(ptr), maxout(mx), mindis2(r * r) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    return vol1.signdis(bXa * vol2) < mindis;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    return vol1.signdis(bXa * obj2.pos) < mindis;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    return (bXa * vol2).signdis(obj1.pos) < mindis;
  }
  bool intersectObjectObject(PtIdx<F> o1, PtIdx<F> o2) {
    bool isect = (o1.pos - bXa * o2.pos).squaredNorm() < mindis2;
    if (isect) {
      if (nout < maxout) {
        // std::cout << "BVH " << nout << " " << o1.idx << " " << o2.idx
        // << std::endl;
        out[2 * nout + 0] = o1.idx;
        out[2 * nout + 1] = o2.idx;
        ++nout;
      } else {
        overflow = true;
      }
    }
    return false;
  }
};

template <typename F>
py::tuple bvh_collect_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                            F maxdist, py::array_t<int32_t> out) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;

  py::buffer_info buf = out.request();
  int nbuf = buf.shape[0];
  if (buf.ndim != 2 || buf.shape[1] != 2)
    throw std::runtime_error("Shape must be (N, 2)");
  if (buf.strides[0] != 2 * sizeof(int32_t))
    throw std::runtime_error("out stride is not 2F");
  int32_t *ptr = (int32_t *)buf.ptr;
  int nout;
  bool overflow;
  {
    py::gil_scoped_release release;
    BVHCollectPairs<F> query(maxdist, pos, ptr, nbuf);
    rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
    nout = query.nout;
    overflow = query.overflow;
  }
  return py::make_tuple(out[py::slice(0, nout, 1)], overflow);
}

template <typename F>
int naive_collect_pairs(BVH<F> &bvh1, BVH<F> &bvh2, M4<F> pos1, M4<F> pos2,
                        F maxdist, py::array_t<int32_t> out) {
  X3<F> x1(pos1), x2(pos2);
  X3<F> pos = x1.inverse() * x2;
  F dist2 = maxdist * maxdist;

  py::buffer_info buf = out.request();
  int nbuf = buf.shape[0], nout = 0;
  if (buf.ndim != 2 || buf.shape[1] != 2)
    throw std::runtime_error("Shape must be (N, 2)");
  int32_t *ptr = (int32_t *)buf.ptr;

  // std::cout << "foo" << bvh1.getRootIndex() << " " << bvh1.objs.size()
  // << std::endl;
  // std::cout << "foo" << bvh2.getRootIndex() << " " << bvh2.objs.size()
  // << std::endl;

  // bounding sphere check
  int ridx1 = bvh1.getRootIndex(), ridx2 = bvh2.getRootIndex();
  if (ridx1 >= 0 && ridx2 >= 0) {
    auto vol1 = bvh1.getVolume(ridx1);
    auto vol2 = bvh1.getVolume(ridx2);
    vol1.cen = x1 * vol1.cen;
    vol2.cen = x2 * vol2.cen;
    if (!vol1.contact(vol2, maxdist)) return 0;
  }

  for (auto o1 : bvh1.objs) {
    for (auto o2 : bvh2.objs) {
      auto d2 = (o1.pos - pos * o2.pos).squaredNorm();
      if (d2 < dist2) {
        assert(nout < nbuf);
        // std::cout << "NAI " << nout << " " << o1.idx << " " << o2.idx
        // << std::endl;

        ptr[2 * nout + 0] = o1.idx;
        ptr[2 * nout + 1] = o2.idx;
        ++nout;
      }
    }
  }

  return nout;
}

template <typename F>
struct BVHCollectPairsVec {
  using Scalar = F;
  using Xform = X3<F>;
  F maxdis = 0.0, maxdis2 = 0.0;
  Xform bXa = Xform::Identity();
  std::vector<int32_t> &out;
  BVHCollectPairsVec(F r, Xform x, std::vector<int32_t> &o)
      : maxdis(r), bXa(x), out(o), maxdis2(r * r) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    return vol1.signdis(bXa * vol2) < maxdis;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    return vol1.signdis(bXa * obj2.pos) < maxdis;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    return (bXa * vol2).signdis(obj1.pos) < maxdis;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < maxdis2;
    if (isect) {
      out.push_back(obj1.idx);
      out.push_back(obj2.idx);
    }
    return false;
  }
};

template <typename F, typename XF>
py::tuple bvh_collect_pairs_vec(BVH<F> &bvh1, BVH<F> &bvh2,
                                py::array_t<XF> pos1, py::array_t<XF> pos2,
                                F maxdist) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1 and pos2 must have same len");

  auto lbub = std::make_unique<Matrix<int, Dynamic, 2, RowMajor>>();
  auto out = std::make_unique<Mx<int32_t>>();
  {
    py::gil_scoped_release release;
    size_t n = std::max(x1.size(), x2.size());
    lbub->resize(n, 2);
    std::vector<int32_t> pairs;
    pairs.reserve(10 * n);
    for (size_t i = 0; i < n; ++i) {
      size_t i1 = x1.size() == 1 ? 0 : i;
      size_t i2 = x2.size() == 1 ? 0 : i;
      X3<F> pos = (x1[i1].inverse() * x2[i2]).template cast<F>();
      BVHCollectPairsVec<F> query(maxdist, pos, pairs);
      (*lbub)(i, 0) = pairs.size() / 2;
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lbub)(i, 1) = pairs.size() / 2;
    }
    out->resize(pairs.size() / 2, 2);
    for (size_t i = 0; i < pairs.size() / 2; ++i) {
      (*out)(i, 0) = pairs[2 * i + 0];
      (*out)(i, 1) = pairs[2 * i + 1];
    }
  }
  return py::make_tuple(*out, *lbub);
}

template <typename F>
struct BVHCollectPairsRangeVec {
  using Scalar = F;
  using Xform = X3<F>;
  F d = 0.0, d2 = 0.0;
  Xform bXa = Xform::Identity();
  int lb1, ub1, lb2, ub2, nasym1, nasym2;
  std::vector<int32_t> &out;
  BVHCollectPairsRangeVec(F r, Xform x, int l1, int u1, int l2, int u2,
                          int _nasym1, int _nasym2, std::vector<int32_t> &o)
      : d(r),
        bXa(x),
        lb1(l1),
        ub1(u1),
        lb2(l2),
        ub2(u2),
        out(o),
        d2(r * r),
        nasym1(_nasym1),
        nasym2(_nasym2) {}
  bool intersectVolumeVolume(Sphere<F> vol1, Sphere<F> vol2) {
    // if (vol1.ub % nasym1 < lb1 || vol1.lb % nasym1 > ub1 || vol2.ub % nasym2
    // < lb2
    // || vol2.lb % nasym2 > ub2)
    // return false;
    return vol1.signdis(bXa * vol2) < d;
  }
  bool intersectVolumeObject(Sphere<F> vol1, PtIdx<F> obj2) {
    if (vol1.ub % nasym1 < lb1 || vol1.lb % nasym1 > ub1 ||
        obj2.idx % nasym2 < lb2 || obj2.idx % nasym2 > ub2)
      return false;
    return vol1.signdis(bXa * obj2.pos) < d;
  }
  bool intersectObjectVolume(PtIdx<F> obj1, Sphere<F> vol2) {
    if (obj1.idx % nasym1 < lb1 || obj1.idx % nasym1 > ub1 ||
        vol2.ub % nasym2 < lb2 || vol2.lb % nasym2 > ub2)
      return false;
    return (bXa * vol2).signdis(obj1.pos) < d;
  }
  bool intersectObjectObject(PtIdx<F> obj1, PtIdx<F> obj2) {
    if (obj1.idx % nasym1 < lb1 || obj1.idx % nasym1 > ub1 ||
        obj2.idx % nasym2 < lb2 || obj2.idx % nasym2 > ub2)
      return false;
    bool isect = (obj1.pos - bXa * obj2.pos).squaredNorm() < d2;
    if (isect) {
      out.push_back(obj1.idx);
      out.push_back(obj2.idx);
    }
    return false;
  }
};

template <typename F, typename XF>
py::tuple bvh_collect_pairs_range_vec(BVH<F> &bvh1, BVH<F> &bvh2,
                                      py::array_t<XF> pos1,
                                      py::array_t<XF> pos2, F maxdist,
                                      Vx<int> lb1, Vx<int> ub1, int nasym1,
                                      Vx<int> lb2, Vx<int> ub2, int nasym2) {
  auto x1 = xform_py_to_eigen(pos1);
  auto x2 = xform_py_to_eigen(pos2);
  if (x1.size() != x2.size() && x1.size() != 1 && x2.size() != 1)
    throw std::runtime_error("pos1/pos2 must be broadcastable");
  if (x1.size() != lb1.size() && x1.size() != 1 && lb1.size() != 1)
    throw std::runtime_error("pos1/lb1 must be broadcastable");
  if (x2.size() != lb2.size() && x2.size() != 1 && lb2.size() != 1)
    throw std::runtime_error("pos2/lb2 must be broadcastable");
  if (lb1.size() != lb2.size() && lb1.size() != 1 && lb2.size() != 1)
    throw std::runtime_error("lb1/lb2 must be broadcastable");
  if (lb1.size() != ub1.size() && lb1.size() != 1 && ub1.size() != 1)
    throw std::runtime_error("lb1/ub1 must be broadcastable");
  if (lb2.size() != ub2.size() && lb2.size() != 1 && ub2.size() != 1)
    throw std::runtime_error("lb2/ub2 must be broadcastable");

  auto lbub = std::make_unique<Matrix<int, Dynamic, 2, RowMajor>>();
  auto out = std::make_unique<Mx<int32_t>>();
  {
    py::gil_scoped_release release;
    size_t n = std::max(std::max(std::max(x1.size(), x2.size()),
                                 std::max(lb1.size(), ub1.size())),
                        std::max(lb2.size(), ub2.size()));
    size_t n0 = std::min(std::min(std::min(x1.size(), x2.size()),
                                  std::min(lb1.size(), ub1.size())),
                         std::min(lb2.size(), ub2.size()));
    if (nasym1 < 0) nasym1 = bvh_max_id(bvh1) + 1;
    if (nasym2 < 0) nasym2 = bvh_max_id(bvh2) + 1;
    // std::cout << bvh1.size() << " " << nasym1 << " "
    // << (float)bvh1.size() / nasym1 << std::endl;
    // std::cout << bvh2.size() << " " << nasym2 << " "
    // << (float)bvh2.size() / nasym2 << std::endl;
    n = n0 ? n : 0;
    lbub->resize(n, 2);
    std::vector<int32_t> pairs;
    pairs.reserve(10 * n);
    for (size_t i = 0; i < n; ++i) {
      size_t ix1 = x1.size() == 1 ? 0 : i;
      size_t ix2 = x2.size() == 1 ? 0 : i;
      int l1 = lb1.size() == 1 ? lb1[0] : lb1[i];
      int l2 = lb2.size() == 1 ? lb2[0] : lb2[i];
      int u1 = ub1.size() == 1 ? ub1[0] : ub1[i];
      int u2 = ub2.size() == 1 ? ub2[0] : ub2[i];

      // std::cout << "cpp " << i << " " << l1 << "-" << u1 << " " << nasym1 <<
      // " "
      // << l2 << "-" << u2 << " " << nasym2 << std::endl;

      X3<F> pos = (x1[ix1].inverse() * x2[ix2]).template cast<F>();
      BVHCollectPairsRangeVec<F> query(maxdist, pos, l1, u1, l2, u2, nasym1,
                                       nasym2, pairs);
      (*lbub)(i, 0) = pairs.size() / 2;
      rpxdock::bvh::BVIntersect(bvh1, bvh2, query);
      (*lbub)(i, 1) = pairs.size() / 2;
    }
    out->resize(pairs.size() / 2, 2);
    for (size_t i = 0; i < pairs.size() / 2; ++i) {
      (*out)(i, 0) = pairs[2 * i + 0];
      (*out)(i, 1) = pairs[2 * i + 1];
    }
  }
  return py::make_tuple(*out, *lbub);
}

template <typename F>
int bvh_print(BVH<F> &bvh) {
  for (auto o : bvh.objs) {
    py::print("BVH PT ", o.idx, o.pos.transpose());
  }
}

template <typename F>
py::array_t<F> bvh_obj_centers(BVH<F> &b) {
  int n = b.objs.size();
  auto shape = std::vector<int>{n, 4};
  py::array_t<F> out(shape);
  py::buffer_info buf = out.request();
  F *ptr = (F *)buf.ptr;
  for (int i = 0; i < n; ++i) {
    ptr[4 * i + 0] = b.objs[i].pos[0];
    ptr[4 * i + 1] = b.objs[i].pos[1];
    ptr[4 * i + 2] = b.objs[i].pos[2];
    ptr[4 * i + 3] = 1;
  }
  return out;
}
template <typename F>
V4<F> bvh_obj_com(BVH<F> &b) {
  py::gil_scoped_release release;
  int n = b.objs.size();
  V4<F> com(0, 0, 0, 1);
  for (int i = 0; i < n; ++i) {
    com[0] += b.objs[i].pos[0];
    com[1] += b.objs[i].pos[1];
    com[2] += b.objs[i].pos[2];
  }
  com /= n;
  return com;
}

template <typename F>
py::tuple BVH_get_state(BVH<F> const &bvh) {
  Vx<int> child(bvh.child.size());
  for (int i = 0; i < bvh.child.size(); ++i) child[i] = bvh.child[i];
  Mx<F> sph(bvh.vols.size(), 4);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    for (int j = 0; j < 3; ++j) sph(i, j) = bvh.vols[i].cen[j];
    sph(i, 3) = bvh.vols[i].rad;
  }
  Mx<int> lbub(bvh.vols.size(), 2);
  for (int i = 0; i < bvh.vols.size(); ++i) {
    lbub(i, 0) = bvh.vols[i].lb;
    lbub(i, 1) = bvh.vols[i].ub;
  }
  Mx<F> pos(bvh.objs.size(), 3);
  for (int i = 0; i < bvh.objs.size(); ++i) {
    for (int j = 0; j < 3; ++j) pos(i, j) = bvh.objs[i].pos[j];
  }
  Vx<int> idx(bvh.objs.size());
  for (int i = 0; i < bvh.objs.size(); ++i) idx[i] = bvh.objs[i].idx;
  return py::make_tuple(child, sph, lbub, pos, idx);
}
template <typename F>
std::unique_ptr<BVH<F>> bvh_set_state(py::tuple state) {
  auto bvh = std::make_unique<BVH<F>>();
  auto child = state[0].cast<Vx<int>>();
  auto sph = state[1].cast<Mx<F>>();
  auto lbub = state[2].cast<Mx<int>>();
  auto pos = state[3].cast<Mx<F>>();
  auto idx = state[4].cast<Vx<int>>();

  for (int i = 0; i < child.size(); ++i) {
    bvh->child.push_back(child[i]);
  }
  for (int i = 0; i < sph.rows(); ++i) {
    Sphere<F> sphere;
    for (int j = 0; j < 3; ++j) sphere.cen[j] = sph(i, j);
    sphere.rad = sph(i, 3);
    sphere.lb = lbub(i, 0);
    sphere.ub = lbub(i, 1);
    bvh->vols.push_back(sphere);
  }
  for (int i = 0; i < idx.size(); ++i) {
    PtIdx<F> pt;
    for (int j = 0; j < 3; ++j) pt.pos[j] = pos(i, j);
    pt.idx = idx[i];
    bvh->objs.push_back(pt);
  }
  return bvh;
}

template <typename F>
void bind_bvh(auto m, std::string name) {
  py::class_<BVH<F>>(m, name.c_str())
      .def(py::init(&bvh_create<F>), "coords"_a, "which"_a = Vx<bool>(),
           "ids"_a = Vx<int>())
      .def("__len__", [](BVH<F> &b) { return b.objs.size(); })
      .def("radius", [](BVH<F> &b) { return b.vols[b.getRootIndex()].rad; })
      .def("center", [](BVH<F> &b) { return b.vols[b.getRootIndex()].cen; })
      .def("centers", &bvh_obj_centers<F>)
      .def("com", &bvh_obj_com<F>)
      .def("max_id", &bvh_max_id<F>)
      .def("min_lb", &bvh_min_lb<F>)
      .def("max_ub", &bvh_max_ub<F>)
      .def("obj_id", &bvh_obj_ids<F>)
      .def("vol_lb", &bvh_vol_lbs<F>)
      .def("vol_ub", &bvh_vol_ubs<F>)
      .def(py::pickle(
          [](const BVH<F> &bvh) { return BVH_get_state<F>((BVH<F> &)bvh); },
          [](py::tuple t) { return bvh_set_state<F>(t); }))
      /**/;
}

PYBIND11_MODULE(bvh, m) {
  // bind_bvh<float>(m, "SphereBVH_float");
  bind_bvh<double>(m, "SphereBVH_double");

  m.def("bvh_min_dist", &bvh_min_dist<double>, "min pair distance", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_vec", &bvh_min_dist_vec<double>, "min pair distance",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  // m.def("bvh_min_dist_32bit", &bvh_min_dist<float>, "intersction test",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a);
  m.def("bvh_min_dist_fixed", &bvh_min_dist_fixed<double>);
  m.def("naive_min_dist", &naive_min_dist<double>);
  m.def("naive_min_dist_fixed", &naive_min_dist_fixed<double>);

  m.def("bvh_isect", &bvh_isect<double>, "intersction test", "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_vec", &bvh_isect_vec<double>, "intersction test", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  // m.def("bvh_isect_32bit", &bvh_isect<float>, "intersction test", "bvh1"_a,
  // "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);
  m.def("bvh_isect_fixed", &bvh_isect_fixed<double>);
  m.def("naive_isect", &naive_isect<double>);
  m.def("naive_isect_fixed", &naive_isect_fixed<double>);

  Vx<int> default_ub(1), default_lb(1);
  default_lb[0] = 0;
  default_ub[0] = 99999999;
  m.def("bvh_isect_fixed_range_vec", &bvh_isect_fixed_range_vec<double>,
        "intersction test with input range", "bvh1"_a, "bvh2"_a, "pos1"_a,
        "pos2"_a, "mindist"_a, "lb1"_a = default_lb, "ub1"_a = default_ub,
        "lb2"_a = default_lb, "ub2"_a = default_ub);

  m.def("isect_range_single", &isect_range_single<double>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a, "maxtrim"_a = -1,
        "maxtrim_lb"_a = -1, "maxtrim_ub"_a = -1, "nasym1"_a = -1);
  m.def("isect_range", &isect_range<double>, "intersction test", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a, "maxtrim"_a = -1,
        "maxtrim_lb"_a = -1, "maxtrim_ub"_a = -1, "nasym1"_a = -1);
  m.def("naive_isect_range", &naive_isect_range<double>, "intersction test",
        "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "mindist"_a);

  m.def("bvh_slide", &bvh_slide<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);
  m.def("bvh_slide_vec", &bvh_slide_vec<double>, "slide into contact", "bvh1"_a,
        "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  // m.def("bvh_slide_32bit", &bvh_slide<float>, "slide into contact",
  // "bvh1"_a, "bvh2"_a, "pos1"_a, "pos2"_a, "rad"_a, "dirn"_a);

  m.def("bvh_collect_pairs", &bvh_collect_pairs<double>);
  m.def("bvh_collect_pairs_vec", &bvh_collect_pairs_vec<double, float>);
  m.def("bvh_collect_pairs_vec", &bvh_collect_pairs_vec<double, double>);
  m.def("naive_collect_pairs", &naive_collect_pairs<double>);
  m.def("bvh_count_pairs", &bvh_count_pairs<double>);
  m.def("bvh_count_pairs_vec", &bvh_count_pairs_vec<double>);

  m.def("bvh_print", &bvh_print<double>);

  m.def("bvh_min_dist_one", &bvh_min_dist_one<double>);

  Vx<int> lb0(1), ub0(1);
  lb0[0] = NL<int>::min();
  ub0[0] = NL<int>::max();
  m.def("bvh_collect_pairs_range_vec",
        &bvh_collect_pairs_range_vec<double, float>, "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "maxdist"_a, "lb1"_a = lb0, "ub1"_a = ub0,
        "nasym1"_a = -1, "lb2"_a = lb0, "ub2"_a = ub0, "nasym2"_a = -1);
  m.def("bvh_collect_pairs_range_vec",
        &bvh_collect_pairs_range_vec<double, double>, "bvh1"_a, "bvh2"_a,
        "pos1"_a, "pos2"_a, "maxdist"_a, "lb1"_a = lb0, "ub1"_a = ub0,
        "nasym1"_a = -1, "lb2"_a = lb0, "ub2"_a = ub0, "nasym2"_a = -1);
}

}  // namespace bvh
}  // namespace rpxdock
