import logging
import numpy as np, xarray as xr, rpxdock as rp
from rpxdock.search import trim_atom_to_res_numbering, hier_search

log = logging.getLogger(__name__)

def make_plugs(plug, hole, hscore, search=hier_search, sampler=None, **kw):

   arg = rp.Bunch(kw)
   arg.nresl = len(hscore.hier) if arg.nresl is None else arg.nresl
   arg.output_prefix = "plug" if arg.output_prefix is None else arg.output_prefix

   t = rp.Timer().start()
   evaluator = PlugEvaluator(plug, hole, hscore, **arg)
   if sampler is None: sampler = _default_samplers[search](plug, hole, hscore)

   xforms, scores, stats = search(sampler, evaluator, **arg)

   ibest = rp.filter_redundancy(xforms, plug, scores, **arg)
   tdump = dump_plugs(xforms, plug, hole, scores, ibest, evaluator, **arg)

   log.debug(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
   log.debug("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
   log.debug("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   scores = scores[ibest]
   wrpx = arg.wts.sub(ncontact=0)
   wnct = arg.wts.sub(rpx=0)
   rpx, lb, ub = evaluator.iface_scores(xforms, arg.nresl - 1, wrpx)
   ncontact, *_ = evaluator.iface_scores(xforms, arg.nresl - 1, wnct)
   ifacescores = rpx + ncontact
   assert np.allclose(np.min(rpx + ncontact, axis=1), scores)
   return rp.Result(
      body_=[] if arg.dont_store_body_in_results else [plug, hole],
      body_label_=[] if arg.dont_store_body_in_results else ['plug', 'hole'],
      attrs=dict(arg=arg, stats=stats, sym=hole.sym, ttotal=t.total, tdump=tdump,
                 output_body='all'),
      scores=(["model"], scores.astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      tot_plug=(["model"], ifacescores[:, 0].astype("f4")),
      tot_hole=(["model"], ifacescores[:, 1].astype("f4")),
      rpx_plug=(["model"], rpx[:, 0].astype("f4")),
      rpx_hole=(["model"], rpx[:, 1].astype("f4")),
      ncontact_plug=(["model"], ncontact[:, 0].astype("f4")),
      ncontact_hole=(["model"], ncontact[:, 1].astype("f4")),
      reslb=(["model"], lb),
      resub=(["model"], ub),
   )

class PlugEvaluator:
   def __init__(self, plug, hole, hscore, **kw):
      self.arg = rp.Bunch(kw)
      self.plug = plug
      self.hole = hole
      self.hscore = hscore
      self.symrot = rp.homog.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)

   def __call__(self, xforms, iresl=-1, wts={}, **_):
      wts = self.arg.wts.sub(wts)
      wts_ph = wts.plug, wts.hole
      iface_scores, plb, pub = self.iface_scores(xforms, iresl, wts)
      scores = self.arg.iface_summary(iface_scores * wts_ph, axis=1)
      return scores, plb, pub

   def iface_scores(self, xforms, iresl=-1, wts={}, **_):
      wts = self.arg.wts.sub(wts)
      xeye = np.eye(4, dtype="f4")
      xforms = xforms.reshape(-1, 4, 4)
      plug, hole, sfxn = self.plug, self.hole, self.hscore.scorepos
      dclsh, max_trim = self.arg.clashdis, self.arg.max_trim
      xsym = self.symrot @ xforms

      # check for "flatness"
      ok = np.abs((xforms @ plug.pcavecs[0])[:, 2]) <= self.arg.max_longaxis_dot_z

      if not self.arg.plug_fixed_olig:
         # check chash in formed oligomer
         ok[ok] &= plug.clash_ok(plug, dclsh, xforms[ok], xsym[ok])

      # check clash olig vs hole, or get non-clash range
      if max_trim > 0:
         ptrim = plug.intersect_range(hole, dclsh, max_trim, xforms[ok])
         ptrim, trimok = trim_atom_to_res_numbering(ptrim, plug.nres, max_trim)
         ok[ok] &= trimok
         # if np.sum(trimok) - np.sum(ntrim == 0):
         # print("ntrim not0", np.sum(trimok) - np.sum(ntrim == 0))
      else:
         ok[ok] &= plug.clash_ok(hole, dclsh, xforms[ok], xeye)
         ptrim = [0], [plug.nres - 1]

      # score everything that didn't clash
      xok = xforms[ok]
      scores = np.zeros((len(xforms), 2))
      scores[ok, 0] = 9999
      if not self.arg.plug_fixed_olig:
         scores[ok, 0] = sfxn(plug, plug, xok, xsym[ok], wts, iresl, (*ptrim, *ptrim))
      scores[ok, 1] = sfxn(plug, hole, xok, xeye[:, ], wts, iresl, ptrim)

      # record ranges used
      plb = np.zeros(len(scores), dtype="i4")
      pub = np.ones(len(scores), dtype="i4") * (plug.nres - 1)
      if ptrim:
         plb[ok], pub[ok] = ptrim[0], ptrim[1]

      return scores, plb, pub

def dump_plugs(xforms, plug, hole, scores, ibest, evaluator, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   fname_prefix = "plug" if arg.output_prefix is None else arg.output_prefix
   nout_debug = min(10 if arg.nout_debug is None else arg.nout_debug, len(ibest))
   for i in range(nout_debug):
      plug.move_to(xforms[ibest[i]])
      wrpx, wnct = (arg.wts.sub(rpx=1, ncontact=0), arg.wts.sub(rpx=0, ncontact=1))
      scoreme = evaluator.iface_scores
      ((pscr, hscr), ), *lbub = scoreme(xforms[ibest[i]], arg.nresl - 1, wrpx)
      ((pcnt, hcnt), ), *lbub = scoreme(xforms[ibest[i]], arg.nresl - 1, wnct)
      fn = fname_prefix + "_%02i.pdb" % i
      log.info(f"{fn} score {scores[ibest[i]]:7.3f} olig: {pscr:7.3f} hole: {hscr:7.3f}" +
               f"resi {lbub[0][0]}-{lbub[1][0]} {pcnt:7.0f} {hcnt:7.0f}")
      rp.io.dump_pdb_from_bodies(fn, [plug], rp.geom.symframes(hole.sym), resbounds=[lbub])
   return t.total

def plug_get_sample_hierarchy(plug, hole, hscore):
   "set up XformHier with appropriate bounds and resolution"
   cart_samp_resl, ori_samp_resl = hscore.base.attr.xhresl
   r0 = max(hole.rg_xy(), 2 * plug.radius_max())
   nr1 = np.ceil(r0 / cart_samp_resl)
   r1 = nr1 * cart_samp_resl
   nr2 = np.ceil(r0 / cart_samp_resl * 2)
   r2 = nr2 * cart_samp_resl / 2
   nh = np.ceil(3 * hole.rg_z() / cart_samp_resl)
   h = nh * cart_samp_resl / 2
   cartub = np.array([+r2, +r2, +h])
   cartlb = np.array([-r2, -r2, -h])
   cartbs = np.array([nr2, nr2, nh], dtype="i")
   xh = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
   assert xh.sanity_check(), "bad xform hierarchy"
   log.info(f"XformHier {xh.size(0):,} {xh.cart_bs} {xh.ori_resl} {xh.cart_lb} {xh.cart_ub}")
   return xh

_default_samplers = {hier_search: plug_get_sample_hierarchy}

def __make_plugs_hier_sample_test__(plug, hole, hscore, **kw):
   arg = rp.Bunch(kw)
   sampler = plug_get_sample_hierarchy(plug, hole, hscore)
   sampler = plug_test_hier_sampler(plug, hole, hscore)

   nresl = kw["nresl"]

   for rpx in [0, 1]:
      arg.wts = rp.Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=rpx)
      evaluator = PlugEvaluator(plug, hole, hscore, **arg)
      iresl = 0
      indices, xforms = expand_samples(**arg.sub(vars()))
      scores, *resbound, t = hier_evaluate(**arg.sub(vars()))
      iroot = np.argsort(-scores)[:10]
      xroot = xforms[iroot]
      sroot = scores[iroot]

      for ibeam in range(6, 27):
         beam_size = 2**ibeam
         indices, xforms, scores = iroot, xroot, sroot
         for iresl in range(1, nresl):
            indices, xforms = expand_samples(**arg.sub(vars()))
            scores, *resbound, t = hier_evaluate(**arg.sub(vars()))
            print(
               f"rpx {rpx} beam {beam_size:9,}",
               f"iresl {iresl} ntot {len(scores):11,} nonzero {np.sum(scores > 0):5,}",
               f"best {np.max(scores)}",
            )
         import _pickle

         fn = "make_plugs_hier_sample_test_rpx_%i_ibeam_%i.pickle" % (rpx, ibeam)
         with open(fn, "wb") as out:
            _pickle.dump((ibeam, iresl, indices, scores), out)
         print()

   assert 0

def plug_test_hier_sampler(plug, hole, hscore):
   r, rori = hscore.base.attr.xhresl
   cartub = np.array([6 * r, r, r])
   cartlb = np.array([-6 * r, 0, 0])
   cartbs = np.array([12, 1, 1], dtype="i")
   xh = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, rori)
   assert xh.sanity_check(), "bad xform hierarchy"
   print(f"XformHier {xh.size(0):,}", xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)
   return xh
