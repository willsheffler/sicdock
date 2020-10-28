import itertools, functools, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, trim_ok
import logging

def make_multicomp(
   bodies,
   spec,
   hscore,
   search=hier_search,
   sampler=None,
   fixed_components=False,
   **kw,
):
   kw = rp.Bunch(kw)
   t = rp.Timer().start()
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else spec.arch
   logging.debug("Docking multicomp")

   assert len(bodies) == spec.num_components
   bodies = list(bodies)
   if not fixed_components:
      for i, b in enumerate(bodies):
         bodies[i] = b.copy_xformed(rp.homog.align_vector([0, 0, 1], spec.axis[i]))

   dotrim = kw.max_trim and kw.trimmable_components and len(bodies) < 3
   Evaluator = TwoCompEvaluatorWithTrim if dotrim else MultiCompEvaluator
   evaluator = Evaluator(bodies, spec, hscore, **kw)

   # do search
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)

   ibest = rp.filter_redundancy(xforms, bodies, scores, **kw)
   tdump = _debug_dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **kw)

   if kw.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = kw.wts.sub(rpx=1, ncontact=0)
   wnct = kw.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   ncontact, ncont_extra = evaluator(xforms, kw.nresl - 1, wnct)

   data = dict(
      attrs=dict(arg=kw, stats=stats, ttotal=t.total, tdump=tdump, output_prefix=kw.output_prefix,
                 output_body='all', sym=spec.arch),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "comp", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
   )
   for k, v in extra.items():
      if not isinstance(v, (list, tuple)) or len(v) > 3:
         v = ['model'], v
      data[k] = v
   if kw.score_self:
      for k, v in ncont_extra.items():
         if not isinstance(v, (list, tuple)) or len(v) > 3:
            v = ['model', v]
         data[f"ncont_{k}"] = v
   for i in range(len(bodies)):
      data[f'disp{i}'] = (['model'], np.sum(xforms[:, i, :3, 3] * spec.axis[None, i, :3], axis=1))
      data[f'angle{i}'] = (['model'], rp.homog.angle_of(xforms[:, i]) * 180 / np.pi)
   default_label = [f'comp{c}' for c in 'ABCDEFD'[:len(bodies)]]

   return rp.Result(
      body_=None if kw.dont_store_body_in_results else bodies,
      body_label_=[] if kw.dont_store_body_in_results else default_label,
      **data,
   )

class MultiCompEvaluatorBase:
   def __init__(self, bodies, spec, hscore, wts=rp.Bunch(ncontact=0.1, rpx=1.0), **kw):
      self.kw = rp.Bunch(kw)
      self.hscore = hscore
      self.symrots = [rp.geom.symframes(n) for n in spec.nfold]
      self.spec = spec
      self.kw.wts = wts
      self.bodies = [b.copy_with_sym(spec.nfold[i], spec.axis[i]) for i, b in enumerate(bodies)]

class MultiCompEvaluator(MultiCompEvaluatorBase):
   def __init__(self, *arg, **kw):
      super().__init__(*arg, **kw)

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      B = self.bodies
      # print(f"docking {len(B)} bodies")
      X = xforms.reshape(-1, xforms.shape[-3], 4, 4)
      xnbr = self.spec.to_neighbor_olig

      # check for "flatness"
      delta_h = np.array(
         [hm.hdot(X[:, i] @ B[i].com(), self.spec.axis[i]) for i in range(len(B))])
      ok = np.max(np.abs(delta_h[None] - delta_h[:, None]), axis=(0, 1)) < kw.max_delta_h
      # ok = np.repeat(True, len(X))

      # check clash, or get non-clash range
      for i in range(len(B)):
         if xnbr[i] is not None:
            ok[ok] &= B[i].clash_ok(B[i], X[ok, i], xnbr[i] @ X[ok, i], **kw)
         for j in range(i):
            ok[ok] &= B[i].clash_ok(B[j], X[ok, i], X[ok, j], **kw)

      if xnbr[0] is None and xnbr[1] is not None and xnbr[2] is not None:  # layer hack
         logging.debug("touch")
         inv = np.linalg.inv
         ok[ok] &= B[0].clash_ok(B[1], X[ok, 0], xnbr[1] @ X[ok, 1], **kw)
         ok[ok] &= B[0].clash_ok(B[2], X[ok, 0], xnbr[2] @ X[ok, 2], **kw)
         ok[ok] &= B[0].clash_ok(B[1], X[ok, 0], xnbr[2] @ X[ok, 1], **kw)
         ok[ok] &= B[0].clash_ok(B[2], X[ok, 0], xnbr[1] @ X[ok, 2], **kw)
         ok[ok] &= B[1].clash_ok(B[2], X[ok, 1], xnbr[2] @ X[ok, 2], **kw)
         ok[ok] &= B[1].clash_ok(B[2], X[ok, 1], xnbr[1] @ X[ok, 2], **kw)
         ok[ok] &= B[0].clash_ok(B[1], X[ok, 0], inv(xnbr[1]) @ X[ok, 1], **kw)
         # ok[ok] &= B[0].clash_ok(B[2], X[ok, 0], inv(xnbr[2]) @ X[ok, 2], **kw)
         # ok[ok] &= B[1].clash_ok(B[2], X[ok, 1], inv(xnbr[2]) @ X[ok, 2], **kw)
         ok[ok] &= B[0].clash_ok(B[2], X[ok, 0], inv(xnbr[1]) @ X[ok, 2], **kw)
         ok[ok] &= B[1].clash_ok(B[2], X[ok, 1], inv(xnbr[1]) @ X[ok, 2], **kw)

      # score everything that didn't clash
      # Behaves normally if arg.score_self is not set
      if not kw.score_self:
         ifscore = list()
         for i in range(len(B)):
            for j in range(i):
               ifscore.append(self.hscore.scorepos(B[j], B[i], X[ok, j], X[ok, i], iresl,
                                                   wts=wts))
               # ifscore = np.stack(ifscore)
               logging.debug(f"ifscore is {len(ifscore)} long and is a {type(ifscore)}")

         scores = np.zeros(len(X))
         logging.debug(f"scores is shaped like {scores.shape} and is a {type(scores)}")
         scores[ok] = kw.iface_summary(ifscore, axis=0)
         logging.debug(f"scores is now shaped like {scores.shape}")
         extra = rp.Bunch()
      else:  #return all of the interface scores
         logging.debug("Scoring self")
         s_ifscore = list()
         ns_ifscore = list()
         logging.debug("stepping into scoring")
         #TO DO: The multiple loops could be simplified into a single loop.
         for i in range(len(B)):
            for j in range(i + 1):
               logging.debug(f"scoring body {i} against body {j}")
               if i == j:
                  logging.debug("found self")
                  Xsym = self.spec.to_neighbor_olig @ X
                  s_ifscore.append(
                     self.hscore.scorepos(B[j], B[i], X[ok, j], Xsym[ok, i], iresl, wts=wts))
               else:
                  ns_ifscore.append(
                     self.hscore.scorepos(B[j], B[i], X[ok, j], X[ok, i], iresl, wts=wts))
         logging.debug(f"self scores is length {len(s_ifscore[0])}")
         logging.debug(f"non-self scores is legnth {len(ns_ifscore[0])}")
         logging.debug(f"OK len is {len(ok)}")
         scores_s = np.zeros((len(B), len(X)))
         #TO DO: Quinton: Make sure this actually works for three-body docking
         scores_ns = np.zeros((len(B) - 1, len(X)))

         #Only keep non-clashing interface scores
         for i, iface_scores in enumerate(ns_ifscore):
            scores_ns[i, ok] = iface_scores
         for i, iface_scores in enumerate(s_ifscore):
            scores_s[i, ok] = iface_scores
         logging.debug(f"Scores self is shape {scores_s.shape}")
         logging.debug(f"Scores not-self is shape {scores_ns.shape}")
         logging.debug("Done Scoring")
         #Normal scoring for consistency in output. This may be the same as one of the cross component scores depending on arg.iface_summary()
         scores = np.zeros(len(X))
         scores[ok] = kw.iface_summary(ns_ifscore, axis=0)

         #Package all scores in a dict that can be bunched
         all_scores = {}
         ind = 0
         for i in range(len(B)):
            all_scores[f'self_score_comp{i}'] = (['model'], scores_s[i])
            for j in range(i + 1):
               if j != i:
                  all_scores[f'cross_comp_score_{i}{j}'] = (['model'], scores_ns[ind])
                  ind += 1

         extra = rp.Bunch(all_scores)
      return scores, extra
      #else:
      #   scores[ok] = arg.iface_summary(ifscore, axis=0)
      #   return scores, rp.Bunch()
      #return scores, rp.Bunch()

class TwoCompEvaluatorWithTrim(MultiCompEvaluatorBase):
   def __init__(self, *arg, trimmable_components="AB", **kw):
      super().__init__(*arg, **kw)
      self.trimmable_components = trimmable_components

   def __call__(self, *arg, **kw):
      if 'A' in self.trimmable_components and 'B' in self.trimmable_components:
         sa, lba, uba = self.eval_trim_one('A', *arg, **kw)
         sb, lbb, ubb = self.eval_trim_one('B', *arg, **kw)
         ia = (sa > sb).reshape(-1, 1)
         lb = np.where(ia, lba, lbb)
         ub = np.where(ia, uba, ubb)
         scores = np.maximum(sa, sb)
      else:
         assert self.trimmable_components.upper() in "AB"
         scores, lb, ub = self.eval_trim_one(self.trimmable_components, *arg, **kw)

      extra = rp.Bunch(reslb=(['model', 'component'], lb), resub=(['model', 'component'], ub))
      return scores, extra

   def eval_trim_one(self, trim_component, x, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      B = self.bodies
      X = x.reshape(-1, 2, 4, 4)
      xnbr = self.spec.to_neighbor_olig

      # check for "flatness"
      d1 = hm.hdot(X[:, 0] @ B[0].com(), self.spec.axis[0])
      d2 = hm.hdot(X[:, 1] @ B[1].com(), self.spec.axis[1])
      ok = abs(d1 - d2) < kw.max_delta_h

      lbA = np.zeros(len(X), dtype='i4')
      lbB = np.zeros(len(X), dtype='i4')
      ubA = np.ones(len(X), dtype='i4') * (self.bodies[0].asym_body.nres - 1)
      ubB = np.ones(len(X), dtype='i4') * (self.bodies[1].asym_body.nres - 1)

      # one-sided trim
      if trim_component == 'A':
         trimA1 = B[0].intersect_range(B[1], X[ok, 0], X[ok, 1], **kw)
         trimA1, trimok = trim_ok(trimA1, B[0].asym_body.nres, **kw)
         ok[ok] &= trimok

         xa = X[ok, 0]
         if xnbr is not None:
            trimA2 = B[0].intersect_range(B[0], xa, xnbr[0] @ xa, **kw)
         trimA2, trimok2 = trim_ok(trimA2, B[0].asym_body.nres, **kw)
         ok[ok] &= trimok2
         lbA[ok] = np.maximum(trimA1[0][trimok2], trimA2[0])
         ubA[ok] = np.minimum(trimA1[1][trimok2], trimA2[1])

         xb = X[ok, 1]
         if xnbr is not None:
            trimB = B[1].intersect_range(B[1], xb, xnbr[1] @ xb, **kw)
         trimB, trimok = trim_ok(trimB, B[1].asym_body.nres, **kw)
         ok[ok] &= trimok
         lbB[ok], ubB[ok] = trimB
      elif trim_component == 'B':
         trimB1 = B[1].intersect_range(B[0], X[ok, 1], X[ok, 0], **kw)
         trimB1, trimok = trim_ok(trimB1, B[1].asym_body.nres, **kw)
         ok[ok] &= trimok

         xb = X[ok, 1]
         if xnbr is not None:
            trimB2 = B[1].intersect_range(B[1], xb, xnbr[1] @ xb, **kw)
         trimB2, trimok2 = trim_ok(trimB2, B[1].asym_body.nres, **kw)
         ok[ok] &= trimok2
         lbB[ok] = np.maximum(trimB1[0][trimok2], trimB2[0])
         ubB[ok] = np.minimum(trimB1[1][trimok2], trimB2[1])

         xa = X[ok, 0]
         if xnbr is not None:
            trimA = B[0].intersect_range(B[0], xa, xnbr[0] @ xa, **kw)
         trimA, trimok = trim_ok(trimA, B[0].asym_body.nres, **kw)
         ok[ok] &= trimok
         lbA[ok], ubA[ok] = trimA
      else:
         raise ValueError('trim_component invalid')

      # score everything that didn't clash
      bounds = lbA[ok], ubA[ok], B[0].asym_body.nres, lbB[ok], ubB[ok], B[1].asym_body.nres
      scores = np.zeros(len(X))
      scores[ok] = self.hscore.scorepos(body1=B[0], body2=B[1], pos1=X[ok, 0], pos2=X[ok, 1],
                                        iresl=iresl, bounds=bounds, **kw)

      # if np.sum(ok):
      # print(iresl, np.sum(ok), np.min(scores[ok]), np.max(scores[ok]), np.mean(lbA),
      # np.mean(ubA), np.mean(lbB), np.mean(ubB))

      return scores, np.stack([lbA, lbB], axis=1), np.stack([ubA, ubB], axis=1)

def _debug_dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **kw):
   kw = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if kw.nout_debug is None else kw.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      bodies[0].move_to(xforms[i, 0])
      bodies[1].move_to(xforms[i, 1])
      wrpx, wnct = (kw.wts.sub(rpx=1, ncontact=0), kw.wts.sub(rpx=0, ncontact=1))
      scr, extra = evaluator(xforms[i], kw.nresl - 1, wrpx)
      cnt, extra = evaluator(xforms[i], kw.nresl - 1, wnct)
      fn = kw.output_prefix + "_%02i.pdb" % iout
      lbub = [extra.lbub] if extra.lbub else []
      if len(lbub) > 1:
         print(
            f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
            f"resi {lbub[0][0]}-{lbub[1][0]}",
         )
      else:
         print(f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}")
      rp.dump_pdb_from_bodies(fn, bodies, rp.geom.symframes(spec.sym, xforms[iout]),
                              resbounds=lbub)
   return t.total
