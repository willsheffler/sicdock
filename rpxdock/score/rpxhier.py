import os, logging, glob, numpy as np, rpxdock as rp
from rpxdock.xbin import xbin_util as xu
from rpxdock.score import score_functions as sfx

log = logging.getLogger(__name__)

"""
RpxHier holds score information at each level of searching / scoring 
Grid search just uses the last/finest scorefunction 
"""
class RpxHier:
   def __init__(self, files, max_pair_dist=8.0, hscore_data_dir=None, **kw):
      kw = rp.Bunch(kw)
      if isinstance(files, str): files = [files]
      if len(files) is 0: raise ValueError('RpxHier given no datafiles')
      if len(files) is 1: files = _check_hscore_files_aliases(files[0], hscore_data_dir)
      if len(files) > 8:
         for f in files:
            log.error(f)
         raise ValueError('too many hscore_files (?)')
      if all(isinstance(f, str) for f in files):
         if "_SSindep_" in files[0]:
            assert all("_SSindep_" in f for f in files)
            self.use_ss = False
         else:
            assert all("_SSdep_" in f for f in files)
            self.use_ss = True
         assert "base" in files[0]
         data = rp.util.load_threads(files, len(files))
         self.base = data[0]
         self.hier = data[1:]
         self.resl = list(h.attr.cart_extent for h in self.hier)
      elif (isinstance(files[0], rp.ResPairScore)
            and all(isinstance(f, rp.Xmap) for f in files[1:])):
         self.base = files[0]
         self.hier = list(files[1:])
         self.use_ss = self.base.attr.opts.use_ss_key
         assert all(self.use_ss == h.attr.cli_args.use_ss_key for h in self.hier)
      else:
         raise ValueError('RpxHier expects filenames or ResPairScore+[Xmap*]')
      # append extra copies of highest resl score to use for higher res search steps
      self.actual_nresl = len(self.hier)
      for i in range(10):
         self.hier.append(self.hier[-1])
      self.cart_extent = [h.attr.cart_extent for h in self.hier]
      self.ori_extent = [h.attr.ori_extent for h in self.hier]
      self.max_pair_dist = [max_pair_dist + h.attr.cart_extent for h in self.hier]
      self.map_pairs_multipos = xu.ssmap_pairs_multipos if self.use_ss else xu.map_pairs_multipos
      self.map_pairs = xu.ssmap_of_selected_pairs if self.use_ss else xu.map_of_selected_pairs
      self.score_only_sspair = kw.score_only_sspair
      self.function = kw.function

   def __len__(self):
      return len(self.hier)

   def hier_mindis(self, iresl):
      return [1.5, 2.0, 2.75, 3.25, 3.5][iresl]

   def score(self, body1, body2, wts, iresl=-1, *bounds):
      return self.scorepos(body1, body2, body1.pos, body2.pos, iresl, *bounds, wts=wts)

   def score_matrix_intra(self, body, wts, iresl=-1):
      pairs, lbub = rp.bvh.bvh_collect_pairs_vec(body.bvh_cen, body.bvh_cen, np.eye(4), np.eye(4),
                                                 self.max_pair_dist[iresl])
      pairs = body.filter_pairs(pairs, self.score_only_sspair)
      assert len(lbub) is 1
      xmap = self.hier[iresl]
      ssstub = body.ssid, body.ssid, body.stub, body.stub
      if not self.use_ss: ssstub = ssstub[2:]
      pscore = self.map_pairs(xmap.xbin, xmap.phmap, pairs, *ssstub)
      m = np.zeros((len(body), ) * 2, dtype='f4')
      m[pairs[:, 0], pairs[:, 1]] += pscore
      m[pairs[:, 1], pairs[:, 0]] += pscore
      m /= 2
      return m

   def score_matrix_inter(self, bodyA, bodyB, wts, symframes=[np.eye(4)], iresl=-1, **kw):
      m = np.zeros((len(bodyA), len(bodyB)), dtype='f4')
      for xsym in symframes[1:].astype('f4'):
         pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
            bodyA.bvh_cen,
            bodyB.bvh_cen,
            bodyA.pos,
            xsym @ bodyB.pos,
            self.max_pair_dist[iresl],
         )
         assert len(lbub) is 1
         pairs = bodyA.filter_pairs(pairs, self.score_only_sspair, other=bodyB, **kw)
         xmap = self.hier[iresl]
         ssstub = bodyA.ssid, bodyB.ssid, bodyA.pos @ bodyA.stub, xsym @ bodyB.pos @ bodyB.stub
         if not self.use_ss: ssstub = ssstub[2:]

         pscore = self.map_pairs(xmap.xbin, xmap.phmap, pairs, *ssstub)
         m[pairs[:, 0], pairs[:, 1]] += pscore

      return m

#  m.def("map_of_selected_pairs", &map_of_selected_pairs_onearray<K, F, double>,
#        "xbin"_a, "phmap"_a, "idx"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);
#  m.def("map_of_selected_pairs",
#        &map_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a, "phmap"_a,
#        "idx"_c, "xform"_c, "pos1"_a = eye4, "pos2"_a = eye4);
#
#  m.def("ssmap_of_selected_pairs",
#        &ssmap_of_selected_pairs_onearray<K, F, double>, "xbin"_a, "phmap"_a,
#        "idx"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);
#  m.def("ssmap_of_selected_pairs",
#        &ssmap_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a,
#        "phmap"_a, "idx"_c, "ss"_c, "xform"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);

   def scorepos(
      self,
      body1,
      body2,
      pos1,
      pos2,
      iresl=-1,
      bounds=(),
      residue_summary=np.mean,  # TODO hook up to options to select
      **kw,
   ):
      '''
      TODO WSH rearrange so ppl can add different ways of scoring
      Get scores for all docks
      :param body1:
      :param body2:
      :param pos1:
      :param pos2:
      :param iresl:
      :param bounds:
      :param kw:
      :return:
      '''
      kw = rp.Bunch(kw)
      pos1, pos2 = pos1.reshape(-1, 4, 4), pos2.reshape(-1, 4, 4)
      # if not bounds:
      # bounds = [-2e9], [2e9], nsym[0], [-2e9], [2e9], nsym[1]
      # if len(bounds) is 2:
      # bounds += nsym[1],
      # if len(bounds) is 3:
      # bounds += [-2e9], [2e9], 1
      bounds = list(bounds)
      if len(bounds) > 2 and (bounds[2] is None or bounds[2] < 0):
         bounds[2] = body1.asym_body.nres
      if len(bounds) > 5 and (bounds[5] is None or bounds[5] < 0):
         bounds[5] = body2.asym_body.nres

      # print('nres asym', body1.asym_body.nres, body2.asym_body.nres)
      # print(bounds[2], bounds[5])

      # calling bvh c++ function that will look at pair of (arrays of) positions, scores pairs that are in contact (ID from maxpair distance)
      # lbub: len pos1
      pairs, lbub = rp.bvh.bvh_collect_pairs_range_vec(
         body1.bvh_cen,
         body2.bvh_cen,
         pos1,
         pos2,
         self.max_pair_dist[iresl],
         *bounds,
      )

      # TODO some output or analysis of distances?

      # pairs, lbub = body1.filter_pairs(pairs, self.score_only_sspair, other=body2, lbub=lbub)

      if bounds: assert len(bounds[0]) in (1, len(lbub))
      # if len(bounds[0]) > 1:
      #    print(len(lbub), len(bounds[0]))

      #    # print(lbub)
      #    for i, (lb, ub) in enumerate(lbub):
      #       asym_res1 = pairs[lb:ub, 0] % body1.asym_body.nres
      #       asym_res2 = pairs[lb:ub, 1] % body2.asym_body.nres
      #       print(i, f'{np.min(asym_res1)}-{np.max(asym_res1)}', bounds[0][i], bounds[1][i])
      #       print(i, f'{np.min(asym_res2)}-{np.max(asym_res2)}', bounds[3][i], bounds[4][i])
      #       assert np.all(asym_res1 >= bounds[0][i])
      #       assert np.all(asym_res1 <= bounds[1][i])
      #       assert np.all(asym_res2 >= bounds[3][i])
      #       assert np.all(asym_res2 <= bounds[4][i])

      #TODO: Figure out if this should be handled in the score functions below.
      if kw.wts.rpx == 0:
         return kw.wts.ncontact * (lbub[:, 1] - lbub[:, 0]) # option to score based on ncontacts only

      xbin = self.hier[iresl].xbin
      phmap = self.hier[iresl].phmap
      ssstub = body1.ssid, body2.ssid, body1.stub, body2.stub
      ssstub = ssstub if self.use_ss else ssstub[2:]

      # hashtable of scores for each pair of res in contact in each dock
      pscore = self.map_pairs_multipos(
         xbin,
         phmap,
         pairs,
         *ssstub,
         # body1.ssid, body2.ssid, body1.stub, body2.stub,
         lbub,
         pos1,
         pos2,
      )

      # summarize pscores for a dock
      lbub1, lbub2, idx1, idx2, ressc1, ressc2 = rp.motif.marginal_max_score(
         lbub,
         pairs,
         pscore,
      )
      score_functions = {"fun2" : sfx.score_fun2, "lin" : sfx.lin, "exp" : sfx.exp, "mean" : sfx.mean, "median" : sfx.median, "stnd" : sfx.stnd, "sasa_priority" : sfx.sasa_priority}
      score_fx = score_functions.get(self.function)

      if score_fx:
         scores = score_fx(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, wts=kw.wts, iresl=iresl)
      else:
         logging.info(f"Failed to find score function {self.function}, falling back to 'stnd'")
         scores = score_functions["stnd"](pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, wts=kw.wts)
      return scores

   def iresls(self):
      return [i for i in range(len(self.hier))]

   def score_all(self, x):
      return np.stack([h[x] for h in self.hier])

   def score_by_resl(self, resl, x_or_k):
      if resl < 0 or resl > self.resl[0] * 2:
         raise ValueError("resl out of bounds")
      iresl = np.argmin(np.abs(resl - self.resl))
      return self.hier[iresl][x_or_k]

   def score_base(self, x_or_k):
      return self.base[x_or_k]

def _check_hscore_files_aliases(alias, hscore_data_dir):
   try:
      pattern = os.path.join(hscore_data_dir, alias, '*.pickle')
      g = sorted(glob.glob(pattern))
      if len(g) > 0:
         return g
   except:
      pass
   raise ValueError(f'hscore datadir {hscore_data_dir} or alias {alias} invalid')
