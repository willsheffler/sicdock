import logging, itertools, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)

def hier_search(sampler, evaluator, **kw):
   '''
   :param sampler:
   :param evaluator:
   :param kw:
   :return:
   gets positions and scores and stuff for sampling
   '''
   kw = rp.Bunch(kw)
   neval, indices, scores = list(), None, None
   nresl = kw.nresl if kw.nresl else evaluator.hscore.actual_nresl
   
   #Uncomment to dump docking metrics at each resolution level of the search.
   #iresl_list = []
   #data_list = []
   #spec = kw.spec
   #bodies = kw.bodies
   for iresl in range(kw.nresl):
      indices, xforms = expand_samples(iresl, sampler, indices, scores, **kw)
      scores, extra, t = rp.search.evaluate_positions(**kw.sub(vars()))
      neval.append((t, len(scores)))
      log.info(f"{kw.output_prefix} iresl {iresl} ntot {len(scores):11,} " +
               f"nonzero {np.sum(scores > 0):5,}")
   #Uncomment to dump docking metrics at each resolution level of the search. 
   """
      iresl_list.append(iresl)
      wrpx = kw.wts.sub(rpx=1, ncontact=0)
      wnct = kw.wts.sub(rpx=0, ncontact=1)
      rpx, rpx_extra = evaluator(xforms, iresl, wrpx)
      ncontact, ncont_extra = evaluator(xforms, iresl, wnct)

      data = dict(
         attrs=dict(arg=kw, output_prefix=kw.output_prefix,
                 output_body='all', sym=spec.arch),
         scores=(["model"], scores.astype("f4")),
         xforms=(["model", "comp", "hrow", "hcol"], xforms),
         rpx=(["model"], rpx),
         ncontact=(["model"], ncontact),
      )

      for k, v in extra.items():
         if not isinstance(v, (list, tuple)) or len(v) > 3:
            v = ['model'], v
         data[k] = v
      for i in range(len(bodies)):
         data[f'disp{i}'] = (['model'], np.sum(xforms[:, i, :3, 3] * spec.axis[None, i, :3], axis=1))
         data[f'angle{i}'] = (['model'], rp.homog.angle_of(xforms[:, i]) * 180 / np.pi)

      data_list.append(data)

   search_data = dict(resl = iresl_list, data = data_list)
   rp.util.dump(search_data, kw.output_prefix + '_iresl_Result.pickle')
   """

   stats = rp.Bunch(ntot=sum(x[1] for x in neval), neval=neval)

   return xforms, scores, extra, stats

def expand_samples(iresl, sampler, indices=None, scores=None, beam_size=None, **kw):
   if iresl == 0:
      indices = np.arange(sampler.size(0), dtype="u8")
      mask, xforms = sampler.get_xforms(0, indices)
      return indices[mask], xforms
   nexpand = max(1, int(beam_size / 2**sampler.dim))
   idx, xforms = sampler.expand_top_N(nexpand, iresl - 1, scores, indices)
   return idx, xforms

# def tccage_slide_hier_samples_depricated(spec, resl=16, max_out_of_plane_angle=16, nstep=1, **kw):
#    tip = max_out_of_plane_angle
#
#    range1 = 180 / spec.nfold1
#    range2 = 180 / spec.nfold2
#    newresl1 = 2 * range1 / np.ceil(2 * range1 / resl)
#    newresl2 = 2 * range2 / np.ceil(2 * range2 / resl)
#    angs1 = np.arange(-range1 + newresl1 / 2, range1, newresl1)
#    angs2 = np.arange(-range2 + newresl2 / 2, range2, newresl2)
#
#    newresl3 = resl
#    angs3 = np.zeros(1)
#    if tip > resl / 8:
#       newresl3 = 2 * tip / np.ceil(2 * tip / resl)
#       angs3 = np.arange(-tip + newresl3 / 2, tip, newresl3)
#    angs3 = np.concatenate([angs3, angs3 + 180])
#    newresls = np.array([newresl1, newresl2, newresl3])
#
#    nr = newresls / 2
#    for i in range(1, nstep):
#       nr /= 2
#       angs1 = (angs1[:, None] + [-nr[0], +nr[0]]).reshape(-1)
#       angs2 = (angs2[:, None] + [-nr[1], +nr[1]]).reshape(-1)
#       angs3 = (angs3[:, None] + [-nr[2], +nr[2]]).reshape(-1)
#
#    rots1 = spec.placements1(angs1)
#    rots2 = spec.placements2(angs2)
#    dirns = spec.slide_dir(angs3)
#
#    return [rots1, rots2, dirns], newresls
#
# def tccage_slide_hier_expand_depricated(spec, pos1, pos2, resls):
#    deltas = resls / 2
#    assert np.min(deltas) >= 0.1, "deltas should be in degrees"
#    deltas = deltas / 180 * np.pi
#    n = len(pos1)
#    x1 = rp.homog.hrot(spec.axis1, [-deltas[0], +deltas[0]])
#    x2 = rp.homog.hrot(spec.axis2, [-deltas[1], +deltas[1]])
#    x3 = rp.homog.hrot(spec.axisperp, [-deltas[2], +deltas[2]])
#    dirn = (pos2[:, :, 3] - pos1[:, :, 3])[:, :, None]
#    dirnorm = np.linalg.norm(dirn, axis=1)
#    assert np.min(dirnorm) > 0.9
#    # print("tccage_slide_hier_expand_depricated", n, dirnorm.shape)
#    dirn /= dirnorm[:, None]
#    newpos1 = np.empty((8 * n, 4, 4))
#    newpos2 = np.empty((8 * n, 4, 4))
#    newdirn = np.empty((8 * n, 3))
#    lb, ub = 0, n
#    for x1, x2, x3 in itertools.product(x1, x2, x3):
#       newpos1[lb:ub] = x1 @ pos1
#       newpos2[lb:ub] = x2 @ pos2
#       newdirn[lb:ub] = (x3 @ dirn)[:, :3].squeeze()
#       lb, ub = lb + n, ub + n
#    newpos1[:, :3, 3] = 0
#    newpos2[:, :3, 3] = 0
#    return [newpos1, newpos2, newdirn]
#
# def tccage_slide_hier_depricated(spec, body1, body2, base_resl=16, nstep=5, base_min_contacts=0,
#                                  prune_frac_sortof=0.875, prune_minkeep=1000, **kw):
#    assert base_resl > 2, "are you sure?"
#    mct = [base_min_contacts]
#    mct_update = prune_frac_sortof
#    npair, pos = [None] * nstep, [None] * nstep
#    samples, newresl = tccage_slide_hier_samples_depricated(spec, resl=base_resl, **kw)
#    nsamp = [np.prod([len(s) for s in samples])]
#    for i in range(nstep):
#       npair[i], pos[i] = rp.search.gridslide.find_connected_2xCyclic_slide(
#          spec, body1, body2, samples, min_contacts=mct[-1], **kw)
#       if len(npair[i]) is 0:
#          return npair[i - 1], pos[i - 1]
#       if i + 1 < nstep:
#          newresl /= 2
#          samples = tccage_slide_hier_expand_depricated(spec, *pos[i], newresl)
#          nsamp.append(len(samples[0]))
#
#          log.debug(mct_update)
#          mct.append(int(np.quantile(npair[i][:, 0], mct_update)))
#          # if len(npair[i]) < prune_minkeep:
#          #     print("same mct")
#          #     mct.append(mct[-1])
#          # else:
#          #     nmct = npair[i][:, 0].partition(-prune_minkeep)
#          #     nmct = npair[i][-prune_minkeep, 0]
#          #     qmct = int(np.quantile(npair[i][:, 0], mct_update))
#          #     nprint("mct update", nmct, qmct)
#          #     mct.append(np.min(nmct, qmct))
#
#    # print("nresult     ", [x.shape[0] for x in npair])
#    # print("samps       ", nsamp)
#    # print("min_contacts", mct)
#    return npair[-1], pos[-1]
