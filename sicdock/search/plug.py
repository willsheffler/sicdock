from time import perf_counter
import numpy as np
import xarray as xr
import homog as hm
from sicdock.search import hier_sample, expand_samples, hier_evaluate
from sicdock.search import hier_evaluate_executor, filter_redundancy
from sicdock.search import trim_atom_to_res_numbering
from sicdock.util import Bunch
from sicdock.sampling import XformHier_f4
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.sym import symframes


def make_plugs(plug, hole, hscore, sampler, **kw):
    args = Bunch(kw)
    ttot = perf_counter()
    args.nresl = len(hscore.hier) if args.nresl is None else args.nresl
    args.out_prefix = "plug" if args.out_prefix is None else args.out_prefix

    evaluator = PlugEvaluator(plug, hole, hscore, **args)
    xforms, scores, stats = hier_sample(sampler, evaluator, **args)
    ibest = filter_redundancy(xforms, plug, scores, **args)
    tdump = dump_plugs(xforms, plug, hole, scores, ibest, evaluator, **args)

    ttot = perf_counter() - ttot
    print(f"rate: {int(stats.ntot / ttot):,}/s ttot {ttot:7.3f} tdump {tdump:7.3f}")
    print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
    print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

    xforms = xforms[ibest]
    wrpx = args.wts.sub(rpx=1, ncontact=0)
    wnct = args.wts.sub(rpx=0, ncontact=1)
    rpx, lb, ub = evaluator.iface_scores(xforms, args.nresl - 1, wrpx)
    ncontact, *_ = evaluator.iface_scores(xforms, args.nresl - 1, wnct)
    return xr.Dataset(
        dict(
            scores=(["model"], scores[ibest].astype("f4")),
            xforms=(["model", "hrow", "hcol"], xforms),
            rpx_plug=(["model"], rpx[:, 0].astype("f4")),
            rpx_hole=(["model"], rpx[:, 1].astype("f4")),
            ncontact_plug=(["model"], ncontact[:, 0].astype("f4")),
            ncontact_hole=(["model"], ncontact[:, 1].astype("f4")),
            reslb=(["model"], lb),
            resub=(["model"], ub),
        )
    )


class PlugEvaluator:
    def __init__(self, plug, hole, hscore, **kw):
        self.args = Bunch(kw)
        self.plug = plug
        self.hole = hole
        self.hscore = hscore
        self.symrot = hm.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)

    def __call__(self, xforms, iresl, wts={}):
        wts = self.args.wts.sub(wts)
        wts_ph = wts.plug, wts.hole
        iface_scores, plb, pub = self.iface_scores(xforms, iresl, wts)
        scores = self.args.iface_summary(iface_scores * wts_ph, axis=1)
        return scores, plb, pub

    def iface_scores(self, xforms, iresl, wts={}):
        wts = self.args.wts.sub(wts)
        xeye = np.eye(4, dtype="f4")
        xforms = xforms.reshape(-1, 4, 4)
        plug, hole, sfxn = self.plug, self.hole, self.hscore.scorepos
        dclsh, max_trim = self.args.clashdis, self.args.max_trim
        xsym = self.symrot @ xforms

        # check for "flatness"
        ok = np.abs((xforms @ plug.pcavecs[0])[:, 2]) <= self.args.max_longaxis_dot_z

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
        scores[ok, 0] = sfxn(iresl, plug, plug, xok, xsym[ok], wts, (*ptrim, *ptrim))
        scores[ok, 1] = sfxn(iresl, plug, hole, xok, xeye[:,], wts, ptrim)

        # record ranges used
        plb = np.zeros(len(scores), dtype="i4")
        pub = np.ones(len(scores), dtype="i4") * (plug.nres - 1)
        if ptrim:
            plb[ok], pub[ok] = ptrim[0], ptrim[1]

        return scores, plb, pub


def dump_plugs(xforms, plug, hole, scores, ibest, evaluator, **kw):
    args = Bunch(kw)
    t = perf_counter()
    fname_prefix = "plug" if args.out_prefix is None else args.out_prefix
    nout = min(10 if args.nout is None else args.nout, len(ibest))
    for i in range(nout):
        plug.move_to(xforms[ibest[i]])
        wrpx, wnct = (args.wts.sub(rpx=1, ncontact=0), args.wts.sub(rpx=0, ncontact=1))
        scoreme = evaluator.iface_scores
        ((pscr, hscr),), *lbub = scoreme(xforms[ibest[i]], args.nresl - 1, wrpx)
        ((pcnt, hcnt),), *lbub = scoreme(xforms[ibest[i]], args.nresl - 1, wnct)
        fn = fname_prefix + "_%02i.pdb" % i
        print(
            f"{fn} score {scores[ibest[i]]:7.3f} olig: {pscr:7.3f} hole: {hscr:7.3f}",
            f"resi {lbub[0][0]}-{lbub[1][0]} {pcnt:7.0f} {hcnt:7.0f}",
        )
        dump_pdb_from_bodies(fn, [plug], symframes(hole.sym), resbounds=[lbub])
        # dump_pdb_from_bodies(fn + "whole.pdb", [plug], symframes(hole.sym))
    # dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))
    return perf_counter() - t


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
    xh = XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
    assert xh.sanity_check(), "bad xform hierarchy"
    print(f"XformHier {xh.size(0):,}", xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)
    return xh


def __make_plugs_hier_sample_test__(plug, hole, hscore, **kw):
    args = Bunch(kw)
    sampler = plug_get_sample_hierarchy(plug, hole, hscore)
    sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore)

    nresl = kw["nresl"]

    for rpx in [0, 1]:
        args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=rpx)
        evaluator = PlugEvaluator(plug, hole, hscore, **args)
        iresl = 0
        indices, xforms = expand_samples(**args.sub(vars()))
        scores, *resbound, t = hier_evaluate(**args.sub(vars()))
        iroot = np.argsort(-scores)[:10]
        xroot = xforms[iroot]
        sroot = scores[iroot]

        for ibeam in range(6, 27):
            beam_size = 2 ** ibeam
            indices, xforms, scores = iroot, xroot, sroot
            for iresl in range(1, nresl):
                indices, xforms = expand_samples(**args.sub(vars()))
                scores, *resbound, t = hier_evaluate(**args.sub(vars()))
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


def ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore):
    r, rori = hscore.base.attr.xhresl
    cartub = np.array([6 * r, r, r])
    cartlb = np.array([-6 * r, 0, 0])
    cartbs = np.array([12, 1, 1], dtype="i")
    xh = XformHier_f4(cartlb, cartub, cartbs, rori)
    assert xh.sanity_check(), "bad xform hierarchy"
    print(f"XformHier {xh.size(0):,}", xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)
    return xh
