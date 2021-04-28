import _pickle, threading, os, sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np, xarray as xr, rpxdock as rp
from rpxdock.geom import symframes
from rpxdock.sampling import RotCart1Hier_f4, grid_sym_axis
from rpxdock.search import concat_results, make_plugs, plug_get_sample_hierarchy

def testarg():
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 3.0
   kw.max_longaxis_dot_z = 0.5
   kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.multi_iface_summary = np.min
   kw.debug = True
   return kw

def test_plug_hier(hscore, plug, hole):
   kw = testarg()
   kw.max_trim = 0

   # kw.output_prefix = "test_plug_hier"
   # kw.nout_debug = 10
   # hole.dump_pdb('ref.pdb', use_body_sym=True)

   sampler = rp.search.plug.plug_test_hier_sampler(plug, hole, hscore, 2)
   # sampler = rp.search.plug.plug_get_sample_hierarchy(plug, hole, hscore)
   # kw.beam_size = 1e5
   result = make_plugs(plug, hole, hscore, rp.hier_search, sampler, **kw)

   # print(result.reslb)
   # print(result.resub)

   # rp.dump(result, 'rpxdock/data/testdata/test_plug_hier.pickle')
   ref = rp.data.get_test_data('test_plug_hier')
   rp.search.assert_results_close(result, ref)

def test_plug_hier_trim(hscore, plug, hole):
   kw = testarg()
   kw.max_trim = 200
   kw.output_prefix = 'plug'

   # kw.output_prefix = "test_plug_hier_trim"
   # kw.nout_debug = 10
   # hole.dump_pdb('ref.pdb', use_body_sym=True)

   sampler = rp.search.plug.plug_test_hier_sampler(plug, hole, hscore, 1)
   # sampler = rp.search.plug.plug_get_sample_hierarchy(plug, hole, hscore)
   # kw.beam_size = 1e5
   result = make_plugs(plug, hole, hscore, rp.hier_search, sampler, **kw)

   # print(result.reslb)
   # print(result.resub)
   result.dump_pdbs_top_score(10)

   # rp.dump(result, 'rpxdock/data/testdata/test_plug_hier_trim.pickle')
   ref = rp.data.get_test_data('test_plug_hier_trim')
   # rp.search.assert_results_close(result, ref)

def test_plug_olig_hier(hscore, body_c3_mono, hole):
   kw = testarg().sub(plug_fixed_olig=True, max_trim=100)
   body_c3_mono.trim_direction = "C"

   # kw.output_prefix = "test_plug_olig_hier"
   # kw.nout_debug = 20
   # hole.dump_pdb('ref.pdb')

   hsamp = RotCart1Hier_f4(-120, 120, 20, 0, 120, 12, [0, 0, 1])
   result = make_plugs(body_c3_mono, hole, hscore, rp.hier_search, hsamp, **kw)

   # rp.dump(result, 'rpxdock/data/testdata/test_plug_olig_hier.pickle')
   ref = rp.data.get_test_data('test_plug_olig_hier')
   rp.search.assert_results_close(result, ref)

def test_plug_olig_grid(hscore, body_c3_mono, hole):
   kw = testarg().sub(plug_fixed_olig=True, max_trim=100)

   # kw.output_prefix = "test_plug_olig_grid"
   # kw.nout_debug = 10
   # hole.dump_pdb('ref.pdb')

   # should match hsamp resl4 grid
   gcart = np.linspace(-119.625, 119.625, 20 * 16)
   gang = np.linspace(0.3125, 119.6875, 12 * 16)
   xgrid = grid_sym_axis(gcart, gang)

   # result = make_plugs(plug, hole, hscore, rp.grid_search, xgrid, **kw)
   result = make_plugs(body_c3_mono, hole, hscore, rp.grid_search, xgrid, **kw)

   # result.dump_pdbs_top_score(10)

   rp.dump(result, 'test_plug_olig_grid.pickle')
   #ref = rp.data.get_test_data('test_plug_olig_grid')
   #rp.search.assert_results_close(result, ref)

if __name__ == "__main__":
   # plug = rp.Body(rp.data.datadir + '/pdb/dhr64.pdb')
   # rp.dump(plug, rp.data.datadir + '/body/dhr64.pickle')
   # c3m = rp.Body(rp.data.datadir + '/pdb/test_c3_mono.pdb')
   # rp.dump(c3m, rp.data.datadir + '/body/test_c3_mono.pickle')
   # hole = rp.Body(rp.data.datadir + '/pdb/small_c3_hole.pdb', sym=3)
   # rp.dump(hole, rp.data.datadir + '/body/small_c3_hole.pickle')

   #hole = rp.data.get_body('/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/cage/i523_z.pdb')
   #plug = rp.data.get_body('/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/plug/C3_HFuse-pH192-3_0046_chA.pdb')
   # hole = rp.Body(
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/cage/i523_z.pdb', sym=3)
   # rp.dump(
   #    hole,
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/cage/i523_z.pickle')
   # hole = rp.data.get_body(
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/cage/i523_z')

   # plug = rp.Body(
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/plug/C3_HFuse-pH192-3_0046_chA.pdb'
   # )
   # rp.dump(
   #    plug,
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/plug/C3_HFuse-pH192-3_0046_chA.pickle'
   # )
   # plug = rp.data.get_body(
   #    '/home/erinyang/projects/ph_plugs/20200427_rpxdock/input/scaffolds/plug/C3_HFuse-pH192-3_0046_chA'   )
   #body_c3_mono = rp.data.get_body('test_c3_mono')

   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ailv_h', hscore_data_dir='/home/erinyang/hscore')
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')

   hole = rp.data.get_body('small_c3_hole_sym3')
   plug = rp.data.get_body('test_c3_mono')

   # hole.dump_pdb('ref.pdb', use_body_sym=True)
   # test_plug_hier(hscore, plug, hole)
   # test_plug_hier_trim(hscore, plug, hole)
   # test_plug_olig_hier(hscore, body_c3_mono, hole)
   test_plug_olig_grid(hscore, plug, hole)
