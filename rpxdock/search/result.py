import copy, logging
from collections import OrderedDict, abc, defaultdict
import numpy as np, xarray as xr, rpxdock as rp
from rpxdock.util import sanitize_for_pickle, num_digits

log = logging.getLogger(__name__)

class Result:
   def __init__(self, data_or_file=None, body_=[], body_label_=None, **kw):
      if isinstance(body_, rp.Body): body_ = [body_]
      self.bodies = [body_]
      self.body_label_ = body_label_ if body_label_ else ['body%i' % i for i in range(len(body_))]
      self.pdb_extra = None
      if len(self.body_label_) != len(body_):
         raise ValueError('body_label_ must match number of bodies')
      if data_or_file:
         assert len(kw) is 0
         if isinstance(data_or_file, xr.Dataset):
            self.data = data_or_file
         else:
            self.load(file_)
      else:
         attrs = OrderedDict(kw['attrs']) if 'attrs' in kw else None
         if attrs: del kw['attrs']
         attrs = sanitize_for_pickle(attrs)
         self.data = xr.Dataset(dict(**kw), attrs=attrs)
      # b/c I always mistype these
      self.dump_pdb_top_score = self.dump_pdbs_top_score
      self.dump_pdb_top_score_each = self.dump_pdbs_top_score_each

   def sortby(self, *args, **kw):
      r = copy.copy(self)
      r.data = self.data.sortby(*args, **kw)
      return r

   def __getattr__(self, name):
      if name == "data":
         raise AttributeError
      return getattr(self.data, name)

   def __getitem__(self, name):
      return self.data[name]

   def __setitem__(self, name, val):
      self.data[name] = val

   def __str__(self):
      return "Result with data = " + str(self.data).replace("\n", "\n  ")

   def copy(self):
      return Result(self.data.copy())

   def getstate(self):
      return self.data.to_dict()

   def setstate(self, state):
      self.data = xr.Dataset.from_dict(state)

   def sel(self, *args, **kw):
      r = copy.copy(self)
      r.data = r.data.sel(*args, **kw)
      return r

   def dump_pdbs_top_score(self, nout_top=10, **kw):
      best = np.argsort(-self.scores)
      return self.dump_pdbs(best[:nout_top], lbl='top', **kw)

   def top_each(self, neach=1):
      which = dict()
      for ijob, imodel in self.scores.groupby(self.ijob).groups.items():
         imodel = np.array(imodel)
         ibest = np.argsort(-self.scores.data[imodel])
         which[ijob] = imodel[ibest[:neach]]
      return which

   def dump_pdbs_top_score_each(self, nout_each=1, **kw):
      if nout_each is 0: return
      which = self.top_each(nout_each)
      if not len(which):
         raise ValueError("can't dump pdbs, no results available")
      ndigijob = num_digits(len(which) - 1)
      dumped = set()
      ndigmdl = max(np.max(num_digits(v)) for v in which.values())
      for ijob, imodel in which.items():
         dumped |= self.dump_pdbs(imodel, lbl=f'job{ijob:0{ndigijob}}_top', ndigmdl=ndigmdl, **kw)
      return dumped

   def dump_pdbs(self, which, ndigwhich=None, ndigmdl=None, lbl='', skip=[], output_prefix='rpx',
                 **kw):
      if len(which) is 0: return set()
      if isinstance(which, abc.Mapping):
         raise ValueError('dump_pdbs takes sequence not mapping')
      if 'fname' in kw and kw['fname'] is not None:
         raise ArgumentError('fname is not a valid argument for dump_pdbs, because multiple files')
      if not output_prefix and 'output_prefix' in self.attrs:
         output_prefix = self.output_prefix
      if ndigwhich is None: ndigwhich = num_digits(len(which) - 1)
      if ndigmdl is None: ndigmdl = num_digits(max(which))

      dumped = set()
      for i, imodel in enumerate(which):
         assert not isinstance(imodel, np.ndarray) or len(imodel) == 1
         if not imodel in skip:
            dumped.add(int(imodel))
            prefix_tmp = f'{output_prefix}_{lbl}{i:0{ndigwhich}}_{int(imodel):0{ndigmdl}}_'
            self.dump_pdb(imodel, output_prefix=prefix_tmp, **kw)
      return dumped

   def dump_pdb(self, imodel, output_prefix='', output_suffix='', fname=None, output_body='ALL',
                sym='', sep='_', skip=[], hscore=None, output_asym_only=False, **kw):
      if not sym and 'sym' in self.attrs: sym = self.attrs['sym']
      if not sym and 'sym' in self.data: sym = self.data.sym.data[imodel]
      sym = sym if sym else "C1"
      if not output_prefix and 'output_prefix' in self.attrs:
         output_prefix = self.output_prefix
      bod = self.bodies[0]
      if 'ijob' in self.data: bod = self.bodies[self.ijob[imodel].values]
      multipos = self.xforms.ndim == 4
      if multipos and self.xforms.shape[1] != len(bod):
         raise ValueError("number of positions doesn't match number of bodies")
      if str(output_body).upper() == 'ALL': output_body = list(range(len(bod)))
      if isinstance(output_body, int): output_body = [output_body]
      if not all(w < len(bod) for w in output_body):
         raise ValueError(f'output_body ouf of bounds {output_body}')
      bod = [bod[i] for i in output_body]
      bodlab = None
      if self.xforms.ndim == 4:
         for x, b in zip(self.xforms[imodel], bod):
            b.move_to(x.data)
      else:
         bod[0].move_to(self.xforms[imodel].data)
      if not fname:
         output_prefix = output_prefix + sep if output_prefix else ''
         body_names = [b.label for b in bod]
         if len(output_body) > 1 and self.body_label_:
            bodlab = [self.body_label_[i] for i in output_body]
            body_names = [bl + '_' + lbl for bl, lbl in zip(bodlab, body_names)]
         middle = '__'.join(body_names)
         output_suffix = sep + output_suffix if output_suffix else ''
         fname = output_prefix + middle + output_suffix + '.pdb'
      log.info(f'dumping pdb {fname} score {self.scores.data[imodel]}')
      bfactor = None
      # hscore scores residue pairs and puts bfactor in pdb
      if hscore and len(bod) == 2:
         sm = hscore.score_matrix_inter(
            bod[0],
            bod[1],
            symframes=rp.geom.symframes(sym, pos=self.xforms.data[imodel], **kw),
            wts=kw['wts'],
         )
         bfactor = [sm.sum(axis=1), sm.sum(axis=0)]
      bounds = np.tile([[-9e9], [9e9]], len(bod)).T
      if 'reslb' in self.data and 'resub' in self.data:
         bounds = np.stack([self.reslb[imodel], self.resub[imodel]], axis=-1)
      symframes = rp.geom.symframes(sym, pos=self.xforms.data[imodel], **kw)
      if output_asym_only: symframes = [np.eye(4)]
      rp.io.dump_pdb_from_bodies(fname, bod, symframes=symframes, resbounds=bounds,
                                 bfactor=bfactor, **kw)
      if self.pdb_extra is not None:
         with open(fname, 'a') as out:
            out.write(self.pdb_extra[int(imodel)])
      if hasattr(self.data, 'helix_n_to_primary'):
         symframes = symframes[np.array(
            [0, self.data.helix_n_to_primary[imodel], self.data.helix_n_to_secondry[imodel]])]
         rp.io.dump_pdb_from_bodies(fname + '_hbase.pdb', bod, symframes=symframes,
                                    resbounds=bounds, bfactor=bfactor, **kw)
         # assert 0, 'testing helix dump'

   def __len__(self):
      return len(self.model)

   @property
   def ndocks(self):
      return len(self.dockinfo)

   def __eq__(self, other):
      return self.data.equals(other.data)

def dict_coherent_entries(alldicts):
   sets = defaultdict(set)
   badkeys = set()
   for d in alldicts:
      for k, v in d.items():
         try:
            sets[k].add(v)
         except:
            badkeys.add(k)
   return {k: v.pop() for k, v in sets.items() if len(v) is 1 and not k in badkeys}

def concat_results(results, **kw):
   if isinstance(results, Result): results = [results]
   assert len(results) > 0
   ijob = np.repeat(np.arange(len(results)), [len(r) for r in results])
   assert max(len(r.bodies) for r in results) == 1
   assert all(r.body_label_ == results[0].body_label_ for r in results)
   allattrs = [r.attrs for r in results]
   common = dict_coherent_entries(allattrs)
   r = Result(xr.concat([r.data for r in results], dim='model', **kw))
   r.bodies = [r.bodies[0] for r in results]
   r.data['ijob'] = (['model'], ijob)
   r.data.attrs = OrderedDict(dockinfo=allattrs, **common)
   r.body_label_ = results[0].body_label_
   if results[0].pdb_extra is not None:
      r.pdb_extra = list()
      for x in results:
         assert len(x.pdb_extra) == len(x.data.scores)
         r.pdb_extra.extend(x.pdb_extra)
   return r

def dummy_result(size=1000):
   from rpxdock.homog import rand_xform
   return Result(
      ijob=(['model'], np.repeat([3, 1, 2, 4, 0], size / 5).astype('i8')),
      scores=(["model"], np.random.rand(size).astype('f4')),
      xforms=(["model", "hrow", "hcol"], rand_xform(size).astype('f4')),
      rpx_plug=(["model"], np.random.rand(size).astype('f4')),
      rpx_hole=(["model"], np.random.rand(size).astype('f4')),
      ncontact_plug=(["model"], np.random.rand(size).astype('f4')),
      ncontact_hole=(["model"], np.random.rand(size).astype('f4')),
      reslb=(["model"], np.random.randint(0, 100, size)),
      resub=(["model"], np.random.randint(100, 200, size)),
   )

def assert_results_close(r, s, n=-1):
   if set(r.keys()) != set(s.keys()):
      print(list(r.keys()))
      print(list(s.keys()))
   print(list(r.keys()))
   print(list(s.keys()))
   assert set(r.keys()) == set(s.keys()), 'results must have same fields'
   assert np.allclose(r.scores[:n], s.scores[:n])
   assert np.allclose(r.xforms[:n], s.xforms[:n], atol=1e-3)
   for k in r.data:
      assert np.allclose(r[k][:n], s[k][:n])
