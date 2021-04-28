from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from rpxdock import Bunch

def grid_search(sampler, evaluator, **kw):
   if (not isinstance(sampler, np.ndarray) or sampler.ndim not in (3, 4) or sampler.shape[-2:] !=
       (4, 4)):
      raise ValueError('sampler for grid_search should be array of samples')
   xforms = sampler
   scores, extra, t = evaluate_positions(evaluator, xforms, **kw)
   stats = Bunch(ntot=len(scores), neval=[(t, len(scores))])
   return xforms, scores, extra, stats

def evaluate_positions(evaluator, xforms, executor=None, **kw):
   t = perf_counter()
   if executor:
      result = evaluate_positions_executor(executor, evaluator, xforms, **kw)
      return (*result, perf_counter() - t)
   scores, extra = evaluator(xforms, **kw)
   return scores, extra, perf_counter() - t

def evaluate_positions_executor(executor, evaluator, xforms, **kw):
   assert isinstance(executor, ThreadPoolExecutor)
   nworkers = executor._max_workers
   assert nworkers > 0
   ntasks = int(len(xforms) / 10000)
   ntasks = max(nworkers, ntasks)
   if int(ntasks) <= 0: ntasks = 1
   futures = list()
   for i, x in enumerate(np.array_split(xforms, ntasks)):
      futures.append(executor.submit(evaluator, x, **kw))
      futures[-1].idx = i
   # futures = [f for f in as_completed(futures)]
   results = [f.result() for f in sorted(futures, key=lambda x: x.idx)]
   scores = np.concatenate([r[0] for r in results])
   extra = dict()
   first_scores, first_extras = results[0]
   for k in first_extras:
      if isinstance(first_extras[k], np.ndarray):
         extra[k] = np.concatenate([r[1][k] for r in results])
      elif isinstance(first_extras[k], tuple) and isinstance(first_extras[k][1], np.ndarray):
         extra[k] = first_extras[k][0], np.concatenate([r[1][k][1] for r in results])
      else:
         print(k, first_extras[k])
         assert all(r[1][k] == first_extras[k] for r in results)
         extra[k] = first_extras[k]
   return scores, extra

# def trim_atom_to_res_numbering(trim, nres, max_trim, **kw):
# trim = ((trim[0] - 1) // 5 + 1, (trim[1] + 1) // 5 - 1)  # to res numbers
# return trim_ok(trim, nres, max_trim, **kw)

def trim_ok(trim, nres, max_trim, **kw):
   ntrim = trim[0] + nres - trim[1] - 1
   # print(nres, max_trim)
   # print('foooo', ntrim, trim[0], trim[1])
   trimok = ntrim <= max_trim
   trimok &= trim[0] >= 0
   trimok &= trim[1] >= 0
   trim = (trim[0][trimok], trim[1][trimok])
   return trim, trimok
