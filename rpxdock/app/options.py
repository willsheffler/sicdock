import sys, os, argparse, re, functools, logging, glob
import numpy as np, rpxdock as rp, functools as ft

log = logging.getLogger(__name__)

_iface_summary_methods = dict(min=np.min, sum=np.sum, median=np.median, mean=np.mean, max=np.max)

def print_options(kw):

   ll = logging.getLogger().level

   print(f'{" COMMAND LINE ":=^80}')
   print('APP:', sys.argv[0], end='')
   for k in sys.argv[1:]:
      if k.startswith('--'): print('\n   ', k, end=' ')
      else: print(k, end=' ')
   print()

   print(f'{" VERSION ":=^80}')
   print('    Date of commit:', rp.util.gitcommit.date)
   print('    Branch:', rp.util.gitcommit.branch)
   print('    Pevious Commit:', rp.util.gitcommit.prev_commit)
   print('    NOTE: Current commit isn\'t possible to record in the code... kinda a')
   print('          chicken/egg problem. You must find it based on the previous commit!')

   print(f'{" SETTINGS ":=^80}')
   maxlen = max(len(_) for _ in kw)
   for k in sorted(kw):
      print('   ', k, '.' * (maxlen - len(k)), kw[k])

   print(f'{" SETTINGS EXTRA INFO ":=^80}')
   for k, v in kw.items():
      vstr = str(v)
      if (vstr.startswith('<') and vstr.count(' at 0x') and vstr.endswith('>')
          and hasattr(v, '__doc__')):
         print(f'    {f" EXTRA INFO ABOUT: {k} ":=^76s}')
         print('   ', k, '.' * (maxlen - len(k)), v)
         doc = v.__doc__

         trunc = [
            _re_find_position('\n.*Parameters', doc) if 25 <= ll else 99999,
            _re_find_position('\n.*See Also', doc) if 15 < ll < 25 else 99999,
         ]
         print(doc[:min(trunc)])
   print(f'{" END SETTINGS EXTRA INFO ":=^80}')

def _re_find_position(pattern, string):
   match = re.search(pattern, string)
   if match:
      return match.start(0)
   return len(string)

def str2bool(v):
   if isinstance(v, (list, tuple)):
      return [str2bool(_) for _ in v]
   if isinstance(v, bool):
      return v
   if v.lower() in ('yes', 'true', 't', 'y', '1'):
      return True
   elif v.lower() in ('no', 'false', 'f', 'n', '0'):
      return False
   else:
      raise argparse.ArgumentTypeError(
         ('Boolean value expected, not "%s" of type %s\n' % (v, type(v)) +
          'Allowed True Vals  (case independant): yes true ty 1\n' +
          'Allowed False Vals (case independant): no false f n 0'))

def parse_list_of_strtuple(s):
   if isinstance(s, list):
      s = ",".join("(%s)" % a for a in s)
   kw = eval(s)
   if isinstance(kw, tuple) and len(kw) == 2 and isinstance(kw[0], int):
      kw = [kw]
   return kw

def add_argument_unless_exists(parser, *arg, **kw):
   if not (arg or kw):
      return functools.partial(add_argument_unless_exists, parser)
   try:
      parser.add_argument(*arg, **kw)
   except argparse.ArgumentError:
      pass

def default_cli_parser(parent=None, **kw):
   parser = parent if parent else argparse.ArgumentParser(allow_abbrev=False)
   addarg = add_argument_unless_exists(parser)
   addarg("--inputs", nargs="*", type=str, default=[], help=argparse.SUPPRESS)
   addarg(
      "--inputs1", nargs="*", type=str, default=[],
      help='input structures for single component protocols or first component for 2+ protocols, plug input structure for plug protocol. Can be inputted as a string or list of strings'
   )
   addarg(
      "--inputs2", nargs="*", type=str, default=[],
      help='input structures for second component for 2+ component protocols, hole input structure for plug protocol. Can be inputted as a string or list of strings'
   )
   addarg(
      "--inputs3", nargs="*", type=str, default=[],
      help='input structures for third component for 3+ component protocols. Can be inputted as a string or list of strings'
   )
   addarg("--allowed_residues", nargs="*", type=str, default=[], help=argparse.SUPPRESS)
   addarg(
      "--allowed_residues1", nargs="*", type=str, default=[],
      help='allowed residues list for single component protocols or first component of 2+ component protocols or the monomeric plug for plug protocol. Takes either nothing (if you leave them out), a single file which applies to all the corresponding inputs, or a list of files which must have the same length as the list of inputs. The files themselves must contain a whitespace separated list of either numbers or ranges.'
   )
   addarg(
      "--allowed_residues2", nargs="*", type=str, default=[],
      help='allowed residues list for second component for 2+ component protocols or the hole for the plug protocol. Takes either nothing (if you leave them out), a single file which applies to all the corresponding inputs, or a list of files which must have the same length as the list of inputs. The files themselves must contain a whitespace separated list of either numbers or ranges.'
   )
   addarg(
      "--allowed_residues3", nargs="*", type=str, default=[],
      help='allowed residues for third component for 3+ component protocols. Takes either nothing (if you leave them out), a single file which applies to all the corresponding inputs, or a list of files which must have the same length as the list of inputs. The files themselves must contain a whitespace separated list of either numbers or ranges.'
   )
   addarg(
      "--ncpu", type=int, default=rp.util.cpu_count(),
      help='number of cpu cores available. defaults to all cores or cores available according to slurm allocation'
   )
   addarg(
      "--nthread", type=int, default=0,
      help='number of threads to use in threaded protocols, default single thread and/or ncpu for some things'
   )
   addarg(
      "--nprocess", type=int, default=0,
      help='number of processes to use for multiprocess protocols, defaults to ncpu most of the time'
   )
   addarg("--trial_run", action="store_true", default=False,
          help='reduce runtime by using minimal samples, smaller score files, etc.')
   addarg(
      "--hscore_files", nargs="+", default=['ilv_h'],
      help='rpx score files using in scoring for most protocols. defaults to pairs involving only ILV and only in helices. Can be only a path-suffix, which will be appended to --hscore_data_dir. Can be a list of files. Score files with various parameters can be generated with rpxdock/app/genrate_motif_scores.py.'
   )
   addarg(
      "--hscore_data_dir", default='/home/sheffler/data/rpx/hscore',
      help='default path to search for hcores_files. defaults to /home/sheffler/data/rpx/hscore')
   addarg(
      "--max_trim", type=int, default=0,
      help='maximum allowed trimming of residues from docking components. specifying 0 will completely disable trimming, and may allow significantly shorter runtimes. defaults to 0.'
   )
   addarg(
      "--trim_direction", type=str, default="NC",
      help='For variable length protocols, if --max_trim > 0, allow trimming from only N or C. If this is not specified, trimming may be allowed from N or C directions, though some protocols trim from only one direction as appropriate. (maybe)'
   )
   addarg(
      "--max_pair_dist", type=float, default=8.0,
      help="maxium distance between centroids for a pair of residues do be considered interacting. In hierarchical protocols, coarser stages will add appropriate amounts to this distance. defaults to 8.0"
   )
   # addarg("--use_fixed_mindis", action='store_true', default=False)
   addarg("--debug", action="store_true", default=False,
          help='Enable potentially expensive debugging checks')
   addarg(
      "--nout_debug", type=int, default=0,
      help='Specify number of pdb outputs for individual protocols to output for each search. This is not the preferred way to get pdb outputs, use --nout_top and --nout_each unless you have a reason not to. defaults to 0'
   )
   addarg(
      "--nout_top", type=int, default=10,
      help='total number of top scoring output structures across all docks. only happens of --dump_pdbs is also specified. defaults to 10'
   )
   addarg(
      "--nout_each", type=int, default=1,
      help='number of top scoring output structurs for each individual dock. only happens if --dump_pdbs is also specfied. defaults to 1'
   )
   addarg("--dump_pdbs", action='store_true', default=False, help='activate output of pdb files.')
   addarg("--output_asym_only", action='store_true', default=False,
          help="dump only asym unit to pdbs")
   addarg("--suppress_dump_results", action='store_true', default=False,
          help="suppress the output of results files")
   addarg(
      "--nresl", type=int, default=None,
      help="number of hierarchical stages to do for hierarchical searches. probably use only for debugging, default is to do all stages"
   )
   addarg("--clashdis", type=float, default=3.5,
          help='minimum distance allowed between heavy atoms. default 3.5')
   addarg(
      "--beam_size", type=int, default=100000,
      help='Maximum number of samples for each stage of a hierarchical search protocol (except the first, coarsest stage, which must sample all available positions. This is the most important parameter for determining rumtime (aside from number of allowed residues list). defaults to 100,000'
   )
   addarg(
      "--max_bb_redundancy", type=float, default=3.0,
      help='mimimum distance between outputs from a single docking run. is more-or-less a non-aligned backbone RMSD. defaults to 3.0'
   )
   addarg(
      "--max_cluster", type=int, default=0,
      help='maximum numer of results to cluster (filter redundancy via max_bb_redundancy) for each dock. defaults to no limit'
   )
   addarg(
      "--max_longaxis_dot_z", type=float, default=1.000001,
      help='maximum dot product of longest input axis (as determined by PCA) with the main symmetry axis. aka the cosine of the angle between the two axes. Can be used to force cyclic oligomers / plugs/ etc to lay flat. defaults to no constraint (1.0)'
   )
   addarg(
      "--max_delta_h", type=float, default=9999,
      help='maximum difference between cartesian component offsets for multicomponent symmetry axis aligned docking like cages and layers.'
   )
   addarg(
      "--iface_summary", default="min",
      help='method to use for summarizing multiple created interface into a single score. For example, a three component cage could have 3 interfaces A/B B/C and C/A, or a monomer-based cage plug will have an oligomer interface and an oligomer / cage interface. default is min. e.g. to take the overall score as the worst of the multiple interfaces'
   )
   addarg("--weight_rpx", type=float, default=1.0,
          help='score weight of the main RPX score component. defaults to 1.0')
   addarg(
      "--weight_ncontact", type=float, default=0.01,
      help='score weight of each contact (pair of centroids within --max_pair_dist). defaults to 0.01'
   )
   addarg(
      "--weight_plug", type=float, default=1.0,
      help='Only for monomer-to-plug docking. score weight of plug oligomer interface. defaults to 1.0'
   )
   addarg(
      "--weight_hole", type=float, default=1.0,
      help='Only for monomer-to-plug docking. score weight of plug / cage hole interface. defaults to 1.0'
   )

   addarg("--weight_sasa", type=float, default=1152,
          help="Desired SASA used to weight dock scoring for sasa_priority scorefunction")
   addarg(
      "--weight_error", type=float, default=4,
      help="Standard deviation used to calculate the distribution of SASA weighting for sasa_priority scorefunction"
   )

   addarg(
      "--output_prefix", nargs="?", default="rpxdock", type=str,
      help="output file prefix. will output pickles for a base ResPairScore plus --hierarchy_depth hier XMaps"
   )
   addarg(
      "--dont_store_body_in_results", action="store_true", default=False,
      help='reduce result output size and maybe runtime by not including structure information in results objects. will not be able to rescore or output pdbs from results objects.'
   )
   addarg("--loglevel", default='INFO',
          help='select log level from CRITICAL, ERROR, WARNING, INFO or DEBUG. defaults to INFO')
   addarg(
      "--score_only_ss", default='EHL',
      help="only consider residues of the specified secondary structure type when scoring. defaults to all (EHL)"
   )
   addarg(
      "--score_only_aa", default='ANYAA',
      help='only consider residues of the specified type when generating score files. Not currently supported in all protocols. defaults to ANYAA'
   )
   addarg(
      "--score_only_sspair", default=[], nargs="+",
      help="only consider pairs with the specified secondary structure types when scoring. may not work in all protocols. defaults to no constraint on scoring."
   )

   addarg(
      "--docking_method", default='hier',
      help='search method to use in docking. available methods may include "hier" for hierarchical search (probably best) "grid" for a flat grid search and "slide" for a lower dimension grid search using slide moves. Not all options available for all protocols (grid is not available for multicomp docking). defaults to "hier"'
   )
   addarg(
      "--cart_bounds", default=[], type=float, nargs='+',
      help='cartesian bounds for various protocols. should probably be replaced with some more general mechanism. no default as protocols specific'
   )
   addarg(
      "--cart_resl", default=10.0, type=float,
      help='resolution of top level cartesian, sometimes ignored, and resl is taken from hscore data instead. default 10'
   )
   addarg(
      "--ori_resl", default=30.0, type=float,
      help='resolution of top level orientation, sometimes ignored, and resl is taken from hscore data instead. default 30'
   )
   addarg("--grid_resolution_cart_angstroms", type=float, default=1,
          help='cartesian resolution in Angstroms during grid search. default 1')
   addarg("--grid_resolution_ori_degrees", type=float, default=1,
          help='rotation orientation resolution in degrees during grid search. default 1')
   # tcdock
   addarg(
      "--architecture", type=str, default=None,
      help='architecture to be produced by docking. Can be cage I32, O43, T32 where larger axis of symmetry is listed first, Cx for cyclic, Dx_y for dihedral, where x is the dihedral symmetry and y is the symmetry of the scaffold, y=2 or y=x. For plug protocol, can be PLUG_Cx. No default value'
   )
   addarg("--trimmable_components", default="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
          help='specify which components "ABC" etc are trimmable. defaults to all components')
   addarg(
      "--flip_components", nargs='+', default=[True], type=str2bool,
      help='list of boolean value or values specifying if and which components should be allowed to flip in axis aligned docking protocols. Defaults to flipping all components'
   )
   addarg(
      "--fixed_rot", nargs='+', type=int, default=[],
      help='list of components (0,1,2 etc) which should be fixed from rotating in hierarchical docking'
   )
   addarg(
      "--fixed_trans", nargs='+', type=int, default=[],
      help='list of components (0,1,2 etc) which should be fixed from translating in hierarchical docking'
   )
   addarg(
      "--fixed_components", nargs='+', type=int, default=[],
      help='list of components (0,1,2 etc) which should be fixed from rotating *and* translating in hierarchical docking'
   )
   addarg(
      "--fixed_wiggle", nargs='+', type=int, default=[],
      help='Similar to fixed_components (input as list 0,1,2,etc) but allows user-inputted translation and rotation wiggling about orientation axis in hierarchical docking'
   )
   addarg("--fw_cartlb", default=-5.0, type=float,
          help='Lower bound for fixed_wiggle translation (in Angstroms) Default 5.0')
   addarg("--fw_cartub", default=5.0, type=float,
          help='Upper bound for fixed_wiggle translation (in Angstroms) Default 5.0')
   addarg("--fw_rotlb", default=-5.0, type=float,
          help='Lower bound for fixed_wiggle rotation (in degrees) Default 5.0')
   addarg("--fw_rotub", default=5.0, type=float,
          help='Upper bound for fixed_wiggle rotation (in degrees) Default 5.0')
   addarg("--use_orig_coords", action='store_true', default=False,
          help='remember and output the original sidechains from the input structures')
   addarg("--primary_iface_cut", default=None, help='score cut for helix primary interface')
   addarg("--symframe_num_helix_repeats", default=10,
          help='number of helix repeat frames to dump. Default 10')
   addarg("--ignored_aas", default='CGP', help='Amino acids to ignore in scoring. Default CGP')
   addarg("--score_self", action='store_true', default=False,
          help='score each interface seperately and dump in output pickle')

   addarg(
      "--function", type=str, default='stnd',
      help='score function to use for scoring. Default is stnd scorefunction. Example: stnd, sasa_priority, mean, exp, median. Full list is defined in score/scorefunctions.py'
   )
   addarg("--sscount_filter", action='store_true', default=False,
          help='calculate the ss_count in the interface')
   addarg(
      "--sscount_confidence", action='store_true', default=False,
      help='If sscount_confidence is set, docks below the threshold number of ss elements in the interface will not be included in the output'
   )
   addarg("--sscount_min_helix_length", default=4, type=int,
          help='Min resis in helix to count as ss element. default 4')
   addarg("--sscount_min_sheet_length", default=3, type=int,
          help='Min resis in sheet to count as ss element. default 3')
   addarg("--sscount_min_loop_length", default=1, type=int,
          help='Min resis in loop to count as ss element. default 1')
   addarg("--sscount_max_dist", default=8, type=float,
          help='Min resis in loop to count as ss element. default 1')
   addarg("--sscount_min_element_resis", default=3, type=int,
          help='Min interface resis in ss_element to include in ss count. default 3')
   addarg("--sscount_sstype", default="EHL", type=str,
          help='Types of secondary structure to include in count. defaults to all (EHL)')
   addarg(
      "--sscount_min_ss_count", default=3, type=int,
      help='If sscount_confidence set, minimum number of ss elements to pass the filter. default 3'
   )
   addarg(
      "--sscount_strict", action='store_true', default=False,
      help='Require that both pairs of residues in the interface are in an SS element meeting the set criteria'
   )

   addarg("--helix_trim_max", default=0,
          help='Allow trimming of N helices from N or C term as specified by --trim_direction')

   addarg("--helix_trim_nres_ignore_end", default=1, help='Trim N extra residues off helix end')

   parser.has_rpxdock_args = True
   return parser

def get_cli_args(argv=None, parent=None, process_args=True, **kw):
   parser = default_cli_parser(parent, **kw)
   argv = sys.argv[1:] if argv is None else argv
   argv = make_argv_with_atfiles(argv, **kw)
   options = rp.Bunch(parser.parse_args(argv))
   if process_args: options = process_cli_args(options, **kw)
   return options

def defaults(**kw):
   return get_cli_args([], **kw)

def set_loglevel(loglevel):
   try:
      numeric_level = int(loglevel)
   except ValueError:
      numeric_level = getattr(logging, loglevel.upper(), None)
   if not isinstance(numeric_level, int):
      raise ValueError('Invalid log level: %s' % loglevel)
   logging.getLogger().setLevel(level=numeric_level)
   log.info(f'set loglevel to {numeric_level}')

def process_cli_args(options, **kw):
   options = rp.Bunch(options)
   kw = rp.Bunch(kw)

   options = _process_inputs(options, **kw)

   options.iface_summary = _iface_summary_methods[options.iface_summary]

   _extract_weights(options)
   _extract_sscount(options)

   set_loglevel(options.loglevel)

   options.score_only_aa = options.score_only_aa.upper()
   options.score_only_ss = options.score_only_ss.upper()

   d = os.path.dirname(options.output_prefix)
   if d: os.makedirs(d, exist_ok=True)

   _process_arg_sspair(options)
   options.trim_direction = options.trim_direction.upper()

   if options.architecture:
      options.architecture = options.architecture.upper()

   if not kw.dont_set_default_cart_bounds:
      options.cart_bounds = _process_cart_bounds(options.cart_bounds)

   options.trimmable_components = options.trimmable_components.upper()

   log.info(str(options))

   return options

def _process_inputs(opt, read_allowed_res_files=True, **kw):

   if opt.inputs:
      msg = "--inputs%i cant be used if --inputs is specified"
      assert not opt.inputs1, msg % 1
      assert not opt.inputs2, msg % 2
      assert not opt.inputs3, msg % 3
   if not opt.inputs1:
      msg = "--inputs%i can only be used if --inputs1 is specified"
      assert not opt.inputs2, msg % 2
      assert not opt.inputs3, msg % 3
   if not opt.inputs2:
      msg = "--inputs%i can only be used if --inputs2 is specified"
      assert not opt.inputs3, msg % 3

   msg = 'allowed_residues must be a single file or match number of inputs'
   assert len(opt.allowed_residues) in (0, 1, len(opt.inputs)), msg
   msg = 'allowed_residues1 must be a single file or match number of inputs1'
   assert len(opt.allowed_residues1) in (0, 1, len(opt.inputs1)), msg
   msg = 'allowed_residues2 must be a single file or match number of inputs2'
   assert len(opt.allowed_residues2) in (0, 1, len(opt.inputs2)), msg
   msg = 'allowed_residues3 must be a single file or match number of inputs3'
   assert len(opt.allowed_residues3) in (0, 1, len(opt.inputs3)), msg

   if not opt.inputs:
      msg = '--allowed_residues cant be used if --inputs not used'
      assert len(opt.allowed_residues) is 0, msg
   if not opt.inputs1:
      msg = '--allowed_residues1 cant be used if --inputs1 not used'
      assert len(opt.allowed_residues1) is 0, msg
   if not opt.inputs2:
      msg = '--allowed_residues2 cant be used if --inputs2 not used'
      assert len(opt.allowed_residues2) is 0, msg
   if not opt.inputs3:
      msg = '--allowed_residues3 cant be used if --inputs3 not used'
      assert len(opt.allowed_residues3) is 0, msg

   if len(opt.allowed_residues) is 1: opt.allowed_residues *= len(opt.inputs)
   if len(opt.allowed_residues1) is 1: opt.allowed_residues1 *= len(opt.inputs1)
   if len(opt.allowed_residues2) is 1: opt.allowed_residues2 *= len(opt.inputs2)
   if len(opt.allowed_residues3) is 1: opt.allowed_residues3 *= len(opt.inputs3)

   if len(opt.allowed_residues) is 0: opt.allowed_residues = [None] * len(opt.inputs)
   if len(opt.allowed_residues1) is 0: opt.allowed_residues1 = [None] * len(opt.inputs1)
   if len(opt.allowed_residues2) is 0: opt.allowed_residues2 = [None] * len(opt.inputs2)
   if len(opt.allowed_residues3) is 0: opt.allowed_residues3 = [None] * len(opt.inputs3)

   if read_allowed_res_files:
      opt.allowed_residues = [_read_allowed_res_file(_) for _ in opt.allowed_residues]
      opt.allowed_residues1 = [_read_allowed_res_file(_) for _ in opt.allowed_residues1]
      opt.allowed_residues2 = [_read_allowed_res_file(_) for _ in opt.allowed_residues2]
      opt.allowed_residues3 = [_read_allowed_res_file(_) for _ in opt.allowed_residues3]

   if not opt.inputs:
      if opt.inputs1:
         opt.inputs.append(opt.inputs1)
         opt.allowed_residues.append(opt.allowed_residues1)
         if opt.inputs2:
            opt.inputs.append(opt.inputs2)
            opt.allowed_residues.append(opt.allowed_residues2)
            if opt.inputs3:
               opt.inputs.append(opt.inputs3)
               opt.allowed_residues.append(opt.allowed_residues3)

   return opt

class DefaultResidueSelector:
   def __init__(self, spec):
      static = set()
      dynamic = list()
      for r in spec.split():
         if r.count(':'):
            lb, ub = [int(x) for x in r.split(':')]
            if lb < 0 or ub < 0:
               dynamic.append((lb, ub))
            else:
               for i in range(lb, ub + 1):
                  static.add(i)
         else:
            static.add(int(r))
      self.static = static
      self.dynamic = dynamic

   def __call__(self, body, **kw):
      residues = {r for r in self.static if r <= len(body)}
      for (lb, ub) in self.dynamic:
         if lb < 0: lb = len(body) + 1 + lb
         if ub < 0: ub = len(body) + 1 + ub
         for i in range(max(1, lb), min(len(body), ub) + 1):
            residues.add(i)
      return residues

# def _default_residue_selector(spec):
#    static = set()
#    dynamic = list()
#    for r in spec.split():
#       if r.count(':'):
#          lb, ub = [int(x) for x in r.split(':')]
#          if lb < 0 or ub < 0:
#             dynamic.append((lb, ub))
#          else:
#             for i in range(lb, ub + 1):
#                static.add(i)
#       else:
#          static.add(int(r))
#
#    def inner(body, **kw):
#       residues = {r for r in static if r <= len(body)}
#       for (lb, ub) in dynamic:
#          if lb < 0: lb = len(body) + 1 + lb
#          if ub < 0: ub = len(body) + 1 + ub
#          for i in range(max(1, lb), min(len(body), ub) + 1):
#             residues.add(i)
#       return residues
#
#    return inner

def _read_allowed_res_file(fname):
   if fname is None: return None
   with open(fname) as inp:
      return DefaultResidueSelector(inp.read())

def _process_cart_bounds(cart_bounds):
   if not cart_bounds: cart_bounds = 0, 500
   elif len(cart_bounds) is 1: cart_bounds = [0, cart_bounds[0]]
   tmp = list()
   for i in range(0, len(cart_bounds), 2):
      tmp.append(cart_bounds[i:i + 2])
   cart_bounds = tmp
   return cart_bounds

def make_argv_with_atfiles(argv=None, **kw):
   if argv is None: argv = sys.argv[1:]
   for a in argv.copy():
      if not a.startswith('@'): continue
      argv.remove(a)
      with open(a[1:]) as inp:
         newargs = []
         for l in inp:
            # last char in l is newline, so [:-1] ok
            newargs.extend(l[:l.find("#")].split())
         argv = newargs + argv
   return argv

def _extract_weights(kw):
   pref = 'weight_'
   wts = rp.Bunch()
   todel = list()
   for k in kw:
      if k.startswith(pref):
         wtype = k.replace(pref, '')
         wts[wtype] = kw[k]
         todel.append(k)
   for k in todel:
      del kw[k]
   kw.wts = wts

def _extract_sscount(kw):
   pref = 'sscount_'
   ssc = rp.Bunch()
   todel = list()
   for k in kw:
      if k.startswith(pref):
         ssc_type = k.replace(pref, '')
         ssc[ssc_type] = kw[k]
         todel.append(k)
   for k in todel:
      del kw[k]
   kw.ssc = ssc

def _process_arg_sspair(kw):
   kw.score_only_sspair = [''.join(sorted(p)) for p in kw.score_only_sspair]
   kw.score_only_sspair = sorted(set(kw.score_only_sspair))
   if any(len(p) != 2 for p in kw.score_only_sspair):
      raise argparse.ArgumentError(None, '--score_only_sspair accepts two letter SS pairs')
   if (any(p[0] not in "EHL" for p in kw.score_only_sspair)
       or any(p[1] not in "EHL" for p in kw.score_only_sspair)):
      raise argparse.ArgumentError(None, '--score_only_sspair accepts only EHL')
