import sys, os, argparse, _pickle
import numpy as np
import sicdock
from sicdock.util import Bunch
from sicdock.motif import ResPairData, make_and_dump_hier_score_tables

def parsearg(s):
   if isinstance(s, list):
      s = ",".join("(%s)" % a for a in s)
   arg = eval(s)
   if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], int):
      arg = [arg]
   return arg

def get_opts():
   parser = sicdock.options.default_cli_parser()
   paa = parser.add_argument
   H = "ResPairData file containing Xarray Dataset with pdb, residue, and pair info"
   paa("respairdat_file", type=str, help=H)
   H = "final sampling resolution"
   paa("--base_sample_resl", default=np.sqrt(2) / 2, type=float, help=H)
   H = "Xbin base translational resolution"
   paa("--base_cart_resl", default=0.7, type=float, help=H)
   H = "Xbin base orientational resoltution"
   paa("--base_ori_resl", default=10.0, type=float, help=H)
   paa("--xbin_max_cart", default=128.0, type=float, help="Xbin max traslation")
   paa("--min_ssep", default=10, type=int, help="min seq sep")
   H = "number of hierarchical tables to make"
   paa("--hierarchy_depth", default=5, type=int, help=H)
   paa("--sampling_lever", default=25, type=float)
   paa("--xhier_cart_fudge_factor", default=1.5, type=float)
   paa("--xhier_ori_fudge_factor", default=2.5, type=float)
   paa("--min_bin_score", default=1.0, type=float)
   paa("--min_pair_score", default=0.5, type=float)
   paa("--smear_params", default=["2,1", "1,1", "1,1", "1,0", "1,0"], type=str, nargs="*")
   # paa("--smear_params", default=[(2, 0)])
   paa("--smear_kernel", default="flat", type=str)
   paa("--only_do_hier", default=-1, type=int)
   paa("--allowed_aas", default="ANYAA", type=str)
   paa("--allowed_ss", default='EHL', type=str)
   paa("--use_ss_key", default=False, action='store_true')
   args = parser.parse_args()
   args.smear_params = parsearg(args.smear_params)
   args.allowed_aas = args.allowed_aas.upper()
   args.allowed_ss = args.allowed_ss.upper()
   return Bunch(args)

def main():
   args = get_opts()

   if args.respairdat_file == "TEST":
      args.respairdat_file = "/home/sheffler/debug/sicdock/respairdat/pdb_res_pair_data_si30_10_rots.pickle"
   if args.out_prefix == "auto":
      # dname = os.path.dirname(args.respairdat_file) + '/hscore'
      dname = "./"
      bname = os.path.basename(args.respairdat_file.replace(".pickle", ""))
      args.out_prefix = dname + "/" + bname + '_'
   if not os.path.exists(os.path.dirname(args.out_prefix)):
      os.mkdir(os.path.dirname(args.out_prefix))

   with open(args.respairdat_file, "rb") as inp:
      rp = ResPairData(_pickle.load(inp))

   if args.allowed_aas.lower() != 'ANYAA':
      rp = rp.subset_by_aa(args.allowed_aas, sanity_check=True)

   if set(args.allowed_ss) != set('EHL'):
      rp = rp.subset_by_ss(args.allowed_ss, sanity_check=True)
      if len(args.allowed_ss) == 1:
         args.use_ss_key = False

   files = make_and_dump_hier_score_tables(rp, **args)

   print(files)

if __name__ == "__main__":
   main()

# ============== full === min_bin_score 2 == min_pair_score 1 ===================
# 0 1 0 cart   8.50   4.25 ori  20.08  13.39 nsmr   4.0M base 118.9K xpnd    33.4
# 0 1 1 cart   8.50   2.83 ori  20.08  13.39 nsmr  13.2M base 125.0K xpnd   105.6
# 0 2 0 cart   8.50   1.70 ori  20.08  13.39 nsmr  29.9M base 128.3K xpnd   233.2
# 0 2 1 cart   8.50   1.42 ori  20.08  13.39 nsmr  63.7M base 128.9K xpnd   494.4

# 1 1 0 cart   4.50   2.25 ori  12.54   8.36 nsmr   8.1M base 130.4K xpnd    62.3
# 1 1 1 cart   4.50   1.50 ori  12.54   8.36 nsmr  27.3M base 130.5K xpnd   208.9
# 1 2 0 cart   4.50   0.90 ori  12.54   8.36 nsmr  54.8M base 131.3K xpnd   417.6
# 1 2 1 cart   4.50   0.75 ori  12.54   8.36 nsmr 121.6M base 131.4K xpnd   925.5

# 2 1 0 cart   2.50   1.25 ori   8.77   5.85 nsmr   9.8M base 131.5K xpnd    74.4
# 2 1 1 cart   2.50   0.83 ori   8.77   5.85 nsmr  33.1M base 131.9K xpnd   251.0
# 2 2 0 cart   2.50   0.50 ori   8.77   5.85 nsmr  62.6M base 132.0K xpnd   474.5
# 2 2 1 cart   2.50   0.42 ori   8.77   5.85 nsmr 141.3M base 132.2K xpnd  1068.4

# 3 1 0 cart   1.50   0.75 ori   6.89   4.59 nsmr  10.2M base 132.1K xpnd    77.3
# 3 1 1 cart   1.50   0.50 ori   6.89   4.59 nsmr  34.5M base 132.1K xpnd   261.5
# 3 2 0 cart   1.50   0.30 ori   6.89   4.59 nsmr  64.6M base 132.1K xpnd   489.1
# 3 2 1 cart   1.50   0.25 ori   6.89   4.59 nsmr 145.8M base 131.9K xpnd  1105.2

# 4 1 0 cart   1.00   0.50 ori   5.94   3.96 nsmr  10.4M base 132.2K xpnd    78.5
# 4 1 1 cart   1.00   0.33 ori   5.94   3.96 nsmr  35.1M base 132.1K xpnd   265.7
# 4 2 0 cart   1.00   0.20 ori   5.94   3.96 nsmr  65.3M base 131.7K xpnd   496.2
# 4 2 1 cart   1.00   0.17 ori   5.94   3.96 nsmr 147.8M base 131.4K xpnd  1124.4

# full Msize 153 781 1731 2201 2399
# 0 1 0 cart   8.50   4.25 ori  20.08  13.39 nsmr  12.0M base 982.1K xpnd    12.2
# 1 1 0 cart   4.50   2.25 ori  12.54   8.36 nsmr  53.7M base 1915.8K xpnd    28.1
# 2 1 0 cart   2.50   1.25 ori   8.77   5.85 nsmr 113.3M base 2096.8K xpnd    54.0
# 3 1 0 cart   1.50   0.75 ori   6.89   4.59 nsmr 144.1M base 2125.9K xpnd    67.8
# 4 1 0 cart   1.00   0.50 ori   5.94   3.96 nsmr 157.1M base 2144.4K xpnd    73.3

# full Msize 497 2729 5815 7416 8097
# 0 1 1 cart   8.50   2.83 ori  20.08  13.39 nsmr  38.8M base 1351.2K xpnd    28.7
# 1 1 1 cart   4.50   1.50 ori  12.54   8.36 nsmr 178.7M base 2017.9K xpnd    88.6
# 2 1 1 cart   2.50   0.83 ori   8.77   5.85 nsmr 380.8M base 2111.4K xpnd   180.4
# 3 1 1 cart   1.50   0.50 ori   6.89   4.59 nsmr 485.7M base 2137.2K xpnd   227.2
# 4 1 1 cart   1.00   0.33 ori   5.94   3.96 nsmr 530.3M base 2152.8K xpnd   246.3

# 0 2 0 cart   8.50   1.70 ori  20.08  13.39 nsmr 104.1M base 1729.7K xpnd    60.2
# 1 2 0 cart   4.50   0.90 ori  12.54   8.36 nsmr 439.6M base 2079.2K xpnd   211.4

# low tot resl 11.313708498984761 0.7071067811865476
# lever  25.000 resoultion  11.314 cart   8.000 ori  37.702 ori actual/request   2.056
# stage 0 samp:  8.000 15.081 score:  8.500 20.081 fugded: 12.000 37.702
# stage 1 samp:  4.000  7.540 score:  4.500 12.540 fugded:  6.000 18.851
# stage 2 samp:  2.000  3.770 score:  2.500  8.770 fugded:  3.000  9.425
# stage 3 samp:  1.000  1.885 score:  1.500  6.885 fugded:  1.500  4.713
# stage 4 samp:  0.500  0.943 score:  1.000  5.943 fugded:  0.750  2.356
# 0 2 1 cart   8.50   1.42 ori  20.08  13.39 nsmr 278.4M base 3997.1K xpnd    69.6
# 1 1 1 cart   4.50   1.50 ori  12.54   8.36 nsmr 287.5M base 4936.2K xpnd    58.2
# 2 1 0 cart   2.50   1.25 ori   8.77   5.85 nsmr 244.3M base 5615.6K xpnd    43.5
# 3 1 0 cart   1.50   0.75 ori   6.89   4.59 nsmr 371.6M base 6062.1K xpnd    61.3
# 4 1 0 cart   1.00   0.50 ori   5.94   3.96 nsmr 439.9M base 6297.9K xpnd    69.8

# =============== 1000 === min_bin_score 1 === min_pair_score 0.5 ===============
# 0 2 0 cart   8.50   1.70 ori  20.08  13.39 nsmr  52.4M base 330.6K xpnd   158.6
# 1 1 1 cart   4.50   1.50 ori  12.54   8.36 nsmr  61.7M base 360.8K xpnd   171.1
# 2 1 0 cart   2.50   1.25 ori   8.77   5.85 nsmr  26.3M base 375.6K xpnd    70.0
# 3 1 0 cart   1.50   0.75 ori   6.89   4.59 nsmr  29.1M base 387.2K xpnd    75.3
# 4 1 0 cart   1.00   0.50 ori   5.94   3.96 nsmr  30.4M base 394.7K xpnd    77.1

# ============== 1000 === min_bin_score 2.0 === min_pair_score 1.0 ==============
# 0 2 0 cart   8.50   1.70 ori  20.08  13.39 nsmr  29.9M base 128.3K xpnd   233.2
# 1 1 1 cart   4.50   1.50 ori  12.54   8.36 nsmr  27.3M base 130.5K xpnd   208.9
# 2 1 0 cart   2.50   1.25 ori   8.77   5.85 nsmr   9.8M base 131.5K xpnd    74.4
# 3 1 0 cart   1.50   0.75 ori   6.89   4.59 nsmr  10.2M base 132.1K xpnd    77.3
# 4 1 0 cart   1.00   0.50 ori   5.94   3.96 nsmr  10.4M base 132.2K xpnd    78.5
