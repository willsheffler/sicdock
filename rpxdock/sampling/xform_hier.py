import rpxdock as rp, numpy as np

class LineHier:
   def __init__(self, lb, ub, nstep, axis):
      self.hier1d = rp.sampling.CartHier1D_f4([lb], [ub], [nstep])
      self.axis = rp.homog.hnormalized(axis).squeeze()[:3]
      self.ncell = nstep
      self.dim = 1
      self.state = lb, ub, nstep, self.axis

   def get_xforms(self, resl, idx):
      ok, disp = self.hier1d.get_trans(resl, idx)
      return ok, rp.homog.htrans(self.axis * disp[:, 0, None])

   def __getstate__(self):
      return self.state

   def __setstate__(self, state):
      self.axis = state[3]
      self.hier1d = rp.sampling.CartHier1D_f4([state[0]], [state[1]], [state[2]])

def hier_axis_sampler(
   nfold,
   lb,
   ub,
   resl,
   angresl,
   axis=[0, 0, 1],
   flipax=[0, 1, 0],
   **kw,
):
   '''
   :param nfold: architecture stuff
   :param lb:
   :param ub:
   :param resl:
   :param angresl:
   :param axis: sample z axis
   :param flipax: flip subunits
   :return: "arrays of pos" to check for a given search resolution where pos are represented by matrices
   '''
   kw=rp.Bunch(kw)
   cart_nstep = int(np.ceil((ub - lb) / resl))
   ang = 360 / nfold
   ang_nstep = int(np.ceil(ang / angresl))
   samp = rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang, ang_nstep, axis[:3])
   if kw.flip_components[0]:
      flip = rp.ZeroDHier([np.eye(4), rp.homog.hrot(flipax, 180)])
      samp = rp.ProductHier(samp, flip)
   return samp

def hier_multi_axis_sampler(
   spec,
   cart_bounds=[25, 200],
   resl=10,
   angresl=10,
   flip_components=True,
   fixed_rot=[],
   fixed_components=[],
   fixed_trans=[],
   fixed_wiggle=[],
   fw_cartlb=-5,
   fw_cartub=5,
   fw_rotlb=-5,
   fw_rotub=5,
   **kw,
):
   if not (hasattr(spec, 'nfold') and hasattr(spec, 'axis') and hasattr(spec, 'xflip')):
      raise ValueError('spec must have nfold, axis and xflip')
   assert len(spec.nfold) == len(spec.axis) == len(spec.xflip)
   if isinstance(flip_components, bool):
      flip_components = [flip_components]
   if len(flip_components) is 1:
      flip_components = flip_components * len(spec.nfold)
   # for i, flip in enumerate(flip_components):
   # flip_components[i] = flip_components[i] and not spec.comp_is_dihedral[i]
   if len(cart_bounds) is 2 and isinstance(cart_bounds[0], int):
      cart_bounds = np.array([cart_bounds] * spec.num_components)
   cart_bounds = np.array(cart_bounds)
   assert len(cart_bounds) in (1, len(spec.nfold))
   cart_bounds = np.tile(cart_bounds, [8, 1])  # for wrapping / repeating

   cart_nstep = np.ceil((cart_bounds[:, 1] - cart_bounds[:, 0]) / resl).astype('i')
   assert np.all(cart_nstep > 0)

   ang = 360 / spec.nfold
   ang_nstep = np.ceil(ang / angresl).astype('i')
   assert np.all(ang_nstep > 0) 
   
   samp = []
   for i in range(len(spec.nfold)):
      if spec.comp_is_dihedral[i]:
         s = LineHier(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], spec.axis[i])
      elif i in fixed_rot: 
         s = LineHier(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], spec.axis[i])
      elif i in fixed_trans: 
         s = rp.sampling.RotHier_f4(0, ang[i], ang_nstep[i], spec.axis[i][:3]) #TODO: MDL try this
      elif i in fixed_components:
         s = rp.ZeroDHier([np.eye(4)])
      elif i in fixed_wiggle: #TODO: MDL try this
         #Samples +/- 3 angstroms along sym axis, and same value around the symaxis
         s = rp.sampling.RotCart1Hier_f4(fw_cartlb,  fw_cartub, cart_nstep[i], fw_rotlb, fw_rotub, ang_nstep[i], spec.axis[i][:3])
      else:
         s = rp.sampling.RotCart1Hier_f4(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], 0,
                                         ang[i], ang_nstep[i], spec.axis[i][:3])
      samp.append(s)

   for i, s in enumerate(samp):
      if flip_components[i]:
         samp[i] = rp.sampling.ProductHier(s, rp.ZeroDHier([np.eye(4), spec.xflip[i]]))

   if spec.type == 'layer':
      sampler = rp.sampling.LatticeHier(samp, spec.directions)
   else:
      sampler = rp.sampling.CompoundHier(*samp)

   sampler.attrs = dict(spec=spec, cart_bounds=cart_bounds, resl=resl, angresl=angresl,
                        flip_components=flip_components)
   return sampler

def hier_mirror_lattice_sampler(
   spec,
   cart_bounds=[0, 100],
   resl=10,
   angresl=10,
   flip_components=True,
   **kw,
):
   '''
   setting cartesian bounds as opposed to lower/upper bounds
   :param spec:
   :param cart_bounds:
   :param resl:
   :param angresl:
   :param flip_components:
   :param kw:
   :return:
   '''
   cart_bounds = np.array(cart_bounds)
   cart_nstep = np.ceil((cart_bounds[:, 1] - cart_bounds[:, 0]) / resl).astype('i')
   ang = 360 / spec.nfold
   ang_nstep = np.ceil(ang / angresl).astype('i')
   # sampling cell type of xtal
   sampcell = LineHier(cart_bounds[0, 0], cart_bounds[0, 1], cart_nstep[0], [1, 0, 0])
   # sampling axis of cage within "context of xtal"
   sampaxis = rp.sampling.RotCart1Hier_f4(cart_bounds[1, 0], cart_bounds[1, 1], cart_nstep[1], 0,
                                          ang[0], ang_nstep[0], [0, 0, 1])
   return rp.sampling.ProductHier(sampcell, sampaxis)
