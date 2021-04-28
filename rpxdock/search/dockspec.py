from abc import abstractmethod
import numpy as np
import rpxdock.homog as hm
from rpxdock.geom import sym

allowed_twocomp_architectures = """
D22 D32 D42 D52 D62 D72 D82
T32 T33 O32 O42 O43 I32 I52 I53
T32D T23D T33D
O32D O23D O42D O24D O43D O34D
I32D I23D I52D I54D I53D I35D
""".split()

class DockSpec:
   @property
   @abstractmethod
   def type(self):
      raise NotImplementedError

class DockSpecHelix(DockSpec):
   @property
   def type(self):
      return 'helix'

class DockSpec1CompCage(DockSpec):
   @property
   def type(self):
      return '1comp_cage'

   def __init__(self, arch):
      assert len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_')
      assert arch[:2] in "T2 T3 O2 O3 O4 I2 I3 I5 D2 D3 D4 D5 D6 D8".split()
      if arch[0] == 'D':
         self.sym = arch[:2]
         if len(arch) is 4:
            assert arch[2] == '_'
            self.nfold = int(arch[3])
            assert self.nfold in (2, int(arch[1]))
         else:
            raise ValueError(f"architecture {arch} invalid, must be Dx_y, where y=2 or y=x")
      else:
         self.sym = arch[0]
         self.nfold = int(arch[1])

      self.arch = arch
      self.num_components = 1

      self.comp_is_dihedral = [False]
      self.axis = sym.axes[self.sym][self.nfold]
      self.axis_second = sym.axes_second[self.sym][self.nfold]
      self.orig = hm.align_vector([0, 0, 1], self.axis)
      self.to_neighbor_olig = sym.to_neighbor_olig[self.sym][self.nfold]
      self.orig_second = self.to_neighbor_olig @ self.orig

      cang = hm.angle(self.axis, self.axis_second)
      aang = (np.pi - cang) / 2.0
      self.slide_to_axis_displacement = np.sin(aang) / np.sin(cang)

      self.symframes_ = sym.symframes(self.sym)

      self.flip_axis = hm.hcross(self.axis, self.axis_second)

      self.nfold = np.array([self.nfold])
      # print(self.sym, self.nfold)

   def slide_dir(self):
      dirn = self.axis_second - self.axis
      return dirn[:3] / np.linalg.norm(dirn)

   def placements(self, angles):
      place1 = hm.hrot(self.axis, angles, degrees=1) @ self.orig
      place2 = place1 @ hm.hrot([1, 0, 0], 180)  # perp to z
      return np.concatenate([place1, place2])

   def placements_second(self, placements):
      return self.to_neighbor_olig @ placements

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def place_along_axis(self, pos, slide_dist):
      origshape = pos.shape
      pos = pos.reshape(-1, 4, 4)
      # pos2 = pos2.reshape(-1, 4, 4)

      adis = -slide_dist * self.slide_to_axis_displacement
      newt = self.axis * adis[:, None]
      newpos = pos.copy()
      newpos[:, :3, 3] = newt[:, :3]

      # newt2 = self.to_neighbor_olig @ newt[:, :, None]
      # newdelta = newt2.squeeze() - newt
      # olddelta = pos2[:, :, 3] - pos[:, :, 3]
      # assert np.allclose(newdelta, olddelta)

      return newpos.reshape(origshape)

class DockSpec2CompCage(DockSpec):
   @property
   def type(self):
      return '2comp_cage'

   def __init__(self, arch):
      self.arch = arch.upper()
      assert self.arch in allowed_twocomp_architectures
      assert len(self.arch) == 3 or self.arch.endswith('D')
      self.sym = arch if arch[0] == 'D' else arch[0]
      self.num_components = 2
      self.comp_is_dihedral = [False, self.arch.endswith('D')]
      self.nfold1 = int(arch[1])
      self.nfold2 = int(arch[2])
      self.nfold = np.array([self.nfold1, self.nfold2])
      self.axis1 = sym.axes[self.sym[0]][self.nfold1]
      self.axis2 = sym.axes[self.sym[0]][self.nfold2]
      self.axis2 = sym.axes[self.sym[0]][33] if arch.startswith('T33') else self.axis2
      self.axis1 = sym.axes['D'][22] if arch.startswith('D22') else self.axis1
      self.axis = np.array([self.axis1, self.axis2])
      self.axisperp = hm.hcross(self.axis1, self.axis2)
      self.orig1 = hm.align_vector([0, 0, 1], self.axis1)
      self.orig2 = hm.align_vector([0, 0, 1], self.axis2)
      self.orig = np.array([self.orig1, self.orig2])
      self.symframes_ = sym.symframes(self.sym)
      self.axis1_second = sym.axes_second[self.sym][self.nfold1]
      self.axis2_second = sym.axes_second[self.sym][self.nfold2]
      self.to_neighbor_olig1 = sym.to_neighbor_olig[self.sym][self.nfold1]
      self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][self.nfold2]
      if arch == 'T33':
         self.axis2_second = sym.axes_second[self.sym][33]
         self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][33]
      if arch == 'D22':
         self.axis1_second = sym.axes_second[self.sym][22]
         self.to_neighbor_olig1 = sym.to_neighbor_olig[self.sym][22]
      self.axis_second = [self.axis1_second, self.axis2_second]
      self.to_neighbor_olig = np.array([self.to_neighbor_olig1, self.to_neighbor_olig2])

      self.compframes = np.array([sym.symframes(self.nfold[i], self.axis[i]) for i in [0, 1]])
      fax1 = hm.hcross(self.axis1, hm.hcross(self.axis1, self.axis2))
      fax2 = hm.hcross(self.axis2, hm.hcross(self.axis2, self.axis1))
      self.xflip = hm.hrot([fax1, fax2], np.pi)

   def __str__(self):
      return f'{self.arch} axis1 {self.axis1[:3]} axis2 {self.axis2[:3]}'

   def slide_dir(self, angles):
      axisdelta = self.axis2 - self.axis1
      axisdelta /= np.linalg.norm(axisdelta)
      dirs = hm.hrot(self.axisperp, angles) @ axisdelta
      return dirs[..., :3]

   def placements1(self, angles, flip=True):
      place1 = hm.hrot(self.axis1, angles, degrees=1) @ self.orig1
      if not flip:
         return place1
      place2 = place1 @ hm.hrot([1, 0, 0], 180)
      return np.concatenate([place1, place2])

   def placements2(self, angles):
      return hm.hrot(self.axis2, angles, degrees=1) @ self.orig2

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def move_to_canonical_unit(self, pos1, pos2):
      origshape = pos1.shape
      pos1 = pos1.reshape(-1, 4, 4).copy()
      pos2 = pos2.reshape(-1, 4, 4).copy()
      xcen = self.symframes_[:, None] @ pos1[:, :, 3, None]
      xcen += self.symframes_[:, None] @ pos2[:, :, 3, None]
      tgt = self.axis1 + self.axis2
      xdot = np.sum(xcen.squeeze(-1) * tgt, axis=2)
      x = self.symframes_[np.argmax(xdot, axis=0)]
      return (x @ pos1).reshape(origshape), (x @ pos2).reshape(origshape)

   def place_along_axes(self, pos1, pos2):
      origshape = pos1.shape
      pos1 = pos1.reshape(-1, 4, 4).copy()
      pos2 = pos2.reshape(-1, 4, 4).copy()
      offset = pos2[:, :3, 3] - pos1[:, :3, 3]
      cang = hm.angle(self.axis1, self.axis2)
      cdis = np.linalg.norm(offset, axis=1)

      offset_norm = offset / cdis[:, None]
      dot1 = hm.hdot(offset_norm, self.axis1)
      dot2 = hm.hdot(offset_norm, self.axis2)
      axisdelta = hm.hnormalized(self.axis2 - self.axis1)
      aofst = np.sign(dot1 + dot2) * hm.angle(offset_norm, axisdelta)
      aang = (np.pi - cang) / 2.0 - aofst
      bang = (np.pi - cang) / 2.0 + aofst

      adis = cdis * np.sin(aang) / np.sin(cang)
      bdis = cdis * np.sin(bang) / np.sin(cang)
      newt1 = self.axis1[:3] * adis[:, None]
      newt2 = self.axis2[:3] * bdis[:, None]
      newoffset = newt2 - newt1

      swap = hm.hdot(offset, newoffset) < 0
      newt1[swap] *= -1
      newt2[swap] *= -1
      assert np.allclose(newt2 - newt1, offset[:, :3])
      pos1[:, :3, 3] = newt1
      pos2[:, :3, 3] = newt2

      return (pos1.reshape(origshape), pos2.reshape(origshape))

class DockSpec3CompCage(DockSpec):
   @property
   def type(self):
      return '3comp_cage'

   def __init__(self, arch):
      assert len(arch) == 4
      assert arch in "O432 I532".split()
      self.arch = arch
      self.sym = arch[0]
      self.num_components = 3
      self.symframes_ = sym.frames[self.sym]
      assert self.sym in "TOI"
      self.comp_is_dihedral = [False] * 3
      self.nfold = np.array([int(arch[1]), int(arch[2]), int(arch[3])], dtype='i')
      self.axis = np.array([sym.axes[self.sym][n] for n in self.nfold])
      self.axisperp = [
         hm.hcross(self.axis[1], self.axis[2]),
         hm.hcross(self.axis[2], self.axis[0]),
         hm.hcross(self.axis[0], self.axis[1])
      ]
      self.orig = [hm.align_vector([0, 0, 1], a) for a in self.axis]
      self.axis_second = [sym.axes_second[self.sym][n] for n in self.nfold]
      self.to_neighbor_olig = [sym.to_neighbor_olig[self.sym][n] for n in self.nfold]
      self.compframes = np.array([sym.symframes(self.nfold[i], self.axis[i]) for i in [0, 1, 2]])
      self.xflip = hm.hrot([
         hm.hcross(self.axis[0], hm.hcross(self.axis[0], self.axis[1])),
         hm.hcross(self.axis[1], hm.hcross(self.axis[1], self.axis[2])),
         hm.hcross(self.axis[2], hm.hcross(self.axis[2], self.axis[0]))
      ], np.pi)

class DockSpecMonomerToCyclic(DockSpec):
   @property
   def type(self):
      return 'cyclic'

   def __init__(self, arch):
      assert len(arch) == 2
      assert arch[0] == "C"
      self.arch = arch
      self.num_components = 1
      self.nfold = int(arch[1])
      assert self.nfold > 1
      self.angle = 2 * np.pi / self.nfold

      self.orig = np.eye(4)
      self.to_neighbor_monomer = hm.hrot([0, 0, 1], self.angle)
      self.orig_second = self.to_neighbor_monomer @ self.orig

      self.symframes_ = [hm.hrot([0, 0, 1], self.angle * i) for i in range(self.nfold)]
      self.tan_half_vertex = np.tan((np.pi - self.angle) / 2)

   def slide_dir(self):
      return [1, 0, 0]

   def placements_second(self, placements):
      return self.to_neighbor_monomer @ placements

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def place_along_axis(self, pos, slide_dist):
      origshape = pos.shape
      pos = pos.reshape(-1, 4, 4)

      print(self.nfold, self.angle, self.tan_half_vertex)
      if self.nfold == 2:
         dx = slide_dist / 2
         dy = 0
      else:
         dx = slide_dist / 2
         dy = dx * self.tan_half_vertex
         print("delta", slide_dist[0], dx[0], dy[0])

      print(np.mean(dx), np.mean(dy))
      newpos = pos.copy()
      newpos[:, 0, 3] = dx
      newpos[:, 1, 3] = dy
      return newpos.reshape(origshape)

_layer_comp_center_directions = dict(
   P6_632=(np.array([0.86602540378, 0.5, 0, 0]), np.array([0.86602540378, 0.0, 0, 0])),
   P4M_4=(np.array([1, 0, 0]), ),
)

class DockSpec1CompMirrorLayer(DockSpec):
   @property
   def type(self):
      return '1comp_mirror_layer'

   def __init__(self, arch):
      arch = arch.upper()
      assert arch.startswith('P')
      self.arch = arch
      self.sym = arch
      self.nfold = np.array(list(arch.split('_')[1]), dtype='i')
      self.directions = _layer_comp_center_directions[arch]
      self.axis = np.array([np.array([0, 0, 1])] * 1)
      self.xflip = [hm.hrot([1, 0, 0], 180)] * 1
      self.comp_is_dihedral = [False]
      self.num_components = 1
      ang = 360 / self.nfold[0]
      self.to_neighbor_olig = [hm.hrot([0, 0, 1], ang)]

class DockSpec3CompLayer(DockSpec):
   @property
   def type(self):
      return '3comp_layer'

   def __init__(self, arch):
      arch = arch.upper()
      assert arch.startswith('P')
      assert 3 == len(arch.split('_')[1])
      self.arch = arch
      self.sym = arch
      self.nfold = np.array(list(arch.split('_')[1]), dtype='i')
      self.directions = _layer_comp_center_directions[arch]
      self.axis = np.array([np.array([0, 0, 1])] * 3)
      self.xflip = [hm.hrot([1, 0, 0], 180)] * 3
      self.comp_is_dihedral = [False, False, False]
      self.num_components = 3
      ang = 360 / self.nfold[0]
      self.to_neighbor_olig = [None, hm.hrot([0, 0, 1], ang), hm.hrot([0, 0, 1], ang)]

class DockSpecAxel:
   @property
   def type(self):
      return 'axel'

   def __init__(self, arch):
      pass
