from rpxdock.sampling import *
from rpxdock.search import *
import rpxdock as rp

spec = rp.search.DockSpec2CompCage('I53')
resl = 1
angresl = 1
cart_bounds = [0,1]
cart_bounds = np.array([cart_bounds] * spec.num_components)
cart_bounds = np.array(cart_bounds)
cart_bounds = np.tile(cart_bounds, [8, 1])
cart_nstep = np.ceil((cart_bounds[:, 1] - cart_bounds[:, 0]) / resl).astype('i')
ang = 3 / spec.nfold
ang_nstep = np.ceil(ang / angresl).astype('i')

def urange(*args):
   return np.arange(*args, dtype="u8")

def test_flip_components():
    flip_components = [0,1]
    samp = []
    for i in range(len(spec.nfold)):
        s = rp.sampling.RotCart1Hier_f4(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], 0,
                                         ang[i], ang_nstep[i], spec.axis[i][:3])
        samp.append(s)

    for i, s in enumerate(samp):
        if flip_components[i]:
            samp[i] = rp.sampling.ProductHier(s, rp.ZeroDHier([np.eye(4), spec.xflip[i]]))

    sampler = rp.sampling.CompoundHier(*samp)

    ok, x = sampler.get_xforms(resl, urange(0, sampler.size(resl), 8))

    #rp.dump(x, 'rpxdock/data/testdata/test_flip_components.pickle')
    ref_x = rp.data.get_test_data('test_flip_components')
    np.allclose(x, ref_x)

def test_fixed_rot(): 
    fixed_rot = [1]
    samp = []
    for i in range(len(spec.nfold)):
        if i in fixed_rot: 
            s = LineHier(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], spec.axis[i])
        else: 
            s = rp.sampling.RotCart1Hier_f4(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], 0,
                                         ang[i], ang_nstep[i], spec.axis[i][:3])
        samp.append(s)

    sampler = rp.sampling.CompoundHier(*samp)
    ok, x = sampler.get_xforms(resl, urange(0, sampler.size(resl), 8))
    #rp.dump(x, 'rpxdock/data/testdata/test_fixed_rot.pickle')
    ref_x = rp.data.get_test_data('test_fixed_rot')
    np.allclose(x, ref_x)

def test_fixed_components():
    fixed_components = [1]
    samp = []

    for i in range(len(spec.nfold)):
        if i in fixed_components:
            s = rp.ZeroDHier([np.eye(4)])
        else: 
            s = rp.sampling.RotCart1Hier_f4(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], 0,
                                         ang[i], ang_nstep[i], spec.axis[i][:3])
        samp.append(s)
    sampler = rp.sampling.CompoundHier(*samp)

    ok, x = sampler.get_xforms(resl, urange(0, sampler.size(resl), 8))
    #rp.dump(x, 'rpxdock/data/testdata/test_fixed_components.pickle')
    ref_x = rp.data.get_test_data('test_fixed_components')
    np.allclose(x, ref_x)

def test_fixed_trans(): 
    fixed_trans = [1]
    samp = []

    for i in range(len(spec.nfold)):
        if i in fixed_trans:
            s = rp.sampling.RotHier_f4(0, ang[i], ang_nstep[i], spec.axis[i][:3]) #TODO: MDL try this
        else: 
            s = rp.sampling.RotCart1Hier_f4(cart_bounds[i, 0], cart_bounds[i, 1], cart_nstep[i], 0,
                                         ang[i], ang_nstep[i], spec.axis[i][:3])
        samp.append(s)

    sampler = rp.sampling.CompoundHier(*samp)

    ok, x = sampler.get_xforms(resl, urange(0, sampler.size(resl), 8))
    #rp.dump(x, 'rpxdock/data/testdata/test_fixed_trans.pickle')
    ref_x = rp.data.get_test_data('test_fixed_trans')
    np.allclose(x, ref_x)

def test_fixed_wiggle():
    fixed_wiggle = [0,1]
    samp = []

    fw_cartlb=-5
    fw_cartub=5
    fw_rotlb=-5
    fw_rotub=5
    
    for i in range(len(spec.nfold)):
        if i in fixed_wiggle:
            s =  rp.sampling.RotCart1Hier_f4(fw_cartlb,  fw_cartub, cart_nstep[i], fw_rotlb, fw_rotub, ang_nstep[i], spec.axis[i][:3])
        samp.append(s) 
    sampler = rp.sampling.CompoundHier(*samp)

    ok, x = sampler.get_xforms(resl, urange(0, sampler.size(resl), 8))
    #rp.dump(x, 'rpxdock/data/testdata/test_fixed_wiggle.pickle')
    ref_x = rp.data.get_test_data('test_fixed_wiggle')
    np.allclose(x, ref_x)

if __name__ == "__main__":

    test_flip_components()
    test_fixed_rot()
    test_fixed_components()
    test_fixed_trans()
    test_fixed_wiggle()


