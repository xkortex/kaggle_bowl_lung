import numpy as np

def padcrop_vol(vol, newshape=(360, 360, 360), padtype='symmetric', value='origin'):
    """
    Pads and crops a volume in order to match the new shape.
    :param vol: 3D array
    :param newshape: Shape that the new volume will have
    :param padtype: {symmetric, origin} - pad symmetrically (on both sides), or only pad from the far side of index.
    :param value: {'origin'}, numeric, or array-like. Value to pad with.
    :return: Array of shape (newshape)
    """

    vol2 = np.array(vol) # clone for safety reasons
    shape = vol.shape
    if value == 'origin':
        constant_values = vol[0, 0, 0]
        print('Origin: ', constant_values)
    else:
        try:
            constant_values = float(value)
        except ValueError:
            raise ValueError('Invalid parameter "value" specified. Cannot coerce to symbol type or float')

    for dim in range(3):
        if shape[dim] < newshape[dim]:
            pad_amt = (newshape[dim] - shape[dim]) // 2
            parity = (shape[dim] & 1) ^ (newshape[dim] & 1)
            if padtype[:3] == 'sym':
                pad_tup = (pad_amt, pad_amt + parity)  #
            elif padtype[:3] == 'ori':
                pad_tup = (0, pad_amt + pad_amt + parity)
            else:
                raise ValueError('Must specify valid padding mode: {"symmetric", "origin"}')
            pad_list = [(0, 0), (0, 0), (0, 0)]
            pad_list[dim] = pad_tup
            vol2 = np.pad(vol2, pad_list, mode='constant', constant_values=constant_values)
        if shape[dim] > newshape[dim]:
            if padtype[:3] != 'sym':
                raise NotImplementedError(
                    'Have not built this feature yet. Crop should be able to handle symmetric or origin')
            slc_amt = (shape[dim] - newshape[dim]) // 2
            parity = (shape[dim] & 1) ^ (newshape[dim] & 1)
            slc_tup = (slc_amt, shape[dim] - slc_amt - parity)  #
            null1, vol2, null2 = np.split(vol2, slc_tup, dim)

    return vol2



def subsplit(vol, edgelengthgth=48, stride=0.5, serialize=True, verbose=False):
    """
    Take a volume and chop it up to equal sized volumes of side edgelengthgth. This automatically pads up to the nearest
    integer mulitple of edgelengthgth in each direction
        serialize: '''

    :param vol:
    :param edgelengthgth: Length of the edge of the cube
    :type edgelengthgth: int
    :param stride: Stride length, as a fraction of edgelengthgth. 0.5 -> each cube has 50% overlap in each direction
    :type stride: float
    :param serialize: Bool. Serialize the output subsections. If true, return an (N, E, E, E) dim array, E=edge,
    if false, return (M,N,P,E,E,E) dim array, where M, N, and P are the coordinates of the subsections in space
    :param verbose:
    :return: Array of shape (N, E,E,E) or (M,N,P, E,E,E)
    """

    nx, ny, nz = vol.shape
    new_idx = [(nn // edgelengthgth) if (nn%edgelengthgth)==0 else (nn// edgelengthgth)+1 for nn in vol.shape] # deal with the edge case of evenly divisible dim length
    if verbose:
        print('New indicies: {}'.format(new_idx))
    new_shape = [edgelengthgth*idx for idx in new_idx]
    a2 = padcrop_vol(vol, newshape=new_shape)
    b = np.array(np.split(a2, new_idx[0], axis=0))
    b = np.array(np.split(b, new_idx[1], axis=2))
    b = np.array(np.split(b, new_idx[2], axis=4))
    if serialize:
        b = np.reshape(b, (-1, edgelengthgth, edgelengthgth, edgelengthgth))

    return b, new_idx

def subslice(vol, coord, edgelengthgth=48, order='zyx'):
    """ Take a volume and return a cube of side edgelengthgth, centered at coord.

    :param vol: 3D array to operate on
    :param coord: Coordinate of the center of the cube to slice out
    :param edgelengthgth: Length of the edge of the cube
    :param order: Dimension coordinate order. DICOM tends to use zyx ordering
    :return: Array of shape (E,E,E)
    """
    assert len(coord) == 3, '"coord" Must be a length 3 array-like'
    m = edgelengthgth // 2
    if order == 'zyx':
        z, y, x = coord
    else:
        x, y, z = coord
    return vol[x - m:x + m, y - m:y + m, z - m:z + m]


def coord_to_ravel_idx3(shape, coord, order='zyx'):
    """ Takes a coordinate (as x y z index notation) and returns the absolute (raveled) single number index.
    3D specific version.
    :param shape: Shape of the volume you are trying to index over (before ravel)
    :param coord: Coordinate of interest
    :param order: {'xyz', 'zyx'} Dimension coordinate order. DICOM tends to use zyx ordering
    :return: 1D-index of coordinate of interest
    """
    assert len(coord) == 3, '"coord" Must be a length 3 array-like'

    n0, n1, n2 = shape
    if order == 'zyx':
        z, y, x = coord
    else:
        x, y, z = coord
    idx = z * n2 * n1 + y * n2 + x
    return idx


def coord_to_ravel_idx(shape, coord):
    '''Takes a coordinate (as x y z index notation) and returns the absolute (raveled) single number index'''

    assert len(shape) == len(coord), 'Must have matching dimension'
    N = len(shape)
    idx = coord[0]
    for i in range(1, N):
        idx += coord[i] * np.prod(shape[N - i:])
        print(i, coord[i], shape[N - i:])

    return idx


def ravel_idx_to_coord(shape, idx):
    """
    Given a shape and the absolute index, return the (x y z) coordinate index
    :param shape: Shape of (spooled) volume
    :param idx: Index of the volume of interest
    :type idx: int
    :return: (x, y, z) coordinate
    """

    N = len(shape)
    coefs = []
    coords = []
    r = idx
    for i in range(N - 1, 0, -1):
        coef = shape[N - i:]
        coefs.append(coef)
        q, r = divmod(r, np.prod(coef))
        coords.append(q)
        print(q, r)
    coords.append(r)
    coords.reverse()

    return coefs, coords


def coord_to_subcoord(subvol_shape, coord):
    """
    Gives the sub-cube 3d index for a subsected volume.
    Given a volume VOL of shape (T, U, V) that has been subsplit to SUBVOL (M,N,P, E,E,E), we want to know which
    subcube (m,n,p) the coordinate of interest lies in, and also the coordinate (x', y', z') in that cube.

    Essentially, this is a 3D division with remainder.
    :param subvol_shape: Shape of the sub-volume (e.g. subcube)
    :param coord: Coordinate of interest in the original reference frame
    :return: new_idx, new_subcoord - new_idx is the coordinate (m,n,p) of the subcube within the meta-volume
    new_subcoord is the coordinate within that subcube
    """

    new_idx = []
    new_subcoord = []
    for i in range(3):
        q, r = divmod(coord[i], subvol_shape[i])
        new_idx.append(q)
        new_subcoord.append(r)
    return new_idx, new_subcoord


def coord_to_ary_idx(coord, origin):
    """
    Helper function to convert the coordinate (from dicom/luna) and origin to numpy indicies
    :param coord: coordinate of interest from dicom/luna reference frame
    :param origin: coordinate of origin of the dicom/luna reference system
    :return: (x, y, z) index of the voxel of interest
    """
    coord = np.array(coord)
    origin = np.array(origin)
    x, y, z = coord - origin
    absidx = x, y, z  # i have no idea why these things use such crazy indexing. but this will match the numpy slicing dims
    print(absidx)
    return list(map(int, absidx))


def get_fiducial_slice(coord, edgelength=48):
    """
    Simple helper function to get a cube-shaped slice. the slicing indicies given a coordinate.
    :param coord: center of cube
    :param edgelength: length of edge of cube
    :return: 
    """
    ''' Gets the slicing indicies given a coordinate and edge length of a cube '''
    x, y, z = map(int, coord)
    m = edgelength // 2
    print(x + m, x - m, y + m, y - m, z + m, z - m)
    return (x - m, x + m, y - m, y + m, z - m, z + m)


def draw_fiducial_cube(ary_shape, coord, edgelength=48, dtype='int16'):
    """
    Draw a cube of size (E,E,E) located at coord, in a volume of shape ary_shape.
    Return a mask of that shape which is 1 for the volume and zero elsewhere
    :param ary_shape: Shape of original volume
    :param coord: Coordinate of interest, center of cube to draw
    :param edgelength: length of edge of cube
    :param dtype: Array type
    :return: array of shape (ary_shape)
    """
    ary = np.ones(ary_shape, dtype=dtype)
    x0, x1, y0, y1, z0, z1 = get_fiducial_slice(coord, edgelength=edgelength)
    ary[:z0] = 0
    ary[z1:] = 0
    ary[:, :y0] = 0
    ary[:, y1:] = 0
    ary[:, :, :x0] = 0
    ary[:, :, x1:] = 0
    print(np.sum(ary))
    return ary