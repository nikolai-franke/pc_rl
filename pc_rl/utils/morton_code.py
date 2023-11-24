import torch

"Modified version of https://github.com/CGuangyan-BIT/PointGPT/blob/V1.2/models/z_order.py"


def round_to_int_32(data):
    """
    Takes a Numpy array of float values between
    -1 and 1, and rounds them to significant
    32-bit integer values, to be used in the
    morton code computation

    :param data: multidimensional numpy array
    :return: same as data but in 32-bit int format
    """
    data = 512 * (
        data + 1.0
    )  # TODO: if we always use the range [-1.0, 1.0], we can just hardcode the 1.0

    # now convert to int
    data = torch.round(2**21 - data).to(dtype=torch.int32)
    return data


def split_by_3(x):
    """
    Method to separate bits of a 32-bit integer
    by 3 positions apart, using the magic bits
    https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations/

    :param x: 32-bit integer
    :return: x with bits separated
    """
    # we only look at 21 bits, since we want to generate
    # a 64-bit code eventually (3 x 21 bits = 63 bits, which # TODO: they actually just use 32bits
    # is the maximum we can fit in a 64-bit code)
    #
    # x = x.to(
    #     torch.int64
    # )  # TODO: if we want higher accuracy (a lot more than 512 values), we need to convert to 64 bit number

    x &= 0x1FFFFF  # only take first 21 bits
    # shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    x = (x | (x << 32)) & 0x1F00000000FFFF
    # shift left 16 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    # shift left 8 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    x = (x | (x << 8)) & 0x100F00F00F00F00F
    # shift left 4 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    x = (x | (x << 4)) & 0x10C30C30C30C30C3
    # shift left 2 bits, OR with self, and 0001001001001001001001001001001001001001001001001001001001001001
    x = (x | (x << 2)) & 0x1249249249249249

    return x


def get_z_order(x, y, z):
    """
    Given 3 arrays of corresponding x, y, z
    coordinates, compute the morton (or z) code for
    each point and return an index array
    We compute the Morton order as follows:
        1- Split all coordinates by 3 (add 2 zeros between bits)
        2- Shift bits left by 1 for y and 2 for z
        3- Interleave x, shifted y, and shifted z
    The mordon order is the final interleaved bit sequence

    :param x: x coordinates
    :param y: y coordinates
    :param z: z coordinates
    :return: index array with morton code
    """
    res = 0
    res |= split_by_3(x) | split_by_3(y) << 1 | split_by_3(z) << 2

    return res


def get_z_values(data):
    """
    Computes the z values for a point array
    :param data: Nx3 array of x, y, and z location

    :return: Nx1 array of z values
    """
    points_round = round_to_int_32(data)  # convert to int
    z = get_z_order(points_round[..., 0], points_round[..., 1], points_round[..., 2])

    return z
