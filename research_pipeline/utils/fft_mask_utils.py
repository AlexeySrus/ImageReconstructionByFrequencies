import numpy as np


def cnorm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor


def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = cnorm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]


def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]


class ssdu_masks():
    """

    Parameters
    ----------
    rho: split ratio for training and loss mask. \ rho = |\Lambda|/|\Omega|
    small_acs_block: keeps a small acs region fully-sampled for training masks
    if there is no acs region, the small acs block should be set to zero
    input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol

    Gaussian_selection:
    -divides acquired points into two disjoint sets based on Gaussian  distribution
    -Gaussian selection function has the parameter 'std_scale' for the standard deviation of the distribution. We recommend to keep it as 2<=std_scale<=4.

    Uniform_selection: divides acquired points into two disjoint sets based on uniform distribution

    Returns
    ----------
    trn_mask: used in data consistency units of the unrolled network
    loss_mask: used to define the loss in k-space

    """

    def __init__(self, rho=0.4, small_acs_block=(4, 4)):
        self.rho = rho
        self.small_acs_block = small_acs_block

    def Gaussian_selection(self, input_data, input_mask, std_scale=4, num_iter=1):

        nrow, ncol = input_data.shape[0], input_data.shape[1]
        center_kx = int(find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 2)))

        if num_iter == 0:
            print(f'\n Gaussian selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0

        loss_mask = np.zeros_like(input_mask)
        count = 0

        while count <= np.int32(np.ceil(np.sum(input_mask[:]) * self.rho)):

            indx = np.int32(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
            indy = np.int32(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))

            if (0 <= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1 and loss_mask[indx, indy] != 1):
                loss_mask[indx, indy] = 1
                count = count + 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask

    def uniform_selection(self, input_data, input_mask, num_iter=1):

        nrow, ncol = input_data.shape[0], input_data.shape[1]

        center_kx = int(find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(find_center_ind(input_data, axes=(0, 2)))

        if num_iter == 0:
            print(f'\n Uniformly random selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2: center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2: center_ky + self.small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                               size=np.int32(np.count_nonzero(pr) * self.rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask

    def apply_fft_matrix_to_channel(self, channel: np.ndarray) -> np.ndarray:
        base_mask = np.ones_like(channel)

        fft_x = np.fft.fft2(channel)
        fft_x = np.fft.fftshift(fft_x)

        mask, _ = self.uniform_selection(fft_x[:, :, None], base_mask[:, :, None])
        mask = mask[..., 0]

        filtered_fft_x= fft_x * mask
        filtered_fft_x = np.fft.ifftshift(filtered_fft_x)
        fn_channel = np.abs(np.fft.ifft2(filtered_fft_x))
        fn_channel = np.clip(fn_channel, 0, 255.0).astype(np.uint8)
        return fn_channel

    def apply_fft_matrix(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return self.apply_fft_matrix_to_channel(image)

        res = image.copy()
        for ch in range(image.shape[2]):
            res[..., ch] = self.apply_fft_matrix_to_channel(res[..., ch])

        return res


if __name__ == '__main__':
    from PIL import Image

    imgp = '/home/alexey/Downloads/61EZ34nFyCL._AC_SL1000_.jpg'
    img = Image.open(imgp).convert('L')
    imgarr = np.array(img)

    ablocks = [2, 4, 8, 16, 32, 64]
    block_size = np.random.choice(ablocks)

    mask_generator = ssdu_masks(rho=1.0, small_acs_block=(block_size, block_size))

    Image.fromarray(mask_generator.apply_fft_matrix(imgarr)).show()

