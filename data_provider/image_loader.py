import os
import guesswhat.train.utils as utils
from PIL import Image

import numpy as np
from guesswhat.data_provider.image_preprocessors import (resize_image, get_spatial_feat,
                                                         scaled_crop_and_pad)

# file renaming
# for file in *.jpg; do mv "$file" "${file/COCO_test2015_/}"; done

# TODO make it available for oracle too
class AbstractImgLoader(object):
    def __init__(self, img_dir):
        self.img_dir = img_dir

    def preload(self,picture_id):
        return self

    def get_image(self, picture_id):
        pass


class DummyImgLoader(AbstractImgLoader):
    def __init__(self, data_dir, size=1000):
        AbstractImgLoader.__init__(self, data_dir)
        self.size = size

    def get_image(self, picture_id):
        return np.zeros(self.size)

"""Trick to avoid serializing complete fc8 dictionary.
We wrap the fc8 into a separate object which does not contain
the fc8 dictionary.
"""
class fcPreloaded(AbstractImgLoader):
    def __init__(self, data_dir, fc8):
        AbstractImgLoader.__init__(self, data_dir)
        self.fc8 = fc8

    def get_image(self, _):
        return self.fc8


class fcLoader(AbstractImgLoader):
    def __init__(self, name, data_dir, year, image_input):
        AbstractImgLoader.__init__(self, data_dir)
        self.data_dir = data_dir
        self.image_path = os.path.join(data_dir, 'vgg_{name}_{year}_{image_input}.pkl'.
                                       format(name=name,year=year, image_input=image_input))
        self.fc8_img = utils.pickle_loader(self.image_path)

    def preload(self, picture_id):
        return fcPreloaded(self.data_dir, self.fc8_img[picture_id])


class ConvLoader(AbstractImgLoader):
    def __init__(self, network, name, data_dir, year, image_input, dim=None, is_sparse=False):
        AbstractImgLoader.__init__(self, data_dir)
        import scipy  # silently used - do not remove
        self.image_path = os.path.join(data_dir, "{network}_{name}_{year}_{img}"
                                       .format(network=network, name=name, year=year, img=image_input), "{}.pkl")
        self.dim = dim
        self.is_sparse = is_sparse

    def get_image(self, picture_id):
        if self.is_sparse:
            scipy_matrix = utils.pickle_loader(self.image_path.format(picture_id))
            matrix = scipy_matrix.toarray()
            matrix = matrix.reshape(self.dim)
        else:
            matrix = utils.pickle_loader(self.image_path.format(picture_id))
        return matrix

class MCBConvLoader(AbstractImgLoader):
    def __init__(self, data_dir, sub_path):
        AbstractImgLoader.__init__(self, data_dir)
        self.image_path = os.path.join(data_dir, sub_path, "{}.jpg.npz")

    def get_image(self, picture_id):

        t_ivec = np.load(self.image_path.format(str(picture_id).zfill(12)))['x']

        #normalize
        t_ivec = (t_ivec / np.sqrt((t_ivec ** 2).sum()))
        t_ivec = np.transpose(t_ivec, axes=[1,2,0])

        return t_ivec





class RawImageLoader(AbstractImgLoader):
    def __init__(self, data_dir, width, height, channel=None):
        AbstractImgLoader.__init__(self, data_dir)
        self.image_path = self.img_dir
        self.width = width
        self.height = height
        self.channel = channel

    def get_image(self, file):
        img = Image.open(file).convert('RGB')

        img = resize_image(img, self.width , self.height)
        img = np.array(img, dtype=np.float32)

        if self.channel is not None:
            img -= self.channel[None, None, :]

        return img
