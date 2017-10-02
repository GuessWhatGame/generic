import os
from PIL import Image

import numpy as np

from generic.data_provider.image_preprocessors import resize_image, scaled_crop_and_pad
from generic.utils.file_handlers import pickle_loader


# Why doing an image loader?
# Well, there are two reasons:
#  - first you want to abstract the kind of picture you are using (raw/conv/feature) etc.
#  One may just want... to load an image!
#  - One must optimize when to load the image, image loader help you doing so





class AbstractImgLoader(object):
    def __init__(self, img_dir, is_raw):
        self.img_dir = img_dir
        self.is_raw = is_raw

    # The goal of the preloading is to pre-st
    def preload(self,picture_id):
        return self

    def get_image(self, picture_id, **kwargs):
        pass

    def is_raw_image(self):
        return self.is_raw


class DummyImgLoader(AbstractImgLoader):
    def __init__(self, data_dir, size=1000):
        AbstractImgLoader.__init__(self, data_dir)
        self.size = size

    def get_image(self, _, **kwargs):
        return np.zeros(self.size)


"""Trick to avoid serializing complete fc8 dictionary.
We wrap the fc8 into a separate object which does not contain
the fc8 dictionary.
"""
class fcPreloaded(AbstractImgLoader):
    def __init__(self, data_dir, fc8):
        AbstractImgLoader.__init__(self, data_dir, is_raw=False)
        self.fc8 = fc8

    def get_image(self, _, **kwargs):
        return self.fc8


class fcLoader(AbstractImgLoader):
    def __init__(self, data_dir):
        AbstractImgLoader.__init__(self, data_dir, is_raw=False)
        self.data_dir = data_dir
        self.image_path = data_dir + ".pkl"
        self.fc8_img = pickle_loader(self.image_path)

    def preload(self, picture_id):
        return fcPreloaded(self.data_dir, self.fc8_img[picture_id])


class ConvLoader(AbstractImgLoader):
    def __init__(self, data_dir):
        AbstractImgLoader.__init__(self, data_dir, is_raw=False)
        self.image_path = os.path.join(data_dir, "{}.npz")

    def get_image(self, picture_id, **kwargs):
        return np.load(self.image_path.format(picture_id)['x'])


class RawImageLoader(AbstractImgLoader):
    def __init__(self, data_dir, width, height, channel=None, extension="jpg"):
        AbstractImgLoader.__init__(self, data_dir, is_raw=True)
        self.image_path = os.path.join(self.img_dir, "{}."+extension)
        self.width = width
        self.height = height
        self.channel = channel

    def get_image(self, picture_id, **kwargs):
        img = Image.open(self.image_path.format(picture_id)).convert('RGB')

        img = resize_image(img, self.width , self.height)
        img = np.array(img, dtype=np.float32)

        if self.channel is not None:
            img -= self.channel[None, None, :]

        return img

class RawCropLoader(AbstractImgLoader):
    def __init__(self, data_dir, width, height, scale, channel=None, extension="jpg"):
        AbstractImgLoader.__init__(self, data_dir, is_raw=True)
        self.image_path = os.path.join(self.img_dir, "{}."+extension)
        self.width = width
        self.height = height
        self.scale = scale
        self.channel = channel

    def get_image(self, object_id, **kwargs):

        bbox = kwargs['bbox']
        image_id = kwargs['image_id']

        img = Image.open(self.image_path.format(image_id)).convert('RGB')

        crop = scaled_crop_and_pad(raw_img=img, bbox=bbox, scale=self.scale)
        crop = resize_image(crop, self.width , self.height)
        crop = np.array(crop, dtype=np.float32)

        if self.channel is not None:
            crop -= self.channel[None, None, :]

        return crop



def get_img_loader(config, image_dir, is_crop=False):

    image_input = config["image_input"]

    if image_input in ["fc7", "fc8"]:
        loader = fcLoader(image_dir)
    elif image_input == "conv":
        loader = ConvLoader(image_dir)
    elif image_input == "raw":
        if is_crop:
            loader = RawCropLoader(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    scale=config["scale"],
                                    channel=config.get("channel", None),
                                    extension=config.get("extension", "jpg"))
        else:
            loader = RawImageLoader(image_dir,
                                    height=config["dim"][0],
                                    width=config["dim"][1],
                                    channel=config.get("channel", None),
                                    extension=config.get("extension", "jpg"))
    else:
        assert False, "incorrect image input: {}".format(image_input)

    return loader