from pprint import pprint

import numpy as np
from time import sleep
from PIL import Image, ImageDraw

from random import randint

from ..utils.module import Module

class AirfoilAugmentor(Module):
    def __init__(self):
        Module.__init__(self, in_label='AirfoilPlot', out_label='AirfoilPlot:Image', connect_labels=('augmented_image', 'augmented_image'))

    def random_color(self):
        return tuple(randint(0, 255) for _ in range(4))

    def noise(self, image, p=1.0):
        width, height = image.size
        noise_map = np.random.randint(int(p*255), size=(height, width, 4,), dtype='uint8')
        image += noise_map
        return Image.fromarray(image)

    def rand_fill(self, image):
        width, height = image.size
        center = int(0.5 * width), int(0.5 * height)
        origin = 0, 0

        ImageDraw.floodfill(image, xy=center, value=self.random_color())
        ImageDraw.floodfill(image, xy=origin, value=self.random_color())
        return image

    def augment(self, filename):
        image = Image.open(filename)
        image = self.rand_fill(image)
        image = self.noise(image, p=0.3)

        filename = filename.replace('.png', '_augmented.png')
        image.save(filename)
        return filename

    def process(self, node, driver=None):
        filename = self.augment(node.data['filename'])
        yield self.default_transaction(data=dict(filename=filename, parent=node.data['parent']))
