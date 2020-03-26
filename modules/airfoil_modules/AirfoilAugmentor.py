from pprint import pprint

import numpy as np
from time import sleep
from PIL import Image, ImageDraw

from random import randint

from ..utils.module import Module

class AirfoilAugmentor(Module):
    def __init__(self, count=10):
        Module.__init__(self, in_label='CleanAirfoilPlot', out_label='AugmentedAirfoilPlot:Image', connect_labels=('augmented_image', 'augmented_image'))
        self.count = count

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

    def rand_translate(self, image):
        minsize    = min(image.size)
        horizontal = randint(minsize)
        vertical   = randint(minsize)
        return image.transform(image.size, Image.AFFINE, (1, 0, horizontal, 0, 1, vertical))

    def flips(self, image):
        return [image] + list(map(Image.fromarray, [np.fliplr(image), np.flipud(image), np.fliplr(np.flipud(image))]))

    def augment(self, filename):
        image = Image.open(filename)
        for j in range(self.count):
            image = self.rand_fill(image)
            image = self.noise(image, p=0.25)

            for i, flipped in enumerate(self.flips(image)):
                aug_file = filename.replace('.png', '_augmented_{}_{}.png'.format(i, j))
                flipped.save(aug_file)
                yield aug_file

    def process(self, node, driver=None):
        for filename in self.augment(node.data['filename']):
            yield self.default_transaction(data=dict(filename=filename, parent=node.data['parent']))
