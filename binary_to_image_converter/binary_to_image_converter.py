import os
import numpy as np
from PIL import Image


def bin_to_image(binary_filepath):
    size = os.path.getsize(binary_filepath)

    f = open(binary_filepath, 'rb')

    # f2 = open(binary_filepath, 'rb')
    # byte2 = int.from_bytes(f2.read(1), byteorder='big')

    # data = f.read()

    # for byte in data:
    #     bytes.append(byte)
    #
    #     assert (byte <= 255)
    data = open(binary_filepath, 'rb').read()

    bytes = np.frombuffer(data, np.uint8)

    # byte = f.read(1)
    # while byte != b"":
    #     byte = int.from_bytes(byte, byteorder='big')
    #     assert (byte <= 255)
    #     bytes.append(byte)
    #     byte = f.read(1)

    f.close()

    image_width = get_image_width(file_size=size)
    image_height = int(size / image_width)
    image_size = image_width * image_height
    number_of_bytes_left = size % image_width

    assert (size == image_size + number_of_bytes_left)

    bytes_image = np.uint8(np.reshape(bytes[:image_size], (image_height, image_width)))

    image = Image.fromarray(bytes_image)
    image = image.resize((224, 224))

    # image_converted = image.convert('RGB')
    # image converted has each pixel repeated three times, one for each channel

    return image


def get_image_width(file_size):
    kb = 1024
    if file_size < 10 * kb:
        return 32
    if file_size < 30 * kb:
        return 64
    if file_size < 60 * kb:
        return 128
    if file_size < 100 * kb:
        return 256
    if file_size < 200 * kb:
        return 384
    if file_size < 500 * kb:
        return 512
    if file_size < 1000 * kb:
        return 768
    return 1024


class BIConverter:

    def __init__(self):
        pass

    def convert(self, path):
        image = bin_to_image(path)
        return image
