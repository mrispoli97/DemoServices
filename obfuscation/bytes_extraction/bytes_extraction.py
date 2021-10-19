import argparse
import os
from pprint import pprint
import random as r
from utility import utils


class BytesExtractor:

    def __init__(self, src):
        self._filepaths = [os.path.join(src, filename) for filename in os.listdir(src)]

    def get_random_bytes(self, num_bytes, verbose=False):

        filepath = self._filepaths[r.randint(0, len(self._filepaths) - 1)]
        data = utils.read_binary(filepath)
        size = len(data)
        if size <= num_bytes / 1000:
            print(f"skipping file {filepath}")
            return self.get_random_bytes(num_bytes, verbose)
        num_bytes_left = num_bytes

        bytes = b""
        percentage = utils.get_percentage(num_bytes - num_bytes_left, num_bytes)
        if verbose:
            print(f"Processing... {percentage}%")
        while num_bytes_left > 0:

            # print(f"num bytes left: {num_bytes_left} < size: {size}")
            if num_bytes_left < size:
                starting = r.randint(0, size - num_bytes_left)
                slice = data[starting: starting + num_bytes_left]

            else:
                slice = data[:]
            bytes += slice
            # print(f"num bytes selected: {len(slice)}")
            num_bytes_left -= len(slice)
            # print(f"num bytes left: {num_bytes_left}")
            # print(f"num bytes: {len(bytes)}")
            new_percentage = utils.get_percentage(num_bytes - num_bytes_left, num_bytes)

            if verbose:
                if new_percentage > percentage:
                    percentage = new_percentage
                    print(f"Processing... {percentage}%")
        return bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='src filepath', required=True)
    args = vars(parser.parse_args())
    src = args['src']
    be = BytesExtractor(src)
    bytes = be.get_random_bytes(10)
    print(len(bytes))


if __name__ == '__main__':
    main()
