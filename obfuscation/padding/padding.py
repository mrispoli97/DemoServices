import os

from obfuscation.dead_code_injection.dead_code_injection import DeadCodeInjector
from obfuscation.bytes_extraction.bytes_extraction import BytesExtractor


class PaddingInjector:

    def __init__(self, src=None):
        self._dci = DeadCodeInjector()
        self._be = BytesExtractor(src)
        self._MODES = ['zeros', 'junk', 'benign', 'random']

    def get_modes(self):
        return self._MODES

    def inject(self, filepath, num_bytes, mode, verbose=False):
        if mode not in self._MODES:
            raise ValueError(f"inject mode {mode} is invalid. Choose among {self._MODES}")
        if mode == 'junk':
            padding = self._dci.get_random_bytes(num_bytes=num_bytes, verbose=verbose)
        elif mode == 'zeros':
            padding = bytes(num_bytes)
        elif mode == 'random':
            padding = os.urandom(num_bytes)
        else:  # benign
            padding = self._be.get_random_bytes(num_bytes=num_bytes, verbose=verbose)

        with open(filepath, 'ab') as file:
            file.write(padding)
