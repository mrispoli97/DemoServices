import os.path
from pprint import pprint
from obfuscation.padding.padding import PaddingInjector
from utility import utils
import shutil


class Obfuscator:

    def __init__(self):
        benign_src = r"C:\Users\mario\PycharmProjects\DemoServices\test\samples\benign"
        self._pi = PaddingInjector(benign_src)
        self._OBFUSCATIONS = self._pi.get_modes()

    def _parse_config(self, config):
        obfuscation = config['obfuscation']
        if obfuscation not in self._OBFUSCATIONS:
            raise ValueError(f"obfuscation {obfuscation} is invalid. Please choose among {self._OBFUSCATIONS}")
        else:
            self._obfuscation = obfuscation

        self._params = config['params']

        severity = self._params['severity']
        if not (0 < severity < 1):
            raise ValueError(f"padding severity {severity} is invalid. Please choose a value in ]0, 1[")

    def obfuscate(self, filepath, config, dst=None, verbose=False):
        self._parse_config(config)
        if dst:
            _, filename = os.path.split(filepath)
            dst_filepath = os.path.join(dst, filename)
            shutil.copy(filepath, dst_filepath)
        else:
            dst_filepath = filepath
        size = os.path.getsize(dst_filepath)
        severity = self._params['severity']
        num_bytes = int(size * severity)

        self._pi.inject(
            filepath=dst_filepath,
            num_bytes=num_bytes,
            mode=self._obfuscation,
            verbose=verbose,
        )

        return dst_filepath


def main():
    config = r"C:\Users\mario\PycharmProjects\DemoServices\obfuscation\obfuscator\configs\benign\benign_025.json"
    filepath = r"C:\Users\mario\PycharmProjects\DemoServices\test\samples\adware\0cb9b7fdc027a8d6a2682bb7c0de4adce8dd8d9f89906919e3969bc453294f39"
    obfuscated = r"C:\Users\mario\PycharmProjects\DemoServices\test\obfuscated"
    obfuscator = Obfuscator()
    obfuscator.obfuscate(filepath=filepath, dst=obfuscated, verbose=False, config={
        "obfuscation": "benign",
        "params": {
            "severity": 0.1
        }
    })


if __name__ == '__main__':
    main()
