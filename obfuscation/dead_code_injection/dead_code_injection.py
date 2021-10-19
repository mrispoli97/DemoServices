from pprint import pprint
import random as r
from utility import utils

DEAD_CODE_INSTRUCTIONS = [
    b'\x90',
    b'\x40\x48',
    b'\x43\x4B',
    b'\x41\x49',
    b'\x42\x4A',
    b'\x83\xC0\x00',
    b'\x83\xC3\x00',
    b'\x83\xC1\x00',
    b'\x83\xC2\x00',
    b'\x8E\xE8\x00',
    b'\x83\xEB\x00',
    b'\x83\xE9\x00',
    b'\x83\xEA\x00'
]


class DeadCodeInjector:

    def __init__(self, instructions=None):
        self._instructions = self._initInstructions(instructions if instructions else DEAD_CODE_INSTRUCTIONS)
        self._addComposedInstructions()
        self._max_length_instruction = 3

    def _initInstructions(self, instructions):
        bytes = {}
        for instruction in instructions:
            num_bytes = len(instruction)
            if num_bytes not in bytes:
                bytes[num_bytes] = []
            bytes[num_bytes].append(instruction)
        return bytes

    def _addComposedInstructions(self):
        for num_bytes in [10, 100, 1000]:
            self._instructions[num_bytes] = [self.get_random_bytes(num_bytes) for i in range(0, num_bytes)]

    def get_instructions(self, length=None, max_length=None):

        if length:
            if length in self._instructions.keys():
                return list(self._instructions[length])
            return []

        if max_length:
            instructions = []
            for length, instructions_list in self._instructions.items():
                if length <= max_length:
                    instructions_list = list(instructions_list)
                    instructions += instructions_list
            return instructions

        instructions = []
        for instructions_list in self._instructions.values():
            instructions_list = list(instructions_list)
            instructions += instructions_list
        return instructions

    def get_random_instruction(self, max_length=None):
        instructions = []
        if not max_length:
            instructions = self.get_instructions()
        else:
            for length in range(1, min(self._max_length_instruction, max_length) + 1):
                instructions += self.get_instructions(length=length)
        return instructions[r.randint(0, len(instructions) - 1)] if len(instructions) > 0 else None

    def get_random_bytes(self, num_bytes, verbose=False):
        percentage = utils.get_percentage(0, num_bytes)
        if verbose:
            print(f"processing random bytes.. {percentage}%")
        bytes = b""
        num_bytes_left = num_bytes
        while num_bytes_left > 0:
            instructions = self.get_instructions(max_length=num_bytes_left)
            if len(instructions) == 0:
                raise Exception(f"No instructions found with max length {num_bytes_left}.")
            byte_sequence = instructions[r.randint(0, len(instructions) - 1)]
            bytes += byte_sequence
            num_bytes_left -= len(byte_sequence)

            new_percentage = utils.get_percentage(num_bytes - num_bytes_left, num_bytes)
            if new_percentage > percentage:
                percentage = new_percentage
                if verbose:
                    print(f"processing random bytes.. {percentage}%")
        return bytes


def main():
    dci = DeadCodeInjector()
    bytes = dci.get_random_bytes(2000000)
    pprint(bytes)


if __name__ == '__main__':
    main()
