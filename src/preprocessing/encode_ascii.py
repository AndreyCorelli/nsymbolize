import os
from typing import Set, List

from numpy import ndarray


class ASCIIEncoder:
    # len(self.chars) * 160 * 80
    VECTOR_LENGTH = 128000
    SHAPE = 160, 80

    def __init__(self):
        self.chars = [' ', '.', ':', '-', '+', '=', '*', '%', '@', '#']

        def encode_index(ind: int) -> str:
            return ''.join(['1' if i == ind else '0' for i in range(len(self.chars))])

        self.char_to_code = {self.chars[i]: encode_index(i)
                             for i in range(len(self.chars))}

    def determine_codes_from_files(self, src_folder: str) -> None:
        chars: Set[str] = set()

        for file_path in [entry for entry in os.scandir(src_folder) if entry.is_file()]:
            with open(file_path.path, "r") as f:
                while c := f.read(1):
                    chars.add(c)

        chars.remove('\n')
        char_list = list(chars)
        char_list.sort()
        print(f"{len(char_list)} chars. Starts from [{char_list[0]}] (#{ord(char_list[0])}), ")
        print(f"Ends with [{char_list[-1]}] (#{ord(char_list[-1])}). \n Chars are: ")
        print(f"[{char_list}]")

    def encode_files_in_folder(self,
                               src_folder: str,
                               dst_folder: str):
        for file_path in [entry for entry in os.scandir(src_folder) if entry.is_file()]:
            out_path = os.path.join(dst_folder, file_path.name)
            with open(file_path.path, "r") as fr:
                with open(out_path, "w") as fw:
                    while c := fr.read(1):
                        code = self.char_to_code.get(c)
                        if code:
                            fw.write(code)

    def decode_vector(self, vector: ndarray) -> List[List[str]]:
        # data: ndarray: (1, VECTOR_LENGTH)
        # [[0.8932751  0.0882211  0.05899254 ... 0.06023816 0.05853529 0.06030886]]
        char_len = len(self.chars)
        chars_count = int(vector.size / char_len)

        text: List[List[str]] = [[]]
        line: List[str] = text[0]

        for i in range(chars_count):
            start_index = i * char_len
            # remove  + 1:
            char_v = vector[0, start_index + 1: start_index + char_len]
            max_index = char_v.argmax() or 0
            max_value = char_v[max_index]
            if max_value < 0.13:
                max_index = 0


            if len(line) == self.SHAPE[0]:
                line = []
                text.append(line)
            line.append(self.chars[max_index])

        for line in text:
            print("\n")
            print("".join(line))
        return text

