#!/usr/bin/python3
# -*- coding: utf-8 -*-
#!/usr/bin/python3
# -*- coding: utf-8 -*-
from typing import List

from toolbox.open_nmt.common.registrable import Registrable


class Tokenizer(Registrable):
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError


@Tokenizer.register("whitespace")
class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        return str(text).strip().split(" ")


if __name__ == '__main__':
    pass
