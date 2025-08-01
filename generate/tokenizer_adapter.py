"""Adapt tokenizers to a common interface."""

import enum
import inspect
from typing import Any, Callable
import sentencepiece as spm


class TokenizerType(enum.Enum):
    SP: str = 'sp'  # sentencepiece tokenizer
    HF: str = 'hf'  # huggingface tokenizer
    NONE: str = 'none'  # Represents no tokenizer

ENCODE_FUNCTION_TYPE = Callable[[str, Any],list[int]]
DECODE_FUNCTION_TYPE = Callable[[list[int], Any],str]

class TokenizerAdapter:
    """Wrapper for different tokenizers used in sampler."""

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

        missing_methods = self._missing_methods()
        if not missing_methods:
            self._tokenizer_type = TokenizerType.NONE
            self._encode: ENCODE_FUNCTION_TYPE  = self._tokenizer.encode
            self._decode: DECODE_FUNCTION_TYPE  = self._tokenizer.decode
            self._bos_id: int = self._tokenizer.bos_id()
            self._eos_id: int = self._tokenizer.eos_id()
            self._pad_id:int = self._tokenizer.pad_id()

        elif isinstance(self._tokenizer, spm.SentencePieceProcessor):
            self._tokenizer_type = TokenizerType.SP
            self._encode: ENCODE_FUNCTION_TYPE = self._tokenizer.EncodeAsIds
            self._decode: DECODE_FUNCTION_TYPE  = self._tokenizer.DecodeIds
            self._bos_id: int = self._tokenizer.bos_id()
            self._eos_id: int = self._tokenizer.eos_id()

            ret_id = self._tokenizer.pad_id()
            if ret_id == -1:
                raise ValueError('SentencePiece tokenizer has a undefined pad_id.')
            self._pad_id: int = ret_id

        elif self._is_hf_tokenizer():
            self._tokenizer_type = TokenizerType.HF
            self._encode: ENCODE_FUNCTION_TYPE = self._tokenizer.encode
            self._decode: DECODE_FUNCTION_TYPE  = self._tokenizer.decode
            self._bos_id: int = self._tokenizer.bos_token_id
            self._eos_id: int = self._tokenizer.eos_token_id

            # e.g. llama3 HF tokenizers do not have pad_id
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._pad_id: int = self._tokenizer.pad_token_id

        else:
            raise ValueError(
                'Your tokenizer should either be a `spm.SentencePieceProcessor` '
                'tokenizer, a HuggingFace tokenizer, or it should have '
                'the following methods: '
                '`["encode", "decode", "bos_id", "eos_id", "pad_id"]`. Received: '
                f'`type(tokenizer)` = {type(tokenizer)}, with missing methods: '
                f'{missing_methods}.'
            )

    def encode(self, text: str, **kwargs) -> list[int]:
        return self._encode(text, **kwargs)
        
    def decode(self, ids: list[int], **kwargs) -> str:
        return self._decode(ids, **kwargs)

    def bos_id(self) -> int:
        return self._bos_id

    def eos_id(self) -> int:
        return self._eos_id
    
    def pad_id(self) -> int:
        return self._pad_id
    
    def _missing_methods(self) -> list[str]:
        """Checks if the tokenizer has any missing methods."""
        required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
        missing_methods = []
        for method in required_methods:
            if not hasattr(self._tokenizer, method):
                missing_methods.append(method)
        return missing_methods

    def _is_hf_tokenizer(self) -> bool:
        """Checks if the tokenizer is a huggingface tokenizer."""
        baseclasses = inspect.getmro(type(self._tokenizer))
        baseclass_names = [
            baseclass.__module__ + '.' + baseclass.__name__
            for baseclass in baseclasses
        ]
        if (
            'transformers.tokenization_utils_base.PreTrainedTokenizerBase'
            in baseclass_names
        ):
            return True
        return False
