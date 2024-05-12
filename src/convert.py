import sys
import torch

from gguf import GGUFWriter, GGMLQuantizationType
import gguf
from typing import TYPE_CHECKING, Any, Callable, ClassVar, IO, Iterable, Literal, Protocol, TypeVar, runtime_checkable
from sentencepiece import SentencePieceProcessor
from pathlib import Path
import json

KEY_PAD_ID = 'tokenizer.ggml.padding_token_id'
KEY_UNK_ID = 'tokenizer.ggml.unknown_token_id'
KEY_BOS_ID = 'tokenizer.ggml.bos_token_id'
KEY_EOS_ID = 'tokenizer.ggml.eos_token_id'
KEY_WORD_PREFIX = 'tokenizer.ggml.word_prefix'
KEY_SUBWORD_PREFIX = 'tokenizer.ggml.subword_prefix'

@runtime_checkable
class BaseVocab(Protocol):
    tokenizer_model: ClassVar[str]
    name: ClassVar[str]


@runtime_checkable
class Vocab(BaseVocab, Protocol):
    vocab_size: int
    added_tokens_dict: dict[str, int]
    added_tokens_list: list[str]
    fname_tokenizer: Path

    def __init__(self, base_path: Path): ...
    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]: ...

class SentencePieceVocab(Vocab):
    tokenizer_model = "deberta"
    name = "spm"

    def __init__(self, base_path: Path):
        added_tokens: dict[str, int] = {}
        if (fname_tokenizer := base_path / 'spm.model').exists():
            ...

        self.sentencepiece_tokenizer = SentencePieceProcessor()
        self.sentencepiece_tokenizer.LoadFromFile(str(fname_tokenizer))
        vocab_size = self.sentencepiece_tokenizer.vocab_size()

        new_tokens       = {id: piece for piece, id in added_tokens.items() if id >= vocab_size}
        expected_new_ids = list(range(vocab_size, vocab_size + len(new_tokens)))
        actual_new_ids   = sorted(new_tokens.keys())

        if expected_new_ids != actual_new_ids:
            raise ValueError(f"Expected new token IDs {expected_new_ids} to be sequential; got {actual_new_ids}")

        # Token pieces that were added to the base vocabulary.
        self.added_tokens_dict  = added_tokens
        self.added_tokens_list  = [new_tokens[id] for id in actual_new_ids]
        self.vocab_size_base    = vocab_size
        self.vocab_size         = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer    = fname_tokenizer
        self.pad_token_id       = 0
        self.cls_token_id       = 1
        self.eos_token_id       = 2
        self.unk_token_id       = 3
        self.sep_token_id       = 2

    def sentencepiece_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(i)
            text         = piece.encode("utf-8")
            score: float = tokenizer.GetScore(i)

            toktype = gguf.TokenType.NORMAL
            if tokenizer.IsUnknown(i):
                toktype = gguf.TokenType.UNKNOWN
            if tokenizer.IsControl(i):
                toktype = gguf.TokenType.CONTROL


            if tokenizer.IsUnused(i):
                toktype = gguf.TokenType.UNUSED
            if tokenizer.IsByte(i):
                toktype = gguf.TokenType.BYTE

            yield text, score, toktype

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

def extract_vocab(vocab):
    tokens, scores, toktypes = [], [], []
    for text, score, toktype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)

    assert len(tokens) == vocab.vocab_size
    return tokens, scores, toktypes


if __name__ == "__main__":
    with open('../model/config.json', 'r') as f_read:
        config = json.load(f_read)

    print(config)
    vocab = SentencePieceVocab(Path("../model"))
    model = torch.load("../model/pytorch_model.bin")

    float_type = "f32"
    qtype = GGMLQuantizationType[float_type.upper()]
    dtype0 = {'f16': torch.float16, 'f32': torch.float32}[float_type]

    param_keys = [
        'vocab_size', 'max_position_embeddings', 'hidden_size', 'intermediate_size',
        'num_attention_heads', 'num_hidden_layers', 'layer_norm_eps'
    ]

    gguf_writer = GGUFWriter("deberta.ggml", 'deberta')
    gguf_writer.add_name("deberta")
    gguf_writer.add_description("ggml deberta model")
    gguf_writer.add_file_type(qtype)

    # writing model parameters
    gguf_writer.add_uint32("vocab_size", config['vocab_size'])
    gguf_writer.add_uint32("max_position_embedding", config['max_position_embeddings'])
    gguf_writer.add_uint32("hidden_size", config['hidden_size'])
    gguf_writer.add_uint32("intermediate_size", config['intermediate_size'])
    gguf_writer.add_uint32("num_attention_heads", config['num_attention_heads'])
    gguf_writer.add_uint32("num_hidden_layers", config['num_hidden_layers']);
    gguf_writer.add_float32("layer_norm_eps", config['layer_norm_eps'])

    # writing vocab parameters
    gguf_writer.add_int32(KEY_PAD_ID, vocab.pad_token_id)
    gguf_writer.add_int32(KEY_UNK_ID, vocab.unk_token_id)
    gguf_writer.add_int32(KEY_BOS_ID, vocab.cls_token_id)
    gguf_writer.add_int32(KEY_EOS_ID, vocab.sep_token_id)

    tokens, scores, toktypes = extract_vocab(vocab)

    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)

    for n, p in model.items():
        if 'LayerNorm' in n or 'bias' in n:
            dtype = torch.float32
        else:
            dtype = dtype0

        shape_str = str(list(p.shape))
        print(f'{n:64s} = {shape_str:16s} {p.dtype} → {dtype}')

        p = p.to(dtype)
        gguf_writer.add_tensor(n, p.numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()


