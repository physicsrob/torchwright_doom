"""Readable text surface + HuggingFace tokenizer for the DOOM token stream.

The transformer ingests a sequence of integer ``W_EMBED`` row ids
(``inference/tokens_bridge.py``). This package gives that stream a single
text representation that is **byte-exact** (re-encoding reproduces the
identical id stream), **readable** (the blog's "this is the real prompt"
figure and a debugging view), and **vanilla** (a stock ``WordLevel``
id<->label map at its core; the readability lives in the labels).

* :mod:`.surface` — the grammar: ``render(tokens) -> text`` /
  ``parse(text) -> tokens``. Baked context-free labels, one contextual rule
  (de-quantize a ``VALUE`` by its preceding marker's range), header-break
  line layout. Works at the ``Token`` / ``(TokenType, values)`` level.
* :mod:`.hf_tokenizer` — ``DoomTokenizer(PreTrainedTokenizer)`` wrapping
  ``surface`` + ``tokens_bridge`` (the id<->token bijection).

Sibling to ``inference/`` and subject to the same import-time-vocab caveat:
``embedding`` builds the screen-sized vocab AT IMPORT, so configure the screen
(``apply_screen_env``) before importing modules that reach it.
"""
