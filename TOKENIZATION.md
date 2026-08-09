# Tokenization and text formats

The short version: a Doom token ID is literally a row of the model's tied
embedding table. The raw text format gives every row one whitespace-free word.
The pretty text format is a reversible view over those words -- it folds numeric
carrier rows into their markers, converts them back to physical values, and lays
the stream out by protocol phase.

This is not natural-language tokenization. There are no subwords, merge rules,
or learned vocabulary. The tokenizer is an exhaustive, data-only `WordLevel`
map from a finite Doom vocabulary to integer row IDs.

## The complete path

```text
Token(type, slot values)
    -> TOKEN_VOCAB row / tokenizer ID
    -> canonical WordLevel word
    -> examples/e1m1_prompt.txt
    -> stock AutoTokenizer
    -> tied W_EMBED row
    -> compiled transformer
    -> greedy next-row ID
       |-> output.ids.json
       `-> stock tokenizer.decode() -> output.txt
                                      |-> tools/pretty_text.py -> pretty text
                                      `-> tools/txt_to_png.py  -> frame.png
```

`examples/e1m1_prompt.txt` and `output.txt` use the same raw format. The first
describes the scene and ends in `begin`; the second contains only newly
generated rows and normally ends in `done`. `output.ids.json` is the
authoritative integer form of the generated stream. `infer.py` verifies that
its decoded `output.txt` re-encodes to exactly those IDs before writing either
artifact.

The pretty text is downstream presentation. Model inference, render-output
validation, and PNG decoding all use row IDs or raw text; none depends on pretty
layout.

## From a typed token to a row

The structured vocabulary starts in `model/vocab.py`. A `TokenType` has a name
and zero or more ordered slots:

- `IntSlot(lo, hi)` contains integers in the half-open interval `[lo, hi)`.
- `FloatSlot(lo, hi, levels=65536)` contains one of 65,536 evenly spaced levels
  in the closed interval `[lo, hi]`.

`model/embedding.py:TokenVocab` walks `VOCAB_TYPES` in its declared order and
enumerates every combination of every type's slot values. Each combination gets
one row. For a multi-slot token this is mixed-radix enumeration in declaration
order, with the last slot changing fastest. `tokenizer/rows.py:row_index`
implements the same arithmetic without searching the table.

That ordering is load-bearing. Changing a type order, slot range, screen size,
or other vocabulary-sized constant can move row IDs. The vocabulary is also
screen-dependent: for example, `setCursorX` and `screenRange` have ranges fixed
by the compiled resolution. Each bundle records its exact row count, screen
configuration, row-vocabulary fingerprint, and ordered-word hash.

Each row of `W_EMBED` contains a type code, normalized slot values, compact
digit-quad payloads, and precomputed derived values. The compiled graph reads an
input ID by looking up this row. On the output side it constructs a residual row
and the tied LM head scores it against `W_EMBED.T`; greedy argmax chooses the
next row ID. So these are three names for the same integer:

```text
token ID == vocabulary row == W_EMBED row index
```

## Raw text

Raw text is the interchange format spoken by the stock tokenizer. Its grammar
is deliberately small:

```text
raw stream = zero or more canonical words separated by whitespace
word       = one exact vocabulary label containing no whitespace
```

The in-memory codec returns words joined by single spaces with no trailing
newline. Artifact writers append exactly one newline. Readers use whitespace
splitting, so spaces and newlines between words are equivalent.

A word is normally a compact functional label:

```text
setCursorDirectionY
bspFront(node=40,depth=0)
pixel(143,2)
texture.mid(STARTAN3)
ds.meta(i=23,wall_kind=portal,silhouette=both)
```

The functional syntax is cosmetic as far as Hugging Face is concerned. It does
not parse `node=40` or the decimal in `value(v=0.121111)`; `WordLevel` treats the
entire whitespace-free string as one atomic vocabulary entry.

There can be no spaces inside a raw word. In particular, commas are not followed
by spaces. `bspFront(node=40, depth=0)` is two whitespace chunks and is therefore
not valid raw tokenizer text. Comments are also a pretty-format convenience,
not part of raw text: a `#` chunk would be an unknown WordLevel token.

Raw does not mean opaque. `tokenizer/display.py` bakes useful names into every
vocabulary word before the tokenizer is saved:

- redundant prefixes are removed or shortened: `node.x` becomes `x`,
  `seg.front.ceiling` becomes `ceil`, `bbox.x1` becomes `bx1`, and
  `drawseg.scale1` becomes `ds.scale1`;
- self-evident slots become positional, as in `R_AddLine(34)` and
  `pixel(143,2)`;
- enums and flags become words such as `portal`, `yes`, and `no`;
- unified BSP IDs become `node17` or `ss5`;
- texture and flat IDs become WAD names such as `STARTAN3` and `FLOOR4_8`;
- a few invented protocol tokens fold a dominant value into their name, such
  as `twoSided`, `floorMark`, and `frontSideResult`.

That last rule is intentionally limited to invented protocol tokens. Real Doom
calls keep their literal `R_*` or `ST_*` name, so `R_CheckPlane` remains
`R_CheckPlane(floor,0)` rather than being renamed around its argument.

### Markers and carriers in raw text

Wide numbers occupy their own rows. A marker says what a number means, and the
following carrier holds the quantized number:

```text
segDcTmidMid value(v=0.121111)
angle1 angleValue(angle=-420)
```

`value` is a 16-bit carrier normalized to `[-1, 1]`. It does not contain a range
ID. The preceding marker selects one of the physical ranges in
`model/value_ranges.py`; for example, ordinary heights use `[-256, 256]` while
wide map coordinates use `[-2048, 3072]`. `angleValue` contains a signed BAM
angle directly. This project uses 8,192 BAM units per revolution.

The pair remains two model positions and two tokenizer IDs. Folding it into one
pretty expression is only a display operation.

## The stock tokenizer and detokenizer

`tokenizer/standard.py` builds a normal fast Hugging Face tokenizer:

1. `canonical_words()` creates one asset-aware word for every semantic row, in
   row order, followed by one zero-semantic `<unk>` word.
2. `WordLevel(vocab={word: row}, unk_token="<unk>")` makes that table the
   complete vocabulary without shifting any semantic row ID.
3. `WhitespaceSplit` is the only pre-tokenizer.
4. `PreTrainedTokenizerFast` registers Doom's `bos` row as BOS and `done` as
   both EOS and padding, without adding new rows.

Automatic BOS insertion is disabled. The prompt builder explicitly emits `bos`
as its first row and `begin` as its last row. Generation stops when the model
emits the `done`/EOS row. Inference passes `add_special_tokens=False`, and decode
passes both `skip_special_tokens=False` and
`clean_up_tokenization_spaces=False`.

The stock tokenizer maps an unrecognized word to the final `<unk>` row. That row
is outside the renderer's semantic token table and has an all-zero embedding;
it exists to satisfy the standard tokenizer/compiler contract. Canonical
artifacts remain strict: the project codec and shipped formatter reject unknown
words, and the bundled prompt never contains `<unk>`. Every canonical input
therefore still names one exact semantic embedding row.

Detokenization is consequently just the inverse table lookup followed by a
single-space join. `tokenizer/codec.py` provides the lightweight project-side
equivalent:

```python
raw = raw_text_from_rows(words, rows)
rows_again = rows_from_raw_text(words, raw)
assert rows_again == rows
```

The codec, the shipped formatter's embedded codec, and the Hugging Face
tokenizer are tested for identical behavior over canonical vocabulary rows.
Negative and out-of-range rows are rejected; an empty row list maps to empty
text; arbitrary inter-word whitespace is accepted. On unknown text, the two
strict codecs raise an error while the stock tokenizer returns `<unk>`.

## Pretty text

The pretty, or beautified, format is produced after tokenization by
`tools/pretty_text.py`. The tool is copied byte-for-byte from
`portable/pretty_text.py` into each published bundle and uses only the Python
standard library. It loads two frozen data files:

- `doom_vocab.json` contains the ordered raw words, pretty per-row labels,
  structured row records, screen identity, and vocabulary fingerprint.
- `doom_tables.json` contains carrier row ranges, marker-to-physical-range
  bindings, angle markers, the scene origin, sentinel rules, and layout headers.

The bundle manifest hashes both files. The formatter refuses an incomplete
bundle, a hash mismatch, a screen mismatch, or a vocabulary-identity mismatch.
This keeps a formatter from quietly interpreting IDs from a different model.

Formatting happens in three steps.

### 1. Recover rows and labels

The formatter whitespace-splits the raw text and maps every canonical word back
to its row. It then looks up the row's pretty label. The label has the same
semantic spelling as the raw word, but uses normal punctuation spacing, for
example `bspFront(node=40, depth=0)`.

### 2. Fold and decode carriers

When a row is followed by `value` or `angleValue`, the formatter consumes both
rows and appends the decoded carrier as the marker's final positional argument.

For `value`, it:

1. finds the physical range associated with the marker;
2. recovers the carrier's exact one-of-65,536 quantization level;
3. maps that level from `[-1, 1]` into the physical range;
4. adds the frozen scene-origin offset for absolute X/Y coordinate markers;
5. prints the shortest decimal that quantizes back to the same carrier row.

The last step is important. A prettier rounded number is not accepted if it
would select an adjacent row. One-sided back-sector height sentinels render as
`none` rather than `-4096`.

For `angleValue`, the formatter converts signed BAM to degrees and likewise
chooses the shortest decimal that rounds back to the same BAM integer. Thus
`angleValue(angle=-420)` becomes `-18.46`, not an approximate value that loses
the original row.

An orphan carrier, a value after a non-marker, or an angle after a non-angle
marker is an error.

### 3. Add semantic whitespace

Header types such as `viewx`, `node`, `seg`, `R_Subsector`, `R_AddLine`,
`R_StoreWallRange`, and `setCursorX` start display groups. Each header has a
static nesting level from 0 through 2. The portable formatter writes the header
at `2 * level` spaces and its following fields on a continuation line four
spaces farther in. This is a shallow protocol-phase layout, not the live BSP
recursion depth -- that depth remains an explicit token slot.

Only whitespace changes in this step. The pretty parser ignores whitespace,
newlines, and `#` comments. It first tries to match each expression as a complete
per-row label; if that fails, it treats the final positional argument as a
folded carrier and reconstructs the two original rows. Therefore:

```text
row IDs -> raw text -> pretty text -> raw text -> row IDs
```

preserves the exact ID stream. Production does not need this inverse path, but
bundle publication smoke-tests it.

`tokenizer/surface.py` is the richer in-repository reference grammar over
structured `Token` objects. It exposes optional raw/physical carrier, BAM/degree,
asset-name, origin, and column-layout knobs. The shipped formatter fixes the
public choices to asset names, physical values, WAD coordinates, degrees, and
the frozen portable layout.

## Examples

### A prompt prefix

This is the beginning of the production E1M1 prompt in raw form:

```text
bos viewx value(v=-0.0549783) viewy value(v=-0.306844) viewz value(v=0.160143) viewangle angleValue(angle=2048) node(j=0) x value(v=0.229725) y value(v=0.582025) dx value(v=-0.0781109) dy value(v=-0.562493) child1(ss1) child0(ss29)
```

Using the bundle's frozen scene origin, the pretty form begins:

```text
bos
viewx(1056)
    viewy(-3616) viewz(41) viewangle(90)
node(j=0)
    x(1384) y(-2592) dx(-40) dy(-288) child1(ss1) child0(ss29)
```

The coordinate values are readable WAD coordinates, but they still reproduce
the exact quantized prompt rows.

### A generated traversal fragment

Raw -- 11 tokenizer IDs, with four angle carriers:

```text
R_Subsector(s=14,depth=7) R_AddLine(34) angle1 angleValue(angle=-420) theta1 angleValue(angle=-2468) angle2 angleValue(angle=-1025) theta2 angleValue(angle=-3073) nextSeg(34)
```

Pretty -- the carriers are folded and the protocol hierarchy is visible:

```text
R_Subsector(s=14, depth=7)
  R_AddLine(34)
      angle1(-18.46) theta1(-108.46) angle2(-45.04) theta2(-135.04)
  nextSeg(34)
```

It still represents 11 IDs. There are fewer printed expressions only because
each marker/carrier pair now reads as one expression.

### A ranged value

```text
# raw
segDcTmidMid value(v=0.121111) ds.uPhase angleValue(angle=0)

# pretty
segDcTmidMid(31.004) ds.uPhase(0)
```

`segDcTmidMid` selects range R3, `[-256, 256]`; the carrier row decodes to the
shortest safe decimal `31.004`. `ds.uPhase` is an angle marker, so its carrier
decodes in degrees.

## Commands and APIs

For a published bundle:

```bash
python infer.py --model . --prompt examples/e1m1_prompt.txt --output out
python tools/pretty_text.py --input out/output.txt --output out/output.pretty.txt
python tools/txt_to_png.py --input out/output.txt --output out/frame.png
```

The formatter also accepts stdin and writes stdout when `--input` or `--output`
is omitted. Outside a bundle, pass `--bundle /path/to/bundle`; otherwise each
tool resolves the bundle as the parent of its own `tools/` directory.

The project-side wrapper is:

```python
from torchwright_doom.interpret.formatter import DoomFormatter

formatter = DoomFormatter.from_bundle(bundle)
pretty = formatter.format_text(raw)
canonical_raw_again = formatter.parse_pretty_text(pretty)
```

For protocol meaning -- what `R_AddLine`, `wallColU`, cursor marks, pixels, and
the other expressions actually do -- see `PROTOCOL.md`. This document covers
only how those tokens become IDs and text.

## Source map and invariants

| Concern | Source of truth |
|---|---|
| Token types and slot domains | `model/vocab.py`, `model/tokens.py` |
| Row enumeration and embedding | `model/embedding.py` |
| Structured token/row conversion | `tokenizer/rows.py` |
| Per-row names and aliases | `tokenizer/display.py` |
| Stock Hugging Face tokenizer | `tokenizer/standard.py` |
| Raw words/rows codec | `tokenizer/codec.py` |
| Marker/range bindings | `model/marker_ranges.py`, `model/value_ranges.py` |
| Reference readable grammar | `tokenizer/surface.py` |
| Frozen formatter data | `tokenizer/freeze.py` |
| Shipped pretty formatter | `portable/pretty_text.py` |
| Inference artifacts | `torchwright_doom/infer.py` (shipped as `infer.py`) |
| Pixel detokenization | `portable/txt_to_png.py`, `interpret/decode.py` |

The principal tests are `tests/tokenizer/test_standard_tokenizer.py` (exhaustive
word/row bijection), `tests/tokenizer/test_codec.py` (codec and stock-tokenizer
parity), `tests/tokenizer/test_surface.py` (row and readable-grammar coverage),
and `tests/interpret/test_formatter.py` (bundle-driven pretty round-trip).
