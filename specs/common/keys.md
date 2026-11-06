# Record Keys

Related: **Read first** [records_dataset](records_dataset.md). **See also** [split](split.md), [similarity](similarity.md), [eda cli](../eda/cli.md).

## Overview

A key spec is a short string that says how to read one value per record, such as the first folder of its path or a column of a CSV. Commands that need to partition records take key specs as data: `split --group-by` keeps records with the same key together, `split --stratify-by` balances per-key counts, `sample --stratify-by` samples inside each key value, and `eda --group-by` draws one panel per key. The tool never learns what a board, a scene or a survey is. A project states that in a regex, an attribute name or a file.

A group is a set of records linked by keys. When several specs are given, two records share a group when they share a value under any spec, so groups are the connected components of that relation.

## Language

- **Key spec**: `kind:argument`, parsed by `parse_key`.
- **Layout path**: the record's relative image path with leading layout folders removed, always with forward slashes.
- **Group**: the set of records that share a key value under at least one of the given specs, directly or through other records.

## Map

- `common/keys.py`: `parse_key`, `layout_path`, `group_records`.


## Contracts

### Key specs

| spec | value of a record |
|---|---|
| `regex:PATTERN` | `re.search` on the layout path including the file name; the first capture group, or the whole match when the pattern has none |
| `dir:N` | the Nth folder of the layout path, counting from 0 at the dataset root; a negative N counts from the end; the file name is not a folder |
| `attr:NAME` | `str(record.attributes[NAME])` |
| `file:PATH.csv` | the second column of the row whose first column equals the layout path, or else the file name when the file name is unique in the dataset |
| `split` | the record's split (`train`, `val` or `test`); valid wherever one value per record is read, such as `sample --stratify-by` |
| `similar:BITS` | the id of the connected component of records whose perceptual hashes are within BITS bits ([similarity](similarity.md)); valid for group specs only |

Examples: `dir:0` on `pcb1/Data/Images/Anomaly/000.JPG` gives `pcb1`; `dir:-1` gives `Anomaly`; `regex:^(\w+)-` on `oil-3-cam2-0042.jpg` gives `oil`.

### Layout path

`layout_path(record, dataset)` takes `record.rel_image_path`, or the image path relative to `dataset.root`, or the file name when neither exists. It then drops leading components that name a split (`train`, `val`, `valid`, `validation`, `test`, `eval`) or `images`, in any order, so `train/images/a.jpg`, `images/train/a.jpg` and `a.jpg` give the same path and `dir:0` is the first folder of the original layout.

### Grouping

```python
group_records(dataset, specs: list[str], *, bits=None) -> list[str]
```

Returns one group id per record, aligned with `dataset.records`. Ids are the smallest member key of the component, so they are the same for any record order. With no specs, each record is its own group. A `similar:` spec hashes the images unless the caller passes `bits`, the records' packed hashes in [similarity](similarity.md) layout.

## Constraints

- A record that a spec cannot resolve MUST raise `ValueError` naming the spec and up to five offending paths. No record gets an empty or default key.
- An unknown spec kind MUST raise and list the valid kinds. `dir:N` with N beyond the path depth MUST raise.
- A `file:` table MUST have two columns, MUST NOT repeat a first-column value, and a file name that is not unique in the dataset MUST NOT be accepted as a match.
- `similar:` MUST raise when it is given to a command that needs one value per record in isolation.
- Group ids MUST NOT depend on record order or on the Python hash seed. This is checked by shuffling the records.

## History

- [What belongs in cvsuite](../logs/2026-11-05_what-belongs-in-cvsuite.md): why group definitions enter as data.
- [The eda branch](../logs/2026-11-05_eda-branch.md): why `--group-by` carries the same meaning in `eda`.
- [Gaps found by migrating the projects](../logs/2026-11-06_migration-gaps.md): why the `split` key exists.
