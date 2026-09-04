# RBC `class_code` mapping

Dataset: `dp_8k_query_hybrid_adaptive_preview/rbc_query.parquet`

Source of truth: `rbc_query_manifest.json` (`classEncoding` and `edgeDetection`).

## Important: `class_code` is packed

One RBC row has one `class_code`. The value combines four classification
fields and the independent edge-cell flag:

```text
class_code = color + size + shape + inclusion + edge
```

Therefore, a whole `class_code` value does not normally belong to only one RBC
type. Each type below contributes its value to the final code.

## Color — bits 0–1

Decode: `class_code & 3`

| Contribution | Stored type |
|---:|---|
| 0 | `normal_color` |
| 1 | `hyperchromatic` |
| 2 | `hypochromatic` |
| 3 | `unclassified_stain` |

## Size — bits 2–4

Decode: `(class_code >> 2) & 7`

| Contribution | Decoded value | Stored type |
|---:|---:|---|
| 0 | 0 | `microcytes` |
| 4 | 1 | `normal_size` |
| 8 | 2 | `macrocytes` |
| 12 | 3 | `unclassified_size` |
| 16 | 4 | `unclassified_stain` |

## Shape — bits 5–8

Decode: `(class_code >> 5) & 15`

| Contribution | Decoded value | Stored type |
|---:|---:|---|
| 0 | 0 | `normal_shape` |
| 32 | 1 | `target_cells` |
| 64 | 2 | `unclassified_shape` |
| 96 | 3 | `ovalocytes` |
| 128 | 4 | `schistocytes` |
| 160 | 5 | `spherocytes` |
| 192 | 6 | `normocytes` |
| 224 | 7 | `stomatocytes` |
| 256 | 8 | `elliptocytes` |
| 288 | 9 | `bite_cells` |
| 320 | 10 | `tear_drop` |
| 352 | 11 | `echinocytes` |
| 384 | 12 | `teardrop_cells` |
| 416 | 13 | `acanthocytes` |
| 448 | 14 | `sickle_cells` |
| 480 | 15 | `unclassified_stain` |

## Inclusion — bit 9

Decode: `(class_code >> 9) & 1`

| Contribution | Decoded value | Stored type |
|---:|---:|---|
| 0 | 0 | `no_inclusion` |
| 512 | 1 | `unclassified_inclusions` |

## Edge cell — bit 10

Decode:

```text
(class_code & 1024) != 0
```

| Contribution | Meaning |
|---:|---|
| 0 | Normal/non-edge cell |
| 1024 | Edge cell |

The edge bit does not replace color, size, shape, or inclusion. It is added to
the same packed code. To remove the edge flag while preserving classification:

```text
classification_code = class_code & 1023
```

## Examples

Normal color + normal size + normal shape + no inclusion:

```text
0 + 4 + 0 + 0 = class_code 4
```

Hyperchromatic + macrocyte + target cell + no inclusion:

```text
1 + 8 + 32 + 0 = class_code 41
```

The same cell touching an FOV edge:

```text
41 + 1024 = class_code 1065
```

Hypochromatic + microcyte + spherocyte + unclassified inclusion:

```text
2 + 0 + 160 + 512 = class_code 674
```

## Filtering rules

Do not filter a classification by comparing the complete code with its
contribution. Extract only that field:

```text
normal_color:      (class_code & 3) = 0
hyperchromatic:    (class_code & 3) = 1
hypochromatic:     (class_code & 3) = 2

microcytes:        ((class_code >> 2) & 7) = 0
normal_size:       ((class_code >> 2) & 7) = 1
macrocytes:        ((class_code >> 2) & 7) = 2

normal_shape:      ((class_code >> 5) & 15) = 0
target_cells:      ((class_code >> 5) & 15) = 1
spherocytes:       ((class_code >> 5) & 15) = 5

no_inclusion:      ((class_code >> 9) & 1) = 0
edge_cell:         (class_code & 1024) != 0
non_edge_cell:     (class_code & 1024) = 0
```

The React UI excludes edge cells from ordinary classification rows and shows
them separately by testing bit 10.
