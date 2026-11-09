# Frame-level possession labelling (H / A / D)

Labels every frame of a tracking-only match as home possession (H), away possession (A) or dead ball (D) and writes
`frame,label` (199,319 rows). Training uses open data only (IDSSE/DFL, Metrica sample-data); the match to label is an input, nothing
else. One variant is released (s4): every tuned value has an external source (IDSSE, Metrica, organisers' documentation).
Expected output fingerprint (first 16 hex digits of the CSV SHA-256): `e1aa6ba433762e0a`.

## Run

```
python -m venv .venv && .venv/bin/pip install --require-hashes -r requirements.lock   # Python 3.11
export MTAH_DATA_DIR=/path/to/match/data        # the provided match, never stored here
python src/setup_external.py                    # clones the 2 repositories at pinned commits (checked)
python src/regenerate.py                        # s4  -> submissions/s4.csv
```
The IDSSE files (2.6 GB) are downloaded from figshare (version 1, HTTPS only, closed list of 21 files) and each file is checked against the SHA-256 committed in `configs/dfl_sha256.csv` by `src/fetch_dfl.py`; any mismatch deletes the file and stops. Training is not repeated: inference
runs on CPU from the text LightGBM weights in `models/v23s4/` (`SHA256SUMS` included, no pickle). The script
prints the SHA-256 prefix of the CSV; it must equal the fingerprint above. Training from scratch: `src/label_match.py --train`
(see its docstring).

## Pipeline
1. Tracking of open matches is degraded to look like automatic tracking (anonymised, fragmented tracks, absences, ball gaps).
2. Per-frame features (ball kinematics, stoppage detector, latent ball, player geometry); 5 LightGBM models (`K=5`).
3. Hazards for possession change, stoppage and restart are learned on DFL + Metrica; Viterbi decoding with a minimum-spell rule.
4. Restart anchors from the provided Smart Tagging file; `out` rule; short-possession repair.

## Settings and where they come from
| Setting | Value | Source |
|---|---|---|
| Training labels | IDSSE `BallStatus`/`BallPossession`, aligned to the 1.0 s spell definition (`dfl_relabel.py`) | IDSSE, task text |
| Metrica labels | pseudo-labels from Metrica events | Metrica sample-data |
| Decoding (`configs/decode.json`) | min spell 1.5 s, min dead 6.0 s, costs 0.5 / 2.0, alpha 3.0 | selected on Metrica development matches; re-check on IDSSE gave no gain |
| `stick` 0.8, `short_poss` 1.6 | grid search (widened until no fold sits on a bound), 7 IDSSE leave-one-match-out folds | IDSSE |
| Track-absence length `mean_dropout` 450 frames, minimum track piece `min_piece` 125 frames | chosen on the 7 IDSSE matches, leave-one-match-out, mean span-F1; rule fixed before the run (best mean wins, no significance threshold): `mean_dropout` in 50 / 150 / 450 / 1000 at `min_piece` 125, then `min_piece` in 25 / 75 / 125 / 250 / 375; scores in `configs/degradation_selection.json` | IDSSE |
| Extra ball-loss rate for Metrica training tracks, 0.04 | a-priori choice, not tuned (0 on IDSSE) | none |
| Track presence (0-0.15) and cut rate (0.5-2.0 per player) | drawn per seed in wide a-priori intervals | none (not fitted) |
| Bias `b` -1.036 | median over the 7 IDSSE folds (decoded share of D = true share of D), search interval widened until no fold at a bound | IDSSE |
| Ball profile, gap lengths, dead-ball missing rates | Metrica sample-data statistics | Metrica |
| Kick-off frames 110808-113741 all D | stated in the official FAQ | organisers' documentation |

Every constant is in `configs/` or at the top of the module that uses it. `score.py` only contains the span-F1 helpers; the
official metric (`possession_metrics`) is imported from the organisers' repository, never reimplemented.

## Data and licences
Code and model weights: MIT (`LICENSE`). **No data is redistributed.** Data keep their own terms:
- IDSSE: Bassek, M. et al. (2025), integrated dataset of spatiotemporal and event data in elite soccer, CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/),
  DOI 10.6084/m9.figshare.28196177.v1 (figshare; files are SHA-256-checked against `configs/dfl_sha256.csv` by `src/fetch_dfl.py`). Models trained with it are derived works: please cite it.
- Metrica Sports, sample-data (https://github.com/metrica-sports/sample-data, commit `e706dd506b360d69d9d123d5b8026e7294b13996`).
  The repository has no licence file; its README asks to be responsible with the data and to acknowledge the source.
- The organisers' repository (commit `3de6b60aeabd2475d95879a651d38b72bc4fba00`) and the provided match are used under the
  competition rules; the match data are read from `MTAH_DATA_DIR` and never copied.
