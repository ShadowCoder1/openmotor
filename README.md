# OpenMotor

Human reaching experiments in one format.

OpenMotor is a database of sensorimotor adaptation experiments: people reach to visual targets while the
experimenter changes how a cursor follows the hand (rotations, error clamps, mirrors, rewards). It holds
97 studies, 12,498 participants and 5,286,546 trials, every one converted to the same trial-by-trial table
with the same 90 columns, units and sign conventions.

Browse and download: https://shadowcoder1.github.io/openmotor/

## Files

- `data/<dataset>.zip`: one file per dataset, with the trial table as CSV, a README (citation, lab, license,
  original deposit) and `columns.csv`.
- `catalog.csv`: every study with its paper, lab, counts, license and download.
- `columns.csv`: what each of the 90 columns means.
- `SHA256SUMS.txt`: checksums of the data files.

## Licenses

Each dataset keeps the license its authors chose; its README states it. Datasets first shared through the
OpenMotor database are CC BY 4.0. Some original deposits list no license; those are shared with credit to
their authors.

**Authors:** to correct, relicense or remove a dataset, open an issue in this repository.

## Citing

A paper describing OpenMotor is in preparation. Until then, cite the original paper for every dataset you
use; each README gives the reference.

OpenMotor is curated by the Tsay lab at Carnegie Mellon University.
