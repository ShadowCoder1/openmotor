# OpenMotor

Open data for motor control and motor learning.

OpenMotor is an open archive of human motor control and motor learning experiments. Each dataset is converted
into a common trial-level format, checked against the results reported in its paper, and free to download. It
starts with 97 studies of reaching and sensorimotor adaptation (12,498 participants and 5,286,546 trials, every
one in the same trial-by-trial table with the same 90 columns, units and sign conventions) and welcomes data from
any motor task.

Browse and download: https://shadowcoder1.github.io/openmotor/

## Files

- `data/<dataset>.zip`: one file per dataset, with the trial table as CSV, a README (citation, lab, license,
  original deposit) and `columns.csv`.
- `catalog.csv`: every study with its paper, lab, counts, license, tags and download.
- `columns.csv`: what each of the 90 columns means.
- `SHA256SUMS.txt`: checksums of the data files.

## Licenses

Each dataset keeps the license its authors chose; its README states it. Datasets first shared through the
OpenMotor database are CC BY 4.0. Some original deposits list no license; those are shared with credit to
their authors.

**Authors:** to correct, relicense or remove a dataset, open an issue in this repository. To add a dataset,
use the Add your data page on the site.

## Citing

A paper describing OpenMotor is in preparation. Until then, cite the original paper for every dataset you
use; each README gives the reference.

OpenMotor is curated by the Tsay lab at Carnegie Mellon University.
