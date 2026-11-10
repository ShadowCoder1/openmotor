# OpenMotor

Open data for motor control and motor learning.

OpenMotor is a place to find open data from human motor control and motor learning experiments. Each study
links to the data its authors shared and to a standardized version: the same trials in one table with the same 90
columns, units and sign conventions, checked against results reported in the paper. It starts with 97 studies of
reaching and sensorimotor adaptation (12,498 participants, 5,286,546 trials) and welcomes data from any motor task.

Browse and download: https://physicalintelligencelab.github.io/openmotor/

## Files

- Dataset ZIPs (one per dataset, linked from the site) are attached to this repository's Releases: the first release's
  reaching datasets under "OpenMotor data, version frozen 4 November 2026", the other tasks under "Standardized datasets".
  Each holds the data as CSV, a README (citation, lab, license, original deposit, how the file was made, the published
  numbers it was checked against) and `columns.csv`.
- On the site: [catalog.csv](https://physicalintelligencelab.github.io/openmotor/catalog.csv) (every study with its paper,
  lab, participants, license, tags and a link to the authors' original data),
  [columns.csv](https://physicalintelligencelab.github.io/openmotor/columns.csv) (the 90 reaching columns) and
  [SHA256SUMS.txt](https://physicalintelligencelab.github.io/openmotor/SHA256SUMS.txt).
- This repository: `source/` (the study list and one record per standardized dataset), `site/` (the page),
  `tools/build_page.py` (builds the site; GitHub Actions runs it on every change) and `tools/std/` (the standardization
  library, the task formats in `modules/` and one converter per dataset in `convert/`).

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
