#!/usr/bin/env bash
# Download the ACS 1-year national PUMS files (person + housing CSV zips) used by the paper:
# 2009-2013 (historical window and 2013 base year), 2018, 2021 and 2024 (scenario references).
# Source: U.S. Census Bureau, https://www2.census.gov/programs-surveys/acs/data/pums/<year>/1-Year/
# The raw files are not redistributed with this repository (about 19 GB unzipped).
#   bash data_prep/download_pums.sh [RAW_DIR]        # default RAW_DIR = data/raw
# File names after unzipping: 2009-2013 ss<yy>{pus,hus}{a,b}.csv; 2018 onward psam_{pus,hus}{a,b}.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${1:-data/raw}
for y in 2009 2010 2011 2012 2013 2018 2021 2024; do
  mkdir -p "$RAW/$y"
  for f in csv_pus.zip csv_hus.zip; do
    wget -q -c -O "$RAW/$y/$f" "https://www2.census.gov/programs-surveys/acs/data/pums/$y/1-Year/$f" && echo "downloaded $y $f"
  done
  (cd "$RAW/$y" && unzip -o -q csv_pus.zip && unzip -o -q csv_hus.zip && echo "unzipped $y")
done
echo "done -> $RAW"
