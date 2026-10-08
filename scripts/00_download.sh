#!/bin/sh
# Fetch every public dataset the pipeline uses into data/raw/ (about 2 GB transient, 300 MB kept).
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw/comstock_meta data/raw/comstock_ts data/raw/kit_industrial data/raw/uci_ld data/raw/isone

# NREL ComStock 2025 release 3 (AMY2018): Massachusetts metadata, one file per county
R="https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/comstock_amy2018_release_3"
for c in 001 003 005 007 009 011 013 015 017 019 021 023 025 027; do
  curl -sf -o "data/raw/comstock_meta/MA_G250${c}0_upgrade0.parquet" \
    "$R/metadata_and_annual_results/by_state_and_county/full/parquet/state=MA/county=G250${c}0/MA_G250${c}0_upgrade0.parquet"
done
.venv/bin/python -I scripts/select_comstock.py   # writes data/processed/comstock_library_meta.csv + ids_all.txt
xargs -P 10 -n 1 -I{} scripts/fetch_comstock.sh {} data/raw/comstock_ts < data/raw/comstock_ts/ids_all.txt

# KIT: load profiles of 50 industrial plants (Zenodo 3899018, CC-BY-4.0)
for f in LoadProfile_20IPs_2016.csv LoadProfile_30IPs_2017.csv; do
  curl -sfL -o "data/raw/kit_industrial/$f" "https://zenodo.org/api/records/3899018/files/$f/content"
done

# UCI ElectricityLoadDiagrams20112014 (CC-BY-4.0)
curl -sfL -o data/raw/uci_ld/ld.zip "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
(cd data/raw/uci_ld && unzip -o -q ld.zip && rm -rf __MACOSX ld.zip)
.venv/bin/python -I -c "import pandas as pd, numpy as np; pd.read_csv('data/raw/uci_ld/LD2011_2014.txt', sep=';', decimal=',', index_col=0, parse_dates=True).astype(np.float32).to_parquet('data/raw/uci_ld/ld.parquet')"
rm -f data/raw/uci_ld/LD2011_2014.txt

# ISO-NE 2018 SMD hourly (system load, used for ConnectedSolutions event days)
curl -sf -A "Mozilla/5.0" -o data/raw/isone/2018_smd_hourly.xlsx "https://www.iso-ne.com/static-assets/documents/2018/02/2018_smd_hourly.xlsx"
echo "done"
