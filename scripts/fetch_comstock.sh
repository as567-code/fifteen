#!/bin/sh
# Download one ComStock 2025 R3 (AMY2018) MA building timeseries by bldg_id into $2.
id="$1"; out="$2"
base="https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/comstock_amy2018_release_3/timeseries_individual_buildings/by_state/upgrade=0/state=MA"
[ -f "$out/$id-0.parquet" ] || curl -sf -o "$out/$id-0.parquet" "$base/$id-0.parquet" || echo "fail $id"
