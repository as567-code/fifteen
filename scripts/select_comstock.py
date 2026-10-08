"""Pick the ComStock Massachusetts buildings a single Powerblock is relevant for (annual peak 300 kW - 3 MW)."""
import glob, os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cols = ["bldg_id", "weight", "in.sqft..ft2", "in.county_name", "in.comstock_building_type", "in.comstock_building_type_group",
        "in.electric_utility_eia_code", "in.weekday_operating_hours..hr", "out.electricity.total.energy_consumption..kwh",
        "out.electricity.total.peak_demand..kw"]
df = pd.concat([pd.read_parquet(f, columns=cols) for f in glob.glob(os.path.join(ROOT, "data/raw/comstock_meta/*.parquet"))])
df.columns = ["bldg_id", "weight", "sqft", "county", "btype", "bgroup", "eia", "wk_hours", "kwh", "peak_kw"]
u = df.sort_values("weight", ascending=False).drop_duplicates("bldg_id")  # buildings are apportioned to several counties
r = u[(u.peak_kw >= 300) & (u.peak_kw <= 3000)].copy()
r["utility"] = r.eia.map({11804: "National Grid", 54913: "Eversource"}).fillna("Municipal/other")
os.makedirs(os.path.join(ROOT, "data/processed"), exist_ok=True)
r.to_csv(os.path.join(ROOT, "data/processed/comstock_library_meta.csv"), index=False)
open(os.path.join(ROOT, "data/raw/comstock_ts/ids_all.txt"), "w").write("\n".join(str(i) for i in r.bldg_id) + "\n")
print(len(r), "buildings")
