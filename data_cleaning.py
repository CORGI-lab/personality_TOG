import os
from pathlib import Path
import pandas as pd
from typing import Optional, List, Dict
## Keys that define the "same hotel"
#KEY_COLS = ["place_name", "location", "star_rating"]  #hotel
# All fields you listed (kept here for clarity; code will work even if some are missing)
# ALL_FIELDS = [
#     "location", "number_of_rooms", "phone_number", "place_name", "price_per_night",
#     "smoking_allowed", "star_rating", "street_address", "check_in_date", "stay_length"
# ]
#DEDUP_KEY_COLS = ["place_name", "location", "star_rating", "street_address", "phone_number"]

##restaurant
KEY_COLS = ["street_address", "restaurant_name", "city"]   #restaurant
ALL_FIELDS = [
    "city", "cuisine", "has_live_music", "phone_number", "price_range",
    "restaurant_name", "serves_alcohol", "street_address", "date", "party_size","time"
]
DEDUP_KEY_COLS = ["cuisine", "restaurant_name", "city", "has_live_music", "price_range","serves_alcohol", "street_address", "phone_number"]

def normalize_hotels_csv(
    csv_path: str,
    out_path: str,
    sync_cols: Optional[List[str]] = None,
) -> dict:
    """
    For each group with same (place_name, location, star_rating),
    if price_per_night differs across rows, make rows identical to the first row
    (for sync_cols). By default, sync ALL non-key columns.

    Returns a small report dict.
    """
    csv_path = str(csv_path)
    df = pd.read_csv(csv_path)

    # Ensure required key cols exist
    missing_keys = [c for c in KEY_COLS if c not in df.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns {missing_keys} in {csv_path}")

    # Default: sync all non-key columns that exist in the file
    if sync_cols is None:
        sync_cols = [c for c in df.columns if c not in KEY_COLS]

    # Only sync columns that actually exist
    sync_cols = [c for c in sync_cols if c in df.columns]

    # Group and detect where price_per_night differs  #hotel
    # if "price_per_night" not in df.columns:
    #     raise ValueError(f"'price_per_night' column not found in {csv_path}")

    if "cuisine" not in df.columns:  #restaurant
        raise ValueError(f"'cuisine' column not found in {csv_path}")
    # Keep original for reporting
    before = df.copy()

    # Use group indices to update only groups with differing price_per_night
    groups = df.groupby(KEY_COLS, dropna=False, sort=False)
    affected_groups = 0
    affected_rows = 0

    for _, idx in groups.indices.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue

        ## If price_per_night differs within group -> canonicalize to first row
        #prices = df.loc[idx, "price_per_night"]
        cuisine = df.loc[idx,"cuisine"]
        # nunique handles NaNs; treat NaNs as a distinct value
        #if prices.nunique(dropna=False) > 1:
        if cuisine.nunique(dropna=False) > 1:
    
            affected_groups += 1
            affected_rows += len(idx)

            canonical = df.loc[idx[0], sync_cols]
            df.loc[idx, sync_cols] = canonical.values
    before = len(df)
    print("before: ", before)
    df = df.drop_duplicates(subset=DEDUP_KEY_COLS, keep="first")
    #df = df.drop_duplicates(subset=KEY_COLS, keep="first")

    after = len(df)
    print("after: ",after)
    # Write output
    if out_path is None:
        stem, ext = os.path.splitext(csv_path)
        out_path = f"{stem}_normalized{ext}"
    df.to_csv(out_path, index=False)

    # Simple sanity: how many rows had price changed?
    # price_changed = (before["price_per_night"] != df["price_per_night"]) & ~(
    #     before["price_per_night"].isna() & df["price_per_night"].isna()
    # )
    # address_changed = (before["cuisine"] != df["cuisine"]) & ~(
    #     before["cuisine"].isna() & df["cuisine"].isna()
    # )
    # #changed_count = int(price_changed.sum())
    # changed_count = int(address_changed.sum())
    return {
        "file": csv_path,
        "out": out_path,
        "affected_groups": affected_groups,
        "affected_rows": affected_rows,
        #"rows_with_price_changed": changed_count,
        "synced_columns": sync_cols,
    }

def normalize_all_csvs_in_dir(input_dir: str, pattern: str = "*.csv") -> list[dict]:
    """
    Normalize all CSV files in a directory. Writes <name>_normalized.csv next to each file.
    """
    input_dir = Path(input_dir)
    reports = []
    for csv_file in input_dir.glob(pattern):
        rep = normalize_hotels_csv(str(csv_file))
        reports.append(rep)
    return reports


# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Single file:
    #input_csv = "./service_results/Hotels_4.csv"
    input_csv = "./service_results/Restaurants_1.csv"
    out_path= "./service_results/Restaurants_1_normalized.csv"

    #out_path= "./service_results/Hotels_4_normalized2.csv"
    #eport = normalize_hotels_csv(input_csv,out_path,sync_cols=["number_of_rooms","price_per_night","street_address","phone_number"])
    report = normalize_hotels_csv(input_csv,out_path,sync_cols=["has_live_music", "price_range","serves_alcohol", "street_address", "phone_number"])

    #print(report)

    # Or all CSVs in a folder:
    # reports = normalize_all_csvs_in_dir("./data/hotels")
    # for r in reports:
    #     print(r)
