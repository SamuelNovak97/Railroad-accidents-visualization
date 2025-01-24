import pandas as pd
import numpy as np

#adding severity and constructing timedata DATE

df = pd.read_csv("filtered_railincidents.csv")


# We'll define a list of columns for which we want to compute z-scores. (We didnt end up using this)
zscore_cols = ["TOTKLD", "TOTINJ", "EQPDMG", "TRKDMG", "ACCDMG", "CARSHZD", "HIGHSPD"]

# Before computing z-scores, ensure the columns exist in df.
# Also handle any potential division-by-zero if std is 0.
for col in zscore_cols:
    if col not in df.columns:
        # If a column doesn't exist, create it (all zero)
        df[col] = 0

# Compute z-score manually or using pandas:
# z = (x - mean) / std
# We'll do ddof=0 for population std
for col in zscore_cols:
    mean_val = df[col].mean()
    std_val  = df[col].std(ddof=0)  # population standard deviation
    if std_val == 0:
        # If std is zero (all values the same), just set z-score to 0
        df[f"z_{col}"] = 0
    else:
        df[f"z_{col}"] = (df[col] - mean_val) / std_val

# Computing score for severity from 1 to 10
def compute_severity(row):
    """
    Approach:
    1. We sum up the scores of TOTKLD, TOTINJ, EQPDMG, TRKDMG, ACCDMG, CARSHZD, HIGHSPD.
    2. We clamp (limit) that sum to the range 1–10.
    3. Return an integer severity score.
    """
    # Grab each z-score from the row: (Not used)
    z_fatalities    = max(row.get("z_TOTKLD", 0), 0) #a
    z_injuries      = max(row.get("z_TOTINJ", 0), 0) #b
    z_eqp_damage    = max(row.get("z_EQPDMG", 0), 0) #c
    z_trk_damage    = max(row.get("z_TRKDMG", 0), 0) #c
    z_acc_damage    = max(row.get("z_ACCDMG", 0), 0) #c
    z_hazmat_cars   = max(row.get("z_CARSHZD", 0), 0) #d
    z_train_speed   = max(row.get("z_HIGHSPD", 0), 0) #e

    # Grab each score from the row:
    fatalities    = max(row.get("TOTKLD", 0), 0) #a
    injuries      = max(row.get("TOTINJ", 0), 0) #b
    eqp_damage    = max(row.get("EQPDMG", 0), 0) #c Not used
    trk_damage    = max(row.get("TRKDMG", 0), 0) #c Not used
    acc_damage    = max(row.get("ACCDMG", 0), 0) #c
    hazmat_cars   = max(row.get("CARSHZD", 0), 0) #d
    train_speed   = max(row.get("HIGHSPD", 0), 0) #e

    # Check if it was a passenger train
    is_passenger = row.get("PASSTRN", "N") #f
    passenger_value = 1 if is_passenger=="N" else 0

    #Coefficients
    k = 1 #Not used

    a = 0.45 * k
    b = 0.15 * k
    c = 0.00008 * k
    d = 0.1 * k
    e = 0.05 * k
    f = 0.125 * k

    
    raw_score = (a * fatalities + b * injuries + c * acc_damage + d * hazmat_cars + e * train_speed + f * passenger_value)
    
    #Limit the range to [1,10]
    if raw_score > 10:
        raw_score = 10
    if raw_score < 1:
        raw_score = 1

    # Round to nearest integer
    return int(round(raw_score))

# Apply the above function to the severity column
df["severity"] = df.apply(compute_severity, axis=1)
print(df["severity"].describe())

# OPTIONAL: preview the result
print(df[[
    "YEAR", "MONTH", "DAY", "TOTKLD", "TOTINJ", "EQPDMG", "TRKDMG", 
    "ACCDMG", "CARSHZD", "HIGHSPD", "severity"
]].head(20))


# Safely create the 'DATE' column
def safe_to_datetime(row):
    try:
        return pd.to_datetime(f"{row['YEAR']}-{row['MONTH']}-{row['DAY']}")
    except ValueError:
        return pd.NaT  # Assign NaT for invalid dates

# Apply the function to create the DATA column
df['DATE'] = df.apply(safe_to_datetime, axis=1)

print(df[["DATE"]])
df.to_csv("filtered_plus_severity.csv", index=False)
