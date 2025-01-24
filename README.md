# Railroad Accidents Visualization

## How to Run:
1. **Download and unzip the project** – This will extract the project into a folder in your chosen path.
2. **Navigate to the folder** – Press `Shift + Right Click` and select **"Open PowerShell window here"**.
3. **Install dependencies** – Copy and paste the following command into PowerShell and press Enter:
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the project** – Execute the following command:
   ```sh
   python3 main.py
   ```
5. **Access the visualization** – Once the script runs, it will output:
   ```
   Running on http://127.0.0.1:8050
   ```
   Copy and paste this link into your browser to view the visualization.

---

## Project Structure

All Python files have been written by me without using any external code, except for the **polygon coordinates** in `extract_coords.py`, which were sourced from:
[PublicaMundi GitHub Repository](https://github.com/PublicaMundi/MappingAPI/blob/master/data/geojson/us-states.json?short_path=1c1ebe5).  
However, the code in this file has been written entirely by me.

There are **two additional files** in the directory that were not written by me:
- **`Railroad_Equipment_Accident_Incident_Source_Data__Form_54__20241113.csv`** – The original data file.
- **`cb_2020_us_nation_20m.zip`** – Contains border coordinates of each US state.

---

## File Descriptions

### `filtering_points.py`
- Cleans the original `.csv` file by **removing rows** with missing latitude/longitude values.
- Discards rows with **incorrect location data** based on their state attribute.
- Uses **`cb_2020_us_nation_20m.zip`** to validate whether a point belongs to the correct state.
- **Output:** `filtered_railIncidents.csv`

### `appending_severity.py`
- Computes and appends a **severity** attribute to `filtered_railIncidents.csv`.
- Constructs a **DATE attribute** in `datetime` format using YEAR, MONTH, and DAY columns.
- **Details on severity computation are available in the report.**
- **Output:** `filtered_plus_severity.csv`

### `extract_coords.py`
- Converts external **polygon/multipolygon** border coordinates into simplified **polygon structures**.
- When called from `main.py` via `get_state_boundaries()`, it returns a **dictionary** where:
  - **Keys** = State index.
  - **Values** = State boundary coordinates.

### `state_codes.py`
- Contains the **index mapping** of each U.S. state.
- Used for **development purposes** and to assist graders in understanding state indexing.

### `main.py`
- The **core script** that runs the **Dash** web application.

---

## Dependencies

```txt
dash
dash_leaflet==0.0.6
pandas
matplotlib
numpy
datetime
plotly
shapely
geopandas
json
```
