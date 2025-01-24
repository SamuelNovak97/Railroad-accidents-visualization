import pandas as pd
import geopandas as gpd
from extract_coords import get_state_boundaries
from shapely.geometry import Point, Polygon

# Load your data into a DataFrame
df = pd.read_csv('Railroad_Equipment_Accident_Incident_Source_Data__Form_54__20241113.csv', low_memory=False)

print("Loading...")

#filter NaN from Latitude and Longitud
df = df.dropna(subset=['Longitud', 'Latitude'])

# Create a GeoDataFrame from the DataFrame
geometry = [Point(xy) for xy in zip(df['Longitud'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

# Load USA boundaries from US Census Bureau TIGER/Line shapefiles
usa_url = 'cb_2020_us_nation_20m.zip'
usa = gpd.read_file(usa_url)

# Ensure both GeoDataFrames use the same coordinate reference system
gdf = gdf.to_crs(usa.crs)

# Use spatial join to filter points within the USA boundaries
usa_union = usa.geometry.union_all()
gdf_in_usa = gdf[gdf.within(usa_union)]

# Convert back to a regular DataFrame if needed
df_in_usa = pd.DataFrame(gdf_in_usa.drop(columns='geometry'))

# Filter out points with incorrect lat, lon, by chekcing whether the state attribute is accurate, for the specified (lat,lon)
state_boundaries = get_state_boundaries()
def state_and_coords_match(row):
    global state_boundaries
    if Polygon(state_boundaries[row["STATE"]]).contains(Point(row["Latitude"], row["Longitud"])):
        return True
    return False


df_in_usa = df_in_usa[df_in_usa.apply(state_and_coords_match, axis=1)].reset_index(drop=True)

df_in_usa.to_csv("filtered_railIncidents.csv", index=False)

print("Filtering completed.")