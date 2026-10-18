import os
import geopandas as gpd
import math
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, Point, LineString
from rasterstats import zonal_stats

#region INITIAL LOADING
current_season = "Dry"  # "Dry" or "Wet"

shapefile_path = "WDPA_WDOECM_Mar2026_Public_884_shp-polygons.shp"  # Downloaded from https://www.protectedplanet.net/884 (thank you reddit)
park = gpd.read_file(shapefile_path)
park = park.to_crs(epsg=32733)

park_geometry = park.geometry.union_all()

minx, miny, maxx, maxy = park_geometry.bounds
grid_size = 5000  # in metres

grid_cells = []
for x0 in np.arange(minx, maxx, grid_size):
    for y0 in np.arange(miny, maxy, grid_size):
        x1 = x0 + grid_size
        y1 = y0 + grid_size
        # Draw a 5x5km square
        cell = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        grid_cells.append(cell)

gridWhole = gpd.GeoDataFrame(geometry=grid_cells, crs=park.crs)

grid_inside = gpd.sjoin(gridWhole, gpd.GeoDataFrame(geometry=[park_geometry], crs=park.crs), how='inner', predicate='intersects')
grid_inside = grid_inside.drop(columns=['index_right'])
#endregion

#region CELL VARIABLES
park_boundary = park_geometry.boundary

grid_inside['Dist_to_Fence'] = grid_inside.geometry.centroid.distance(park_boundary) / 1000.0  # is basically distance to boundary
grid_inside['Coord_X'] = ((grid_inside.geometry.centroid.x - minx) // grid_size).astype(int)
grid_inside['Coord_Y'] = ((grid_inside.geometry.centroid.y - miny) // grid_size).astype(int)

grid_inside['Cell_ID'] = 'C_' + grid_inside['Coord_X'].astype(str) + '_' + grid_inside['Coord_Y'].astype(str)

grid_inside['Terrain_Type'] = "Tree Savanna"
grid_inside['Has_Water'] = 0
grid_inside['Dist_to_Road'] = 0.0
grid_inside['Dist_to_Camp'] = 0.0
grid_inside['Dist_to_Saltpan'] = 0.0
grid_inside['Has_Road'] = 0  # 0 = no road, 1 = road

grid_inside['Y_Burn'] = 0  # 0 = No burn, 1 = Controlled burn conducted

grid_inside['APS'] = 0.0
grid_inside['Threat_Score'] = 0.0

grid_inside['Count_FP'] = 0
grid_inside['Count_4x4'] = 0
grid_inside['Count_Drones'] = 0
grid_inside['Count_Cams'] = 0

grid_inside['Zone_Number'] = 0

grid_inside['T50'] = 0
grid_inside['k'] = 0
#endregion

#region LOAD INFRASTRUCTURE AND WATER HOLES AND OTHER THINGS THAT MAKE MY MIND GO WROOM
roads = gpd.read_file("etosha_roads.geojson")  # roads are from OpenStreetMap (downloaded from overpass-turbo.eu)
roads = roads.to_crs(epsg=32733)
all_roads = roads.geometry.union_all()

# load the csv that we made (we made it using x and y coordinate of every waterhole pixel on original map on (https://www.etoshanationalpark.org/media/Etosha-Map-2025.jpg) and we did the same for camps
camps_df = pd.read_csv("etosha_camps.csv")
waterholes_df = pd.read_csv("etosha_waterholes.csv")
gates_df = pd.read_csv("etosha_gates.csv")

P_xmin = 182  # x pixel of the furthest left tip
P_xmax = 9795  # x pixel of the furthest right tip
P_ymin = 675  # y pixel of the highest tip
P_ymax = 4180  # y pixel of the lowest tip
scale_x = (maxx - minx) / (P_xmax - P_xmin)
scale_y = (maxy - miny) / (P_ymax - P_ymin)

def pixels_to_meters(df):
    df['Easting'] = minx + ((df['X_pixel'] - P_xmin) * scale_x)
    df['Northing'] = maxy - ((df['Y_pixel'] - P_ymin) * scale_y)
    return df


camps_df = pixels_to_meters(camps_df)
waterholes_df = pixels_to_meters(waterholes_df)
gates_df = pixels_to_meters(gates_df)

# EPSG:32733 (it's the best for namibia goat apparently)
camps = gpd.GeoDataFrame(camps_df, geometry=gpd.points_from_xy(camps_df['Easting'], camps_df['Northing']), crs="EPSG:32733")

waterholes = gpd.GeoDataFrame(waterholes_df, geometry=gpd.points_from_xy(waterholes_df['Easting'], waterholes_df['Northing']), crs="EPSG:32733")

gates = gpd.GeoDataFrame(gates_df, geometry=gpd.points_from_xy(gates_df['Easting'], gates_df['Northing']),crs="EPSG:32733")

# turning on all waterholes. 1 = pump on, 0 = pump off so area dry nearby
# permanent are always 1, while seasonal are 0 in the dry season [Default Setting]
waterholes['Pump_Status'] = np.where(waterholes['Validity'] == 'P', 1, 0)
waterholes['Change'] = 0

all_camps = camps.geometry.union_all()
all_waterholes = waterholes.geometry.union_all()
all_gates = gates.geometry.union_all()

grid_inside['Dist_to_Road'] = grid_inside.geometry.centroid.distance(all_roads) / 1000.0
grid_inside['Dist_to_Camp'] = grid_inside.geometry.centroid.distance(all_camps) / 1000.0
grid_inside['Dist_to_Water'] = grid_inside.geometry.centroid.distance(all_waterholes) / 1000.0

grid_inside['Has_Road'] = grid_inside.geometry.intersects(all_roads).astype(int)
grid_inside['Has_Camp'] = grid_inside.geometry.intersects(all_camps).astype(int)
grid_inside['Has_Gate'] = grid_inside.geometry.intersects(all_gates).astype(int)

# if a cell is less than or equal to 10km from a waterhole, it has water (value of 1). This is a binary variable, and not the distance wala variable :D
grid_inside['Has_Water'] = (grid_inside['Dist_to_Water'] <= 10).astype(int)

# we made these zones using our custom zone maker
zones_df = pd.read_csv("manual_zones.csv")

if 'Zone_Number' in grid_inside.columns:  # this stupid thing costed me so much time.
    grid_inside = grid_inside.drop(columns=['Zone_Number'])

grid_inside = grid_inside.merge(zones_df[['Cell_ID', 'Zone_Number']], on='Cell_ID', how='left')
#endregion

#region TERRAIN & FORAGE CLASSIFICATION
# load manually digitized data we made manually; the biome data is based on (https://www.researchgate.net/figure/Map-of-the-study-area-Etosha-National-Park-Namibia-The-locations-of-sampling-sites_fig1_318895272)
manual_data = pd.read_csv("manual_biomes.csv")

if 'Terrain_Type' in grid_inside.columns:
    grid_inside = grid_inside.drop(columns=['Terrain_Type'])

grid_inside = grid_inside.merge(manual_data, on='Cell_ID', how='left')

# dist to salt pan
salt_pan_cells = grid_inside[grid_inside['Terrain_Type'] == 'Salt Pan']
if not salt_pan_cells.empty:
    all_salt_pans = salt_pan_cells.geometry.union_all()
    grid_inside['Dist_to_SaltPan'] = grid_inside.geometry.centroid.distance(all_salt_pans) / 1000.0
else:
    grid_inside['Dist_to_SaltPan'] = 999.0

# we got our both ndvi and ndmi files from NASA Earthdata AppEEARS.
# FOR NDVI
# load up ndvi tif files of both seasons and both historical files
ndviFiles = {
    'Wet': 'etosha_ndvi_wet.tif',
    'Dry': 'etosha_ndvi_dry.tif',
    'Min': 'etosha_ndvi_min.tif',
    'Max': 'etosha_ndvi_max.tif'
}

for name, tiff_path in ndviFiles.items():
    with rasterio.open(tiff_path) as src:
        tiff_crs = src.crs

    grid_for_extraction = grid_inside.to_crs(tiff_crs)
    stats = zonal_stats(grid_for_extraction, tiff_path, stats=['mean'], nodata=-3000.0)

    # This da lovely and mwah mwah MODIS data (thank you nasa i love you)
    raw_ndvi = [s['mean'] if s['mean'] is not None else 0.0 for s in stats]
    raw_array = np.array(raw_ndvi) / 10000.0
    raw_array = np.where(raw_array < -1.0, 0.0, raw_array)
    grid_inside[f'Raw_NDVI_{name}'] = raw_array

    min_ndvi, max_ndvi = 0.1, 0.45
    scaled_forage = (raw_array - min_ndvi) / (max_ndvi - min_ndvi)
    grid_inside[f'Base_Forage_{name}'] = np.clip(scaled_forage, 0.0, 1.0)

# initialize the active Base_Forage and Raw_NDVI based on the starting Current_Season (which happens to be dry season)
grid_inside['Base_Forage'] = grid_inside[f'Base_Forage_{current_season}']
grid_inside['Raw_NDVI'] = grid_inside[f'Raw_NDVI_{current_season}']

# FOR NDMI
# load up ndmi tif files of both seasons (SWIR and NIR bands)
ndmiFiles = {
    'Wet': {'swir': 'modis_swir_wet.tif', 'nir': 'modis_nir_wet.tif'},
    'Dry': {'swir': 'modis_swir_dry.tif', 'nir': 'modis_nir_dry.tif'},
    'Max': {'swir': 'modis_swir_max.tif', 'nir': 'modis_nir_max.tif'},
    'Min': {'swir': 'modis_swir_min.tif', 'nir': 'modis_nir_min.tif'}
}

for name, bands in ndmiFiles.items():
    with rasterio.open(bands['swir']) as swir_src, rasterio.open(bands['nir']) as nir_src:
        swir = swir_src.read(1).astype('float32')
        nir = nir_src.read(1).astype('float32')
        affine = swir_src.transform
        tiff_crs = swir_src.crs

    nir[nir < 0] = np.nan
    swir[swir < 0] = np.nan
    # again, thank you nasa mwah (and reddit too)
    grid_for_extraction = grid_inside.to_crs(tiff_crs)

    # NDMI = (NIR - SWIR) / (NIR + SWIR)
    np.seterr(divide='ignore', invalid='ignore')  # ignore divide by zero for empty pixels
    ndmi_array = (nir - swir) / (nir + swir)

    ndmi_array[(ndmi_array < -1.0) | (ndmi_array > 1.0)] = np.nan
    stats = zonal_stats(grid_for_extraction, ndmi_array, affine=affine, stats=['mean'], nodata=np.nan)

    # if a cell has no data, we default it to -1.0 (i noticed someimtes therea re cells missing data)
    grid_inside[f'NDMI_{name}'] = [s['mean'] if s['mean'] is not None else 0.0 for s in stats]

# initialize the active NDMI based on the starting Current_Season (which ofcourse is our beloved dry season)
grid_inside['NDMI'] = grid_inside[f'NDMI_{current_season}']

def Update_Forage():
    # Fxy = min(1.0, Base_Forage + (Y_Burn * 0.25))  this da 1st formula (this for vegetation attractiveness while f1 is for F_Friction
    grid_inside['Forage_Quality'] = np.minimum(1.0, grid_inside['Base_Forage'] + (grid_inside['Y_Burn'] * 0.25))


Update_Forage()

# This da time to camp for each cell
if os.path.exists('routing_wet.csv') and os.path.exists('routing_dry.csv'):  # added this cuz if file doesnt exist we have a chance to generate it without getting smacked with an error
    r_wet = pd.read_csv('routing_wet.csv')
    r_dry = pd.read_csv('routing_dry.csv')

    grid_inside = grid_inside.merge(r_wet, on='Cell_ID', how='left')
    grid_inside = grid_inside.merge(r_dry, on='Cell_ID', how='left')

    # initialize the active Time_to_Camp column based on current season
    grid_inside['Time_to_Camp'] = grid_inside[f'Time_to_Camp_{current_season}']

# ***END OF TERRAIN THINGY TYPA THINGY THAT MAKES MIND GO MORE WROOM***
#endregion

#region CALCULATIONS OF VARIABLES USING FORMULAS
def Calculate_Time_To_Camp():
    nodes_df = pd.read_csv('routing_nodes.csv')
    nodes_df = nodes_df.merge(grid_inside[['Cell_ID', 'geometry']], on='Cell_ID', how='inner')

    times_wet = []
    times_dry = []
    spatial_index = grid_inside.sindex

    def calculate_segment_time(ray, season, matches=None):
        if matches is None:
            possible_match_idx = list(spatial_index.intersection(ray.bounds))
            possible_matches = grid_inside.iloc[possible_match_idx]
            matches = possible_matches[possible_matches.intersects(ray)]

        total_time = 0.0
        for _, cr in matches.iterrows():
            segment = cr.geometry.intersection(ray)
            dist_km = segment.length / 1000.0

            if cr['Has_Road'] == 1:
                v = 70.0
            else:
                base_f = cr[f'Base_Forage_{season}']
                v = 10.0 + 30.0 * (1.0 - base_f)

            total_time += (dist_km / v) * 60.0
        return total_time, matches

    grid_inside['Speed_Dry'] = np.where(grid_inside['Has_Road'] == 1,70.0, 10.0 + 30.0 * (1.0 - grid_inside['Base_Forage_Dry']))

    grid_inside['Speed_Wet'] = np.where((grid_inside['Terrain_Type'] == 'Salt Pan') & (grid_inside['Has_Road'] == 0), 0.0, np.where(grid_inside['Has_Road'] == 1, 70.0, 10.0 + 30.0 * (1.0 - grid_inside['Base_Forage_Wet'])))

    Avg_Speed_Dry = grid_inside['Speed_Dry'].mean()
    Avg_Speed_Wet = grid_inside['Speed_Wet'].mean()

    print(f"Avg Speed (Dry): {Avg_Speed_Dry:.2f} km/h")
    print(f"Avg Speed (Wet): {Avg_Speed_Wet:.2f} km/h")

    for i, origin_row in grid_inside.iterrows():
        origin_centroid = origin_row.geometry.centroid

        distances_to_camps = camps.distance(origin_centroid)
        nearest_idx = distances_to_camps.idxmin()
        nearest_camp_geom = camps.loc[nearest_idx].geometry

        ray_dry_direct = LineString([origin_centroid, nearest_camp_geom])
        time_dry_direct, j = calculate_segment_time(ray_dry_direct, "Dry")
        best_dry_time = time_dry_direct

        for j, node_row in nodes_df.iterrows():
            node_centroid = node_row.geometry.centroid
            ray_node = LineString([origin_centroid, node_centroid])

            time_to_node, j = calculate_segment_time(ray_node, "Dry")
            total_route_time = time_to_node + node_row['Static_Time']

            if total_route_time < best_dry_time:
                best_dry_time = total_route_time

        times_dry.append(best_dry_time)

        if origin_row['Terrain_Type'] == 'Salt Pan' and origin_row['Has_Road'] == 0:
            times_wet.append(999.0)
        else:
            best_wet_time = np.inf

            ray_wet_direct = LineString([origin_centroid, nearest_camp_geom])
            time_wet_direct, matches_wet = calculate_segment_time(ray_wet_direct, "Wet")
            pan_count_direct = ((matches_wet['Terrain_Type'] == 'Salt Pan') & (matches_wet['Has_Road'] == 0)).total()

            if pan_count_direct < 2:
                best_wet_time = time_wet_direct

            for j, node_row in nodes_df.iterrows():
                node_centroid = node_row.geometry.centroid
                ray_node = LineString([origin_centroid, node_centroid])

                j, matches_node = calculate_segment_time(ray_node, "Wet")
                node_pan_count = ((matches_node['Terrain_Type'] == 'Salt Pan') & (matches_node['Has_Road'] == 0)).total()

                if node_pan_count < 2:
                    time_to_node, j = calculate_segment_time(ray_node, "Wet", matches=matches_node)
                    total_route_time = time_to_node + node_row['Static_Time']

                    if total_route_time < best_wet_time:
                        best_wet_time = total_route_time

            if best_wet_time == np.inf:
                times_wet.append(888.0)
            else:
                times_wet.append(best_wet_time)

    grid_inside['Time_to_Camp_Wet'] = times_wet
    grid_inside['Time_to_Camp_Dry'] = times_dry

    df_wet = grid_inside[['Cell_ID', 'Time_to_Camp_Wet', 'Speed_Wet']]
    df_dry = grid_inside[['Cell_ID', 'Time_to_Camp_Dry', 'Speed_Dry']]

    df_wet.to_csv('routing_wet.csv', index=False)
    df_dry.to_csv('routing_dry.csv', index=False)

    grid_inside['Time_to_Camp'] = grid_inside[f'Time_to_Camp_{current_season}']

def Calculate_k_T50():
    T_Harvest, V_Poacher, T_Delay = 1, 4, 0.5  # Static according to doc
    grid_inside['T50'] = T_Harvest + (0.8 - 0.6*(grid_inside['Forage_Quality']))*(grid_inside["Dist_to_Fence"]/V_Poacher) - T_Delay

    T_Slack_dict = {
        'Salt Pan': 3.0,
        'Grass Savannah': 1.5,
        'Shrub Savannah': 0.75,
        'Tree Savannah': 0.25
    }
    grid_inside['T_Slack'] = grid_inside['Terrain_Type'].map(T_Slack_dict)
    grid_inside['k'] = 2.89 / grid_inside['T_Slack']

    Avg_T50 = grid_inside['T50'].mean()
    print(f"Average T50 of all cells: {Avg_T50}")

    Avg_k = grid_inside['k'].mean()
    print(f"Average k of all cells: {Avg_k}")


Calculate_k_T50()
lam = 0.5  # this is our lambda for R_Total

def Calculate_Baseline_Threat():
    kw, km, B_sat = 0.230, 0.096, 4.0  # lambdas for APS

    #lambda_a, lambda_b, lambda_r = 5.544, 0.046, 0.154
    lambda_b, lambda_r = 0.046, 0.154

    # non collar APS weights
    omega_w, omega_f, omega_m = 0.47, 0.41, 0.12

    if 'n_blocks' not in grid_inside.columns:
        grid_inside['n_blocks'] = 0

    grid_inside['W'] = np.exp(-kw * grid_inside['Dist_to_Water'])

    grid_inside['F'] = grid_inside['Forage_Quality']

    salt_pan_decay = np.exp(-km * grid_inside['Dist_to_SaltPan'])
    block_boost = grid_inside['n_blocks'] / B_sat
    grid_inside['M'] = np.clip(salt_pan_decay + block_boost, 0.0, 1.0)

    # non colar aps calc kerna pehle
    grid_inside['C'] = 0.0

    grid_inside['APS'] = (omega_w * grid_inside['W']) +  (omega_f * grid_inside['F']) + (omega_m * grid_inside['M'])

    # threat multipliers (A, B, R)
    #grid_inside['A'] = 1.0 - np.exp(-lambda_a * grid_inside['APS'])  since we r not calculating here

    grid_inside['B'] = np.exp(-lambda_b * grid_inside['Dist_to_Fence'])
    grid_inside['R'] = np.exp(-lambda_r * grid_inside['Dist_to_Road'])

    # we arent calculating threat score here now since ab collar distribute kertei waqt hi kerein gai
    #grid_inside['Threat_Score'] = (omega_a * grid_inside['A']) + (omega_b * grid_inside['B']) + (omega_r * grid_inside['R'])
    #grid_inside['Risk_Score'] = grid_inside['Threat_Score']

def HM_APS():
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='APS',
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
        legend=True,
        legend_kwds={'label': "APS"},
        edgecolor='gray',  # Softens the grid lines
        linewidth=0.05,
        ax=ax
    )
    ax.set_title(f'APS [{current_season.upper()} SEASON]', fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("UTM Easting (meters)", fontsize=14)
    ax.set_ylabel("UTM Northing (meters)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig('hm_aps.pdf', format='pdf', bbox_inches='tight')
    plt.show()


#Calculate_Baseline_Threat()
#HM_APS()
print(f"Total cell count: {len(grid_inside)}")
alv = 0

def Distribute_Collars():
    print(f"\nDistributing Collars...")

    nc_aps = grid_inside['APS'].copy()  # non collar

    # animals cant be on salt pan while its flooded
    if current_season == "Wet":
        nc_aps = np.where(grid_inside['Terrain_Type'] == 'Salt Pan', 0.0, nc_aps)

    total_nc_aps = nc_aps.sum()
    if total_nc_aps > 0:
        prob_dist = nc_aps / total_nc_aps
    else:
        prob_dist = np.ones(len(grid_inside)) / len(grid_inside)

    collar_tiers = {
        'Tier_1': {'count': 100, 'value': 10},
        'Tier_2': {'count': 60, 'value': 5},
        'Tier_3': {'count': 30, 'value': 2},
        'Tier_4': {'count': 10, 'value': 1}
    }

    grid_inside['Collar_Value_Total'] = 0.0

    # we set seed to get same results everytime so we can properly see effects of our before after and our sensitivity analysis
    np.random.seed(42)

    for tn, td in collar_tiers.items():
        chosen_indices = np.random.choice(grid_inside.index, size=td['count'], p=prob_dist)

        for idx in chosen_indices:
            grid_inside.at[idx, 'Collar_Value_Total'] += td['value']

    c_threshold = 15.0
    grid_inside['C'] = np.clip(grid_inside['Collar_Value_Total'] / c_threshold, 0.0, 1.0)

    # with collar aps omegas
    omega_w, omega_f, omega_m, omega_c = 0.40, 0.35, 0.10, 0.15

    grid_inside['APS'] = (omega_w * grid_inside['W']) + (omega_f * grid_inside['F']) + (omega_m * grid_inside['M']) + (omega_c * grid_inside['C'])

    grid_inside['A'] = 1.0 - np.exp(-5.544 * grid_inside['APS'])

    omega_a, omega_b, omega_r = 0.5, 0.3, 0.2
    grid_inside['Threat_Score'] = (omega_a * grid_inside['A']) + (omega_b * grid_inside['B']) + (omega_r * grid_inside['R'])

    grid_inside['R_Poacher_Mitigated'] = grid_inside['Threat_Score']
    grid_inside['R_Poacher_Baseline'] = grid_inside['R_Poacher_Mitigated']

    total_collars = sum(t['count'] for t in collar_tiers.values())
    print(f"{total_collars} Collars Distributed")

def Pre_Deploy_Jeeps(budget_security):
    global alv
    print(f"\nDeploying Jeeps at camps...")

    budget_security += alv

    camp_cells = []
    for _, cr in camps.iterrows():
        matching_cells = grid_inside[grid_inside.geometry.intersects(cr.geometry)]
        if not matching_cells.empty:
            camp_cells.append(matching_cells.iloc[0]['Cell_ID'])

    camp_cells = list(set(camp_cells))  # to remove duplicates (if any) common practice

    capital_cost = 40000
    running_cost = 25000
    jeeps = 0
    for cell_id in camp_cells:
        if budget_security >= 2*(capital_cost + running_cost):
            grid_inside.loc[grid_inside['Cell_ID'] == cell_id, 'Count_4x4'] = 2
            jeeps += 2
        elif budget_security >= (capital_cost + running_cost):
            grid_inside.loc[grid_inside['Cell_ID'] == cell_id, 'Count_4x4'] = 1
            jeeps += 1
        else:
            break

    # cuts from security budget
    alv += jeeps * capital_cost
    total_pre_cost = jeeps * (capital_cost + running_cost)
    remaining_budget = budget_security - total_pre_cost

    print(f"Remaining budget from deploying {jeeps} jeeps: ${remaining_budget}")

    return remaining_budget

def Calculate_Wildfires_Threat(budget_wildfires):
    global grid_inside
    print(f"\nWildfires Optimization Started...")

    activeWh = waterholes[waterholes['Pump_Status'] == 1]
    allActiveWh = activeWh.geometry.union_all()
    grid_inside['Has_Pump'] = grid_inside.geometry.intersects(allActiveWh).astype(int)

    c_camp, c_pump, c_gate, c_road = 45, 15, 9, 3  # basev infrastrcutre  constants

    # unnormalised
    grid_inside['BaseV_Unnorm'] = grid_inside['APS'] + (c_camp * grid_inside['Has_Camp']) + (c_pump * grid_inside['Has_Pump']) + (c_gate * grid_inside['Has_Gate']) + (c_road * grid_inside['Has_Road'])

    V_max = grid_inside['BaseV_Unnorm'].max()
    if V_max <= 0:
        V_max = 1.0

    # Calculate the final Normalized V_x,y
    grid_inside['BaseV'] = grid_inside['BaseV_Unnorm'] / V_max

    grid_inside['WCI'] = np.maximum(0, np.minimum(1, (grid_inside['NDMI_Max'] - grid_inside['NDMI']) / (grid_inside['NDMI_Max'] - grid_inside['NDMI_Min'])))
    grid_inside['VCI'] = np.maximum(0, np.minimum(1, (grid_inside['Raw_NDVI_Max'] - np.maximum(0, grid_inside['Raw_NDVI'])) / (grid_inside['Raw_NDVI_Max'] - grid_inside['Raw_NDVI_Min'])))

    w_veg = {
        'Grass Savanna': 0.8,
        'Salt Pan': 0.0,
        'Tree Savanna': 0.2,
        'Shrub Savanna': 0.5
    }

    yearsSinceFire = 5

    grid_inside['T_Fire'] = (grid_inside['Terrain_Type'].map(w_veg) * (grid_inside['Raw_NDVI_Wet'] * min(5, yearsSinceFire)) * (grid_inside['WCI'] * grid_inside['VCI'])) / 4.0

    firezones = grid_inside.groupby('Zone_Number').agg(
        BaseV_Zone_Mean=('BaseV', 'mean'),
        BaseV_Zone_Sum=('BaseV', 'sum'),
        T_Fire_Zone=('T_Fire', 'mean')
    ).reset_index()

    conditions = [
        (firezones['BaseV_Zone_Mean'] >= 0.0316) & (firezones['T_Fire_Zone'] >= 0.231),
        (firezones['BaseV_Zone_Mean'] >= 0.0158) & (firezones['T_Fire_Zone'] >= 0.116) & (firezones['T_Fire_Zone'] < 0.231)
    ]

    choices = [1, 2]
    firezones['Priority'] = np.select(conditions, choices, default=3)

    # firebreak costs
    costs = {1: 45000, 2: 28125}

    firezones['R_Firezone_Baseline'] = firezones['BaseV_Zone_Sum'] * firezones['T_Fire_Zone']

    firezones.sort_values(by=['Priority', 'R_Firezone_Baseline'], ascending=[True, False], inplace=True)

    # m values
    m_Firezones = {1: 0.90, 2: 0.75}
    firezones['M_Firebreak'] = 0.0
    firebreaksDeployed = 0

    for i, z in firezones.iterrows():
        if budget_wildfires <= 0:
            break

        p = z['Priority']
        if p == 1 or p == 2:
            if budget_wildfires >= costs[p]:
                firezones.at[i, 'M_Firebreak'] = m_Firezones[p]
                budget_wildfires -= costs[p]
                firebreaksDeployed += 1

    print(f"Remaining budget from deploying {firebreaksDeployed} firebreaks: ${budget_wildfires}")

    firezones['R_Firezone_Mitigated'] = firezones['BaseV_Zone_Sum'] * firezones['T_Fire_Zone'] * (1 - firezones['M_Firebreak'])

    # grid_inside = grid_inside.merge(firezones[['Zone_Number', 'R_Firezone_Baseline', 'R_Firezone_Mitigated']], on='Zone_Number', how='left')  # merge baseline and mitigated Rs to original cells
    # commented cuz copilot has something better

    if 'M_Firebreak' in grid_inside.columns:
        grid_inside.drop(
            columns=['M_Firebreak', 'Priority', 'R_Firezone_Baseline', 'R_Firezone_Mitigated', 'BaseV_Zone_Sum'],
            inplace=True, errors='ignore')

    grid_inside = grid_inside.merge(
        firezones[['Zone_Number', 'R_Firezone_Baseline', 'R_Firezone_Mitigated', 'M_Firebreak', 'Priority',
                   'BaseV_Zone_Sum']],
        on='Zone_Number', how='left')

    return budget_wildfires

def Recalculate_Water_Distances(temp_waterholes_df):
    active_wh = temp_waterholes_df[temp_waterholes_df['Pump_Status'] == 1]
    all_active_geom = active_wh.geometry.union_all()
    new_distances = grid_inside.geometry.centroid.distance(all_active_geom) / 1000.0
    return new_distances

def Apply_Conservation_Law(base_aps, proposed_aps, changed_mask):
    total_park_aps = base_aps.sum()
    unaffected_mask = ~changed_mask

    new_changed_sum = proposed_aps[changed_mask].sum()
    target_pool_aps = total_park_aps - new_changed_sum

    if target_pool_aps < 0:
        return None

    pool_aps = base_aps[unaffected_mask].sum()

    if pool_aps <= 0:  # prevents division by zero
        if target_pool_aps < pool_aps:
            return None
        multiplier = 1.0
    else:
        multiplier = target_pool_aps / pool_aps

    # create a copy before directly changing proposed_aps
    final_aps = proposed_aps.copy()

    # multplier is multiplied to all unaffected cell (we got rid of the subtraction wali logic)
    final_aps[unaffected_mask] = base_aps[unaffected_mask] * multiplier

    k_max_limit = 0.05 * total_park_aps  # 5% wala kmax

    if (final_aps > k_max_limit).any():
        return None

    return final_aps

def Greedy_Terrain_Optimizer(budget_terrain):
    print(f"\nStarting Terrain Optimization...")

    # costs for E1, E2, E3, E4 [costs of E3 and E4 are same]
    costs = {
        1: 500,
        2: 2000,
        3: 100,
        4: 100
    }
    itr = 1
    waterholes['Change'] = 0
    pump_count = 0

    while budget_terrain > 0:
        best_bcr = 0.0
        best_move = None
        current_global_threat = grid_inside['Threat_Score'].sum()

        def Evaluate_Move(proposed_aps, cost):
            if proposed_aps is None:  # if kmax denied it
                return -1.0

            new_A = 1.0 - np.exp(-5.544 * proposed_aps)
            new_threat = (0.5 * new_A) + (0.3 * grid_inside['B']) + (0.2 * grid_inside['R'])

            risk_reduction = current_global_threat - new_threat.sum()
            return risk_reduction / cost

       # test of E1 and E2
        for i, r in grid_inside.iterrows():
            if budget_terrain >= costs[1] and r['n_blocks'] < 4:  # e1
                old_M = r['M']  # in case needed ager
                new_blocks = r['n_blocks'] + 1
                new_M = min(1.0, np.exp(-0.767 * r['Dist_to_SaltPan']) + (new_blocks / 4.0))

                new_aps = (0.47 * r['W']) + (0.41 * r['F']) + (0.12 * new_M)

                proposed_aps_raw = grid_inside['APS'].copy()
                proposed_aps_raw.at[i] = new_aps

                changed_mask = pd.Series(False, index=grid_inside.index)
                changed_mask.at[i] = True

                balanced_aps = Apply_Conservation_Law(grid_inside['APS'], proposed_aps_raw, changed_mask)
                bcr = Evaluate_Move(balanced_aps, costs[1])

                if bcr > best_bcr:
                    best_bcr = bcr
                    best_move = {'type': 'E1', 'cell_idx': i, 'bcr': bcr, 'cost': costs[1],
                                 'new_aps': balanced_aps, 'new_M': new_M}

            if budget_terrain >= costs[2] and r['Y_Burn'] == 0 and r['Raw_NDVI'] >= 0.1:  # e2
                new_F = min(1.0, r['F'] + 0.25)

                new_aps = (0.47 * r['W']) + (0.41 * new_F) + (0.12 * r['M'])

                proposed_aps_raw = grid_inside['APS'].copy()
                proposed_aps_raw.at[i] = new_aps

                changed_mask = pd.Series(False, index=grid_inside.index)
                changed_mask.at[i] = True

                balanced_aps = Apply_Conservation_Law(grid_inside['APS'], proposed_aps_raw, changed_mask)
                bcr = Evaluate_Move(balanced_aps, costs[2])

                if bcr > best_bcr:
                    best_bcr = bcr
                    best_move = {'type': 'E2', 'cell_idx': i, 'bcr': bcr, 'cost': costs[2],
                                 'new_aps': balanced_aps, 'new_F': new_F}

        # test E3 and E4
        for i, wr in waterholes.iterrows():
            if pump_count >= 10:
                break

            if wr['Type'] == 'A' and budget_terrain >= costs[3] and wr['Change'] == 0:
                is_active = (wr['Pump_Status'] == 1)

                temp_wh = waterholes.copy()
                temp_wh.at[i, 'Pump_Status'] = 0 if is_active else 1

                new_d_water = Recalculate_Water_Distances(temp_wh)

                # Create a mask of every cell that was affected by the pump toggle
                changed_mask = (new_d_water != grid_inside['Dist_to_Water'])

                if changed_mask.any():
                    new_W = np.exp(-0.230 * new_d_water)
                    proposed_aps_raw = (0.47 * new_W) + (0.41 * grid_inside['F']) + (0.12 * grid_inside['M'])

                    balanced_aps = Apply_Conservation_Law(grid_inside['APS'], proposed_aps_raw, changed_mask)
                    bcr = Evaluate_Move(balanced_aps, costs[3])

                    if bcr > best_bcr:
                        best_bcr = bcr
                        best_move = {'type': 'E3' if is_active else 'E4', 'wh_idx': i,
                                     'bcr': bcr, 'cost': costs[3], 'new_aps': balanced_aps,
                                     'new_W': new_W, 'new_d_water': new_d_water,
                                     'new_status': 0 if is_active else 1}

        # finally do the best move
        if best_move is None or best_bcr <= 0:
            print("Optimization Complete")
            break

        x = float(f"{best_bcr:.6f}")
        if x == 0.000000:
            print("Optimization Complete")
            break

        if itr >= 100:
            print("Optimization Complete")
            break

        budget_terrain -= best_move['cost']

        grid_inside['APS'] = best_move['new_aps']
        grid_inside['A'] = 1.0 - np.exp(-5.544 * grid_inside['APS'])
        grid_inside['Threat_Score'] = (0.5 * grid_inside['A']) + (0.3 * grid_inside['B']) + (0.2 * grid_inside['R'])

        if best_move['type'] == 'E1':
            i = best_move['cell_idx']
            grid_inside.at[i, 'n_blocks'] += 1
            grid_inside.at[i, 'M'] = best_move['new_M']
            print(f"Iter {itr} | E1 (Mineral) added to Cell {i} | Cost: {costs[1]} | BCR: {best_bcr:.6f}")

        elif best_move['type'] == 'E2':
            i = best_move['cell_idx']
            grid_inside.at[i, 'Y_Burn'] = 1
            grid_inside.at[i, 'F'] = best_move['new_F']
            grid_inside.at[i, 'Forage_Quality'] = best_move['new_F']
            print(f"Iter {itr} | E2 (Burn) executed on Cell {i} | Cost: {costs[2]} | BCR: {best_bcr:.6f}")

        elif best_move['type'] in ['E3', 'E4']:
            i = best_move['wh_idx']
            waterholes.at[i, 'Pump_Status'] = best_move['new_status']
            waterholes.at[i, 'Change'] = 1
            pump_count += 1
            grid_inside['Dist_to_Water'] = best_move['new_d_water']
            grid_inside['W'] = best_move['new_W']
            action_str = "Closed" if best_move['type'] == 'E3' else "Opened"
            print(
                f"Iter {itr} | {best_move['type']} ({action_str} Pump) at WH {i} | Cost: {costs[3]} | BCR: {best_bcr:.6f}")

        itr += 1

    print(f"Terrain Optimization Completed with remaining budget: ${budget_terrain}")


# dict to store and cache neighbors so we don't be dumbos to calc it multiple times (i got this sleek technique from youtube + copilot refined it properly)
nei = {}
nei2 = {}
id_to_idx = {}

def Build_Neighbor_Dictionary():
    global nei, nei2, id_to_idx
    nei.clear()
    nei2.clear()
    id_to_idx.clear()

    coord_to_id = {}
    for idx, row in grid_inside.iterrows():
        coord_to_id[(row['Coord_X'], row['Coord_Y'])] = row['Cell_ID']
        id_to_idx[row['Cell_ID']] = idx

    for idx, row in grid_inside.iterrows():
        cx, cy = row['Coord_X'], row['Coord_Y']
        cell_id = row['Cell_ID']

        adj_coords = [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]
        diag_coords = [(cx + 1, cy + 1), (cx + 1, cy - 1), (cx - 1, cy + 1), (cx - 1, cy - 1)]

        adj_ids = [coord_to_id[c] for c in adj_coords if c in coord_to_id]
        diag_ids = [coord_to_id[c] for c in diag_coords if c in coord_to_id]

        nei[cell_id] = {'adj': adj_ids, 'diag': diag_ids}

        # 2 step radius for cache
        radius_2 = []
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                target_coord = (cx + dx, cy + dy)
                if target_coord in coord_to_id:
                    radius_2.append(id_to_idx[coord_to_id[target_coord]])

        nei2[idx] = list(set(radius_2))

def Calculate_P_Catch(target_cell_id):
    idx = id_to_idx[target_cell_id]

    base_forage = grid_inside.at[idx, 'Base_Forage']
    y_burn = grid_inside.at[idx, 'Y_Burn']
    F1 = max(0.1, base_forage - (y_burn * 0.40))

    x1_count = grid_inside.at[idx, 'Count_FP']
    x2_count = grid_inside.at[idx, 'Count_4x4']
    x3_count = grid_inside.at[idx, 'Count_Drones']
    x4_count = grid_inside.at[idx, 'Count_Cams']

    neighbors = nei[target_cell_id]

    # p_detect
    p_detects = []

    if x1_count > 0:
        p_detects.append(0.30 + (0.40 * F1))
    if x4_count > 0:
        for _ in range(x4_count):
            p_detects.append(0.05 + (0.20 * F1))
    if x2_count > 0:
        p_detects.append(0.70 - (0.40 * F1))
    if x3_count > 0:
        p_detects.append(0.80 - (0.50 * F1))

    for n_id in neighbors['adj']:
        n_idx = id_to_idx[n_id]
        if grid_inside.at[n_idx, 'Count_4x4'] > 0:
            p_detects.append(0.70 - (0.40 * F1) * math.exp(-(5 ** 2) / (2 * (2.5 * (1 - 0.7 * F1)) ** 2)))
        if grid_inside.at[n_idx, 'Count_Drones'] > 0:
            p_detects.append(0.80 - (0.50 * F1) * math.exp(-(5 ** 2) / (2 * (5.0 * (1 - 0.7 * F1)) ** 2)))

    for n_id in neighbors['diag']:
        n_idx = id_to_idx[n_id]
        if grid_inside.at[n_idx, 'Count_4x4'] > 0:
            p_detects.append(0.70 - (0.40 * F1) * math.exp(-(7.07 ** 2) / (2 * (2.5 * (1 - 0.7 * F1)) ** 2)))
        if grid_inside.at[n_idx, 'Count_Drones'] > 0:
            p_detects.append(0.80 - (0.50 * F1) * math.exp(-(7.07 ** 2) / (2 * (5.0 * (1 - 0.7 * F1)) ** 2)))

    total_p_detect = 1.0
    for p in p_detects:
        p_clean = max(0.0, min(1.0, p))
        total_p_detect *= (1.0 - p_clean)
    total_p_detect = 1.0 - total_p_detect

    # p_interdiction
    I_options = [0.0]

    if x1_count > 0: I_options.append(0.8)
    if x2_count > 0: I_options.append(0.9)

    has_adj_x2 = any(grid_inside.at[id_to_idx[n], 'Count_4x4'] > 0 for n in neighbors['adj'])
    has_diag_x2 = any(grid_inside.at[id_to_idx[n], 'Count_4x4'] > 0 for n in neighbors['diag'])

    is_dry = (current_season == "Dry")
    if has_adj_x2: I_options.append(0.8471 if is_dry else 0.8063)
    if has_diag_x2: I_options.append(0.8452 if is_dry else 0.8023)

    if (x3_count > 0 or x4_count > 0) and not (x1_count > 0 or x2_count > 0 or has_adj_x2 or has_diag_x2):
        T_escape = grid_inside.at[idx, 'Dist_to_Fence'] / 5.0
        T_arrive = grid_inside.at[idx, 'Time_to_Camp'] / 60.0
        T_50 = grid_inside.at[idx, 'T50']
        k = grid_inside.at[idx, 'k']

        if T_arrive <= T_escape:  # i'd ve never done it if it wasnt for copilot
            try:
                camp_response = 0.9 / (1.0 + math.exp(k * (T_arrive - T_50)))
                I_options.append(camp_response)
            except OverflowError:
                I_options.append(0.0)

    max_I = max(I_options)
    return min(1.0, total_p_detect * max_I)

def Update_Drone_Safe_Zones():
    sz = set()

    # check camps
    camp_drone_limit = 5
    for _, cr in camps.iterrows():
        dists = grid_inside.geometry.centroid.distance(cr.geometry) / 1000.0
        cells_in_radius = grid_inside[dists <= 10.0]

        if cells_in_radius['Count_Drones'].sum() < camp_drone_limit:
            sz.update(cells_in_radius['Cell_ID'].tolist())

    # check jeeps
    x2_cells = grid_inside[grid_inside['Count_4x4'] > 0]
    jeep_drone_limit = 2
    for _, jr in x2_cells.iterrows():
        dists = grid_inside.geometry.centroid.distance(jr.geometry.centroid) / 1000.0
        cells_in_radius = grid_inside[dists <= 10.0]

        if cells_in_radius['Count_Drones'].sum() < jeep_drone_limit:
            sz.update(cells_in_radius['Cell_ID'].tolist())

    # check foot patrols
    x1_cells = grid_inside[grid_inside['Count_FP'] > 0]
    fp_drone_limit = 1
    for _, fpr in x1_cells.iterrows():
        dists = grid_inside.geometry.centroid.distance(fpr.geometry.centroid) / 1000.0
        cells_in_radius = grid_inside[dists <= 10.0]

        if cells_in_radius['Count_Drones'].sum() < fp_drone_limit:
            sz.update(cells_in_radius['Cell_ID'].tolist())

    return list(sz)

def Get_Local_Risk(target_cell_id):
    neighbors = nei[target_cell_id]
    cluster_ids = [target_cell_id] + neighbors['adj'] + neighbors['diag']

    local_risk = 0.0
    for cid in cluster_ids:
        idx = id_to_idx[cid]

        t_score = grid_inside.at[idx, 'Threat_Score']
        p_catch = Calculate_P_Catch(cid)
        local_risk += t_score * (1.0 - p_catch)

    return local_risk

def Greedy_Security_Optimizer(budget):
    global alv
    print(f"\nStarting Security Optimization...")

    capital_costs = {'x1': 0, 'x2': 40000, 'x3': 10000, 'x4': 700}
    running_costs = {'x1': 10000, 'x2': 25000, 'x3': 5000, 'x4': 150}
    costs = {'x1': capital_costs['x1'] + running_costs['x1'],
             'x2': capital_costs['x2'] + running_costs['x2'],
             'x3': capital_costs['x3'] + running_costs['x3'],
             'x4': capital_costs['x4'] + running_costs['x4']}

    iteration = 1
    drone_safe_zones = set(Update_Drone_Safe_Zones())

    dirty_cells = set(grid_inside.index.tolist())

    bcr_cache = {}  # this rmrs best move for every cell

    while budget >= min(costs.values()):
        for i in list(dirty_cells):
            cell_id = grid_inside.at[i, 'Cell_ID']
            current_local_risk = Get_Local_Risk(cell_id)

            best_bcr_for_cell = 0.0
            best_move_for_cell = None

            # Test x1
            if budget >= costs['x1'] and grid_inside.at[i, 'Count_FP'] < 1 and Calculate_P_Catch(cell_id) < 0.8:
                # the idea is we first add then check risk then remove and the one that lowers the risk the most (per dollar) [so it is budget efficient] will be our best move that we will permanently apply
                grid_inside.at[i, 'Count_FP'] += 1
                new_local_risk = Get_Local_Risk(cell_id)
                grid_inside.at[i, 'Count_FP'] -= 1

                bcr = (current_local_risk - new_local_risk) / costs['x1']
                if bcr > best_bcr_for_cell:
                    best_bcr_for_cell, best_move_for_cell = bcr, {'type': 'x1', 'idx': i, 'cost': costs['x1'], 'bcr': bcr, 'capital': capital_costs['x1']}

            # Test x2
            if budget >= costs['x2'] and grid_inside.at[i, 'Count_4x4'] < 1:
                grid_inside.at[i, 'Count_4x4'] += 1
                new_local_risk = Get_Local_Risk(cell_id)
                grid_inside.at[i, 'Count_4x4'] -= 1

                bcr = (current_local_risk - new_local_risk) / costs['x2']
                if bcr > best_bcr_for_cell:
                    best_bcr_for_cell, best_move_for_cell = bcr, {'type': 'x2', 'idx': i, 'cost': costs['x2'], 'bcr': bcr, 'capital': capital_costs['x2']}

            # Test x3
            if budget >= costs['x3'] and grid_inside.at[i, 'Count_Drones'] < 1 and cell_id in drone_safe_zones:
                grid_inside.at[i, 'Count_Drones'] += 1
                new_local_risk = Get_Local_Risk(cell_id)
                grid_inside.at[i, 'Count_Drones'] -= 1

                bcr = (current_local_risk - new_local_risk) / costs['x3']
                if bcr > best_bcr_for_cell:
                    best_bcr_for_cell, best_move_for_cell = bcr, {'type': 'x3', 'idx': i, 'cost': costs['x3'], 'bcr': bcr, 'capital': capital_costs['x3']}

            # Test x4
            if budget >= costs['x4'] and grid_inside.at[i, 'Count_Cams'] < 5:
                grid_inside.at[i, 'Count_Cams'] += 1
                new_local_risk = Get_Local_Risk(cell_id)
                grid_inside.at[i, 'Count_Cams'] -= 1

                bcr = (current_local_risk - new_local_risk) / costs['x4']
                if bcr > best_bcr_for_cell:
                    best_bcr_for_cell, best_move_for_cell = bcr, {'type': 'x4', 'idx': i, 'cost': costs['x4'], 'bcr': bcr, 'capital': capital_costs['x4']}

            bcr_cache[i] = best_move_for_cell

        dirty_cells.clear()

        best_bcr = 0.0
        best_move = None

        for i, move in bcr_cache.items():
            if move and move['cost'] <= budget and move['bcr'] > best_bcr:
                best_bcr = move['bcr']
                best_move = move

        # permanently apply move (break if there is no moves left)
        if best_move is None or best_bcr <= 0.0:
            break

        budget -= best_move['cost']
        alv += best_move['capital']
        target_idx = best_move['idx']

        if best_move['type'] == 'x1':
            grid_inside.at[target_idx, 'Count_FP'] += 1
        elif best_move['type'] == 'x2':
            grid_inside.at[target_idx, 'Count_4x4'] += 1
        elif best_move['type'] == 'x3':
            grid_inside.at[target_idx, 'Count_Drones'] += 1
        elif best_move['type'] == 'x4':
            grid_inside.at[target_idx, 'Count_Cams'] += 1

        cell_id_placed = grid_inside.at[target_idx, 'Cell_ID']
        print(
            f"Deploy #{iteration} | {best_move['type']} placed at Cell {cell_id_placed} | Cost: ${best_move['cost']:,} | BCR: {best_bcr:.6f}")

        dirty_cells.update(nei2[target_idx])

        if best_move['type'] in ['x1', 'x2']:
            new_sz = set(Update_Drone_Safe_Zones())
            changed_sz = new_sz ^ drone_safe_zones
            if changed_sz:
                for cid in changed_sz:
                    dirty_cells.add(id_to_idx[cid])

            drone_safe_zones = new_sz

        iteration += 1

    grid_inside['P_catch'] = grid_inside['Cell_ID'].apply(Calculate_P_Catch)
    grid_inside['R_Poacher_Mitigated'] = grid_inside['Threat_Score'] * (1.0 - grid_inside['P_catch'])

    protected_aps = grid_inside[grid_inside['P_catch'] > 0.70]['APS'].sum()
    total_aps = grid_inside['APS'].sum()
    protection_ratio = (protected_aps / total_aps) * 100 if total_aps > 0 else 0

    print(f"Security Optimization Completed with remaining budget: ${budget} and total alv: {alv}; with {protection_ratio:.2f}% of animals secured.")
    return budget


phi = 0
def Calculate_Protection_Score():
    global lam
    global phi

    # rtotal of all cells
    grid_inside['R_Total'] = (lam * (grid_inside['R_Poacher_Mitigated'] / grid_inside['R_Poacher_Baseline'])) + ((1 - lam) * (grid_inside['R_Firezone_Mitigated'] / grid_inside['R_Firezone_Baseline']))

    # phi
    poach_base = grid_inside['R_Poacher_Baseline'].sum()

    zones = grid_inside[grid_inside['Zone_Number'] > 0].drop_duplicates(subset=['Zone_Number'])
    fire_base = zones['R_Firezone_Baseline'].sum()

    poach_mitig = grid_inside['R_Poacher_Mitigated'].sum()
    fire_mitig = zones['R_Firezone_Mitigated'].sum()

    Poach_Risk_Remaining = (lam * (poach_mitig / poach_base))
    Fire_Risk_Remaining = ((1.0 - lam) * (fire_mitig / fire_base))
    R_total = (lam * (poach_mitig / poach_base)) + ((1.0 - lam) * (fire_mitig / fire_base))
    phi = 1 - R_total

    print(f"phi of all cells: {phi}")

    # delta risk
    grid_inside['R_Baseline_xy'] = (lam * (grid_inside['R_Poacher_Baseline'] / poach_base)) + ((1.0 - lam) * (grid_inside['R_Firezone_Baseline'] / fire_base))

    grid_inside['R_Mitigated_xy'] = (lam * (grid_inside['R_Poacher_Mitigated'] / poach_base)) + ((1.0 - lam) * (grid_inside['R_Firezone_Mitigated'] / fire_base))

    grid_inside['Delta_Risk'] = grid_inside['R_Baseline_xy'] - grid_inside['R_Mitigated_xy']
    return phi, Poach_Risk_Remaining, Fire_Risk_Remaining

def Run_Master_Simulation():
    total_budget = 2000000
    reserve_budget = total_budget * 0.10  # reserve 10% budget for C_Reserve
    total_budget -= reserve_budget

    split_s, split_f, split_e = 80, 10, 10
    split_t = split_s + split_f + split_e

    Budget_Terrain = total_budget * split_e / split_t
    Budget_Security = total_budget * split_s / split_t
    Budget_Wildfires = total_budget * split_f / split_t

    Calculate_Baseline_Threat()
    Remaining_Security_Budget = Pre_Deploy_Jeeps(Budget_Security)

    remains = Calculate_Wildfires_Threat(Budget_Wildfires)
    Budget_Security += remains * split_s / (split_s + split_e)
    Budget_Terrain += remains * split_e / (split_s + split_e)

    Distribute_Collars()

    Build_Neighbor_Dictionary()
    remains = Greedy_Security_Optimizer(Remaining_Security_Budget)
    Budget_Terrain += remains

    Greedy_Terrain_Optimizer(Budget_Terrain)

    Calculate_Protection_Score()

#endregion

#region CONTROLS
def Update_Hydrology():
    active_waterholes = waterholes[waterholes['Pump_Status'] == 1]
    if not active_waterholes.empty:
        all_active_waterholes = active_waterholes.geometry.union_all()
        grid_inside['Dist_to_Water'] = grid_inside.geometry.centroid.distance(all_active_waterholes) / 1000.0
        grid_inside['Has_Water'] = (grid_inside['Dist_to_Water'] <= 10).astype(int)
    else:
        grid_inside['Dist_to_Water'] = 999.0
        grid_inside['Has_Water'] = 0

    Update_Forage()

def Toggle_Season():
    global current_season

    # Flip the season
    current_season = "Wet" if current_season == "Dry" else "Dry"

    # Update the status of Seasonal waterholes
    new_status = 1 if current_season == "Wet" else 0
    waterholes.loc[waterholes['Validity'] == 'S', 'Pump_Status'] = new_status

    # Update forage values according to current season wali tif file
    grid_inside['Base_Forage'] = grid_inside[f'Base_Forage_{current_season}']
    grid_inside['Raw_NDVI'] = grid_inside[f'Raw_NDVI_{current_season}']
    grid_inside['NDMI'] = grid_inside[f'NDMI_{current_season}']

    # Update Time_to_Camp based on current season
    grid_inside['Time_to_Camp'] = grid_inside[f'Time_to_Camp_{current_season}']

    Update_Hydrology()
    Calculate_k_T50()

#endregion

#region SENSITIVITY ANALYSIS
def Reset_Simulation_State():
    global alv
    alv = 0

    grid_inside['Count_FP'] = 0
    grid_inside['Count_4x4'] = 0
    grid_inside['Count_Drones'] = 0
    grid_inside['Count_Cams'] = 0
    if 'P_catch' in grid_inside.columns:
        grid_inside['P_catch'] = 0.0

    grid_inside['n_blocks'] = 0
    grid_inside['Y_Burn'] = 0

    grid_inside['Collar_Value_Total'] = 0.0
    grid_inside['C'] = 0.0

    if 'M_Firebreak' in grid_inside.columns:
        grid_inside['M_Firebreak'] = 0.0
    if 'Priority' in grid_inside.columns:
        grid_inside['Priority'] = 3

    grid_inside['Threat_Score'] = 0.0
    grid_inside['R_Poacher_Baseline'] = 0.0
    grid_inside['R_Poacher_Mitigated'] = 0.0

    if 'R_Firezone_Baseline' in grid_inside.columns:
        grid_inside['R_Firezone_Baseline'] = 0.0
    if 'R_Firezone_Mitigated' in grid_inside.columns:
        grid_inside['R_Firezone_Mitigated'] = 0.0

    if 'R_Baseline_xy' in grid_inside.columns:
        grid_inside['R_Baseline_xy'] = 0.0
    if 'R_Mitigated_xy' in grid_inside.columns:
        grid_inside['R_Mitigated_xy'] = 0.0
    if 'Delta_Risk' in grid_inside.columns:
        grid_inside['Delta_Risk'] = 0.0
    if 'R_Total' in grid_inside.columns:
        grid_inside['R_Total'] = 0.0

    if 'Baseline_Poaching_Risk' in grid_inside.columns:
        grid_inside['Baseline_Poaching_Risk'] = 0.0
    if 'Phi' in grid_inside.columns:
        grid_inside['Phi'] = 0.0

    waterholes['Pump_Status'] = np.where(waterholes['Validity'] == 'P', 1, 0)

    if current_season == "Wet":
        waterholes.loc[waterholes['Validity'] == 'S', 'Pump_Status'] = 1

    waterholes['Change'] = 0

    Update_Hydrology()

def Run_Sensitivity_Analysis():
    global lam, grid_inside, waterholes

    Reset_Simulation_State()

    total_budget = 2000000
    reserve_budget = total_budget * 0.10
    total_budget -= reserve_budget

    split_s, split_f, split_e = 80, 10, 10
    split_t = split_s + split_f + split_e

    Budget_Terrain_Base = total_budget * split_e / split_t
    Budget_Security = total_budget * split_s / split_t
    Budget_Wildfires = total_budget * split_f / split_t

    Calculate_Baseline_Threat()
    Remaining_Security_Budget = Pre_Deploy_Jeeps(Budget_Security)

    remains_fire = Calculate_Wildfires_Threat(Budget_Wildfires)
    Budget_Security += remains_fire * split_s / (split_s + split_e)
    Budget_Terrain_Base += remains_fire * split_e / (split_s + split_e)

    Distribute_Collars()
    Build_Neighbor_Dictionary()

    remains_sec = Greedy_Security_Optimizer(Remaining_Security_Budget)
    Final_Terrain_Budget = Budget_Terrain_Base + remains_sec

    grid_snapshot = grid_inside.copy(deep=True)
    waterholes_snapshot = waterholes.copy(deep=True)

    sensitivity_results = []
    lambdas_to_test = [0.0, 0.25, 0.5, 0.75, 1.0]

    for current_lam in lambdas_to_test:
        print(f"\n--- Simulating Terrain Tactics with λ = {current_lam:.1f} ---")

        grid_inside = grid_snapshot.copy(deep=True)
        waterholes = waterholes_snapshot.copy(deep=True)
        lam = current_lam

        Greedy_Terrain_Optimizer(Final_Terrain_Budget)

        phi, poach_frac, fire_frac = Calculate_Protection_Score()

        sensitivity_results.append({
            'Lambda': current_lam,
            'Global_Phi': phi,
            'Poach_Risk_Remaining': poach_frac,
            'Fire_Risk_Remaining': fire_frac
        })

    results_df = pd.DataFrame(sensitivity_results)
    results_df.to_csv('Sensitivity_Results_Fast.csv', index=False)
    print("\nData successfully saved to 'Sensitivity_Results_Fast.csv'!")

    plt.figure(figsize=(10, 6))

    plt.plot(results_df['Lambda'], results_df['Global_Phi'],
             marker='^', markersize=10, color='forestgreen', linewidth=3, label='Global Protection Score (Φ)')

    best_idx = results_df['Global_Phi'].idxmax()
    best_lam = results_df.loc[best_idx, 'Lambda']
    max_phi = results_df.loc[best_idx, 'Global_Phi']

    # Draw a golden dot at the absolute peak
    plt.scatter([best_lam], [max_phi], color='gold', s=200, edgecolor='black', zorder=5,
                label=f'Optimal Policy (λ={best_lam})')

    plt.title('System Efficiency (Φ) vs. Policy Weight (λ)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Lambda (λ) : 0.0 (Pro-Infrastructure)  -->  1.0 (Pro-Wildlife)', fontsize=14)
    plt.ylabel('Overall Protection Score (Φ)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower center', frameon=True, edgecolor='black')
    plt.tight_layout()

    # Save it directly as a high-res Vector Graphic for your paper!
    plt.savefig('Phi_Parabola_Analysis.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def Run_Budget_Analysis():
    total_budget = 2000000
    reserve_budget = total_budget * 0.10  # reserve 10% budget for C_Reserve
    total_budget -= reserve_budget

    splits = [0, 20, 40, 60, 80, 100]
    sensitivity_results = []

    for split_s in splits:
        print(f"\n--- Simulating Sen Analysis with Split of Security Budget = {split_s} ---")
        split_f = (100 - split_s) / 2
        split_e = split_f

        Budget_Terrain = total_budget * split_e / 100
        Budget_Security = total_budget * split_s / 100
        Budget_Wildfires = total_budget * split_f / 100

        Reset_Simulation_State()
        Calculate_k_T50()
        Calculate_Baseline_Threat()
        Remaining_Security_Budget = Pre_Deploy_Jeeps(Budget_Security)

        remains = Calculate_Wildfires_Threat(Budget_Wildfires)
        Budget_Security += remains * split_s / (split_s + split_e)
        Budget_Terrain += remains * split_e / (split_s + split_e)

        Distribute_Collars()

        Build_Neighbor_Dictionary()
        remains = Greedy_Security_Optimizer(Remaining_Security_Budget)
        Budget_Terrain += remains

        Greedy_Terrain_Optimizer(Budget_Terrain)

        phi, poach_frac, fire_frac = Calculate_Protection_Score()

        sensitivity_results.append({
            'Split_S': split_s,
            'Global_Phi': phi,
            'Poach_Risk_Remaining': poach_frac,
            'Fire_Risk_Remaining': fire_frac
        })

    results_df = pd.DataFrame(sensitivity_results)
    results_df.to_csv('Sensitivity_Results_Fast.csv', index=False)
    print("\nData successfully saved to 'Sensitivity_Results_Fast.csv'!")

    plt.figure(figsize=(10, 6))

    plt.plot(results_df['Lambda'], results_df['Global_Phi'],
             marker='^', markersize=10, color='forestgreen', linewidth=3, label='Global Protection Score (Φ)')

    best_idx = results_df['Global_Phi'].idxmax()
    Split_S = results_df.loc[best_idx, 'Split_S']
    max_phi = results_df.loc[best_idx, 'Global_Phi']

    # Draw a golden dot at the absolute peak
    plt.scatter([Split_S], [max_phi], color='gold', s=200, edgecolor='black', zorder=5,
                label=f'Optimal Policy (λ={Split_S})')

    plt.title('System Efficiency (Φ) vs. split of security', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Lambda (λ) : 0.0 (Pro-Infrastructure)  -->  1.0 (Pro-Wildlife)', fontsize=14)
    plt.ylabel('Overall Protection Score (Φ)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower center', frameon=True, edgecolor='black')
    plt.tight_layout()

    plt.savefig('Phi_Parabola_Analysis.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def Load_Run_Budget_Analysis():
    file_path = 'Sensitivity_Results_Fast.csv'
    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found. Please run the analysis first.")
        return

    results_df = pd.read_csv(file_path)
    print(f"\nData successfully loaded from '{file_path}'!")

    plt.figure(figsize=(10, 6))

    plt.plot(results_df['Split_S'], results_df['Global_Phi'],
             marker='^', markersize=10, color='forestgreen', linewidth=3, label='Global Protection Score (Φ)')

    best_idx = results_df['Global_Phi'].idxmax()
    Split_S = results_df.loc[best_idx, 'Split_S']
    max_phi = results_df.loc[best_idx, 'Global_Phi']

    # Draw a golden dot at the absolute peak
    plt.scatter([Split_S], [max_phi], color='gold', s=200, edgecolor='black', zorder=5,
                label=f'Optimal Policy (λ={Split_S})')

    plt.title('System Efficiency (Φ) vs. split of security', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Security Split: 0 -> 100', fontsize=14)
    plt.ylabel('Overall Protection Score (Φ)', fontsize=14)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12, loc='lower center', frameon=True, edgecolor='black')
    plt.tight_layout()

    plt.savefig('Phi_Parabola_Analysis.pdf', format='pdf', bbox_inches='tight')
    plt.show()


Load_Run_Budget_Analysis()

#endregion

#region CODE FOR DATA DISTRIBUTION VISUALS AND SIMULATIONS AND DEBUGGING AND ANYTHING TO INTERACT WITH
    #region Maps
def M_CollarMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    # 1. Park boundary on top
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    empty_cells = grid_inside[grid_inside['Collar_Value_Total'] == 0]
    collar_cells = grid_inside[grid_inside['Collar_Value_Total'] > 0]

    if not empty_cells.empty:
        empty_cells.plot(
            ax=ax,
            color='#f0f0f0',
            edgecolor='black',
            linewidth=0.1,
            alpha=0.5,
            zorder=1
        )

    if not collar_cells.empty:
        collar_cells.plot(
            column='Collar_Value_Total',
            cmap='YlOrRd',  # Yellow -> Orange -> Red
            legend=True,
            legend_kwds={
                'label': "Collar Value Total",
                'orientation': "vertical"
            },
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            ax=ax,
            zorder=2
        )

        for idx, row in collar_cells.iterrows():
            centroid = row.geometry.centroid
            val = int(row['Collar_Value_Total'])
            ax.text(centroid.x, centroid.y, str(val), fontsize=8, fontweight='bold', color='black', ha='center',
                    va='center', zorder=3)

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.3, zorder=4)
    camps.plot(ax=ax, color='blue', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=5)

    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0), frameon=True, edgecolor='black')

    plt.title(f'Collars Distribution [{current_season.upper()} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def M_ZoneMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    barren_cells = grid_inside[grid_inside['Zone_Number'] == 0]  # salt pan walei cells
    managed_cells = grid_inside[grid_inside['Zone_Number'] > 0]

    barren_cells.plot(
        ax=ax,
        color='#333333',
        edgecolor='black',
        hatch='///',
        linewidth=0.5,
        alpha=0.95,
        zorder=2
    )
    zone_colors = {
        1: '#0ac213',
        2: '#eb5f47',
        3: '#a2c20a',
        4: '#e09052',
        5: '#b87414',
        6: '#f5c43d',
        7: '#ad9c1f',
        8: '#e7eb47',
        9: '#c2400a',
        10: '#b3e052',
        11: '#6cb814',
        12: '#b8142c',
        13: '#b414b8',
        14: '#57eb47',
        15: '#ad1f1f',
        16: '#52e06d',
        17: '#0a2ec2',
        18: '#3df597',
        19: '#1fad79',
        20: '#47ebc7',
        21: '#0ac2b4',
        22: '#ad1f87',
        23: '#1494b8',
        24: '#3db2f5',
        25: '#1f64ad',
        26: '#477feb',
        27: '#14b84c',
        28: '#5259e0',
        29: '#2414b8',
        30: '#6a3df5',
        31: '#561fad',
        32: '#9f47eb',
        33: '#880ac2',
        34: '#c852e0',
        35: '#41ad1f',
        36: '#f53dde',
        37: '#52d6e0',
        38: '#eb47a7',
        39: '#c20a5b',
        40: '#e0527b',
        41: '#85f53d',
        42: '#ED760E'
    }
    cell_colors = managed_cells['Zone_Number'].map(zone_colors).fillna('#000000')

    managed_cells.plot(
        ax=ax,
        color=cell_colors,
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        zorder=3
    )

    dissolved_zones = managed_cells.dissolve(by='Zone_Number')

    for zone_id, row in dissolved_zones.iterrows():
        centroid = row.geometry.centroid
        ax.text(centroid.x, centroid.y, str(int(zone_id)), fontsize=10, fontweight='bold', color='white', ha='center', va='center', zorder=7)

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.6, zorder=5)
    camps.plot(ax=ax, color='red', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=6)

    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0), frameon=True, edgecolor='black')

    plt.title('Zones', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def M_FirebreakMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    managed_cells = grid_inside[grid_inside['Zone_Number'] > 0]
    barren_cells = grid_inside[grid_inside['Zone_Number'] == 0]

    if not barren_cells.empty:
        barren_cells.plot(ax=ax, color='#333333', edgecolor='black', hatch='///', linewidth=0.5, alpha=0.95, zorder=1)

    if not managed_cells.empty:
        managed_cells.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.2, zorder=2)

    # M_Firebreak == 0.9 is priority 1, 0.75 is priority 2
    heavy_breaks = grid_inside[grid_inside['M_Firebreak'] == 0.9]
    std_breaks = grid_inside[grid_inside['M_Firebreak'] == 0.75]

    legend_patches = []
    # 4. Plot Priority 1 (Heavy Mitigations)
    if not heavy_breaks.empty:
        heavy_breaks.plot(
            ax=ax,
            color='darkred',
            hatch='XX',
            edgecolor='black',
            linewidth=1,
            alpha=0.85,
            zorder=3
        )
        legend_patches.append(mpatches.Patch(facecolor='darkred', edgecolor='black', hatch='XX',
                                             label='Priority 1 Firebreak (90% Mitigated)'))

    # 5. Plot Priority 2 (Standard Mitigations)
    if not std_breaks.empty:
        std_breaks.plot(
            ax=ax,
            color='darkorange',
            hatch='\\\\',
            edgecolor='black',
            linewidth=1,
            alpha=0.85,
            zorder=3
        )
        legend_patches.append(mpatches.Patch(facecolor='darkorange', edgecolor='black', hatch='\\\\',
                                             label='Priority 2 Firebreak (75% Mitigated)'))

    legend_patches.append(mpatches.Patch(facecolor='#e0e0e0', edgecolor='white', label='No Firebreak Deployed'))
    legend_patches.append(mpatches.Patch(facecolor='#333333', edgecolor='black', hatch='///', label='0 (Barren)'))

    # 6. Add infrastructure
    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.4, zorder=4)
    camps.plot(ax=ax, color='blue', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=5)

    # 7. Legends and formatting
    ax.legend(handles=legend_patches, title="Wildfire Defenses", loc='upper right', bbox_to_anchor=(1.25, 1),
              frameon=True, edgecolor='black')

    plt.title(f'Firebreak Deployment Map [{current_season.upper()} SEASON]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def M_SecurityDeploymentMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)
    grid_inside.plot(ax=ax, color='#e5e5e5', edgecolor='white', linewidth=0.2, zorder=1)

    # use only cells that have the assets
    c_x1 = grid_inside[grid_inside['Count_FP'] > 0]
    c_x2 = grid_inside[grid_inside['Count_4x4'] > 0]
    c_x3 = grid_inside[grid_inside['Count_Drones'] > 0]
    c_x4 = grid_inside[grid_inside['Count_Cams'] > 0]

    if not c_x2.empty:
        ax.scatter(c_x2.geometry.centroid.x, c_x2.geometry.centroid.y,
                   color='dodgerblue', marker='s', s=120, edgecolor='black',
                   label='x2: 4x4 Jeep', zorder=3)

    if not c_x1.empty:
        ax.scatter(c_x1.geometry.centroid.x, c_x1.geometry.centroid.y,
                   color='forestgreen', marker='o', s=70, edgecolor='black',
                   label='x1: Foot Patrol', zorder=4)

    if not c_x3.empty:
        ax.scatter(c_x3.geometry.centroid.x, c_x3.geometry.centroid.y,
                   color='purple', marker='P', s=90, edgecolor='white',
                   label='x3: Drone', zorder=5)

    if not c_x4.empty:
        cam_sizes = 40 + (c_x4['Count_Cams'] * 30)
        ax.scatter(c_x4.geometry.centroid.x, c_x4.geometry.centroid.y,
                   color='darkorange', marker='^', s=cam_sizes, edgecolor='black',
                   label='x4: Camera(s)', zorder=6)

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.3, zorder=2)
    camps.plot(ax=ax, color='red', marker='*', markersize=200, edgecolor='white', label='Base Camps', zorder=7)

    ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1), frameon=True, edgecolor='black', title="Security Assets")

    plt.title(f'Active Security Deployments [{current_season.upper()} SEASON]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

    print("\nCount Summary of Deployments")
    print(f"Foot Patrols (x1): {grid_inside['Count_FP'].sum()}")
    print(f"4x4 Jeeps (x2)   : {grid_inside['Count_4x4'].sum()}")
    print(f"Drones (x3)      : {grid_inside['Count_Drones'].sum()}")
    print(f"Cameras (x4)     : {grid_inside['Count_Cams'].sum()}")

def M_MineralMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='M',
        cmap='copper',  # Earthy tones for mineral deposits
        vmin=0,
        vmax=1,
        legend=True,
        legend_kwds={'label': "Mineral Attractiveness Score (M)"},
        edgecolor='black',
        linewidth=0.1,
        alpha=0.9,
        ax=ax,
        zorder=2
    )

    # cells jahan minerals sach mai hai
    blocks_placed = grid_inside[grid_inside['n_blocks'] > 0]
    if not blocks_placed.empty:
        ax.scatter(
            blocks_placed.geometry.centroid.x,
            blocks_placed.geometry.centroid.y,
            color='cyan',
            marker='+',
            s=100,
            label='Artificial Mineral Blocks (E1)',
            zorder=5
        )

        for idx, row in blocks_placed.iterrows():
            ax.text(
                row.geometry.centroid.x, row.geometry.centroid.y,
                f"x{int(row['n_blocks'])}",
                fontsize=9, color='cyan', fontweight='bold', ha='right', va='bottom'
            )

    roads.plot(ax=ax, color='white', linewidth=1, alpha=0.3, zorder=3)

    salt_pan = grid_inside[grid_inside['Terrain_Type'] == 'Salt Pan']
    if not salt_pan.empty:
        salt_pan.boundary.plot(ax=ax, color='white', linewidth=0.5, alpha=0.5, zorder=4)

    plt.title(f'Mineral Attractiveness (M) & Supplement Sites', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def M_Mastermap():
    color_map = {
        'Salt Pan': '#cccccc',  # Gray
        'Grass Savanna': '#fdbf6f',  # Orange
        'Shrub Savanna': '#b2df8a',  # Light Green
        'Tree Savanna': '#33a02c'  # Dark Green
    }

    # Dynamic Salt Pan color based on the season
    if current_season == "Wet":
        color_map['Salt Pan'] = '#a6cee3'  # Light blue for flooded pan
        pan_label = 'Salt Pan (Flooded)'
    else:
        color_map['Salt Pan'] = '#cccccc'  # Gray for dry pan
        pan_label = 'Salt Pan (Dry)'

    # 1. Setup high-res canvas
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # 2. Plot Biomes with the BORDER FIX (Microscopic gray lines so colors pop)
    grid_inside.plot(
        color=grid_inside['Terrain_Type'].map(color_map),
        edgecolor='gray',
        linewidth=0.05,
        ax=ax,
        zorder=1
    )

    # Build Biome Legend with TEXT FIX (scaled fonts for LaTeX)
    legend_patches = [
        mpatches.Patch(color=hex_code, label=pan_label if biome_name == 'Salt Pan' else biome_name)
        for biome_name, hex_code in color_map.items()
    ]
    first_legend = ax.legend(
        handles=legend_patches,
        title="Biomes",
        loc='upper right',
        bbox_to_anchor=(1.15, 1),
        frameon=True,
        edgecolor='black',
        fontsize=11,
        title_fontsize=12
    )

    # 3. Add Infrastructure
    roads.plot(ax=ax, color='black', linewidth=1.5, alpha=0.6, zorder=5)

    # Plot waterholes based on pump status and validity
    active_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 1)]
    natural_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 1)]
    closed_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 0)]
    dry_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 0)]

    # Increased markersizes slightly so they don't disappear when LaTeX shrinks the image
    if not active_wh.empty:
        active_wh.plot(ax=ax, color='dodgerblue', marker='o', markersize=60, edgecolor='black',
                       label='Active WH (Art.)', zorder=6)
    if not natural_wh.empty:
        natural_wh.plot(ax=ax, color='cyan', marker='o', markersize=60, edgecolor='black', label='Active WH (Nat.)',
                        zorder=6)
    if not closed_wh.empty:
        closed_wh.plot(ax=ax, color='red', marker='X', markersize=60, edgecolor='black', label='Closed WH', zorder=6)
    if not dry_wh.empty:
        dry_wh.plot(ax=ax, color='saddlebrown', marker='o', markersize=60, edgecolor='black', label='Dry WH', zorder=6)

    camps.plot(ax=ax, color='red', marker='*', markersize=200, edgecolor='white', label='Camps', zorder=7)
    gates.plot(ax=ax, color='green', marker='^', markersize=200, edgecolor='white', label='Gates', zorder=7)

    # Secondary Legend with LaTeX font scaling
    ax.legend(title=f"Current Season: {current_season}", loc='lower left', bbox_to_anchor=(1.02, 0),
              frameon=True, edgecolor='black', fontsize=11, title_fontsize=12)

    # Re-add the first legend (Matplotlib overwrites legends by default if you call ax.legend twice)
    ax.add_artist(first_legend)

    # 4. The Title and Axis Text Fix
    ax.set_title("Master Biome & Infrastructure Map", fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel("UTM Easting (meters)", fontsize=14)
    ax.set_ylabel("UTM Northing (meters)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 5. Export as Vector Graphic (PDF)
    plt.tight_layout()
    plt.savefig('Master_Biome_Map.pdf', format='pdf', bbox_inches='tight')
    plt.show()
#endregion

    #region Heatmaps
def HM_DistanceToWater():
    # Find max distance to make bins
    max_km = int(np.ceil(grid_inside['Dist_to_Water'].max()))

    bins = [-0.1] + list(range(5, max_km + 5, 5)) + [np.inf]

    # Generate the string labels to match the bins exactly
    labels = [f"0-5 km"] + [f"{bins[i]}-{bins[i + 1]} km" for i in range(1, len(bins) - 2)] + [f">{bins[-2]} km"]

    # Apply za categories
    grid_inside['Water_Intensity'] = pd.cut(grid_inside['Dist_to_Water'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    num_labels = len(labels)
    num_blues = min(6, num_labels)  # The first 5 bins represent 0 to 25km
    num_reds = num_labels - num_blues  # The rest are > 25km

    # Sample 5 evenly spaced colors from the Blues_r colormap (0.0 to 0.7 keeps it from becoming totally white)
    blue_colors = plt.cm.Blues_r(np.linspace(0.0, 0.7, num_blues))

    # If the map has cells further than 25km, sample from the Reds colormap (0.3 to 1.0 to avoid pure white)
    if num_reds > 0:
        red_colors = plt.cm.Reds(np.linspace(0.3, 1.0, num_reds))
        combined_colors = np.vstack((blue_colors, red_colors))
    else:
        combined_colors = blue_colors

    # Generate the final custom map
    custom_cmap = mcolors.ListedColormap(combined_colors)

    # GeoPandas automatically slices this to match the number of 5km categories
    grid_inside.plot(
        column='Water_Intensity',
        cmap=custom_cmap,  # just like Reds_r
        categorical=True,  # Tells GeoPandas to treat these as distinct categories, not a scale
        legend=True,
        legend_kwds={
            'title': "Water Availability",
            'loc': 'upper right',
            'bbox_to_anchor': (1.15, 1)  # Pushes legend slightly outside the map, so it doesn't cover cells
        },
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        ax=ax
    )

    first_legend = ax.get_legend()

    # Plot waterholes based on pump status and validity (seasonal or permanent)
    active_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 1)]
    natural_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 1)]
    closed_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 0)]
    dry_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 0)]

    if not active_wh.empty:
        active_wh.plot(ax=ax, color='dodgerblue', marker='o', markersize=40, edgecolor='black', label='Active Waterholes (Artificial)', zorder=5)
    if not natural_wh.empty:
        natural_wh.plot(ax=ax, color='cyan', marker='o', markersize=40, edgecolor='black', label='Active Waterholes (Natural)', zorder=5)
    if not closed_wh.empty:
        closed_wh.plot(ax=ax, color='red', marker='X', markersize=40, edgecolor='black', label='Closed Waterholes (Artificial)', zorder=5)
    if not dry_wh.empty:
        dry_wh.plot(ax=ax, color='saddlebrown', marker='o', markersize=40, edgecolor='black', label='Dry Waterholes (Natural)', zorder=5)

    for idx, row in waterholes.iterrows():
        color = 'black'  # default value
        # Assign text color based on state for readability
        if row['Type'] == 'A' and row['Pump_Status'] == 1:
            color = 'dodgerblue'
        elif row['Type'] == 'N' and row['Pump_Status'] == 1:
            color = 'cyan'
        elif row['Type'] == 'A' and row['Pump_Status'] == 0:
            color = 'darkred'
        elif row['Type'] == 'N' and row['Pump_Status'] == 0:
            color = 'saddlebrown'

        label_text = f"{idx}"
        ax.annotate(label_text, (row.geometry.x, row.geometry.y), textcoords="offset points", xytext=(5, 5), ha='left',
                    fontsize=8, fontweight='bold', color=color, zorder=7)

    ax.legend(title=f"Current Season: {current_season}", loc='lower left', bbox_to_anchor=(1.01, 0), frameon=True, edgecolor='black')

    plt.title('Distance to Freshwater Heatmap', fontsize=16)
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    ax.add_artist(first_legend)
    plt.show()

    print("\n--- Water Intensity Cell Counts ---")
    print(grid_inside['Water_Intensity'].value_counts(sort=False))

def HM_DistanceToRoad():
    # Find max distance to make bins
    max_km = int(np.ceil(grid_inside['Dist_to_Road'].max()))

    bins = [-0.1] + list(range(5, max_km + 5, 5)) + [np.inf]

    # Generate the string labels to match the bins exactly
    labels = [f"0-5 km"] + [f"{bins[i]}-{bins[i + 1]} km" for i in range(1, len(bins) - 2)] + [f">{bins[-2]} km"]

    # Apply za categories
    grid_inside['Road_Intensity'] = pd.cut(grid_inside['Dist_to_Road'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # Use a built-in colormap ('Reds_r' reverses it so closest is the darkest red)
    # GeoPandas automatically slices this to match the number of 5km categories
    grid_inside.plot(
        column='Road_Intensity',
        cmap='Reds_r',
        categorical=True,
        legend=True,
        legend_kwds={
            'title': "Road Proximity",
            'loc': 'upper right',
            'bbox_to_anchor': (1.15, 1)
        },
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        ax=ax
    )

    roads.plot(ax=ax, color='white', linewidth=1.5, alpha=1.0, zorder=5)

    plt.title('Distance to Road Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

    print("\n--- Road Intensity Cell Counts ---")
    print(grid_inside['Road_Intensity'].value_counts(sort=False))

def HM_DistanceToCamp():
    # made just like distance to road
    # Find max distance to make bins
    max_km = int(np.ceil(grid_inside['Dist_to_Camp'].max()))

    bins = [-0.1] + list(range(5, max_km + 5, 5)) + [np.inf]

    # Generate the string labels to match the bins exactly
    labels = [f"0-5 km"] + [f"{bins[i]}-{bins[i + 1]} km" for i in range(1, len(bins) - 2)] + [f">{bins[-2]} km"]

    # Apply za categories
    grid_inside['Camp_Intensity'] = pd.cut(grid_inside['Dist_to_Camp'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # Using 'Purples_r' to differentiate camp proximity from road proximity visually
    # GeoPandas automatically slices this to match the number of 5km categories
    grid_inside.plot(
        column='Camp_Intensity',
        cmap='Purples_r',
        categorical=True,
        legend=True,
        legend_kwds={
            'title': "Camp Proximity",
            'loc': 'upper right',
            'bbox_to_anchor': (1.15, 1)
        },
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        ax=ax
    )
    first_legend = ax.get_legend()

    camps.plot(ax=ax, color='red', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=5)

    plt.title('Distance to Camp Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    ax.add_artist(first_legend)
    plt.legend(loc="upper left")
    plt.show()

    print("\n--- Camp Intensity Cell Counts ---")
    print(grid_inside['Camp_Intensity'].value_counts(sort=False))

def HM_DistanceToSaltPan():
    global salt_pan_cells
    # Find max distance to make bins
    max_km = int(np.ceil(grid_inside['Dist_to_SaltPan'].max()))

    bins = [-0.1] + list(range(5, max_km + 5, 5)) + [np.inf]

    # Generate the string labels to match the bins exactly
    labels = [f"0-5 km"] + [f"{bins[i]}-{bins[i + 1]} km" for i in range(1, len(bins) - 2)] + [f">{bins[-2]} km"]

    # Apply za categories
    grid_inside['SaltPan_Intensity'] = pd.cut(grid_inside['Dist_to_SaltPan'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # GeoPandas automatically slices this to match the number of 5km categories
    grid_inside.plot(
        column='SaltPan_Intensity',
        cmap='Oranges_r',
        categorical=True,
        legend=True,
        legend_kwds={
            'title': "Salt Pan Proximity",
            'loc': 'upper right',
            'bbox_to_anchor': (1.15, 1)
        },
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        ax=ax
    )

    # Overlay a subtle hatched pattern on the actual Salt Pan cells
    if not salt_pan_cells.empty:
        salt_pan_cells.plot(ax=ax, facecolor='#cccccc', edgecolor='black', linewidth=0.5, alpha=1.0, zorder=5)

    plt.title('Distance to Salt Pan Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

    print("\n--- Salt Pan Intensity Cell Counts ---")
    print(grid_inside['SaltPan_Intensity'].value_counts(sort=False))

def HM_DistanceToBoundary():
    # made just like distance to road
    # Find max distance to make bins
    max_km = int(np.ceil(grid_inside['Dist_to_Fence'].max()))

    bins = [-0.1] + list(range(5, max_km + 5, 5)) + [np.inf]

    # Generate the string labels to match the bins exactly
    labels = [f"0-5 km"] + [f"{bins[i]}-{bins[i + 1]} km" for i in range(1, len(bins) - 2)] + [f">{bins[-2]} km"]

    # Apply za categories
    grid_inside['Fence_Intensity'] = pd.cut(grid_inside['Dist_to_Fence'], bins=bins, labels=labels)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # Using 'Purples_r' to differentiate camp proximity from road proximity visually
    # GeoPandas automatically slices this to match the number of 5km categories
    grid_inside.plot(
        column='Fence_Intensity',
        cmap='viridis_r',
        categorical=True,
        legend=True,
        legend_kwds={
            'title': "Boundary/Fence Proximity",
            'loc': 'upper right',
            'bbox_to_anchor': (1.15, 1)
        },
        edgecolor='black',
        linewidth=0.2,
        alpha=0.85,
        ax=ax
    )

    plt.title('Distance to Boundary/Fence Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

    print("\n--- Boundary/Fence Intensity Cell Counts ---")
    print(grid_inside['Fence_Intensity'].value_counts(sort=False))

def HM_ForageQuality():
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='Forage_Quality',
        cmap='YlGn',
        legend=True,
        legend_kwds={'label': "Forage Quality"},
        edgecolor='black',
        linewidth=0.1,
        ax=ax
    )
    # Red hatched pattern, so you still know where burns are active
    active_burns = grid_inside[grid_inside['Y_Burn'] == 1]
    if not active_burns.empty:
        active_burns.plot(
            ax=ax,
            facecolor='none',
            edgecolor='red',
            hatch='///',
            linewidth=1.5,
            zorder=5
        )
        burn_patch = mpatches.Patch(facecolor='none', edgecolor='red', hatch='///', label='Active Burn')
        ax.legend(handles=[burn_patch], loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True, edgecolor='black')

    plt.title(f'Forage Quality [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_TimeToCamp():
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # Split data to isolate salt pan cells
    valid_cells = grid_inside[grid_inside['Time_to_Camp'] < 800.0]
    blocked_cells = grid_inside[grid_inside['Time_to_Camp'] >= 800.0]

    # plasma_r: Yellow = Fast (low time), Dark Purple = Slow (high time)
    valid_cells.plot(
        column='Time_to_Camp',
        cmap='plasma_r',
        vmin=0,
        vmax=190,
        legend=True,
        legend_kwds={'label': "Response Time (Minutes)"},
        edgecolor='black',
        linewidth=0.1,
        alpha=0.85,
        ax=ax
    )

    # salt pan cells which are colored dark gray in wet season
    if not blocked_cells.empty:
        blocked_cells.plot(ax=ax, facecolor='#222222', edgecolor='black', linewidth=0.1, alpha=0.85, zorder=2)

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.6, zorder=4)

    camps.plot(ax=ax, color='red', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=5)

    plt.title(f'Jeep Arrival Time ($T_{{arrive}}$) Heatmap [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_PThreatScore():
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='Threat_Score',
        cmap='YlOrRd',  # yellow to dark red (low to high threat)
        legend=True,
        legend_kwds={'label': "Threat Score"},
        edgecolor='black',
        linewidth=0.1,
        ax=ax
    )

    plt.title(f'Threat Score Heatmap [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_FThreatScore():
    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='T_Fire',
        cmap='YlOrRd',  # yellow to dark red (low to high threat)
        legend=True,
        legend_kwds={'label': "Threat Score"},
        edgecolor='black',
        linewidth=0.1,
        ax=ax
    )

    plt.title(f'Threat Score Heatmap [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_RiskMap():
    fig, ax = plt.subplots(figsize=(15, 8))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    grid_inside.plot(
        column='R_Total',
        cmap='YlOrRd',  # yellow to dark red (low to high risk)
        legend=True,
        legend_kwds={
            'label': "Final Poaching Risk Score",
            'orientation': "vertical"
        },
        edgecolor='black',
        linewidth=0.1,
        alpha=0.9,
        ax=ax
    )

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.5, zorder=4)
    camps.plot(ax=ax, color='blue', marker='*', markersize=150, edgecolor='white', label='Camps', zorder=5)

    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0), frameon=True, edgecolor='black')

    plt.title(f'Risk Map [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_BaselineParkValue():
    if 'Has_Pump' not in grid_inside.columns:
        activeWh = waterholes[waterholes['Pump_Status'] == 1]
        grid_inside['Has_Pump'] = grid_inside.geometry.intersects(activeWh.geometry.union_all()).astype(int)

    c_camp, c_pump, c_gate, c_road = 45, 15, 9, 3

    unnorm_V = grid_inside['APS'] + (c_camp * grid_inside['Has_Camp']) + (c_pump * grid_inside['Has_Pump']) + (c_gate * grid_inside['Has_Gate']) + (c_road * grid_inside['Has_Road'])

    avg_aps = grid_inside['APS'].mean()
    v_max = unnorm_V.max()

    print(f"Average Animal Presence Score (APS) : {avg_aps:.4f}")
    print(f"Maximum Cell Value (V_max)          : {v_max:.4f}")

    if 'BaseV' not in grid_inside.columns:
        grid_inside['BaseV'] = unnorm_V / (v_max if v_max > 0 else 1.0)

    fig, ax = plt.subplots(figsize=(15, 8))
    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    # We use the 'viridis' colormap: Dark Purple = Empty Dirt, Bright Yellow = High Value (Camps/Rhinos)
    grid_inside.plot(
        column='BaseV',
        cmap='viridis',
        legend=True,
        legend_kwds={
            'label': "Normalized Park Asset Value (0.0 to 1.0)",
            'orientation': "vertical"
        },
        edgecolor='black',
        linewidth=0.1,
        alpha=0.9,
        ax=ax
    )

    # Add Infrastructure Context
    roads.plot(ax=ax, color='white', linewidth=1.2, alpha=0.3, zorder=4)
    camps.plot(ax=ax, color='red', marker='*', markersize=150, edgecolor='black', label='Tourist Camps', zorder=5)

    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0), frameon=True, edgecolor='black')

    plt.title(f'Baseline Park Asset Value ($V_{{x,y}}$) [{current_season} Season]', fontsize=16,
              fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

def HM_RiskDeltaMap():
    fig, ax = plt.subplots(figsize=(18, 12))

    park.boundary.plot(ax=ax, color='black', linewidth=2.5, zorder=10)

    max_abs = max(abs(grid_inside['Delta_Risk'].min()), abs(grid_inside['Delta_Risk'].max()))
    if max_abs == 0:
        max_abs = 1.0

    grid_inside.plot(
        column='Delta_Risk',
        cmap='coolwarm',  # Blue = Risk Decrease (Good), Red = Risk Increase (Bad)
        vmin=-max_abs,
        vmax=max_abs,
        legend=True,
        legend_kwds={
            'label': "Δ T_Risk (Mitigated - Baseline)",
            'orientation': "vertical"
        },
        edgecolor='black',
        linewidth=0.1,
        alpha=0.8,
        ax=ax,
        zorder=1
    )

    roads.plot(ax=ax, color='black', linewidth=1.2, alpha=0.5, zorder=2)

    active_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 1)]
    natural_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 1)]
    closed_wh = waterholes[(waterholes['Type'] == 'A') & (waterholes['Pump_Status'] == 0)]
    dry_wh = waterholes[(waterholes['Type'] == 'N') & (waterholes['Pump_Status'] == 0)]

    if not active_wh.empty:
        active_wh.plot(ax=ax, color='dodgerblue', marker='o', markersize=50, edgecolor='white', label='Active WH (Art.)', zorder=3)
    if not natural_wh.empty:
        natural_wh.plot(ax=ax, color='cyan', marker='o', markersize=50, edgecolor='black', label='Active WH (Nat.)', zorder=3)
    if not closed_wh.empty:
        closed_wh.plot(ax=ax, color='red', marker='X', markersize=50, edgecolor='black', label='Closed WH', zorder=3)
    if not dry_wh.empty:
        dry_wh.plot(ax=ax, color='saddlebrown', marker='o', markersize=50, edgecolor='white', label='Dry WH', zorder=3)

    # 5. E1 (Minerals) & E2 (Controlled Burns)
    c_e1 = grid_inside[grid_inside['n_blocks'] > 0]
    c_e2 = grid_inside[grid_inside['Y_Burn'] > 0]

    if not c_e1.empty:
        ax.scatter(c_e1.geometry.centroid.x, c_e1.geometry.centroid.y, color='cyan', marker='P', s=100, edgecolor='black', label='E1: Minerals', zorder=4)
    if not c_e2.empty:
        c_e2.plot(ax=ax, facecolor='none', edgecolor='red', hatch='///', linewidth=1.5, zorder=4)
        ax.add_patch(mpatches.Patch(facecolor='none', edgecolor='red', hatch='///', label='E2: Active Burn'))

    c_x1 = grid_inside[grid_inside['Count_FP'] > 0]
    c_x2 = grid_inside[grid_inside['Count_4x4'] > 0]
    c_x3 = grid_inside[grid_inside['Count_Drones'] > 0]
    c_x4 = grid_inside[grid_inside['Count_Cams'] > 0]

    if not c_x1.empty:
        ax.scatter(c_x1.geometry.centroid.x, c_x1.geometry.centroid.y, color='forestgreen', marker='o', s=80, edgecolor='white', label='x1: Foot Patrol', zorder=5)
    if not c_x2.empty:
        ax.scatter(c_x2.geometry.centroid.x, c_x2.geometry.centroid.y, color='dodgerblue', marker='s', s=100, edgecolor='black', label='x2: 4x4 Jeep', zorder=5)
    if not c_x3.empty:
        ax.scatter(c_x3.geometry.centroid.x, c_x3.geometry.centroid.y, color='purple', marker='X', s=90, edgecolor='white', label='x3: Drone', zorder=5)
    if not c_x4.empty:
        ax.scatter(c_x4.geometry.centroid.x, c_x4.geometry.centroid.y, color='darkorange', marker='^', s=80, edgecolor='black', label='x4: Camera', zorder=5)

    camps.plot(ax=ax, color='red', marker='*', markersize=250, edgecolor='white', label='Camps', zorder=6)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=True, edgecolor='black')

    plt.title(f'Risk Delta (Mitigated - Baseline) + Master Assets [{current_season} Season]', fontsize=16, fontweight='bold')
    plt.xlabel('UTM Easting (meters)')
    plt.ylabel('UTM Northing (meters)')
    plt.tight_layout()
    plt.show()

#endregion

    #region Manual Generation
def G_Biomes():
    b_roads = gpd.read_file("etosha_roads.geojson").to_crs(epsg=32733)

    if 'Terrain_Type' not in grid_inside.columns:
        grid_inside['Terrain_Type'] = 'Grass Savanna'
    else:
        grid_inside['Terrain_Type'] = 'Grass Savanna'

    color_map = {
        'Grass Savanna': '#fdbf6f',  # Orange/Yellow (Base)
        'Salt Pan': '#cccccc',  # Gray
        'Tree Savanna': '#33a02c',  # Dark Green
        'Shrub Savanna': '#b2df8a'  # Light Green
    }

    cycle_logic = {
        'Grass Savanna': 'Salt Pan',
        'Salt Pan': 'Tree Savanna',
        'Tree Savanna': 'Shrub Savanna',
        'Shrub Savanna': 'Grass Savanna'
    }

    current_brush = ['Salt Pan']

    fig, ax = plt.subplots(figsize=(18, 10))
    plt.title("Left Click: Cycle Biome | Right Clik: Paint with last used color\nCLOSE window to save.", fontsize=14,
              fontweight='bold')

    park.boundary.plot(ax=ax, color='black', linewidth=3, zorder=10)
    b_roads.plot(ax=ax, color='white', linewidth=2, zorder=5)

    patches = []
    for idx, row in grid_inside.iterrows():
        x, y = row.geometry.exterior.coords.xy
        xy = np.column_stack((x, y))

        poly = mpatches.Polygon(xy, facecolor=color_map[row['Terrain_Type']], edgecolor='black', linewidth=0.5, alpha=0.9, zorder=1)
        ax.add_patch(poly)
        patches.append((idx, poly))

    ax.autoscale()
    ax.set_aspect('equal')

    def onclick(event):
        if event.inaxes != ax:
            return
        click_point = Point(event.xdata, event.ydata)

        for l_idx, poly_patch in patches:
            if grid_inside.at[l_idx, 'geometry'].contains(click_point):

                # Rmb: Cycle the color and update the brush
                if event.button == 1:
                    current = grid_inside.at[l_idx, 'Terrain_Type']
                    nxt = cycle_logic[current]

                    grid_inside.at[l_idx, 'Terrain_Type'] = nxt
                    poly_patch.set_facecolor(color_map[nxt])
                    current_brush[0] = nxt  # Save this color to your active brush

                # Lmb: Paint instantly with the active brush
                elif event.button == 3:
                    nxt = current_brush[0]

                    grid_inside.at[l_idx, 'Terrain_Type'] = nxt
                    poly_patch.set_facecolor(color_map[nxt])

                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("\nSaving your hand-crafted ecology...")
    # export_df = grid_inside[['Cell_ID', 'Terrain_Type']].copy()
    # export_df.to_csv("manual_biomes_2.csv", index=False)  # changed name to 'manual_biomes_2' cuz I don't want to lose the real manual_biomes and I commented it to be extra safe

def G_Zones():
    b_roads = gpd.read_file("etosha_roads.geojson").to_crs(epsg=32733)

    color_map = {
        'Grass Savanna': '#fdbf6f',  # Orange/Yellow (Base)
        'Salt Pan': '#cccccc',  # Gray
        'Tree Savanna': '#33a02c',  # Dark Green
        'Shrub Savanna': '#b2df8a'  # Light Green
    }

    currentZone = 1
    fig, ax = plt.subplots(figsize=(18, 10))
    plt.title("Left Click: Add to Zone | Right Clik: Incremeant Zone Number\nCLOSE window to save.", fontsize=14,
              fontweight='bold')

    park.boundary.plot(ax=ax, color='black', linewidth=3, zorder=10)
    b_roads.plot(ax=ax, color='white', linewidth=2, zorder=5)

    patches = []
    for idx, row in grid_inside.iterrows():
        x, y = row.geometry.exterior.coords.xy
        xy = np.column_stack((x, y))

        poly = mpatches.Polygon(xy, facecolor=color_map[row['Terrain_Type']], edgecolor='black', linewidth=0.5, alpha=0.9, zorder=1)
        ax.add_patch(poly)
        patches.append((idx, poly))

    text = ax.text(0.5, 0.05, f"Zone Number: {currentZone}", transform=ax.transAxes, clip_on=False)

    ax.autoscale()
    ax.set_aspect('equal')

    def onclick(event):
        if event.inaxes != ax:
            return
        nonlocal currentZone
        click_point = Point(event.xdata, event.ydata)

        for l_idx, poly_patch in patches:
            if grid_inside.at[l_idx, 'geometry'].contains(click_point):

                # Lmb: Cycle the color and update the brush
                if event.button == 1:
                    grid_inside.at[l_idx, 'Zone_Number'] = currentZone
                    poly_patch.set_facecolor('red')

                # Rmb: Paint instantly with the active brush
                elif event.button == 3:
                    currentZone += 1
                    text.set_text(f"Zone Number: {currentZone}")

                fig.canvas.draw_idle()
                break

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    export_df = grid_inside[['Cell_ID', 'Zone_Number']].copy()
    export_df.to_csv("manual_zones.csv", index=False)

def G_CsvGen():
    name = input("Enter CSV name (without .csv): ").strip() + ".csv"
    print(name)
    img = mpimg.imread('Etosha-Map-2025.jpg')

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    plt.title("Click on every landmark. Close the window when finished.")  # click on only one type of landmark for each file (yani click only on waterholes for etosha_waterholes.csv file)

    coords = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            coords.append({'X_pixel': x, 'Y_pixel': y})
            print(f"Point recorded: X={x}, Y={y}")

            # Draw a tiny red dot on click
            ax.plot(x, y, 'ro', markersize=4)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    # 5. Show the image (The script pauses here until you close the window)
    plt.show()

    c = int(input("Press 1 for save: "))
    if c != 1:
        return

    if len(coords) > 0:
        df_landmarks = pd.DataFrame(coords)
        df_landmarks.to_csv(name, index=False)
        print(f"\nSuccess! Saved {len(coords)} locations to {name}.")
    else:
        print("\nNo points were clicked. Nothing saved.")

    # Note: You can manually add names of waterholes and camps (3rd Column), on the csv by hand by comparing the points with the original map at (https://www.etoshanationalpark.org/media/Etosha-Map-2025.jpg)
    # Note: Likewise, you can also add info manually into another column regarding which waterholes are artificial and which are natural using names. (thank you haarith for spending 13 mins from your life)

#endregion

    #region Toggle Season
def C_Toggle_Season():
    Toggle_Season()

    print(f"\n" + "=" * 40)
    print(f"   SEASON CHANGED TO: {current_season.upper()}")
    print("=" * 40)

    # Show the user how many waterholes were affected
    affected = (waterholes['Validity'] == 'S').sum()
    state_str = "FILLED" if current_season == "Wet" else "DRIED UP"
    print(f">> {affected} seasonal waterholes have {state_str}.")
#endregion

    #region Loader Code
def Heatmaps():
    print(f"\n"
          "1: Distance to Roads or Infrastructure Heatmaps")
    print("2: Forage Quality    | 3: Time to Camp")
    print("4: APS               | 5: Threat Score (Poacher)")
    print("6: Threat (Fires)    | 7: Risk")
    print("8: Baseline Park Val | 9: Delta Risk")
    print("0: Return")
    c = int(input())
    if c == 1:
        Heatmaps_Distance()
    elif c == 2:
        HM_ForageQuality()
    elif c == 3:
        HM_TimeToCamp()
    elif c == 4:
        HM_APS()
    elif c == 5:
        HM_PThreatScore()
    elif c == 6:
        HM_FThreatScore()
    elif c == 7:
        HM_RiskMap()
    elif c == 8:
        HM_BaselineParkValue()
    elif c == 9:
        HM_RiskDeltaMap()

def Heatmaps_Distance():
    print(f"\n"
          "1: Distance to Water | 2: Distance to Road")
    print("3: Distance to Camp  | 4: Distance to Boundary/Fence")
    print("5: Distance to Salt Pan")
    print("0: Return")
    c = int(input())
    if c == 1:
        HM_DistanceToWater()
    elif c == 0:
        Heatmaps()
    elif c == 2:
        HM_DistanceToRoad()
    elif c == 3:
        HM_DistanceToCamp()
    elif c == 4:
        HM_DistanceToBoundary()
    elif c == 5:
        HM_DistanceToSaltPan()

def Maps():
    print(f"\n"
         f"1: Collars Map         | 2: Master Map")
    print("3: Firebreaks Map      | 4: Zone Map")
    print("5: Security Deployment | 6: Minerals Map")
    print("0: Return")
    c = int(input())
    if c == 1:
        M_CollarMap()
    elif c == 2:
        M_Mastermap()
    elif c == 3:
        M_FirebreakMap()
    elif c == 4:
        M_ZoneMap()
    elif c == 5:
        M_SecurityDeploymentMap()
    elif c == 6:
        M_MineralMap()

def Manual_Gen():
    print("Enter 1 for Manual Biomes Gen, 2 for Manual CSV Gen from og map, 3 for Calculation of time to camp, 4 for Manual Zones gen, 0 to return")
    c = int(input())
    if c == 1:
        G_Biomes()
    elif c == 2:
        G_CsvGen()
    elif c == 3:
        Calculate_Time_To_Camp()
    elif c == 4:
        G_Zones()

def Choice():
    choice = -1
    while choice != 0:
        global current_season

        print(f"\n--- ETOSHA SIMULATOR (Season: {current_season}) ---")
        print("1: Heatmaps          | 2: Other Maps")
        print("3: Manual Generation | 4: Toggle Season")
        print("5: Run Simulation    | 0: Exit")
        try:
            choice = int(input())
            if choice == 2:
                Maps()
            elif choice == 1:
                Heatmaps()
            elif choice == 3:
                Manual_Gen()
            elif choice == 4:
                Toggle_Season()
            elif choice == 5:
                Run_Master_Simulation()
        except ValueError:
            print("Invalid Input")


Choice()
#endregion
#endregion

