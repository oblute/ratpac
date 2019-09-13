import geopandas as gpd
from os.path import join as op
​

df = gpd.read_file('/xView_train.geojson')
output_dir = "/planes"
​

def split_geojson(gdf, outdir, groupby_col='type_id', remaps=None):
    if remaps == None:
        for name, group in gdf.groupby([groupby_col]):
            group.to_file(op(outdir, "{}.geojson".format(name)), driver="GeoJSON")
    else:
        gdf[groupby_col] = gdf[groupby_col].map(remaps).fillna(gdf[groupby_col])
        for name, group in gdf.groupby([groupby_col]):
            group.to_file(op(outdir, "{}.geojson".format(name)), driver="GeoJSON")
​
split_geojson(df, outdir=output_dir)
#split_geojson(df, outdir=od, remaps = {12: 11, 13 : 11})