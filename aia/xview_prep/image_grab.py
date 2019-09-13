import geopandas as gpd
from os.path import join as op
from os.path import splitext as sp
from os.path import basename as base
from shutil import copyfile
import csv


geojson = ('12.geojson')
image_folder = 'train_images'
subset_folder = 'xview_subset/12'
image_id_field = 'image_id'

def image_intercept(bbox_geojson, image_folder, out_folder, image_id_field, create_subset=False):
    gdf = gpd.read_file(bbox_geojson)
    unique = gdf[image_id_field].unique()
    if create_subset == True:
        for i in image_id_field:
            src = op(image_folder, i)
            dst = op(subset_folder, i)
            copyfile(src, dst)
    with open(op(out_folder, sp(base(bbox_geojson))[0]+"_images.csv"),'w') as f:
        writer = csv.writer(f);
        writer.writerows(zip(unique))

image_intercept(geojson, image_folder, subset_folder, image_id_field)
