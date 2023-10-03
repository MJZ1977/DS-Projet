import rasterio
import numpy as np
import glob
import plotly.express as px
import plotly.graph_objs as go
import cv2
import pandas as pd
import geopandas as gpd

def load_tabfile(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    coords = []
    for line in lines:
        if '(' in line and ')' in line:
            coord_str = line.split('(')[1].split(')')[0]
            coords.append(tuple(map(float, coord_str.split(','))))

    return rasterio.transform.from_bounds(*coords[0], *coords[2], len(coords[0]) // 2, len(coords) // 2)

def load_image(path):
    with rasterio.open(path) as src:
        value = np.transpose(src.read(), [1,2,0])
        resolution = src.res
        bounds = src.bounds
        
    
    # transfrom = load_tabfile(path[:-3]+"tab")
    xrange = [bounds.left, bounds.right]
    yrange = [bounds.bottom, bounds.top]
    return value, xrange, yrange

def load_shp(path):
    # Load data
    bdtopo = gpd.read_file(path)
    # Add Category and Type
    bdtopo.insert(0, 'Category', path.split('\\')[-2])
#     bdtopo['Category'] = path.split('\\')[-2]
    bdtopo.insert(1, 'Type', path.split('\\')[-1][:-4])
#     bdtopo['Type'] = path.split('\\')[-1][:-4]
    return bdtopo


def isInMap(xrange, yrange):
    def my_function(polynom):
        x, y = polynom.centroid.x, polynom.centroid.y
        if xrange[0]<x and xrange[1]>x and yrange[0]<y and yrange[1]>y:
            return True
        else :
            return False
        
    return my_function

def convert_centroid(map_size, xrange, yrange):
    def my_function(polygon):
        x, y = polygon.centroid.x, polygon.centroid.y
        x_new = (x - xrange[0])/(xrange[1]-xrange[0])*map_size[0]
        y_new = map_size[1] - (y - yrange[0])/(yrange[1]-yrange[0])*map_size[1]
        return [x_new, y_new]
    
    return my_function

def convert_polygon(map_size, xrange, yrange):
    def my_function(polygon):
        if polygon.wkt.lower()[:7]=="polygon":
            x, y = polygon.exterior.coords.xy
            x = x.tolist()
            y = y.tolist()
        elif polygon.wkt[:10]=="LINESTRING":
            x, y = polygon.coords.xy
            x = x.tolist()
            x += x[::-1]
            y = y.tolist()
            y += y[::-1]
        
        else :
            x = [1,2]
            y = [1,2]
        x = np.array(x)
        y = np.array(y) 
        x_new = (x - xrange[0])/(xrange[1]-xrange[0])*map_size[0]
        y_new = map_size[1] - (y - yrange[0])/(yrange[1]-yrange[0])*map_size[1]
        return [x_new, y_new]
    
    return my_function

def generate_xy_polygons(bdtopo_area, map_size):

    list_x = []
    for xpoly in bdtopo_area['xpolygon']:
        list_x.extend(xpoly.tolist() + [None])
    list_x = list_x[:-1]
    
    list_y = []
    for ypoly in bdtopo_area['ypolygon']:
        ypoly = map_size[1]-ypoly
        list_y.extend(ypoly.tolist() + [None])
    list_y = list_y[:-1]
    
    return list_x, list_y

def generate_x_polygons(xdata):
    list_x = []
    for xpoly in xdata:
        list_x.extend(xpoly.tolist() + [None])
    list_x = list_x[:-1]
    
    return list_x

def create_map(value, scatters_data, scatters_list_name, points_data, points_list_name, resolution):
    # image = px.imshow(cv2.resize(value, resolution))
    image = go.Image(z=cv2.resize(value, resolution))
    
    points = []
    
    for i, (list_x, list_y) in enumerate(scatters_data):
        # Ajouter des points
        point = go.Scatter(
            x=list_x,
            y=list_y,
            fill="toself",
            name=scatters_list_name[i],
        #     fillcolor="blue"

        )
        points.append(point)
        
    for i, (list_x, list_y) in enumerate(points_data):
        # Ajouter des points
        point = go.Scatter(
            x=list_x,
            y=list_y,
            mode='markers',
            marker=dict(size=5),
            name=points_list_name[i]
        )
        points.append(point)

    # Créer la figure
    fig = go.Figure(data=[image]+points)
    # fig = go.Figure(data=image)
    fig.update_xaxes(range=[0,resolution[0]])
    fig.update_yaxes(range=[0,resolution[1]])
    
    fig.update_layout(
        # autosize=False,
        width=1000,
        height=1000,)

    # Afficher la figure
    return fig


# ---------------------------------
# ---- Début du script


# 1---- Charger l'orthophoto
Dossier = r'BDORTHO\1_DONNEES_LIVRAISON_2018-03-00388\BDO_IRC_0M50_JP2-E080_UTM20W84GUAD_D977-2013\*.jp2'
Num_img = 5
Reduction = 2	# Entier >= 1 de préférence

paths = glob.glob(Dossier)
value, xrange, yrange = load_image(paths[Num_img])
Xbords = [xrange[0]+0000,xrange[1]-3000]	#Possibilité de réduire les bords pour alléger
Ybords = [yrange[0]+3000,yrange[1]-0000]

print(xrange, yrange)
# print(value.shape)

Xlim=((np.array(Xbords)-xrange[0])*value.shape[0]/(xrange[1]-xrange[0])).astype(int)
Ylim=((np.array(Ybords)-yrange[0])*value.shape[1]/(yrange[1]-yrange[0])).astype(int)
Ylim=value.shape[0]-Ylim[::-1]	
print(Xlim,Ylim)
value = value[(Ylim[0]):(Ylim[1]),(Xlim[0]):(Xlim[1])] #Il semble que le tableau value inverse les x et les y
map_size = [value.shape[0]//Reduction,value.shape[1]//Reduction]
# map_size=[1000,1000]
print("Résolution de la carte = ",map_size)
# px.imshow(cv2.resize(value, map_size)).show()

# 2---- Charger les formes (shapes)
paths_shp = glob.glob(r'BDTOPO\1_DONNEES_LIVRAISON_2022-12-00159\BDT_3-3_SHP_RGAF09UTM20_D977-ED2022-12-15\*\*.shp')
bdtopo = load_shp(paths_shp[0])
for path in paths_shp[1:]:
    bdtopo = bdtopo.append(load_shp(path))
bdtopo.reset_index(drop=True, inplace=True)

# 3--- Conversion des formes aux coordonnées de l'image

# Sélection
bdtopo_zone = bdtopo[bdtopo['geometry'].apply(isInMap(Xbords, Ybords))].copy()
print('Avant :', bdtopo.shape, 'Après :', bdtopo_zone.shape)

# Conversion
bdtopo_zone['centroid'] = bdtopo_zone['geometry'].apply(convert_centroid(map_size, Xbords, Ybords))
bdtopo_zone['xcentroid'] = bdtopo_zone['centroid'].apply(lambda x : x[0])
bdtopo_zone['ycentroid'] = bdtopo_zone['centroid'].apply(lambda x : x[1])

bdtopo_point = bdtopo_zone[bdtopo['geometry'].apply(lambda x : x.wkt.lower()[:5]=="point")]
bdtopo_zone = bdtopo_zone[bdtopo_zone['geometry'].apply(lambda x : x.wkt.lower()[:7]=="polygon" or x.wkt[:10]=="LINESTRING")]

bdtopo_zone['polygon'] = bdtopo_zone['geometry'].apply(convert_polygon(map_size, Xbords, Ybords))
bdtopo_zone['xpolygon'] = bdtopo_zone['polygon'].apply(lambda x : x[0])
bdtopo_zone['ypolygon'] = bdtopo_zone['polygon'].apply(lambda x : x[1])

bdtopo_zone_agregate = bdtopo_zone.groupby('Type').agg({'xpolygon':list, 'ypolygon':list})
bdtopo_zone_agregate['xpolygon_ready'] = bdtopo_zone_agregate['xpolygon'].apply(generate_x_polygons)
bdtopo_zone_agregate['ypolygon_ready'] = bdtopo_zone_agregate['ypolygon'].apply(generate_x_polygons)

bdtopo_point_agregate = bdtopo_point.groupby('Type').agg({'xcentroid':list, 'ycentroid':list})


# 4--- Affichage
fig = create_map(value, bdtopo_zone_agregate[['xpolygon_ready', 'ypolygon_ready']].values, bdtopo_zone_agregate.index, 
         bdtopo_point_agregate[['xcentroid', 'ycentroid']].values, bdtopo_point_agregate.index, map_size)
# fig = px.imshow(cv2.resize(value, map_size))
fig.write_html('Figure.html')