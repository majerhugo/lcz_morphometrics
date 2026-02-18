import rasterio as rio
import rioxarray
from rasterio.features import rasterize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, f1_score, accuracy_score
from rasterio.plot import reshape_as_raster, reshape_as_image
import geopandas as gpd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import shapely
from shapely.geometry import Polygon
from rasterio.enums import Resampling
from shapely.geometry import box
from rasterio.transform import from_origin
import pandas as pd
from rasterio.warp import reproject, Resampling

def extract_fold_data(data, splited_ref, fold):
    
    test_polygons = splited_ref[splited_ref["fold"] == fold]
    train_polygons = splited_ref[splited_ref["fold"] != fold]

    train_cells = gpd.sjoin(data, train_polygons, predicate="within", how="inner")
    test_cells = gpd.sjoin(data, test_polygons, predicate="within", how="inner")

    train_cells["gridcode"] = train_cells["gridcode"].astype("int32")
    test_cells["gridcode"] = test_cells["gridcode"].astype("int32")

    # cleaning
    y_train = train_cells[['gridcode']]
    X_train = train_cells.drop(['focal','index_right', 'gridcode', 'class_name', 'geometry', 'fold'], axis=1)
    
    y_test = test_cells[['gridcode']]
    X_test = test_cells.drop(['focal','index_right', 'gridcode', 'class_name', 'geometry', 'fold'], axis=1)

    fold_data = {"fold": fold,
                 "X_train": X_train,
                 "y_train": y_train,
                 "X_test": X_test,
                 "y_test": y_test}
    
    return fold_data

def finetune_height(X_train, y_train, X_test, y_test, max_height, class_weight=False):
    depth = [d+1 for d in range(max_height)]
    for d in depth:
        if class_weight:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=d, class_weight='balanced')    
        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=d)
        
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        
        # evaluation
        train_accuracy = accuracy_score(y_train, pred_train)
        train_wf1 = f1_score(y_train, pred_train, average='weighted', labels=np.unique(y_train))
        pred_test = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, pred_test)
        test_wf1 = f1_score(y_test, pred_test, average='weighted', labels=np.unique(y_test))

        # compare
        print(f'{d},OA Train: {round(train_accuracy*100, 2)}, OA Test: {round(test_accuracy*100, 2)}, OA Gap: {round((train_accuracy-test_accuracy)*100, 2)}, wF1 Train: {round(train_wf1*100, 2)}, wF1 Test: {round(test_wf1*100, 2)}, wF1 Gap: {round((train_wf1-test_wf1)*100, 2)}') 

def finetune_max_features(X_train, y_train, X_test, y_test, height, max_features, class_weight=False):
    max_feats = [d+1 for d in range(max_features)]
    train_accuracy_list = []
    test_accuracy_list = []
    train_accuracy = 0
    test_accuracy = 0
    best_param = 0
    for f in max_feats:
        if class_weight:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=height, class_weight='balanced', max_features=f)    
        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, max_depth=height, max_features=f)
    
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
    
        # evaluation
        train_accuracy_temp = accuracy_score(y_train, pred_train)
        train_wf1 = f1_score(y_train, pred_train, average='weighted', labels=np.unique(y_train))
        pred_test = model.predict(X_test)
        test_accuracy_temp = accuracy_score(y_test, pred_test)
        test_wf1 = f1_score(y_test, pred_test, average='weighted', labels=np.unique(y_test))
    
        print(f'{f},OA Train: {round(train_accuracy_temp*100, 2)}, OA Test: {round(test_accuracy_temp*100, 2)}, OA Gap: {abs(round((train_accuracy_temp-test_accuracy_temp)*100, 2))}, wF1 Train: {round(train_wf1*100, 2)}, wF1 Test: {round(test_wf1*100, 2)}, wF1 Gap: {round((train_wf1-test_wf1)*100, 2)}')
        train_accuracy_list.append(train_accuracy_temp)
        test_accuracy_list.append(test_accuracy_temp)
        if (test_accuracy_temp>test_accuracy) and (train_accuracy_temp-test_accuracy_temp<=0.05) and (train_accuracy_temp>=test_accuracy_temp):
            train_accuracy=train_accuracy_temp
            test_accuracy=test_accuracy_temp
            best_param=f

    return best_param, round(train_accuracy*100,2), round(test_accuracy*100,2), round((train_accuracy*100)-(test_accuracy*100),2)
   
def rasterize_top20_morphometrics(imagery_to_match, output_path, data, model):
    with rio.open(imagery_to_match) as src:
        meta = src.meta
        transform = src.transform
        out_shape = (src.height, src.width)
        crs = src.crs

    feature_importance = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_
    )

    # select top 20 features
    top20 = feature_importance.sort_values(ascending=False).head(20)

    top_features = top20.index
    importances = top20.values
    num_bands = len(top_features)

    with rio.open(
        output_path,
        'w',
        driver='GTiff',
        height=out_shape[0],
        width=out_shape[1],
        count=num_bands,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:

        for i, (column, importance) in enumerate(zip(top_features, importances), start=1):

            rasterized = rasterize(
                ((geom, value) for geom, value in zip(data.geometry, data[column])),
                out_shape=out_shape,
                transform=transform,
                fill=0,
                dtype='float32'
            )

            dst.write(rasterized, i)

            print(f'Band {i:02d} | {column} | importance = {importance:.5f}')

    print(f"Rasterization complete. Saved to {output_path}.")

def create_grid(gdf, size):
    # get total bounds (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # create ranges for x and y
    x_coords = np.arange(minx, maxx + size, size)
    y_coords = np.arange(miny, maxy + size, size)
    
    # generate grid polygons
    grid_polys = []
    for x in x_coords:
        for y in y_coords:
            # create a 100x100 box
            grid_polys.append(box(x, y, x + size, y + size))
    grid = gpd.GeoDataFrame({'geometry': grid_polys}, crs=gdf.crs)
    
    # add a unique ID to every grid cell
    grid['grid_id'] = range(len(grid))
    
    return grid

def aggregate(INPUT_FILE, CLASS_COLUMN, GRID_SIZE, OUTPUT_FILE):

    print("1. Loading data...")
    #data = gpd.read_file(INPUT_FILE)
    data = gpd.read_parquet(INPUT_FILE)
        
    print(f"2. Generating {GRID_SIZE}m grid...")
    grid = create_grid(data, GRID_SIZE)
    
    print("3. Calculating intersections")
    overlay = gpd.overlay(grid, data, how='intersection')
        
    # Calculate the area of every resulting fragment
    overlay['area_m2'] = overlay.geometry.area
        
    print("4. determining majority class per cell...")
    area_sums = overlay.groupby(['grid_id', CLASS_COLUMN])['area_m2'].sum().reset_index()
        
    # Sort so the largest area is at the top for each grid_id
    area_sums_sorted = area_sums.sort_values(['grid_id', 'area_m2'], ascending=[True, False])
        
    # Drop duplicates to keep only the top (majority) row for each grid_id
    winners = area_sums_sorted.drop_duplicates(subset=['grid_id'], keep='first')
        
    # Keep only the columns we need
    winners = winners[['grid_id', CLASS_COLUMN]]
        
    print("5. Joining results back to the master grid...")
    final_grid = grid.merge(winners, on='grid_id', how='left')
        
    # remove grid cells that are purely empty
    # final_grid = final_grid.dropna(subset=[CLASS_COLUMN])
    
    print(f"6. Saving to {OUTPUT_FILE}...")
    #final_grid.to_file(OUTPUT_FILE, driver="GPKG")
    final_grid.to_parquet(OUTPUT_FILE, compression='zstd')
    print("Done!")

def rasterize_reference_polygons(reference_polygons, satellite_image, output):
    with rio.open(satellite_image) as src:
        raster_crs = src.crs
        raster_transform = src.transform
        raster_width = src.width
        raster_height = src.height
        raster_bounds = src.bounds

    if reference_polygons.crs != raster_crs:
        reference_polygons = reference_polygons.to_crs(raster_crs)

    rasterized_array = rasterize(
        [(geom, value) for geom, value in zip(reference_polygons.geometry, reference_polygons['gridcode'])],
        out_shape=(raster_height, raster_width), 
        transform=raster_transform,
        fill=0,
        dtype=np.uint8)

    with rio.open(
        output, 
        'w', 
        driver='GTiff', 
        count=1, 
        dtype=rasterized_array.dtype, 
        crs=raster_crs, 
        transform=raster_transform, 
        width=raster_width, 
        height=raster_height
    ) as dst:
        dst.write(rasterized_array, 1)
        
def match_rasters(raster_to_change_path, raster_path):
    raster_to_change = rioxarray.open_rasterio(raster_to_change_path, masked=True)
    raster = rioxarray.open_rasterio(raster_path, masked=True)

    raster_to_change = raster_to_change.drop_vars("band").squeeze()
    raster = raster.drop_vars("band").squeeze()

    raster_to_change = raster_to_change.astype('int64')
    raster = raster.astype('int64')
    
    raster_matched = raster_to_change.rio.reproject_match(raster)

    return raster_matched
    
def lcz_map(offset_left, offset_top, image, prediction, output):
    tile_size = (32, 32)  
    stride = 10
    center_size = 10

    with rio.open(image) as src:
        W_image, H_image = src.width, src.height
        transform = src.transform 
        crs = src.crs 

    T_width, T_height = tile_size
    tiles_per_row = (W_image - offset_left - T_width) // stride + 1
    tiles_per_column = (H_image - offset_top - T_height) // stride + 1

    total_tiles = tiles_per_row * tiles_per_column
    if len(prediction) != total_tiles:
        raise ValueError(f"Number of labels ({len(prediction)}) does not match expected tiles ({total_tiles}).")

    classification_map = np.full((H_image, W_image), -1, dtype=int)
    half_center_size = center_size // 2

    for idx, label in enumerate(prediction):
        row_idx = idx // tiles_per_row
        col_idx = idx % tiles_per_row

        top = offset_top + row_idx * stride
        left = offset_left + col_idx * stride
    
        center_row_start = top + (T_height // 2) - half_center_size
        center_row_end = top + (T_height // 2) + half_center_size
        center_col_start = left + (T_width // 2) - half_center_size
        center_col_end = left + (T_width // 2) + half_center_size
    
        classification_map[
            max(center_row_start, 0):min(center_row_end, H_image),
            max(center_col_start, 0):min(center_col_end, W_image)
        ] = label

    with rio.open(
        output,
        'w',
        driver='GTiff',
        height=classification_map.shape[0],
        width=classification_map.shape[1],
        count=1,
        dtype=classification_map.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(classification_map, 1)

    print(f" saved to {output}")
    
def resample_lcz_map(input_path, output_path, target_resolution_meters=100):
    with rio.open(input_path) as src:
        scale_x = src.res[0] / target_resolution_meters
        scale_y = src.res[1] / target_resolution_meters

        new_width = int(src.width * scale_x)
        new_height = int(src.height * scale_y)
        
        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTiff',
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

        with rio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=new_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )
                
        src.close()
        dst.close()
        print(f" saved to {output_path}")

def perpixel_validation(output, test_polygons_path, splited_ref_data):
    metrics = []
    confusion_matrices = {}

    all_possible_classes = list(range(1, 18))
    urban_classes = list(range(1, 11))
    natural_classes = list(range(11, 18))

    for fold_idx, (pred_path, gt_path) in enumerate(zip(output, test_polygons_path)):
        with rio.open(pred_path) as src_pred:
            pred = src_pred.read(1)
            pred_transform = src_pred.transform
            pred_crs = src_pred.crs
            pred_nodata = src_pred.nodata
            pred_shape = (src_pred.height, src_pred.width)
    
        with rio.open(gt_path) as src_gt:
            gt = src_gt.read(1)
            gt_nodata = src_gt.nodata
            gt_resampled = np.empty(pred_shape, dtype=gt.dtype)
            reproject(
                source=gt,
                destination=gt_resampled,
                src_transform=src_gt.transform,
                src_crs=src_gt.crs,
                dst_transform=pred_transform,
                dst_crs=pred_crs,
                resampling=Resampling.nearest
            )

        mask_test = np.ones(pred.shape, dtype=bool)
        if pred_nodata is not None:
            mask_test &= (pred != pred_nodata)
        if gt_nodata is not None:
            mask_test &= (gt_resampled != gt_nodata)
        mask_test &= (gt_resampled != -1) & (gt_resampled != 0)
        mask_test &= (pred != -1) & (pred != 0)
    
        y_true_test = gt_resampled[mask_test].astype(int)
        y_pred_test = pred[mask_test].astype(int)
    
        cm_test = confusion_matrix(y_true_test, y_pred_test, labels=sorted(splited_ref_data['gridcode'].unique()))

        oa_test = accuracy_score(y_true_test, y_pred_test)
        wf1_test = f1_score(y_true_test, y_pred_test, average='weighted')
    
        urban_mask_test = np.isin(y_true_test, urban_classes)
        wf1_urban_test = f1_score(y_true_test[urban_mask_test], y_pred_test[urban_mask_test], average='weighted') if np.any(urban_mask_test) else np.nan
    
        natural_mask_test = np.isin(y_true_test, natural_classes)
        wf1_natural_test = f1_score(y_true_test[natural_mask_test], y_pred_test[natural_mask_test], average='weighted') if np.any(natural_mask_test) else np.nan

        record = {
            "Fold": fold_idx,
            "OA": round(oa_test*100, 2),
            "wF1": round(wf1_test*100, 2),
            "wF1_Urban": round(wf1_urban_test*100, 2),
            "wF1_Natural": round(wf1_natural_test*100, 2)
        }

        for c in all_possible_classes:
            record[f"F1_Class_{c}"] = np.nan

        f1_per_class_array = f1_score(y_true_test, y_pred_test, average=None, labels=np.unique(y_true_test))
        for label, score in zip(np.unique(y_true_test), f1_per_class_array):
            if label in all_possible_classes:
                record[f"F1_Class_{int(label)}"] = round(score * 100, 2)

        metrics.append(record)
        
        confusion_matrices[fold_idx] = {
            "Confusion_Matrix": cm_test,
            "labels": sorted(splited_ref_data['gridcode'].unique())
        }

    return metrics, confusion_matrices