"""Rebuild an EPSG:4326 DEM from cached AWS Terrarium XYZ tiles."""

import argparse
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

from .terrarium import XYZ_TILE_SIZE, terrarium_rgb_to_elevation, xyz_tile_transform


def lonlat_to_xyz(lon, lat, zoom):
    n = 2 ** zoom
    lat = min(max(lat, -85.05112878), 85.05112878)
    x = math.floor((lon + 180.0) / 360.0 * n)
    y = math.floor((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi)
                   * 0.5 * n)
    return x, y


def rebuild_dem(tile_cache, output_path, bbox, zoom=12, overwrite=False):
    """Warp cached Web Mercator tiles to a regular geographic grid.

    ``bbox`` is ``(min_lon, min_lat, max_lon, max_lat)``. The source tile rows
    remain in EPSG:3857 until GDAL reprojects them; they are never interpreted
    as equally spaced latitude rows.
    """
    cache = Path(tile_cache)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    min_lon, min_lat, max_lon, max_lat = map(float, bbox)
    x0, y0 = lonlat_to_xyz(min_lon, max_lat, zoom)
    x1, y1 = lonlat_to_xyz(max_lon, min_lat, zoom)
    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    tile_paths = {
        (x, y): cache / f"t_{zoom}_{x}_{y}.png"
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
    }
    missing = [str(path) for path in tile_paths.values() if not path.is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} XYZ tiles are missing: {preview}")

    source_transform = xyz_tile_transform(zoom, x0, y0)
    source_width, source_height = nx * XYZ_TILE_SIZE, ny * XYZ_TILE_SIZE
    source_profile = {
        "driver": "GTiff", "width": source_width, "height": source_height,
        "count": 1, "dtype": "float32", "crs": "EPSG:3857",
        "transform": source_transform, "nodata": -32768.0,
        "compress": "lzw", "tiled": True,
        "blockxsize": XYZ_TILE_SIZE, "blockysize": XYZ_TILE_SIZE,
        "BIGTIFF": "IF_SAFER",
    }

    with tempfile.TemporaryDirectory(prefix="terrarium-warp-", dir=output.parent) as temp_dir:
        source_path = Path(temp_dir) / "terrarium_3857.tif"
        warped_path = Path(temp_dir) / output.name
        with rasterio.open(source_path, "w", **source_profile) as source:
            for (tile_x, tile_y), tile_path in tile_paths.items():
                with Image.open(tile_path) as tile_image:
                    rgba = np.asarray(tile_image.convert("RGBA"))
                values = terrarium_rgb_to_elevation(rgba)
                invalid = ((rgba[..., 3] == 0) | (values < -100.0) | (values > 9000.0))
                values[invalid] = source_profile["nodata"]
                source.write(values, 1, window=Window(
                    (tile_x - x0) * XYZ_TILE_SIZE,
                    (tile_y - y0) * XYZ_TILE_SIZE,
                    XYZ_TILE_SIZE, XYZ_TILE_SIZE,
                ))
            source.update_tags(
                source="AWS Terrain Tiles Terrarium XYZ cache",
                zoom=str(zoom),
                tile_range=f"x={x0}-{x1};y={y0}-{y1}",
                elevation_encoding="R*256 + G + B/256 - 32768 metres",
                vertical_datum="not verified from the cached PNG files",
            )

        with rasterio.open(source_path) as source:
            left, bottom, right, top = transform_bounds(
                source.crs, "EPSG:4326", *source.bounds, densify_pts=21)
            center_lat = 0.5 * (bottom + top)
            xres = 360.0 / (XYZ_TILE_SIZE * (2 ** zoom))
            yres = xres * math.cos(math.radians(center_lat))
            width = math.ceil((right - left) / xres)
            height = math.ceil((top - bottom) / yres)
            destination_transform = Affine(xres, 0.0, left, 0.0, -yres, top)
            destination_profile = {
                "driver": "GTiff", "width": width, "height": height,
                "count": 1, "dtype": "float32", "crs": "EPSG:4326",
                "transform": destination_transform, "nodata": -32768.0,
                "compress": "lzw", "tiled": True,
                "blockxsize": XYZ_TILE_SIZE, "blockysize": XYZ_TILE_SIZE,
                "BIGTIFF": "IF_SAFER",
            }
            with rasterio.open(warped_path, "w", **destination_profile) as destination:
                reproject(
                    source=rasterio.band(source, 1),
                    destination=rasterio.band(destination, 1),
                    src_transform=source.transform, src_crs=source.crs,
                    src_nodata=source.nodata,
                    dst_transform=destination_transform, dst_crs="EPSG:4326",
                    dst_nodata=destination_profile["nodata"],
                    resampling=Resampling.bilinear, num_threads=2,
                )
                destination.update_tags(
                    source="AWS Terrain Tiles Terrarium XYZ cache",
                    source_crs="EPSG:3857",
                    source_zoom=str(zoom),
                    source_tile_range=f"x={x0}-{x1};y={y0}-{y1}",
                    georeferencing="Web Mercator XYZ tiles reprojected to EPSG:4326",
                    resampling="bilinear",
                    vertical_datum="not verified from the cached PNG files",
                )

        os.replace(warped_path, output)

    with rasterio.open(output) as result:
        print(f"Wrote {output}: {result.width}x{result.height}, "
              f"EPSG:4326, bounds={tuple(round(v, 6) for v in result.bounds)}")


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile-cache", required=True, help="directory containing t_Z_X_Y.png")
    parser.add_argument("--output", default=str(project_dir / "data" / "taiwan_dem_35m_georef_fixed.tif"))
    parser.add_argument("--bbox", nargs=4, type=float, default=(119.4, 21.7, 122.3, 25.5),
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--zoom", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rebuild_dem(args.tile_cache, args.output, args.bbox, args.zoom, args.overwrite)


if __name__ == "__main__":
    main()
