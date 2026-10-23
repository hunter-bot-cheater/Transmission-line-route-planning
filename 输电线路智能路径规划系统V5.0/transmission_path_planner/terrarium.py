"""Helpers for correctly locating XYZ Terrarium elevation pixels."""

import math

import numpy as np
from affine import Affine


WEB_MERCATOR_RADIUS_M = 6378137.0
XYZ_TILE_SIZE = 256


def terrarium_rgb_to_elevation(rgb):
    """Decode Terrarium RGB pixels into metres, preserving float fractions."""
    pixels = np.asarray(rgb)
    if pixels.ndim < 3 or pixels.shape[-1] < 3:
        raise ValueError("Terrarium pixels must have at least three RGB channels")
    values = pixels[..., :3].astype(np.float64, copy=False)
    return (values[..., 0] * 256.0 + values[..., 1]
            + values[..., 2] / 256.0 - 32768.0).astype(np.float32)


def xyz_tile_transform(zoom, tile_x, tile_y):
    """Return the source-pixel affine transform in EPSG:3857 for an XYZ tile."""
    scale = XYZ_TILE_SIZE * (2 ** int(zoom))
    resolution = 2.0 * math.pi * WEB_MERCATOR_RADIUS_M / scale
    left = -math.pi * WEB_MERCATOR_RADIUS_M + tile_x * XYZ_TILE_SIZE * resolution
    top = math.pi * WEB_MERCATOR_RADIUS_M - tile_y * XYZ_TILE_SIZE * resolution
    return Affine(resolution, 0.0, left, 0.0, -resolution, top)


def xyz_pixel_to_lonlat(zoom, tile_x, tile_y, pixel_x, pixel_y):
    """Return the WGS84 center of a pixel in a Web Mercator XYZ tile."""
    x, y = xyz_tile_transform(zoom, tile_x, tile_y) * (pixel_x + 0.5, pixel_y + 0.5)
    lon = math.degrees(x / WEB_MERCATOR_RADIUS_M)
    lat = math.degrees(math.atan(math.sinh(y / WEB_MERCATOR_RADIUS_M)))
    return lon, lat
