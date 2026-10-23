"""Fetch official Taiwan protection layers and export planner-ready GeoJSON."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile


MOA_MANIFEST = "https://data.moa.gov.tw/api/FileToJson.ashx?DataId=350&DataTable=OpenDataList"
MOENV_API = "https://data.moenv.gov.tw/api/v2/gisepa_p_13"
# Current file published under dataset 6377 on 2026-07-20; use an API key to resolve updates.
MOENV_CURRENT_ZIP = "https://newcdx.moenv.gov.tw/api/files/GISEPA_P_13/220bedf3-6d6c-4685-8d0f-ac64042bb226"
OCA_MARINE_HABITAT_GEOJSON = (
    "https://iocean.oca.gov.tw/oca_datahub/WebService/GetData.ashx?"
    "id=fba2361b-6ac5-4f7c-827c-631ce78cc91f"
)
MAX_ARCHIVE_SIZE = 512 * 1024 * 1024
MAX_EXTRACTED_SIZE = 2 * 1024 * 1024 * 1024
DATASETS = {
    "nature_reserve": ("Ministry of Agriculture", "24914", "https://data.gov.tw/dataset/24914"),
    "drinking_water_protection": ("Ministry of Environment", "6377", "https://data.gov.tw/dataset/6377"),
    "marine_wildlife_habitat": (
        "Ocean Conservation Administration, Ocean Affairs Council",
        "157562", "https://data.gov.tw/dataset/157562"),
}


def _json(url):
    request = Request(url, headers={"User-Agent": "Taiwan-Route-Planner/1.0", "Accept": "application/json"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _date_key(record, field):
    return re.sub(r"\D", "", str(record.get(field, "")))


def latest_moa_shapefile(records):
    records = [row for row in records if isinstance(row, dict)
               and "shp" in str(row.get("FileName", "")).lower() and row.get("FileUrl")]
    if not records:
        raise ValueError("农业部数据清单没有可下载的 SHP")
    return max(records, key=lambda row: _date_key(row, "CreateDate"))


def latest_moenv_zip(records):
    records = [row for row in records if row.get("url", "").startswith("https://") and ".zip" in row.get("filename", "").lower()]
    if not records:
        raise ValueError("环境部数据清单没有可下载的 ZIP")
    return max(records, key=lambda row: _date_key(row, "upload_date"))


def validate_weight(value):
    weight = float(value)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("软避让权重必须是非负有限数值")
    return weight


def make_feature(geometry, properties, *, source, category, dataset_id, updated_at, layer, weight):
    properties.update({
        "mode": "soft", "weight": validate_weight(weight), "source": source,
        "category": category, "dataset_id": dataset_id,
        "source_updated_at": updated_at or None, "source_layer": layer,
    })
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def _download(url, cache_dir, category):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    path = cache_dir / f"{category}_{key}.zip"
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    request = Request(url, headers={"User-Agent": "Taiwan-Route-Planner/1.0"})
    partial = path.with_suffix(".part")
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"下载失败：{url} ({exc})") from exc
    if partial.stat().st_size > MAX_ARCHIVE_SIZE or not zipfile.is_zipfile(partial):
        partial.unlink(missing_ok=True)
        raise ValueError(f"下载结果不是有效且小于 512 MiB 的 ZIP：{url}")
    partial.replace(path)
    return path


def _download_geojson(url, cache_dir, category):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    path = cache_dir / f"{category}_{key}.geojson"
    if path.is_file():
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("type") == "FeatureCollection":
                return path
        except (OSError, json.JSONDecodeError):
            pass
    request = Request(url, headers={"User-Agent": "Taiwan-Route-Planner/1.0"})
    partial = path.with_suffix(".part")
    try:
        with urlopen(request, timeout=120) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"下载失败：{url} ({exc})") from exc
    if partial.stat().st_size > MAX_ARCHIVE_SIZE:
        partial.unlink(missing_ok=True)
        raise ValueError(f"下载 GeoJSON 超过 512 MiB：{url}")
    try:
        document = json.loads(partial.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        partial.unlink(missing_ok=True)
        raise ValueError(f"下载结果不是有效 GeoJSON：{url}") from exc
    if document.get("type") != "FeatureCollection" or not isinstance(document.get("features"), list):
        partial.unlink(missing_ok=True)
        raise ValueError(f"下载结果不是 GeoJSON FeatureCollection：{url}")
    partial.replace(path)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip(archive_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    extracted_size = 0
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            extracted_size += item.file_size
            if extracted_size > MAX_EXTRACTED_SIZE:
                raise ValueError("ZIP 解压后超过 2 GiB")
            relative = PurePosixPath(item.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"ZIP 包含不安全路径：{item.filename}")
            target = (output_dir / Path(*relative.parts)).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"ZIP 包含不安全路径：{item.filename}")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return sorted(output_dir.rglob("*.shp"))


def _read_layers(shapefiles, *, agency, category, dataset_id, updated_at, weight):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("转换图层需要安装 requirements.txt 中的 geopandas") from exc

    features, layer_info = [], []
    for path in shapefiles:
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError(f"图层缺少 CRS：{path.name}")
        source_crs = frame.crs.to_string()
        frame = frame.to_crs("EPSG:4326")
        input_count = len(frame)
        if len(frame):
            min_lon, min_lat, max_lon, max_lat = frame.total_bounds
            if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
                raise ValueError(f"转换后的图层超出 WGS84 范围：{path.name}")
        invalid_count = int((~frame.geometry.is_valid).sum())
        frame.geometry = frame.geometry.make_valid()
        frame = frame[frame.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
        for item in json.loads(frame.to_json(na="null"))["features"]:
            geometry = item.get("geometry")
            if not geometry:
                continue
            features.append(make_feature(
                geometry, item.get("properties") or {},
                source=f"{agency} (dataset {dataset_id}; updated {updated_at or 'unknown'})",
                category=category, dataset_id=dataset_id, updated_at=updated_at,
                layer=path.stem, weight=weight,
            ))
        layer_info.append({"layer": path.stem, "source_crs": source_crs,
                           "input_count": input_count, "polygon_count": len(frame),
                           "repaired_count": invalid_count, "skipped_count": input_count - len(frame)})
    return features, layer_info


def _read_marine_habitat(path, *, weight):
    """Convert the OCA GeoJSON, whose current coordinate values contradict its CRS84 tag."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("转换图层需要安装 requirements.txt 中的 geopandas") from exc

    frame = gpd.read_file(path)
    if frame.crs is None:
        raise ValueError("海洋栖息地图层缺少 CRS")
    source_document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    declared_crs = str((source_document.get("crs", {}).get("properties") or {}).get(
        "name", frame.crs.to_string()))
    min_x, min_y, max_x, max_y = map(float, frame.total_bounds)
    coordinate_note = "source CRS honored"
    source_crs = declared_crs
    if min_x < -180 or max_x > 180 or min_y < -90 or max_y > 90:
        if not (0 <= min_x < max_x <= 500_000 and 2_000_000 <= min_y < max_y <= 3_000_000):
            raise ValueError("海洋栖息地图层坐标既非经纬度，也不符合已核验的 TWD97 TM2 121 范围")
        frame = frame.set_crs("EPSG:3826", allow_override=True)
        source_crs = "EPSG:3826"
        coordinate_note = "GeoJSON declares CRS84 but coordinate extent is projected TWD97 TM2 121"
    frame = frame.to_crs("EPSG:4326")
    min_lon, min_lat, max_lon, max_lat = map(float, frame.total_bounds)
    if not (110 <= min_lon < max_lon <= 130 and 15 <= min_lat < max_lat <= 35):
        raise ValueError("海洋栖息地图层转换后不落在台湾周边核验范围内")
    input_count = len(frame)
    invalid_count = int((~frame.geometry.is_valid).sum())
    frame.geometry = frame.geometry.make_valid()
    frame = frame[frame.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
    for column in frame.columns:
        if column != frame.geometry.name and getattr(frame[column].dtype, "kind", None) == "M":
            frame[column] = frame[column].astype(str)
    features = []
    for item in json.loads(frame.to_json(na="null"))["features"]:
        geometry = item.get("geometry")
        if geometry:
            features.append(make_feature(
                geometry, item.get("properties") or {},
                source="Ocean Conservation Administration (dataset 157562; catalog updated 2026-01-15)",
                category="marine_wildlife_habitat", dataset_id="157562",
                updated_at="2026-01-15", layer=Path(path).stem, weight=weight,
            ))
    if not features:
        raise ValueError("海洋栖息地图层没有有效多边形")
    layer_info = [{
        "layer": Path(path).stem, "declared_crs": declared_crs,
        "source_crs_used": source_crs, "crs_note": coordinate_note,
        "input_count": input_count, "polygon_count": len(frame),
        "repaired_count": invalid_count, "skipped_count": input_count - len(frame),
    }]
    return features, layer_info


def _latest_moenv(api_key):
    query = urlencode({"api_key": api_key, "format": "JSON", "limit": 1000, "sort": "ImportDate desc"})
    rows = _json(f"{MOENV_API}?{query}")
    if not isinstance(rows, list):
        raise ValueError("环境部 API 返回格式错误")
    return latest_moenv_zip(rows)


def fetch_constraints(output, *, weight, cache_dir, api_key=None, file_url=None,
                      include_nature_reserve=True):
    weight = validate_weight(weight)
    if file_url:
        moenv_url, moenv_date = file_url, "not provided"
    elif api_key:
        record = _latest_moenv(api_key)
        moenv_url, moenv_date = record["url"], record.get("upload_date", "")
    else:
        moenv_url, moenv_date = MOENV_CURRENT_ZIP, "20260720T153652"

    sources = []
    if include_nature_reserve:
        moa = latest_moa_shapefile(_json(MOA_MANIFEST))
        sources.append((moa["FileUrl"], "nature_reserve", *DATASETS["nature_reserve"],
                        moa.get("CreateDate", "")))
    sources.append((moenv_url, "drinking_water_protection",
                    *DATASETS["drinking_water_protection"], moenv_date))
    features, data_sources = [], []
    with tempfile.TemporaryDirectory(prefix="taiwan_constraints_") as temporary:
        for index, (url, category, agency, dataset_id, dataset_url, updated_at) in enumerate(sources):
            archive = _download(url, cache_dir, category)
            digest = _sha256(archive)
            layers = _extract_zip(archive, Path(temporary) / str(index))
            if not layers:
                raise ValueError(f"{agency} ZIP 中没有 Shapefile")
            converted, layer_info = _read_layers(
                layers, agency=agency, category=category, dataset_id=dataset_id,
                updated_at=updated_at, weight=weight,
            )
            if not converted:
                raise ValueError(f"{agency} ZIP 中没有有效的多边形")
            features.extend(converted)
            data_sources.append({
                "agency": agency, "dataset_id": dataset_id, "dataset_url": dataset_url,
                "download_url": url, "source_updated_at": updated_at or None,
                "license": "Open Government Data License, version 1.0",
                "archive_sha256": digest, "layers": layer_info,
            })

        marine_path = _download_geojson(
            OCA_MARINE_HABITAT_GEOJSON, cache_dir, "marine_wildlife_habitat")
        marine_features, marine_layer_info = _read_marine_habitat(marine_path, weight=weight)
        features.extend(marine_features)
        data_sources.append({
            "agency": DATASETS["marine_wildlife_habitat"][0],
            "dataset_id": DATASETS["marine_wildlife_habitat"][1],
            "dataset_url": DATASETS["marine_wildlife_habitat"][2],
            "download_url": OCA_MARINE_HABITAT_GEOJSON,
            "source_updated_at": "2026-01-15",
            "license": "Open Government Data License, version 1.0",
            "file_sha256": _sha256(marine_path), "layers": marine_layer_info,
        })

    document = {
        "type": "FeatureCollection", "features": features, "data_sources": data_sources,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": "OGC:CRS84 (WGS84 longitude, latitude)",
            "constraint_mode": "soft", "soft_weight": weight, "feature_count": len(features),
            "important_note": "Screening layers only; verify current legal status before use.",
        },
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    partial.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    partial.replace(output)
    return len(features)


def main(argv=None):
    parser = argparse.ArgumentParser(description="下载台湾官方保护区图层并转为 WGS84 GeoJSON")
    package_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("--output", type=Path, default=package_dir / "data" / "official_constraints.geojson")
    parser.add_argument("--cache-dir", type=Path, default=package_dir / "data" / "official_constraints_cache")
    parser.add_argument("--soft-weight", type=float, required=True,
                        help="路线筛查权重，须按项目确定；不是法定避让距离")
    parser.add_argument("--moenv-api-key", default=os.environ.get("MOENV_API_KEY"))
    parser.add_argument("--moenv-file-url", default=None)
    parser.add_argument("--skip-nature-reserve", action="store_true",
                        help="仅获取环境部和海保署图层；适用于农业部清单暂不可访问时")
    args = parser.parse_args(argv)
    try:
        count = fetch_constraints(args.output, weight=args.soft_weight, cache_dir=args.cache_dir,
                                  api_key=args.moenv_api_key, file_url=args.moenv_file_url,
                                  include_nature_reserve=not args.skip_nature_reserve)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    print(f"已输出 {count} 个软约束面：{args.output}")


if __name__ == "__main__":
    main()
