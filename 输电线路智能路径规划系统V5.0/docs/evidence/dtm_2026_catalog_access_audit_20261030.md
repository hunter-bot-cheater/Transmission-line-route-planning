# Official 2026 DTM catalog and access audit (2026-10-30)

## Catalog record

The Ministry of the Interior's Land Administration Department lists the **2026 Taiwan 20 m grid DTM** at [dataset 178729](https://data.gov.tw/dataset/178729). The catalog says it covers Taiwan and some offshore islands, with tile production years varying by map sheet, and describes the data as suitable for terrain-relief display and approximate analysis. It lists an open-data license, free access, and a 2026-09-30 publication date.

The resource row is labeled `CSV` and its description says `2026 ... DTM DSM`. That label does not identify a raster tile directly. The resource endpoint exposed by the catalog is:

`https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/B6219841-E247-4743-958D-77FB61067092/resource/D788C60B-33CB-4A84-A438-EC6E8B7060F1/download`

## Retrieval result

- The web retrieval service returned HTTP 400, `Unsupported content-type: text/csv`, when asked to open the resource. This indicates that the catalog resource is a CSV payload the page reader cannot parse; it does not establish that the underlying tile files are unavailable.
- From the authenticated planning server, `curl -sSIL --max-time 30 <resource-url>` timed out after 30 seconds.
- From the workstation, Python's default TLS validation rejected the host certificate with `Missing Subject Key Identifier`; `curl.exe` timed out after 20 seconds. TLS validation was not bypassed.
- No official CSV, tile, or ZIP was downloaded by this audit, and no server file was created.

## What the current route experiments used

The 12-route hazard-vector experiment already used `/data1/B24041626/pp/dtm_2026_planner_input_20261028_01/dtm_2026_20m_geographic_4326.tif`. Its MVT and SHP arms produced essentially identical routes, with median symmetric F1 within 100 m of 3.19%; replacing hazard representation did not improve geometry. The separate three-case native-20 m experiment also used this raster and found median symmetric P95 near 4.50 km against historical references. These results are documented in the [12-route post-hoc comparison](official_hazard_vector_sensitivity_20261030_01/posthoc_5m_20261030_02/README.md) and [20 m DTM experiment](dtm_2026_native_20m_3case_6400_20261028/README.md).

The existing raster was copied from a public Docker image and its SHA-256 matches that image's source record, but the official catalog's original data was not byte-compared. Its TIFF GeoKey is user-defined, the EPSG:3826 assignment was made explicitly during preprocessing, and the vertical datum was not independently confirmed from the raster metadata. Therefore, the existing 20 m result is a useful high-resolution sensitivity test, not yet an independently authenticated official-data baseline.

## Planning consequence

Do not rerun the same 20 m route experiment as if it were a new data-quality improvement. First retrieve and validate the official catalog CSV or its referenced tiles through a trusted network path, inspect the tile list and source metadata, and compare source checksums, CRS, and vertical datum with the existing mirror. Keep historical line geometry out of planning; use it only after candidate routes are frozen for post-hoc evaluation. The current three benchmark endpoints also remain unverified proxies, so their spatial error is not a station-connection accuracy estimate.
