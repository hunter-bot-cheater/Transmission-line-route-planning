# Official constraint data review (2026-10-27)

This review identifies public sources worth verifying for Taiwan corridor screening. Listing a source does not mean the file was downloaded, its coverage validated, or its categories treated as legally prohibitive.

## Higher-value hazard source candidates

The Ministry of Agriculture's 2026 (ROC year 115) open data catalogue lists official SHP downloads for:

- [1,753 debris-flow potential streams](https://data.gov.tw/dataset/176524), with stable stream IDs, risk class, length, watershed area, and geology fields.
- [1,753 debris-flow potential stream impact areas](https://data.gov.tw/dataset/176526), with stream IDs, overflow point identifiers, risk class, and exposed-household fields.
- [94 large-scale landslide potential areas](https://data.gov.tw/dataset/176527), with stable zone IDs, risk class, geology/fault, land ownership, watershed, and date fields.
- [94 large-scale landslide potential-area impact zones](https://data.gov.tw/dataset/176528), with potential-area identifiers, risk class, and exposed-household fields.

The catalogue identifies the Ministry of Agriculture as the provider and lists Government Data Open License v1. Direct file links discovered in the catalogue are:

- `https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=SHP&RID=71084&id=J71`
- `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=71085&id=J73`
- `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=66656&id=J74`
- `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=66657&id=J75`

The web retrieval tool timed out on all four direct file links during this review. No file is treated as present or verified. The next retrieval attempt should run on the server and write only to a new, uniquely named folder under `/data1/B24041626/pp`. After download, preserve the untouched archives, record each exact URL, byte count and SHA-256, inspect sidecar completeness, CRS, feature count, IDs, geometries, and spatial extent, then compare these source polygons with the currently used generalized z10 hazard layer. Do not overwrite existing hazard data. Potential-area and impact-zone polygons are risk indicators, not automatically statutory no-build boundaries.

### Resource-link mapping rechecked 2026-10-29

The official catalogue pages were reopened and their resource links followed. This corrects the stream/impact RID mapping above; the file endpoints themselves still timed out, so these are confirmed catalogue links, not downloaded or validated files:

| Dataset | Catalogue resource label | Catalogue-linked file URL | Result |
| --- | --- | --- | --- |
| [115年度 1753 條土石流潛勢溪流圖](https://data.gov.tw/dataset/176524) | `debrisstream1753_20260126_twd97` | `https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=SHP&RID=71084&id=J71` | Official page link confirmed; file fetch timed out |
| [115年度 1753 條土石流潛勢溪流影響範圍圖](https://data.gov.tw/dataset/176526) | `debris1753_20260126_twd97` | `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=71085&id=J73` | Official page link confirmed; file fetch timed out |
| [115年度 94 處大規模崩塌潛勢區](https://data.gov.tw/dataset/176527) | `115年_94處警戒發布區潛勢範圍_TWD97_UTF8` | `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=66656&id=J74` | Official page link confirmed; file fetch timed out |
| [115年度 94 處大規模崩塌潛勢區影響範圍](https://data.gov.tw/dataset/176528) | `115年_94處警戒發布區影響範圍_TWD97_UTF8` | `https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?FileType=SHP&RID=66657&id=J75` | Official page link confirmed; file fetch timed out |

The catalogue lists the layers as SHP downloads under Government Open Data License 1.0. Treat them as official screening-risk inputs after their contents are validated; do not convert potential or impact-area polygons directly into hard exclusions.

## Transmission asset data remains insufficient

The [Taipower pole-coordinate dataset](https://data.gov.tw/dataset/33305) contains administrative location, pole form, pole number, and TWD97 coordinates. A prior local audit found 393,462 records across the downloaded Taichung and Miaoli files, of which only three were typed as towers; there is no circuit, voltage, line ID, tower sequence, or station-side topology. See the [station endpoint data gap audit](../台电站间规划端点数据缺口审计_20261027.md). It must not be converted into a transmission network by spatial clustering.

Taipower also publishes annual national station attribute tables: [ultra-high-voltage and primary substation transformer capacity/load](https://data.gov.tw/dataset/16874), [secondary substation location, supply area, and capacity/load](https://data.gov.tw/dataset/16960), and [substation electromagnetic-field measurements with administrative area, station name, and measurement year](https://data.gov.tw/dataset/16877). These can help cross-check a station's name, class, service area, and operating context, but the catalogue fields do not provide a verified point geometry or circuit-side terminal. Use them as identity/attribute checks only; do not geocode a textual location and treat it as the line-side route endpoint.

The [Taipei public utility transmission-system station layer](https://data.gov.tw/dataset/156863) exposes station geometry, name, operating status, managing unit, and installation date. Its related [transmission-system pipeline layer](https://data.gov.tw/dataset/156861) has geometry and pipe/segment identifiers, start/end identifiers, installation and status fields, and pipe dimensions. The catalog's XML resource endpoints are `https://data.taipei/api/dataset/f2d95aa3-671e-4ff6-97d4-d21953311957/resource/6b62553d-74c7-4e01-8736-6111afab4052/download` (stations) and `https://data.taipei/api/dataset/d506226a-a236-4b05-850f-d656ef53162d/resource/e02cc7b9-1918-4f49-9441-3de5e669c868/download` (pipelines). The web reader could not parse/cache these XML downloads; server retrieval and data validation remain pending. Both are Taipei-only reference layers; the catalogue warns that exact records must be confirmed with the utility. They may help identify candidate urban stations and underground transmission corridors where a study window overlaps Taipei, but they do not provide circuit voltage, line-side terminal geometry, overhead tower sequence, or a national network. Do not use these public utility geometries as a route prior or treat them as authoritative station connection points.

The [Tainan public utility map listing for transmission pipes](https://data.gov.tw/dataset/122514) covers roadway areas across Tainan's 37 districts, exposes a WMS service, and says detailed data must be obtained from the relevant utility; the catalogue reports no current update. It is a local reference layer, not a national overhead transmission route dataset.

## Land use and terrain

The [National Land Surveying and Mapping Center service cloud](https://maps.nlsc.gov.tw/S09SOA/pro/intro.jsp) documents WMS/WMTS/WFS and API services. Some general map services are open without application; the land-use WFS is listed for government agencies and academic units conducting short-term analysis research. WMS imagery can support visual review but is not a precise vector exclusion boundary. Any WFS access or download should record the requested layer, vintage, CRS, license/terms, and actual coverage.

The official [2026 20 m DTM catalogue entry](https://data.gov.tw/dataset/178729) states that production years vary by quadrangle and describes the data as suitable for terrain display and rough analysis. Its downloadable resource is an index CSV, but the catalogue currently labels that resource "DSM" while the dataset itself is titled "DTM"; inspect the index and each tile's metadata before use. The official [MOI DTM service](https://dtm.moi.gov.tw/) says 20 m services are open without application while its 1 m service is restricted to government agencies. A separate [2025 20 m DTM catalogue entry](https://data.gov.tw/dataset/176927) says its index covers locally updated areas and its map sheets also vary in production year. Neither index proves a uniform national bare-earth surface. The NLSC's [DTM overview](https://www.nlsc.gov.tw/cp.aspx?Create=1&n=1853) distinguishes terrain DEM from surface DSM and documents historical higher-resolution products in some areas. Before substituting terrain data, verify each tile's grid spacing, production date, bare-earth versus surface semantics, horizontal/vertical datum, accuracy, and access conditions.

The catalogue resource index URLs are `https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/B6219841-E247-4743-958D-77FB61067092/resource/D788C60B-33CB-4A84-A438-EC6E8B7060F1/download` (2026) and `https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/A964612F-0D64-4C81-BFE5-6C1F2BA61DED/resource/A0B94F67-8ADF-48A1-8DF0-C60719AD2B28/download` (2025). The web tool identified both resources as CSV but cannot display their contents. No tile has been downloaded or accepted as a valid DEM in this review.

If these indexes resolve to valid DTM tiles for the three test windows, the next resolution experiment should compare the current approximately 35 m DEM against the verified 20 m source using the same frozen endpoints, constraint hashes, cost weights, and one fixed search method; disable population and AI to isolate the terrain input. Use common valid coverage, never silently replace DTM with DSM, and size the grid so the 20 m source is not silently downsampled. The same area at 20 m has about 3.1 times as many cells as at 35 m, so run cases sequentially with a server memory guard. Freeze candidate hashes before any historical-line scoring. This tests terrain-source sensitivity; it does not establish that a finer grid will match a historical line or prove engineering feasibility.

## Engineering gate

The official [Transmission and Distribution Equipment Installation Rules](https://law.moea.gov.tw/LawContent.aspx?id=FL011046) provide spacing and clearance tables that depend on voltage, structure, conductor, and crossing object. Generic rules cannot certify a route without the selected circuit's line-side station connection points, voltage, conductor and tower type, span limits, sag/load/temperature assumptions, crossing elevations, and the applicable design criteria from the asset owner. Until those are supplied and checked, report engineering feasibility as **not evaluated**.
