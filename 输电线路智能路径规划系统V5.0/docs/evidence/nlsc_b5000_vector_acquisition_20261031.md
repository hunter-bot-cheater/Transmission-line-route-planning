# NLSC 1:5,000 vector data acquisition check

Checked: 2026-10-31

## Finding

The National Land Surveying and Mapping Center (NLSC) states that its 1:5,000 digital topographic map has vector deliverables in SHP, DWG, and DXF. The listed layers include the public-utility network, specifically high-voltage line towers and high-voltage power lines. This is the strongest identified candidate for independent, post-hoc route-shape verification.

Official source: [NLSC FAQ, question 19](https://www.nlsc.gov.tw/News_Toggle.aspx?_CSN=2033&n=1564&sms=9708).

## Access and fee boundary

The NLSC map e-commerce service accepts applications by individuals, groups, and agencies. Online application requires a mobile natural-person certificate, natural-person certificate, or business certificate; application proceeds through ordering and payment before electronic files become available for download. See [NLSC application FAQ, question 18](https://www.nlsc.gov.tw/News_Toggle.aspx?_CSN=2033&n=1564&sms=9708).

The current public fee page does not unambiguously price the specific 1:5,000 high-voltage tower/wire vector product. Its NT$150-per-sheet line describes the 1:5,000 photo-map digital product without orthophoto and contours; that price must not be assumed to cover the separate numerical-topographic-map vector deliverable. The specific catalog item, licensing category, requested sheets, and total fee must be confirmed in the official e-shop before any order. See [NLSC fees and supply terms](https://www.nlsc.gov.tw/cp.aspx?n=1549).

No account was accessed, no order was submitted, and no payment or restricted data was obtained in this check.

## Permitted evaluation role

If acquired and licensed for this project, the vector layers should be kept out of planner inputs and used only after candidate geometries are frozen and hashed. Before scoring, verify that the mapped line can be assigned to the target circuit and that the map coverage and revision date cover the route. Tower/wire geometry alone may not identify voltage, circuit, operational status, or station-side connection points; absent those attributes, report the comparison as map-feature similarity rather than verified as-built-circuit error.

The public B5000 WMS remains a visualization service. Its raster display is not equivalent to downloadable vector truth and does not by itself identify a circuit. Do not digitize it into planner priors.

## Next acquisition check

1. Select one benchmark route whose station connection points have independent sources.
2. Use the official 1:5,000 sheet index to enumerate only sheets intersecting a narrow evaluation corridor.
3. Confirm the e-shop product name, access classification, allowed academic use, per-sheet price, and current map vintage with NLSC.
4. After authorized acquisition, hash the source archives and inspect CRS, layer names, attributes, geometry validity, coverage, and voltage/circuit identifiability before any post-hoc score is produced.
