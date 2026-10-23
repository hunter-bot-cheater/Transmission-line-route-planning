# transmission_path_planner/cost.py
"""26 维特征栈、CostUNet 建设代价面预测、启发式代价。"""

from .config import *
from .geometry import *
from .data import *

def _minmax01(band):
    b = np.asarray(band, np.float32)
    bmin, bmax = np.nanmin(b), np.nanmax(b)
    if not np.isfinite(bmin) or not np.isfinite(bmax) or (bmax - bmin) < 1e-9:
        return np.zeros_like(b)
    return np.clip((b - bmin) / (bmax - bmin), 0, 1).astype(np.float32)


def build_feature_stack(dem, terrain, osm, dist_existing=None):
    H, W = dem.shape
    stack = np.zeros((H, W, N_FEATURES), np.float32)
    # 13 维地形
    stack[:, :, 0] = _minmax01(terrain["elevation"])
    stack[:, :, 1] = _minmax01(terrain["slope"])
    stack[:, :, 2] = (terrain["aspect_cos"] * 0.5 + 0.5).astype(np.float32)
    stack[:, :, 3] = (terrain["aspect_sin"] * 0.5 + 0.5).astype(np.float32)
    stack[:, :, 4] = _minmax01(terrain["tri"])
    stack[:, :, 5] = _minmax01(terrain["tpi_100"])
    stack[:, :, 6] = _minmax01(terrain["tpi_300"])
    stack[:, :, 7] = _minmax01(terrain["tpi_900"])
    stack[:, :, 8] = _minmax01(np.abs(terrain["profile_curvature"]))
    stack[:, :, 9] = _minmax01(np.abs(terrain["plan_curvature"]))
    stack[:, :, 10] = _minmax01(terrain["roughness_3"])
    stack[:, :, 11] = _minmax01(terrain["roughness_9"])
    stack[:, :, 12] = _minmax01(terrain["roughness_27"])
    # 13-17 距离波段（归一化到 0-1）
    stack[:, :, 13] = np.clip(osm["dist_road"] / 5000.0, 0, 1)
    stack[:, :, 14] = np.clip(osm["dist_water"] / 5000.0, 0, 1)
    if dist_existing is not None:
        stack[:, :, 15] = np.clip(dist_existing / 10000.0, 0, 1)
    stack[:, :, 16] = np.clip(osm["dist_railway"] / 5000.0, 0, 1)
    stack[:, :, 17] = np.clip(osm["dist_fault"] / 10000.0, 0, 1)
    # 18-20 分类/密度/植被
    stack[:, :, 18] = osm["landuse_code"].astype(np.float32) / 8.0
    p99 = np.nanpercentile(osm["building_density"], 99) if np.any(osm["building_density"] > 0) else 1.0
    stack[:, :, 19] = np.clip(osm["building_density"] / max(p99, 1.0), 0, 1)
    stack[:, :, 20] = np.clip(osm["vegetation_height"] / 30.0, 0, 1)
    # 21-25 风险
    for i, name in enumerate(["typhoon_risk", "seismic_risk", "landslide_risk", "ice_cover_risk", "lightning_risk"]):
        stack[:, :, 21 + i] = np.clip(osm[name], 0, 1).astype(np.float32)
    return stack


if HAS_TORCH:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE.type == "cuda":
        print(f"  [GPU] 使用 CUDA 设备: {torch.cuda.get_device_name(0)}")
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch, dropout=0.0):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            return self.dropout(self.conv(x))

    class ResidualBlock(nn.Module):
        def __init__(self, channels, dropout=0.0):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)
            self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            return F.relu(out + x)

    class CostUNet(nn.Module):
        def __init__(self, n_features=N_FEATURES, enc=None, dec=None, bn=512, dropout=0.1):
            super().__init__()
            enc = enc or [32, 64, 128, 256, 512]
            dec = dec or [256, 128, 64, 32]
            self.encoder_blocks = nn.ModuleList()
            in_ch = n_features
            for o in enc:
                self.encoder_blocks.append(DoubleConv(in_ch, o, dropout))
                in_ch = o
            self.bottleneck = nn.Sequential(
                DoubleConv(enc[-1], bn, dropout),
                ResidualBlock(bn, dropout), ResidualBlock(bn, dropout))
            self.decoder_blocks = nn.ModuleList()
            self.up_convs = nn.ModuleList()
            for i, o in enumerate(dec):
                sk = enc[-(i + 1)] if i < len(enc) else enc[-1]
                self.up_convs.append(nn.ConvTranspose2d(in_ch, o, 2, stride=2))
                self.decoder_blocks.append(DoubleConv(o + sk, o, dropout))
                in_ch = o
            self.final_conv = nn.Sequential(
                nn.Conv2d(dec[-1], 16, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1), nn.Sigmoid())

        def forward(self, x):
            inp = x.shape[2:]
            skips = []
            for e in self.encoder_blocks:
                x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
            x = self.bottleneck(x)
            for i, (up, d) in enumerate(zip(self.up_convs, self.decoder_blocks)):
                x = up(x); sk = skips[-(i + 1)]
                if x.shape[2:] != sk.shape[2:]:
                    x = F.interpolate(x, size=sk.shape[2:], mode="bilinear", align_corners=True)
                x = d(torch.cat([x, sk], 1))
            x = self.final_conv(x)
            if x.shape[2:] != inp:
                x = F.interpolate(x, size=inp, mode="bilinear", align_corners=True)
            return x

        def predict(self, stack):
            self.eval()
            H, W, C = stack.shape
            x = torch.from_numpy(np.ascontiguousarray(stack)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            if H <= 512 and W <= 512:
                with torch.no_grad():
                    return self.forward(x).squeeze().cpu().numpy().astype(np.float32)
            ps, ov = 256, 32
            res = np.zeros((H, W), np.float32)
            w = np.zeros((H, W), np.float32)
            for r in range(0, H, ps - ov):
                for c in range(0, W, ps - ov):
                    re = min(r + ps, H); ce = min(c + ps, W)
                    r0 = max(0, re - ps); c0 = max(0, ce - ps)
                    pt = x[:, :, r0:re, c0:ce].to(DEVICE)
                    with torch.no_grad():
                        p = self.forward(pt).squeeze().cpu().numpy().astype(np.float32)
                    res[r0:re, c0:ce] += p
                    w[r0:re, c0:ce] += 1.0
            return (res / np.clip(w, 1e-6, None)).astype(np.float32)


def load_costunet():
    for p in MODEL_WEIGHTS:
        if p.exists():
            try:
                try:
                    ckpt = torch.load(p, map_location="cpu", weights_only=True)
                except Exception:
                    ckpt = torch.load(p, map_location="cpu", weights_only=False)
                sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
                model = CostUNet(n_features=N_FEATURES)
                model.load_state_dict(sd)
                model.to(DEVICE)
                model.eval()
                print(f"  [AI] 已加载 CostUNet 权重: {p.name}")
                return model
            except Exception as ex:
                print(f"  [warn] 加载 {p.name} 失败: {ex}")
    return None


def heuristic_cost(dem, terrain):
    slope_n = _minmax01(terrain["slope"])
    rough_n = np.clip(_minmax01(terrain["roughness_27"]), 0, 1)
    return np.clip(0.65 * slope_n ** 1.5 + 0.20 * rough_n, 0, 1).astype(np.float32)


def build_route_cost(terrain, osm, settlement_weight=2.0, model_cost=None,
                     ridge_penalty_weight=0.10,
                     landcover_penalty_weight=0.0):
    """Build an interpretable screening cost with a soft built-up proxy.

    The proxy is the maximum of normalized mapped-building density, mapped
    residential coverage, and land-use code 4 (residential/retail/commercial/
    urban). Taking the maximum combines overlapping OSM signals without
    double-counting them. Missing bands contribute no proxy; their availability
    must be read from OSM layer metadata. A nonzero land-cover weight adds a
    separate soft cost to mapped forest and wetland codes; missing land use
    remains neutral. Neither proxy is a hard exclusion or a substitute for
    legal environmental data.
    """
    slope = np.nan_to_num(terrain["slope"], nan=0.0, posinf=90.0, neginf=0.0)
    roughness = np.nan_to_num(terrain["roughness_27"], nan=0.0, posinf=0.0, neginf=0.0)
    terrain_cost = 0.75 * np.clip(slope / 45.0, 0.0, 1.0) ** 1.5
    terrain_cost += 0.25 * np.clip(roughness / 100.0, 0.0, 1.0)
    # Positive TPI marks terrain above its local 300 m neighborhood; a weak
    # penalty prefers lower saddles over exposed crests without using route data.
    tpi_300 = np.broadcast_to(np.asarray(
        terrain.get("tpi_300", np.zeros_like(terrain_cost)), dtype=np.float32),
        terrain_cost.shape)
    ridge_proxy = np.clip(
        np.nan_to_num(tpi_300, nan=0.0, posinf=200.0, neginf=0.0) / 200.0,
        0.0, 1.0,
    )
    terrain_cost += max(float(ridge_penalty_weight), 0.0) * ridge_proxy

    density = np.broadcast_to(np.asarray(
        osm.get("building_density", np.zeros_like(terrain_cost)), dtype=np.float32),
        terrain_cost.shape)
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    positive = density[density > 0]
    if positive.size:
        p99 = max(float(np.percentile(positive, 99)), 1.0)
        building_proxy = np.clip(np.log1p(density) / np.log1p(p99), 0.0, 1.0)
    else:
        building_proxy = np.zeros_like(terrain_cost)
    residential = np.broadcast_to(np.asarray(
        osm.get("residential_landuse_mask", np.zeros_like(terrain_cost)), dtype=np.float32),
        terrain_cost.shape)
    residential_proxy = np.clip(np.nan_to_num(residential, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    landuse_code = np.broadcast_to(np.asarray(
        osm.get("landuse_code", np.zeros_like(terrain_cost)), dtype=np.float32),
        terrain_cost.shape)
    landuse_code = np.nan_to_num(landuse_code, nan=0.0, posinf=0.0, neginf=0.0)
    builtup_proxy = (landuse_code == 4.0).astype(np.float32)
    settlement_proxy = np.maximum(np.maximum(building_proxy, residential_proxy), builtup_proxy)
    landcover_proxy = np.isin(landuse_code, (1.0, 7.0)).astype(np.float32)

    if model_cost is not None:
        model = np.nan_to_num(model_cost, nan=0.0, posinf=1.0, neginf=0.0)
        terrain_cost = 0.5 * terrain_cost + 0.5 * np.clip(model, 0.0, 1.0)
    return (
        terrain_cost
        + max(float(settlement_weight), 0.0) * settlement_proxy
        + max(float(landcover_penalty_weight), 0.0) * landcover_proxy
    ).astype(np.float32)
