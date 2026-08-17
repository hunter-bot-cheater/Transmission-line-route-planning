# -*- coding: utf-8 -*-
"""
独立审计工具 (Auditor agent 的核查引擎)
=========================================
不信任 runner 的 real_comparison_metrics.json, 而是从最原始产物重新计算误差:
  - 规划路径: <run_dir>/<case>/paths.json  (runner 直接落盘的 [lon,lat] 坐标, 未经二次处理)
  - 真实路径: <real_dir>/<safe_name>_real_path.shp  (地面真值 SHP)
独立重算:
  len_err% = |L规划 - L真实| / L真实 * 100
  haus%    = (规划点到最近真实线最大距 km) / L真实 * 100    <-- 与 runner 的单边 Hausdorff 定义一致
  worst     = max(len_err%, haus%)
  pass8     = worst <= 8.0
并把独立结果与 runner 的声明值对比, 标出差异(>1.0pct 视为可疑)。
同时判定: 最终交付路线(最优集成)是否真 <=8%, 以及该最终路线是不是 AI 方法(VIN-Grad/PPO)还是退化成了 A*。
"""
import argparse, json, math, sys, os
from pathlib import Path

import numpy as np

# 复用 runner 的只读加载函数 (仅文件读取, 不引入其误差计算逻辑)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_path_planning_v4 import load_real_path, haversine_km, path_length_km, real_length_km

ERROR_TOL = 8.0
R_EARTH = 6371.0


def safe_name(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def point_to_polyline_km(lon, lat, polylines):
    """点到折线集合(多条折线)的最小 haversine 距离 km。polylines: [[(lon,lat),...],...]"""
    best = 1e9
    for poly in polylines:
        for i in range(len(poly) - 1):
            lon0, lat0 = poly[i]
            lon1, lat1 = poly[i + 1]
            # 线段 densify, 步长约 30m
            d_m = math.hypot((lon1 - lon0) * 111000 * math.cos(math.radians((lat0 + lat1) / 2)),
                             (lat1 - lat0) * 111000)
            n = max(int(d_m / 30.0), 1)
            for k in range(n + 1):
                t = k / n
                plon = lon0 + (lon1 - lon0) * t
                plat = lat0 + (lat1 - lat0) * t
                d = haversine_km(lon, lat, plon, plat)
                if d < best:
                    best = d
    return best


def audit_case(case_name, run_dir, real_dir):
    # 1) 真值
    real_shp = Path(real_dir) / f"{case_name}_real_path.shp"
    if not real_shp.exists():
        real_shp = Path(real_dir) / f"{safe_name(case_name)}_real_path.shp"
    real_geo = load_real_path(str(real_shp)) if real_shp.exists() else None
    if not real_geo:
        return {"case": case_name, "error": "no ground-truth shp", "methods": {}}
    real_len = real_length_km(real_geo)

    # 2) 规划路径
    case_dir = Path(run_dir) / case_name
    if not case_dir.exists():
        case_dir = Path(run_dir) / safe_name(case_name)
    pj = case_dir / "paths.json"
    if not pj.exists():
        return {"case": case_name, "error": "no paths.json", "methods": {}}
    paths = json.load(open(pj, encoding="utf-8"))

    methods = {}
    for m, pts in paths.items():
        if not pts or len(pts) < 2:
            methods[m] = {"plan_len_km": 0.0, "worst_err_pct": None, "pass8": False,
                          "note": "empty path"}
            continue
        plan_len = path_length_km([(p[0], p[1]) for p in pts])
        # 对称 Hausdorff: 正向(规划点->最近真实线) + 反向(真实点->最近规划线)
        planned_poly = [[(p[0], p[1]) for p in pts]]
        real_polys = real_geo if isinstance(real_geo[0][0], (list, tuple)) else [real_geo]
        max_fwd = 0.0
        for p in pts:
            d = point_to_polyline_km(p[0], p[1], real_geo)
            if d > max_fwd:
                max_fwd = d
        max_rev = 0.0
        for poly in real_polys:
            for (lon, lat) in poly:
                d = point_to_polyline_km(lon, lat, planned_poly)
                if d > max_rev:
                    max_rev = d
        max_d = max(max_fwd, max_rev)
        len_err = abs(plan_len - real_len) / real_len * 100.0 if real_len > 0 else 1e9
        haus = (max_d / real_len) * 100.0 if real_len > 0 else 1e9
        worst = max(len_err, haus)
        methods[m] = {
            "plan_len_km": round(plan_len, 3),
            "real_len_km": round(real_len, 3),
            "len_err_pct": round(len_err, 3),
            "haus_pct": round(haus, 3),
            "haus_fwd_pct": round((max_fwd / real_len) * 100.0, 3) if real_len > 0 else 1e9,
            "dev_max_km": round(max_d, 3),
            "worst_err_pct": round(worst, 3),
            "pass8": bool(worst <= ERROR_TOL),
        }

    # 3) 最终路线 = 误差最小者 (复刻 runner 的集成逻辑, 但用独立重算值)
    ranked = sorted(((m, v["worst_err_pct"]) for m, v in methods.items()
                     if v.get("worst_err_pct") is not None), key=lambda x: x[1])
    final_method = ranked[0][0] if ranked else None
    all_ai_pass = all(methods.get(m, {}).get("pass8", False)
                      for m in ("VIN-Grad", "PPO") if m in methods)
    return {
        "case": case_name,
        "real_len_km": round(real_len, 3),
        "methods": methods,
        "final_route_method": final_method,
        "final_route_pass8": methods.get(final_method, {}).get("pass8", False) if final_method else False,
        "ai_methods_all_pass": all_ai_pass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cases-csv", required=True)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--runner-json", default=None, help="runner 的 real_comparison_metrics.json, 用于比对差异")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(args.cases_csv, encoding="utf-8-sig")))
    names = [r["name"].strip() for r in rows]

    results = [audit_case(n, args.run_dir, args.real_dir) for n in names]

    # 比对 runner 声明值
    runner = None
    if args.runner_json and Path(args.runner_json).exists():
        runner = json.load(open(args.runner_json, encoding="utf-8"))
    discrepancies = []
    if runner:
        rmap = {r["case"]: r for r in runner}
        for res in results:
            if "error" in res:
                continue
            rr = rmap.get(res["case"])
            if not rr:
                continue
            for m, v in res["methods"].items():
                if "worst_err_pct" not in v or v["worst_err_pct"] is None:
                    continue
                rmet = rr.get("metrics", {}).get(m)
                if not rmet:
                    continue
                rworst = rmet.get("worst_err_pct")
                if rworst is None:
                    continue
                diff = abs(v["worst_err_pct"] - rworst)
                if diff > 1.0:
                    discrepancies.append({
                        "case": res["case"], "method": m,
                        "audit_worst": v["worst_err_pct"],
                        "runner_worst": rworst, "diff": round(diff, 3),
                    })

    # 汇总
    n_total = len([r for r in results if "error" not in r])
    n_final_pass = len([r for r in results if r.get("final_route_pass8")])
    n_ai_pass = len([r for r in results if r.get("ai_methods_all_pass")])
    summary = {
        "n_cases": n_total,
        "n_final_route_pass8": n_final_pass,
        "n_ai_methods_all_pass": n_ai_pass,
        "final_route_all_pass": n_final_pass == n_total,
        "ai_all_pass": n_ai_pass == n_total,
        "n_discrepancies": len(discrepancies),
        "consensus_ready": (n_final_pass == n_total) and (len(discrepancies) == 0),
    }

    out_obj = {"summary": summary, "discrepancies": discrepancies, "cases": results}
    if args.out:
        json.dump(out_obj, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 打印
    print("=" * 78)
    print(f"AUDIT 独立复核 (run_dir={args.run_dir})")
    print("=" * 78)
    for res in results:
        if "error" in res:
            print(f"  [ERR] {res['case']}: {res['error']}")
            continue
        flag = "OK" if res["final_route_pass8"] else "FAIL"
        ai = "AI-PASS" if res["ai_methods_all_pass"] else "AI-NOT-ALL"
        print(f"\n● {res['case']}  (真实 {res['real_len_km']}km)  [最终:{flag} | {ai}]")
        for m, v in res["methods"].items():
            w = v.get("worst_err_pct")
            if w is None:
                print(f"    {m:18s} {v.get('note','')}")
                continue
            mark = "✓" if v["pass8"] else "✗"
            print(f"    {m:18s} worst={w:6.2f}%  len_err={v['len_err_pct']:6.2f}%  haus={v['haus_pct']:6.2f}%  {mark}")
        print(f"    >> 最终路线方法 = {res['final_route_method']}")
    print("\n" + "-" * 78)
    print(f"SUMMARY: cases={summary['n_cases']} 最终路线全过={summary['final_route_all_pass']} "
          f"AI方法全过={summary['ai_all_pass']} 与runner差异数={summary['n_discrepancies']}")
    if discrepancies:
        print("差异明细(>1pct, 需核查):")
        for d in discrepancies:
            print(f"   {d['case']} {d['method']}: audit={d['audit_worst']} runner={d['runner_worst']} diff={d['diff']}")
    print(f"CONSENSUS_READY = {summary['consensus_ready']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
