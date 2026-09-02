#!/usr/bin/env python3
"""Rank OpenVid-1M zip parts by clips-per-GB WITHOUT downloading them.

Reads each zip's End-Of-Central-Directory + central directory via HTTP Range requests
(~1 MB per part) and reports: zip size, #mp4, median MB/clip. Writes JSON to --out.
"""
import json, struct, sys, urllib.request, concurrent.futures as cf, statistics, argparse

REPO = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/"

def _get(url, rng):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng}", "User-Agent": "probe"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read(), r.headers

def probe(part):
    url = REPO + f"OpenVid_part{part}.zip"
    try:
        tail, h = _get(url, "-65536")
        total = int(h["Content-Range"].split("/")[1])
        i = tail.rfind(b"PK\x05\x06")
        if i < 0: return {"part": part, "error": "no EOCD"}
        n, cd_size, cd_off = struct.unpack("<HII", tail[i+10:i+20])
        if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:   # zip64
            j = tail.rfind(b"PK\x06\x06")
            n, = struct.unpack("<Q", tail[j+32:j+40]); cd_size, cd_off = struct.unpack("<QQ", tail[j+40:j+56])
        cd, _ = _get(url, f"{cd_off}-{cd_off+cd_size-1}")
        sizes = []; p = 0
        while p + 46 <= len(cd) and cd[p:p+4] == b"PK\x01\x02":
            usz, = struct.unpack("<I", cd[p+24:p+28]); nlen, xlen, clen = struct.unpack("<HHH", cd[p+28:p+34])
            name = cd[p+46:p+46+nlen]
            if usz == 0xFFFFFFFF:   # zip64 extra
                q = p+46+nlen; end = q+xlen
                while q + 4 <= end:
                    hid, hsz = struct.unpack("<HH", cd[q:q+4])
                    if hid == 1: usz, = struct.unpack("<Q", cd[q+4:q+12]); break
                    q += 4 + hsz
            if name.lower().endswith(b".mp4"): sizes.append(usz)
            p += 46 + nlen + xlen + clen
        return {"part": part, "zip_gb": round(total/1e9, 1), "n_mp4": len(sizes),
                "median_mb": round(statistics.median(sizes)/1e6, 2) if sizes else None,
                "clips_per_gb": round(len(sizes)/(total/1e9), 1)}
    except Exception as e:
        return {"part": part, "error": f"{type(e).__name__}: {e}"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--parts", default="0-185"); ap.add_argument("--out", required=True); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(); lo, hi = map(int, a.parts.split("-"))
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(probe, range(lo, hi+1)))
    res = [r for r in res if "error" not in r or "404" not in r["error"]]
    json.dump(res, open(a.out, "w"), indent=1)
    ok = [r for r in res if "n_mp4" in r]
    ok.sort(key=lambda r: -r["clips_per_gb"])
    print(f"probed {len(ok)} parts, {sum(r['n_mp4'] for r in ok)} clips, {sum(r['zip_gb'] for r in ok):.0f} GB total")
    for r in ok[:25]: print(r)
    print("PROBE_DONE")
