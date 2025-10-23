#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgrade dataset CSV by adding brand_raw, brand, timestamp (additive, idempotent).
Input  CSV columns: id,stem,label,url_text,html_path,img_path,domain,source,split
Output CSV columns: ... + brand_raw, brand, timestamp
"""

import csv
import os
import re
import sys
import datetime
from html import unescape

try:
    from bs4 import BeautifulSoup  # optional
except Exception:
    BeautifulSoup = None

DATE_PAT = re.compile(
    r"(?P<y>20[1-4]\d)[/\-]?(?P<m>0[1-9]|1[0-2])[/\-]?(?P<d>0[1-9]|[12]\d|3[01])"
)

GENERIC_WORDS = {
    "login",
    "sign in",
    "signin",
    "account",
    "home",
    "control-center",
    "kundenshop",
    "webmail",
    "help",
    "hilfe",
    "service",
    "download",
    "portal",
    "dashboard",
    "center",
    "centre",
    "welcome",
    "main",
    "index",
    "default",
    "error",
    "404",
    "500",
}
BLOCK_HOSTS = {
    "blogspot",
    "wordpress",
    "github",
    "gitlab",
    "bitbucket",
    "wix",
    "weebly",
    "tumblr",
    "medium",
    "shopify",
    "s3",
    "cloudfront",
    "googleusercontent",
    "firebaseapp",
    "azurewebsites",
    "herokuapp",
    "pages",
    "netlify",
    "vercel",
    "surge",
    "firebase",
    "heroku",
    "aws",
    "azure",
    "gcp",
}
AMP = re.compile(r"\s*&\s*", re.I)
NONAN = re.compile(r"[^a-z0-9]+")


def norm_brand(s):
    if not s:
        return ""
    s = unescape(s).strip().lower()
    s = AMP.sub(" and ", s)
    s = NONAN.sub("", s)
    return s


def pick_from_html(html_text):
    if not html_text:
        return ""
    title = ""
    site = ""
    h1 = ""
    if BeautifulSoup:
        soup = BeautifulSoup(html_text, "lxml") if html_text else None
        if soup:
            if soup.title and soup.title.string:
                title = soup.title.string
            # meta site_name
            meta = soup.find("meta", attrs={"property": "og:site_name"}) or soup.find(
                "meta", attrs={"name": "application-name"}
            )
            if meta and meta.get("content"):
                site = meta["content"]
            h1tag = soup.find("h1")
            if h1tag:
                h1 = h1tag.get_text(" ", strip=True)
    else:
        # crude regex fallback for <title>...</title>
        m = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
        title = m.group(1) if m else ""
    candidates = [title, site, h1]

    def clean(x):
        x = (x or "").strip()
        x = re.sub(r"\s+", " ", x)
        return x

    for c in map(clean, candidates):
        cl = c.lower()
        if not c:
            continue
        if any(w in cl for w in GENERIC_WORDS):  # skip generic UI words
            continue
        # take first reasonably branded-looking token phrase
        return c
    return ""


def pick_from_domain(domain):
    if not domain:
        return ""
    host = domain.split(":")[0]
    host = host.lower()
    parts = re.split(r"[\.\-]+", host)
    # remove tld and common blocks
    parts = [p for p in parts if p and p not in BLOCK_HOSTS]
    if len(parts) >= 2:
        core = parts[-2]  # sld
    else:
        core = parts[0]
    return core


def parse_url_for_date(url):
    if not url:
        return ""
    m = DATE_PAT.search(url)
    if not m:
        return ""
    y, mn, d = m.group("y", "m", "d")
    try:
        dt = datetime.datetime(
            int(y), int(mn), int(d), 0, 0, 0, tzinfo=datetime.timezone.utc
        )
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def file_mtime_iso(path):
    try:
        ts = os.path.getmtime(path)
        dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


def choose_brand_raw(html_text, url_text, domain):
    # 1) from HTML
    b_html = pick_from_html(html_text)
    if b_html:
        return b_html
    # 2) from domain/url
    b_dom = pick_from_domain(domain) if domain else ""
    if not b_dom and url_text:
        try:
            # host from URL
            host = re.sub(r"^https?://", "", url_text, flags=re.I).split("/")[0]
            b_dom = pick_from_domain(host)
        except Exception:
            pass
    if b_dom:
        return b_dom
    # 3) fallback empty
    return ""


def choose_timestamp(url_text, html_path, img_path):
    # prefer file mtime (image first)
    for p in [img_path, html_path]:
        iso = file_mtime_iso(p)
        if iso:
            return iso
    # then a date pattern in URL
    iso = parse_url_for_date(url_text)
    if iso:
        return iso
    # else empty (let temporal split downgrade)
    return ""


def main(in_csv):
    out_csv = os.path.splitext(in_csv)[0] + "_v2.csv"
    n_total = 0
    n_brand = 0
    n_time = 0
    with open(in_csv, newline="", encoding="utf-8") as f, open(
        out_csv, "w", newline="", encoding="utf-8"
    ) as g:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames.copy()
        for col in ["brand_raw", "brand", "timestamp"]:
            if col not in fieldnames:
                fieldnames.append(col)
        w = csv.DictWriter(g, fieldnames=fieldnames)
        w.writeheader()
        for row in r:
            n_total += 1
            # read HTML text (best effort)
            html_text = ""
            hp = row.get("html_path", "")
            if hp and os.path.exists(hp):
                try:
                    with open(hp, "r", encoding="utf-8", errors="ignore") as hf:
                        html_text = hf.read(200000)  # cap read
                except Exception:
                    pass
            brand_raw = row.get("brand_raw", "") or choose_brand_raw(
                html_text=html_text,
                url_text=row.get("url_text", ""),
                domain=row.get("domain", ""),
            )
            brand = row.get("brand", "") or norm_brand(brand_raw)
            ts = row.get("timestamp", "") or choose_timestamp(
                url_text=row.get("url_text", ""),
                html_path=hp,
                img_path=row.get("img_path", ""),
            )
            if brand_raw:
                n_brand += 1
            if ts:
                n_time += 1
            row["brand_raw"] = brand_raw
            row["brand"] = brand
            row["timestamp"] = ts
            w.writerow(row)
    print(f"[DONE] wrote: {out_csv}")
    print(f"rows: {n_total}, brand_raw filled: {n_brand}, timestamp filled: {n_time}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python upgrade_dataset.py <input.csv>")
        sys.exit(1)
    main(sys.argv[1])
