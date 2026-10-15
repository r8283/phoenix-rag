"""
crawl.py — איסוף תוכן ציבורי מ-fnx.co.il (יום 1)

זרימה:
  1. קריאת robots.txt (כיבוד Disallow + Crawl-delay)
  2. איסוף כל ה-<loc> מ-sitemap.xml (כולל sitemaps מקוננים)
  3. סינון INCLUDE/EXCLUDE + seed_urls.txt ידני
  4. מיון לפי עדיפות (תביעות/דירה קודם)
  5. הורדת HTML, חילוץ טקסט עם trafilatura, שמירה ל-data/pages.jsonl
  6. דילוג על URL שכבר ב-pages.jsonl (המשך מאיפה שנעצר)

שימוש:
  python crawl.py --discover          # דוח robots + sitemap בלבד
  python crawl.py --limit 10          # 10 עמודים לבדיקה (לפני סריקה מלאה)
  python crawl.py                     # עד MAX_PAGES (600)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import httpx
import trafilatura

import config

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

# קודי HTTP שמרמזים על חסימה זמנית / האטה - מנסים שוב
RETRY_STATUS = (418, 429, 503)
RETRY_WAIT_SEC = 10
RETRY_ATTEMPTS = 2


def _log(msg: str) -> None:
    """הדפסה בטוחה גם כשהטרמינל לא תומך בעברית."""
    try:
        print(msg)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((msg + "\n").encode("utf-8", errors="replace"))


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        follow_redirects=True,
        timeout=60.0,
        verify=config.HTTP_SSL_VERIFY,
    )


def normalize_url(url: str) -> str | None:
    """מנקה fragment ומוודא שהדומיין מותר."""
    url = urldefrag(url.strip())[0]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.netloc not in config.ALLOWED_HOSTS:
        return None
    path = parsed.path or "/"
    return f"https://{parsed.netloc}{path}" + (f"?{parsed.query}" if parsed.query else "")


def url_passes_filters(url: str) -> bool:
    """INCLUDE: חייב להתאים (אם הרשימה לא ריקה). EXCLUDE: אסור להתאים."""
    lower = url.lower()
    haystack = f"{lower} {url}"
    for ex in config.URL_EXCLUDE_KEYWORDS:
        if ex.lower() in haystack:
            return False
    if config.URL_INCLUDE_KEYWORDS:
        if not any(kw in haystack for kw in config.URL_INCLUDE_KEYWORDS):
            return False
    return True


def load_manual_seeds() -> list[str]:
    """קורא data/seed_urls.txt — רק השלמה, לא מקור ראשי."""
    if not config.SEED_URLS_PATH.is_file():
        return []
    urls: list[str] = []
    for line in config.SEED_URLS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        u = normalize_url(line)
        if u and url_passes_filters(u):
            urls.append(u)
    return urls


def _parse_sitemap_locs(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """מחזיר (רשימת sitemaps מקוננים, רשימת כתובות עמודים)."""
    child_sitemaps: list[str] = []
    page_urls: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return child_sitemaps, page_urls

    root_tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag

    for elem in root.iter():
        if not elem.tag.endswith("loc") or not elem.text:
            continue
        loc = elem.text.strip()
        if root_tag == "sitemapindex":
            if loc.endswith(".xml") or "sitemap" in loc.lower():
                child_sitemaps.append(loc)
            else:
                page_urls.append(loc)
        elif root_tag == "urlset":
            page_urls.append(loc)
        else:
            if loc.endswith(".xml"):
                child_sitemaps.append(loc)
            else:
                page_urls.append(loc)

    return child_sitemaps, page_urls


def fetch_all_sitemap_urls(client: httpx.Client) -> list[str]:
    """מוריד sitemap ראשי וכל sitemap מקונן, מחזיר כתובות עמודים."""
    to_visit = [
        f"{config.BASE_URL}/sitemap.xml",
        f"{config.BASE_URL}/sitemap_index.xml",
    ]
    seen_sitemaps: set[str] = set()
    all_pages: set[str] = set()

    while to_visit:
        sm_url = to_visit.pop()
        if sm_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sm_url)
        try:
            r = client.get(sm_url)
            if r.status_code != 200:
                continue
            child_sms, pages = _parse_sitemap_locs(r.content)
            for c in child_sms:
                c_norm = c if c.startswith("http") else f"{config.BASE_URL.rstrip('/')}/{c.lstrip('/')}"
                if c_norm not in seen_sitemaps:
                    to_visit.append(c_norm)
            for p in pages:
                u = normalize_url(p)
                if u:
                    all_pages.add(u)
        except httpx.HTTPError as e:
            _log(f"  sitemap skip {sm_url}: {e}")

    return sorted(all_pages)


def load_robots(client: httpx.Client) -> RobotFileParser:
    """טוען robots.txt לבדיקת can_fetch."""
    rp = RobotFileParser()
    robots_url = f"{config.BASE_URL}/robots.txt"
    try:
        r = client.get(robots_url)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            rp.allow_all = True
    except httpx.HTTPError:
        rp.allow_all = True
    rp.set_url(robots_url)
    return rp


def robots_delay_sec(rp: RobotFileParser) -> float:
    """Crawl-delay מ-robots (אם קיים) או DELAY_SEC מה-config, הגדול מביניהם."""
    delay = config.DELAY_SEC
    try:
        cd = rp.crawl_delay(config.USER_AGENT)
        if cd is not None and cd > delay:
            delay = float(cd)
    except (AttributeError, TypeError):
        pass
    return delay


def categorize_urls(urls: list[str]) -> dict[str, int]:
    """ספירה גסה של אזורים לפי מילות מפתח בנתיב (לדוח discover)."""
    buckets = {
        "תביעות/נזק": ("תביע", "claim", "נזק", "מים"),
        "שאלות/FAQ": ("שאל", "faq", "q-and-a", "questions"),
        "מוצרים/ביטוח": ("ביטוח", "מוצר", "כיסוי", "פוליס"),
        "פנסיה/גמל": ("פנס", "גמל", "השתלמות"),
        "בריאות": ("בריאות", "health"),
        "רכב": ("רכב", "car", "auto"),
        "דירה/משכנתא": ("דירה", "משכנת", "home"),
        "שירות/לקוחות": ("שירות", "לקוח", "צור-קשר", "contact"),
        "אחר": (),
    }
    counts: Counter[str] = Counter()
    for url in urls:
        hay = url.lower()
        matched = False
        for name, kws in buckets.items():
            if name == "אחר":
                continue
            if any(k in hay or k in url for k in kws):
                counts[name] += 1
                matched = True
                break
        if not matched:
            counts["אחר"] += 1
    return dict(counts)


def url_priority(url: str) -> int:
    """דרגה נמוכה = נסרק קודם. לפי URL_PRIORITY_PATTERNS."""
    lower = url.lower()
    for rank, patterns in config.URL_PRIORITY_PATTERNS:
        if any(p in lower or p in url for p in patterns):
            return rank
    return config.URL_PRIORITY_DEFAULT


def _filtered_urls(sitemap_urls: list[str]) -> list[str]:
    return [
        u for u in sitemap_urls
        if url_passes_filters(u)
        and not any(x in u.lower() for x in config.URL_EXCLUDE_KEYWORDS_EXTRA)
    ]


def discover_report() -> dict:
    """דוח robots + sitemap."""
    with _client() as client:
        raw_urls = fetch_all_sitemap_urls(client)
        filtered = _filtered_urls(raw_urls)
        seeds = load_manual_seeds()
        merged = sorted(set(filtered) | set(seeds))

        robots_text = ""
        try:
            r = client.get(f"{config.BASE_URL}/robots.txt")
            robots_text = r.text[:2000] if r.status_code == 200 else "(לא נטען)"
        except httpx.HTTPError:
            robots_text = "(שגיאת רשת)"

    return {
        "robots_preview": robots_text,
        "sitemap_total_raw": len(raw_urls),
        "after_include_exclude": len(filtered),
        "manual_seeds": len(seeds),
        "merged_unique": len(merged),
        "areas": categorize_urls(filtered),
        "sample_urls": sorted(filtered, key=lambda u: (url_priority(u), u))[:15],
    }


def load_crawled_urls() -> set[str]:
    """URL שכבר נשמרו — להמשך מאיפה שנעצר."""
    done: set[str] = set()
    if not config.PAGES_PATH.is_file():
        return done
    with config.PAGES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("url"):
                    done.add(row["url"])
            except json.JSONDecodeError:
                continue
    return done


def extract_page(html: str, url: str) -> tuple[str, str]:
    """
    trafilatura מוציא רק תוכן מרכזי (בלי תפריט/פוטר).
    title מ-metadata או מ-<title>.
    """
    meta = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        output_format="json",
        with_metadata=True,
    )
    title = url
    text = ""
    if meta:
        try:
            data = json.loads(meta)
            title = (data.get("title") or title).strip()
            text = (data.get("text") or "").strip()
        except json.JSONDecodeError:
            text = trafilatura.extract(html, url=url) or ""
    else:
        text = trafilatura.extract(html, url=url) or ""
    if not title or title == url:
        m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I | re.S)
        if m:
            title = m.group(1).strip()
    return title, text


def append_page(record: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with config.PAGES_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_url_queue() -> list[str]:
    """תור סריקה: מסונן, ממוין לפי עדיפות (תביעות/דירה קודם), חתוך ל-MAX_PAGES."""
    with _client() as client:
        sitemap_urls = fetch_all_sitemap_urls(client)
    filtered = _filtered_urls(sitemap_urls)
    seeds = load_manual_seeds()
    queue = sorted(set(filtered) | set(seeds), key=lambda u: (url_priority(u), u))
    return queue[: config.MAX_PAGES]


def fetch_with_retry(client: httpx.Client, url: str) -> httpx.Response | None:
    """GET עם ניסיון חוזר על 418/429/503. מחזיר None אם נכשל ברשת."""
    try:
        r = client.get(url)
    except httpx.HTTPError as e:
        _log(f"  network error: {e}")
        return None

    if r.status_code in RETRY_STATUS:
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            _log(f"  HTTP {r.status_code}, waiting {RETRY_WAIT_SEC}s (retry {attempt}/{RETRY_ATTEMPTS})")
            time.sleep(RETRY_WAIT_SEC)
            try:
                r = client.get(url)
            except httpx.HTTPError as e:
                _log(f"  network error on retry: {e}")
                return None
            if r.status_code < 400:
                break
    return r


def run_crawl(limit: int | None = None) -> None:
    queue = build_url_queue()
    done = load_crawled_urls()
    pending = [u for u in queue if u not in done]

    if limit is not None:
        pending = pending[:limit]

    _log(f"Queue: {len(queue)} URLs (max {config.MAX_PAGES}), already crawled: {len(done)}, to fetch: {len(pending)}")

    if not pending:
        _log("Nothing to crawl.")
        return

    empty_streak = 0
    success = 0
    skipped_short = 0
    blocked_robots = 0
    failed_http = 0

    with _client() as client:
        rp = load_robots(client)
        delay = robots_delay_sec(rp)

        for i, url in enumerate(pending, start=1):
            if not rp.can_fetch(config.USER_AGENT, url):
                blocked_robots += 1
                _log(f"[{i}/{len(pending)}] robots disallow: {url}")
                continue

            _log(f"[{i}/{len(pending)}] GET {url}")
            r = fetch_with_retry(client, url)
            if r is None:
                failed_http += 1
                time.sleep(delay)
                continue

            if r.status_code >= 400:
                failed_http += 1
                _log(f"  HTTP {r.status_code} - skipping")
                time.sleep(delay)
                continue

            ctype = r.headers.get("content-type", "")
            if "html" not in ctype.lower() and "<html" not in r.text[:800].lower():
                _log("  not HTML")
                time.sleep(delay)
                continue

            title, text = extract_page(r.text, url)
            text_len = len(text)

            if text_len < config.MIN_PAGE_TEXT_LEN:
                skipped_short += 1
                empty_streak += 1
                _log(f"  skip short/empty ({text_len} chars) — possible JS-only page")
                if limit is not None and empty_streak >= min(5, len(pending)):
                    _log(
                        "\n*** STOP: many pages returned almost no text. "
                        "Site may require JavaScript. ***"
                    )
                    sys.exit(2)
                time.sleep(delay)
                continue

            empty_streak = 0
            record = {
                "url": url,
                "title": title,
                "text": text,
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            }
            append_page(record)
            success += 1
            _log(f"  ok ({text_len} chars)")

            if i < len(pending):
                time.sleep(delay)

    _log(
        f"Done. saved={success}, skipped_short={skipped_short}, "
        f"robots_blocked={blocked_robots}, http_failed={failed_http}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl fnx.co.il for RAG (day 1)")
    parser.add_argument("--discover", action="store_true", help="Report robots + sitemap only")
    parser.add_argument("--limit", type=int, default=None, help="Max new pages to crawl (e.g. 10 for test)")
    args = parser.parse_args()

    if args.discover:
        report = discover_report()
        _log("=== robots.txt (preview) ===")
        _log(report["robots_preview"])
        _log("")
        _log(f"Sitemap page URLs (raw): {report['sitemap_total_raw']}")
        _log(f"After INCLUDE/EXCLUDE filters: {report['after_include_exclude']}")
        _log(f"Manual seeds: {report['manual_seeds']}")
        _log(f"Merged unique (cap {config.MAX_PAGES} at crawl time): {report['merged_unique']}")
        _log("")
        _log("Areas (keyword buckets on filtered URLs):")
        for area, count in sorted(report["areas"].items(), key=lambda x: -x[1]):
            _log(f"  {area}: {count}")
        _log("")
        _log("Sample URLs (by priority):")
        for u in report["sample_urls"]:
            _log(f"  {u}")
        return

    run_crawl(limit=args.limit)


if __name__ == "__main__":
    main()