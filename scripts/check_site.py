#!/usr/bin/env python3
"""Check the static project site in site/ before it is published to GitHub Pages.

Stdlib only, read-only, a second or two. Exit status 1 lists every problem:

1. every local href/src (and the og:image under the site URL) names a file in site/;
2. every in-page #anchor names an element id, and ids are unique;
3. nothing the page loads at runtime (scripts, stylesheets, fonts, icons, images) comes
   from another origin: the site needs no CDN or tracker;
4. every link into this repository (github.com/johnnynunez/cascade/blob|tree/main/<path>)
   names a path that exists in the checkout, so a moved doc fails here and not on the web;
5. every <img> has an alt attribute, and the site stays under the size budget.
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "site"
SITE_URL = "https://johnnynunez.github.io/cascade/"
REPO_URL = re.compile(r"^https://github\.com/johnnynunez/cascade/(?:blob|tree)/main/(?P<path>[^#?]+)(?:#(?P<frag>.+))?")
BUDGET_BYTES = 20 * 1024 * 1024
RUNTIME_ATTRS = {("script", "src"), ("link", "href"), ("img", "src"), ("source", "srcset"),
                 ("img", "srcset"), ("video", "src"), ("audio", "src"), ("iframe", "src")}


class _Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.refs: list[tuple[str, str, str]] = []  # (tag, attr, value)
        self.imgs_without_alt: list[str] = []
        self.inline_css: list[str] = []
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.ids.append(d["id"] or "")
        for attr in ("href", "src", "srcset", "content"):
            value = d.get(attr)
            if value is None:
                continue
            if attr == "content" and d.get("property") not in ("og:image", "og:url") \
                    and d.get("name") != "twitter:image":
                continue
            self.refs.append((tag, attr, value))
        if tag == "img" and "alt" not in d:
            self.imgs_without_alt.append(d.get("src") or "?")
        if tag == "style":
            self._in_style = True
        style = d.get("style")
        if style:
            self.inline_css.append(style)

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False

    def handle_data(self, data):
        if self._in_style:
            self.inline_css.append(data)


def _css_urls(text: str) -> list[str]:
    return [u.strip("'\" ") for u in re.findall(r"url\(([^)]+)\)", text)]


def check(site: Path = SITE, repo: Path = REPO) -> list[str]:
    problems: list[str] = []
    pages = sorted(site.rglob("*.html"))
    if not pages:
        return [f"no HTML page under {site}"]
    for page in pages:
        rel_page = page.relative_to(site)
        parser = _Page()
        parser.feed(page.read_text(encoding="utf-8"))
        ids = parser.ids
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            problems.append(f"{rel_page}: duplicate ids {dupes}")
        for src in parser.imgs_without_alt:
            problems.append(f"{rel_page}: <img src={src!r}> has no alt attribute")
        values = [(t, a, v) for t, a, v in parser.refs]
        values += [("style", "url", u) for css in parser.inline_css for u in _css_urls(css)]
        for tag, attr, value in values:
            for url in ([p.strip().split(" ")[0] for p in value.split(",")] if attr == "srcset" else [value]):
                problems += _check_ref(page, site, repo, tag, attr, url, set(ids))
    for css in sorted(site.rglob("*.css")):
        for url in _css_urls(css.read_text(encoding="utf-8")):
            if url.startswith("data:"):
                continue
            if urlparse(url).scheme:
                problems.append(f"{css.relative_to(site)}: external url({url}) in CSS")
            elif not (css.parent / unquote(url)).resolve().is_file():
                problems.append(f"{css.relative_to(site)}: url({url}) does not exist")
    size = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    if size > BUDGET_BYTES:
        problems.append(f"site/ is {size / 1e6:.1f} MB, over the {BUDGET_BYTES / 1e6:.0f} MB budget")
    return problems


def _check_ref(page: Path, site: Path, repo: Path, tag: str, attr: str, url: str, ids: set[str]) -> list[str]:
    rel_page = page.relative_to(site)
    if not url or url.startswith(("mailto:", "data:", "javascript:")):
        return []
    if url.startswith("#"):
        return [] if url == "#" or url[1:] in ids else [f"{rel_page}: anchor {url} has no target id"]
    if url.startswith(SITE_URL):
        local = url[len(SITE_URL):].split("#")[0] or "index.html"
        if not (site / unquote(local)).is_file():
            return [f"{rel_page}: {url} is not a file in site/"]
        return []
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        if (tag, attr) in RUNTIME_ATTRS and not (tag == "link" and _is_navigation_link(url)):
            return [f"{rel_page}: <{tag} {attr}> loads {url} from another origin"]
        m = REPO_URL.match(url)
        if m:
            target = repo / unquote(m.group("path")).rstrip("/")
            if not target.exists():
                return [f"{rel_page}: repo link {url} names a path missing from the checkout"]
            frag = m.group("frag")
            if frag and target.suffix == ".md" and not frag.startswith("L") \
                    and unquote(frag) not in _markdown_anchors(target):
                return [f"{rel_page}: repo link {url}: no heading or anchor #{frag} in {m.group('path')}"]
        return []
    path = unquote(url.split("#")[0].split("?")[0])
    if path and not (page.parent / path).resolve().is_file():
        return [f"{rel_page}: {url} does not exist"]
    return []


def _markdown_anchors(path: Path) -> set[str]:
    """GitHub's heading slugs (lowercase, punctuation dropped, spaces to hyphens, -N for repeats)
    plus explicit <a id> / <a name> anchors."""
    text = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    seen: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        m = None if in_fence else re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        title = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1))
        title = re.sub(r"<[^>]+>", "", title)
        slug = re.sub(r"[^\w\- ]", "", title.strip().lower()).replace(" ", "-")
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def _is_navigation_link(url: str) -> bool:
    return url.startswith(SITE_URL)


def main() -> int:
    problems = check()
    for p in problems:
        print(f"site check: {p}", file=sys.stderr)
    if problems:
        return 1
    print("site check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
