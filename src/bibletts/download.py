"""Downloading source files and matching audio files to book and chapter."""

from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from tqdm.auto import tqdm

from .books import CHAPTERS, book_code

AUDIO_EXTENSIONS = (".mp3", ".ogg", ".opus", ".m4a", ".wav", ".flac")


def download(url: str, dest: str | Path, retries: int = 4) -> Path:
    """Download url to dest, skipping if dest already exists. Retries with backoff."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                                desc=dest.name, leave=False) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            return dest
        except (requests.RequestException, OSError):
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    return dest


def extract_zip(path: str | Path, dest: str | Path,
                members: list[str] | None = None) -> list[Path]:
    """Extract all (or the named) members of a zip. Returns extracted paths."""
    dest = Path(dest)
    with zipfile.ZipFile(path) as z:
        names = members if members is not None else z.namelist()
        for name in names:
            if not (dest / name).exists():
                z.extract(name, dest)
    return [dest / n for n in names if not n.endswith("/")]


def zip_listing(path: str | Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        return [n for n in z.namelist() if not n.endswith("/")]


def find_links(page_url: str, extensions: tuple[str, ...] = AUDIO_EXTENSIONS + (".zip",),
               follow_pages: bool = True) -> list[str]:
    """Every link on a web page ending in one of `extensions`.

    With follow_pages, also looks one level deeper into linked .htm/.html pages
    and folder listings on the same site, because audio sites often have one
    page per book.
    """
    def links_on(url: str) -> list[str]:
        html = requests.get(url, timeout=60).text
        hrefs = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html, flags=re.I)
        return [urljoin(url, h) for h in hrefs]

    found: list[str] = []
    pages = [page_url]
    if follow_pages:
        site = urlparse(page_url).netloc
        for link in links_on(page_url):
            path = urlparse(link).path
            same_site = urlparse(link).netloc == site
            is_page = path.endswith((".htm", ".html")) or path.endswith("/")
            if same_site and is_page and link.startswith(page_url.rsplit("/", 1)[0]):
                pages.append(link)
    for page in dict.fromkeys(pages):
        try:
            for link in links_on(page):
                if urlparse(link).path.lower().endswith(extensions):
                    found.append(link)
        except requests.RequestException:
            continue
    return list(dict.fromkeys(found))


def guess_book_chapter(name: str, regex: str | None = None) -> tuple[str, int] | None:
    """Work out (book code, chapter) from an audio file name or URL.

    Tries, in order:
    1. `regex`, if given, with named groups "book" and "chapter".
    2. A book code or book name followed later by a number: "MAT_05",
       "Matthew 5", "1Corinthians_13".
    3. Two numbers, book number (1-66) then chapter: "40_005".
    Returns None if nothing matches. Always check the result by eye.
    """
    stem = Path(unquote(urlparse(name).path or name)).stem
    if regex:
        m = re.search(regex, stem)
        if m:
            code = book_code(m.group("book"))
            if code:
                return code, int(m.group("chapter"))
        return None

    # Strategy 2: split into alternating word / number pieces and look for a
    # book name made of one or two consecutive pieces ("1", "Corinthians").
    pieces = re.findall(r"[A-Za-z]+|\d+", stem)
    for i in range(len(pieces)):
        for width in (3, 2, 1):
            cand = "".join(pieces[i : i + width])
            if cand.isdigit():
                continue
            code = book_code(cand)
            if not code:
                continue
            numbers = [int(p) for p in pieces[i + width :] if p.isdigit()]
            for n in numbers:
                if 1 <= n <= CHAPTERS[code]:
                    return code, n
            if CHAPTERS[code] == 1:
                return code, 1

    # Strategy 3: book number then chapter number.
    numbers = [int(p) for p in pieces if p.isdigit()]
    for a, b in zip(numbers, numbers[1:]):
        code = book_code(str(a))
        if code and 1 <= b <= CHAPTERS[code]:
            return code, b
    return None


def catalogue(names: list[str], regex: str | None = None) -> tuple[dict, list[str]]:
    """Map many audio names to (book, chapter).

    Returns ({(book, chapter): name}, [names that could not be matched]).
    Two files claiming the same chapter is reported as an error, because it
    usually means the guesser read the name wrongly.
    """
    found: dict[tuple[str, int], str] = {}
    unmatched: list[str] = []
    for name in names:
        key = guess_book_chapter(name, regex)
        if key is None:
            unmatched.append(name)
        elif key in found:
            raise ValueError(f"Both {found[key]} and {name} look like {key}. "
                             "Set source.audio_filename_regex in the config.")
        else:
            found[key] = name
    return found, unmatched
