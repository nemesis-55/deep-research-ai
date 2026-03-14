"""
Deep Page Scraper — Deep Research AI
- Full article text via trafilatura + BeautifulSoup fallback
- Post-trafilatura cleanup: strip nav/cookie/footer boilerplate
- Skip pages < 1000 chars, login walls, and product/checkout pages
- Returns structured {url, title, domain, text, images, ...}
- Real images extracted with absolute URLs and captions
- YouTube transcript extraction
- PDF / DOCX / image local file parsing
- Recursive follow-links support
"""
import logging
import re
import tempfile
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict
from urllib.parse import urljoin, urlparse

from backend.constants import SCRAPER_USER_AGENT, SCRAPER_TIMEOUT_S

logger = logging.getLogger(__name__)

# ── Minimum text length for a page to be considered useful ────────────────────
_MIN_USEFUL_CHARS = 1000

# ── Login / paywall / product page signals (checked in raw HTML) ───────────────
_LOGIN_WALL_RE = re.compile(
    r'(sign[-\s]?in to continue|log in to read|subscribe to (access|read|view)|'
    r'create an? (free )?account|access denied|403 forbidden|'
    r'members only|premium content|paywall)',
    re.IGNORECASE,
)
_PRODUCT_PAGE_RE = re.compile(
    r'(add to (cart|bag|basket)|buy now|checkout|'
    r'in stock|out of stock|free shipping|'
    r'product description|customer reviews?\s+\d|\$[\d,]+\.?\d*\s+USD)',
    re.IGNORECASE,
)

# ── Boilerplate patterns to strip after trafilatura extraction ─────────────────
# These appear as literal text lines in trafilatura output on noisy sites.
_BOILERPLATE_RE = re.compile(
    r'^('
    r'accept (all )?cookies?|cookie (policy|preferences|settings|notice|consent)|'
    r'we use cookies|your privacy|privacy (settings|choices)|'
    r'(skip to|jump to) (main )?content|'
    r'(main |top |site |global )?(nav(igation)?|menu|header|footer)|'
    r'breadcrumb|you are here|'
    r'share (this|on|via)|follow us (on|at)|'
    r'subscribe (to our)? (newsletter|updates|alerts)|'
    r'(all )?rights? reserved|copyright \d{4}|'
    r'advertisement|sponsored content|'
    r'loading\.\.\.|please wait|'
    r'back to top|scroll to top'
    r')[\s:.\-]*$',
    re.IGNORECASE,
)

HEADERS = {"User-Agent": SCRAPER_USER_AGENT}

YOUTUBE_DOMAINS = {"youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"}

_SKIP_IMG_KEYWORDS = {
    "pixel", "tracking", "beacon", "icon", "logo", "sprite",
    "blank", "spacer", "avatar", "placeholder", "badge",
    "button", "arrow", "close", "menu", "social",
}


# ── Post-trafilatura text cleanup ─────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Remove boilerplate lines (nav/cookie/footer noise) that trafilatura
    sometimes leaves in extracted text.
    Also collapses excessive blank lines.
    """
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip empty lines (we'll re-add paragraph breaks below)
        if not stripped:
            # Preserve paragraph structure: keep a single blank line
            if cleaned_lines and cleaned_lines[-1] != "":
                cleaned_lines.append("")
            continue
        # Drop boilerplate lines
        if _BOILERPLATE_RE.match(stripped):
            continue
        # Drop very short isolated lines (< 4 words) that are likely UI chrome
        if len(stripped.split()) < 4 and len(stripped) < 30:
            continue
        cleaned_lines.append(line)

    # Collapse runs of blank lines into a single blank line
    result_lines: List[str] = []
    prev_blank = False
    for line in cleaned_lines:
        is_blank = line.strip() == ""
        if is_blank and prev_blank:
            continue
        result_lines.append(line)
        prev_blank = is_blank

    return "\n".join(result_lines).strip()


def _is_login_wall(html: str, text: str) -> bool:
    """Return True if the page appears to be a login wall or paywall."""
    sample = (html[:3000] + " " + text[:1000]).lower()
    return bool(_LOGIN_WALL_RE.search(sample))


def _is_product_page(text: str) -> bool:
    """Return True if the page appears to be an e-commerce product page."""
    sample = text[:2000]
    return bool(_PRODUCT_PAGE_RE.search(sample))


# ── YouTube ───────────────────────────────────────────────────────────────────

def _is_youtube(url: str) -> bool:
    return urlparse(url).netloc in YOUTUBE_DOMAINS


def _get_youtube_transcript(url: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        m = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", url)
        if not m:
            return ""
        vid_id = m.group(1)
        transcript = YouTubeTranscriptApi.get_transcript(vid_id)
        text = " ".join(t["text"] for t in transcript)
        logger.info(f"YouTube transcript: {vid_id} ({len(text)} chars)")
        return f"[YouTube Transcript — {url}]\n{text}"
    except Exception as e:
        logger.debug(f"YouTube transcript failed {url}: {e}")
        return ""


# ── Image extraction ──────────────────────────────────────────────────────────

def _extract_images(html: str, base_url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    images = []
    seen: set = set()

    for img in soup.find_all("img"):
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
            or ""
        )
        if not src:
            continue

        abs_url = urljoin(base_url, src)
        if abs_url in seen:
            continue
        seen.add(abs_url)

        # Skip tracking/icon images
        lower = abs_url.lower()
        if any(kw in lower for kw in _SKIP_IMG_KEYWORDS):
            continue

        # Skip tiny images
        width = img.get("width", "")
        height = img.get("height", "")
        try:
            if int(width) < 100 or int(height) < 100:
                continue
        except (TypeError, ValueError):
            pass

        # Skip non-image extensions
        parsed_path = urlparse(abs_url).path.lower()
        if parsed_path and not any(
            parsed_path.endswith(ext)
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".svg", "")
        ):
            continue

        alt = img.get("alt", "").strip()
        # Caption from nearby figcaption
        caption = ""
        fig = img.find_parent("figure")
        if fig:
            cap = fig.find("figcaption")
            if cap:
                caption = cap.get_text(strip=True)

        images.append({"url": abs_url, "alt": alt, "caption": caption})

    return images[:25]


# ── Follow links ──────────────────────────────────────────────────────────────

def _extract_follow_links(html: str, base_url: str, max_links: int = 3) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_domain = urlparse(base_url).netloc
    links = []
    seen: set = set()
    skip_exts = {".pdf", ".zip", ".exe", ".dmg", ".jpg", ".png", ".gif", ".mp4"}

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc != base_domain:
            continue
        if abs_url in seen:
            continue
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_exts):
            continue
        seen.add(abs_url)
        links.append(abs_url)
        if len(links) >= max_links:
            break

    return links


# ── PDF / DOCX parsing ────────────────────────────────────────────────────────

def _parse_pdf(content: bytes) -> str:
    try:
        import pypdf
        import io
        reader = pypdf.PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages[:30]:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        logger.debug(f"PDF parse failed: {e}")
        return ""


def _parse_docx(content: bytes) -> str:
    try:
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        logger.debug(f"DOCX parse failed: {e}")
        return ""


# ── Local file parsing ────────────────────────────────────────────────────────

def _parse_local_file(file_path: str) -> Dict:
    """Parse local file. Returns {"text": ..., "images": [...]}"""
    path = Path(file_path)
    result: Dict = {"text": "", "images": []}

    if not path.exists():
        logger.warning(f"File not found: {file_path}")
        return result

    if path.is_dir():
        texts = []
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() in (
                ".pdf", ".docx", ".txt", ".md", ".csv", ".rst"
            ):
                sub = _parse_local_file(str(child))
                if sub["text"]:
                    texts.append(f"[File: {child.name}]\n{sub['text'][:5000]}")
        result["text"] = "\n\n---\n\n".join(texts)
        return result

    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".pdf":
        result["text"] = _parse_pdf(raw)
    elif suffix == ".docx":
        result["text"] = _parse_docx(raw)
    elif suffix in (".txt", ".md", ".csv", ".rst", ".json"):
        result["text"] = raw.decode("utf-8", errors="replace")
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
        try:
            import pytesseract
            from PIL import Image
            import io as _io
            img = Image.open(_io.BytesIO(raw))
            text = pytesseract.image_to_string(img)
            result["text"] = text
            result["images"] = [{"url": f"file://{file_path}", "alt": path.name, "caption": ""}]
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            result["images"] = [{"url": f"file://{file_path}", "alt": path.name, "caption": ""}]

    return result


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape_page(
    url: str,
    timeout: int = SCRAPER_TIMEOUT_S,
    follow_links: bool = False,
    follow_links_depth: int = 1,
    max_follow_links: int = 3,
) -> Dict:
    """
    Scrape a URL and return:
    {
        "url": str,
        "title": str,
        "domain": str,
        "text": str,
        "images": [{"url", "alt", "caption"}],
        "youtube_embeds": [{"url", "transcript"}],
        "followed_sources": [{"url", "title", "domain", "text", "images"}],
    }
    Pages shorter than _MIN_USEFUL_CHARS, login walls, and product pages are
    returned with empty "text" so callers can skip them.
    Also handles local file paths and file:// URIs.
    """
    from urllib.parse import urlparse as _urlparse
    _domain = _urlparse(url).netloc.lower().removeprefix("www.")

    result: Dict = {
        "url":              url,
        "title":            "",
        "domain":           _domain,
        "text":             "",
        "images":           [],
        "youtube_embeds":   [],
        "followed_sources": [],
    }

    # ── Local file ────────────────────────────────────────────────────────
    if url.startswith("file://") or (not url.startswith("http") and Path(url).exists()):
        local_path = url.replace("file://", "")
        parsed = _parse_local_file(local_path)
        result["text"]   = parsed["text"]
        result["images"] = parsed["images"]
        result["title"]  = Path(local_path).name
        return result

    # ── YouTube direct link ───────────────────────────────────────────────
    if _is_youtube(url):
        transcript = _get_youtube_transcript(url)
        result["text"]           = transcript
        result["youtube_embeds"] = [{"url": url, "transcript": transcript}]
        return result

    # ── Fetch web page ────────────────────────────────────────────────────
    import requests
    html: str = ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        raw_bytes = resp.content

        if "application/pdf" in content_type:
            result["text"] = _parse_pdf(raw_bytes)
            return result

        if "application/vnd.openxmlformats" in content_type:
            result["text"] = _parse_docx(raw_bytes)
            return result

        html = resp.text

    except Exception as e:
        logger.warning(f"Fetch failed {url}: {e}")
        return result

    # ── Extract page title ────────────────────────────────────────────────
    try:
        _soup_title = BeautifulSoup(html[:4096], "html.parser")
        _t = _soup_title.find("title")
        if _t:
            result["title"] = _t.get_text(strip=True)[:200]
    except Exception:
        pass

    # ── Login wall / product page guard ──────────────────────────────────
    # Do a quick pre-check on raw HTML before spending time extracting text
    if _is_login_wall(html, ""):
        logger.info(f"[Scraper] ⏭ Login wall detected — skipping {url[:70]}")
        return result

    # ── Extract text ──────────────────────────────────────────────────────
    text: str = ""
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            include_links=False,
        ) or ""
    except Exception:
        text = ""

    # BeautifulSoup fallback
    if not text or len(text) < 150:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except Exception:
            text = ""

    # ── Post-trafilatura cleanup ──────────────────────────────────────────
    text = _clean_text(text)

    # ── Quality gates ─────────────────────────────────────────────────────
    if len(text) < _MIN_USEFUL_CHARS:
        logger.info(
            f"[Scraper] ⏭ Too short ({len(text)} chars < {_MIN_USEFUL_CHARS}) — {url[:70]}"
        )
        result["text"] = ""
        return result

    if _is_product_page(text):
        logger.info(f"[Scraper] ⏭ Product page detected — skipping {url[:70]}")
        result["text"] = ""
        return result

    result["text"] = text.strip()

    # ── Extract images ────────────────────────────────────────────────────
    result["images"] = _extract_images(html, url)

    # ── Extract embedded YouTube iframes ─────────────────────────────────
    try:
        soup_yt = BeautifulSoup(html, "html.parser")
        for iframe in soup_yt.find_all("iframe", src=True):
            src = iframe["src"]
            if _is_youtube(src) or "youtube" in src:
                abs_yt = urljoin(url, src)
                transcript = _get_youtube_transcript(abs_yt)
                result["youtube_embeds"].append({"url": abs_yt, "transcript": transcript})
    except Exception as e:
        logger.debug(f"YouTube embed extraction failed: {e}")

    # ── Follow outgoing same-domain links ─────────────────────────────────
    if follow_links and follow_links_depth > 0 and html:
        follow_urls = _extract_follow_links(html, url, max_links=max_follow_links)
        for furl in follow_urls:
            try:
                fsub = scrape_page(
                    furl,
                    timeout=timeout,
                    follow_links=False,
                    follow_links_depth=0,
                )
                if fsub.get("text") and len(fsub["text"]) > 150:
                    result["followed_sources"].append({
                        "url":    furl,
                        "title":  fsub.get("title", furl),
                        "domain": fsub.get("domain", ""),
                        "text":   fsub["text"][:5000],
                        "images": fsub.get("images", [])[:5],
                    })
            except Exception as e:
                logger.debug(f"Follow-link failed {furl}: {e}")

    logger.info(
        f"Scraped {url[:70]}: {len(result['text'])} chars | "
        f"{len(result['images'])} imgs | "
        f"{len(result['youtube_embeds'])} YT | "
        f"{len(result['followed_sources'])} followed"
    )
    return result
