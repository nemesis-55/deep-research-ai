"""
Deep Page Scraper — Deep Research AI
- Full article text via trafilatura + BeautifulSoup fallback
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

HEADERS = {"User-Agent": SCRAPER_USER_AGENT}

YOUTUBE_DOMAINS = {"youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com"}

_SKIP_IMG_KEYWORDS = {
    "pixel", "tracking", "beacon", "icon", "logo", "sprite",
    "blank", "spacer", "avatar", "placeholder", "badge",
    "button", "arrow", "close", "menu", "social",
}


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
        "text": str,
        "images": [{"url", "alt", "caption"}],
        "youtube_embeds": [{"url", "transcript"}],
        "followed_sources": [{"url", "text", "images"}],
    }
    Also handles local file paths and file:// URIs.
    """
    result: Dict = {
        "url": url,
        "text": "",
        "images": [],
        "youtube_embeds": [],
        "followed_sources": [],
    }

    # ── Local file ────────────────────────────────────────────────────────
    if url.startswith("file://") or (not url.startswith("http") and Path(url).exists()):
        local_path = url.replace("file://", "")
        parsed = _parse_local_file(local_path)
        result["text"] = parsed["text"]
        result["images"] = parsed["images"]
        return result

    # ── YouTube direct link ───────────────────────────────────────────────
    if _is_youtube(url):
        transcript = _get_youtube_transcript(url)
        result["text"] = transcript
        result["youtube_embeds"] = [{"url": url, "transcript": transcript}]
        return result

    # ── Fetch web page ────────────────────────────────────────────────────
    import requests
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()
        raw_bytes = resp.content
        html = None

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

    # ── Extract text ──────────────────────────────────────────────────────
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            include_links=False,
        )
    except Exception:
        text = None

    # BeautifulSoup fallback
    if not text or len(text) < 150:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
        except Exception:
            text = ""

    result["text"] = (text or "").strip()

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
                        "url": furl,
                        "text": fsub["text"][:5000],
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
