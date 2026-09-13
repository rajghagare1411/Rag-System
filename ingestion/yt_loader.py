from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """Extract YouTube video ID from various URL formats."""
    parsed = urlparse(url)

    # format: https://www.youtube.com/watch?v=VIDEO_ID
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]

    # format: https://youtu.be/VIDEO_ID
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    raise ValueError(f"Could not extract video ID from URL: {url}")


def load_youtube(url: str) -> list[dict]:
    """
    Fetch transcript from a YouTube video.
    Returns a list of dicts with timestamp and text.
    """
    video_id = extract_video_id(url)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id)
    except Exception as e:
        raise RuntimeError(f"[YouTube] Failed to fetch transcript: {e}")

    chunks = []
    for entry in transcript:
        chunks.append({
            "source": url,
            "timestamp": round(entry.start, 2),
            "content": entry.text.strip()
        })

    print(f"[YouTube] Loaded {len(chunks)} transcript segments from '{url}'")
    return chunks


# ---------- quick test ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python youtube_loader.py <youtube_url>")
        sys.exit(1)

    segments = load_youtube(sys.argv[1])
    for s in segments[:5]:  # preview first 5 segments
        print(f"\n[{s['timestamp']}s] {s['content']}")