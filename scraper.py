import os
import re
import requests
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "twitter241.p.rapidapi.com"

_yt_api = YouTubeTranscriptApi()


def extract_tweet_id(url):
    """X/TwitterのURLからツイートIDを抽出する"""
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else None


def fetch_tweet_via_api(tweet_id):
    """RapidAPI (twitter241) を使ってツイート/記事データを取得する"""
    resp = requests.get(
        f"https://{RAPIDAPI_HOST}/tweet",
        params={"pid": tweet_id},
        headers={
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def parse_article_blocks(blocks):
    """記事のブロックデータをプレーンテキストに変換する"""
    lines = []
    for block in blocks:
        text = block.get("text", "").strip()
        if not text:
            lines.append("")
            continue

        btype = block.get("type", "unstyled")

        if btype.startswith("header-"):
            lines.append(text)
            lines.append("")
        elif btype == "unordered-list-item":
            lines.append(f"・{text}")
        elif btype == "ordered-list-item":
            lines.append(f"- {text}")
        elif btype == "blockquote":
            lines.append(text)
            lines.append("")
        elif btype == "atomic":
            continue
        else:
            lines.append(text)
            lines.append("")

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def extract_from_api_response(data):
    """APIレスポンスからテキスト・メタ情報を抽出する"""
    instructions = (
        data.get("data", {})
        .get("threaded_conversation_with_injections_v2", {})
        .get("instructions", [])
    )

    for inst in instructions:
        if "entries" not in inst:
            continue
        for entry in inst["entries"]:
            result = (
                entry.get("content", {})
                .get("itemContent", {})
                .get("tweet_results", {})
                .get("result", {})
            )
            if not result:
                continue

            core_legacy = (
                result.get("core", {})
                .get("user_results", {})
                .get("result", {})
                .get("legacy", {})
            )
            display_name = core_legacy.get("name", "")
            username = core_legacy.get("screen_name", "")

            if "article" in result:
                article = result["article"]["article_results"]["result"]
                title = article.get("title", "")
                blocks = article.get("content_state", {}).get("blocks", [])
                text = parse_article_blocks(blocks)

                return {
                    "text": text,
                    "meta": {
                        "article_title": title,
                        "display_name": display_name,
                        "username": username,
                        "author": f"{display_name} (@{username})" if username else display_name,
                    },
                }

            note = result.get("note_tweet", {})
            if note:
                text = (
                    note.get("note_tweet_results", {})
                    .get("result", {})
                    .get("text", "")
                )
            else:
                text = result.get("legacy", {}).get("full_text", "")

            text = re.sub(r"https?://t\.co/\S+", "", text).strip()

            return {
                "text": text,
                "meta": {
                    "article_title": "",
                    "display_name": display_name,
                    "username": username,
                    "author": f"{display_name} (@{username})" if username else display_name,
                },
            }

    return {"text": "", "meta": {}}


def fetch_x_content(url):
    """X/TwitterのURLから記事/ポストのテキストとメタ情報を取得する"""
    tweet_id = extract_tweet_id(url)
    if not tweet_id:
        raise ValueError(f"URLからツイートIDを抽出できません: {url}")

    data = fetch_tweet_via_api(tweet_id)
    return extract_from_api_response(data)


def is_youtube_url(url):
    """YouTube URLかどうか判定する"""
    return bool(re.match(
        r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/", url
    ))


def extract_youtube_video_id(url):
    """YouTube URLから動画IDを抽出する"""
    patterns = [
        r"(?:youtube\.com/watch\?.*v=|youtu\.be/|youtube\.com/embed/|youtube\.com/v/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


def fetch_youtube_transcript(url):
    """YouTube動画の字幕を取得してテキストとメタ情報を返す"""
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise ValueError(f"YouTube動画IDを抽出できません: {url}")

    try:
        transcript = _yt_api.fetch(video_id, languages=["ja", "en"])
    except Exception:
        try:
            transcript = _yt_api.fetch(video_id)
        except Exception as e:
            raise ValueError(f"字幕を取得できませんでした（字幕が無効な動画の可能性があります）: {e}")

    text = " ".join(snippet.text for snippet in transcript)
    text = re.sub(r"\s+", " ", text).strip()

    title = _get_youtube_title(video_id)

    return {
        "text": text,
        "meta": {
            "article_title": title,
            "display_name": "",
            "username": "",
            "author": "",
            "source_type": "youtube",
            "video_id": video_id,
        },
    }


def _get_youtube_title(video_id):
    """YouTube動画のタイトルをoEmbed APIで取得する"""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("title", "")
    except Exception:
        return ""


def fetch_sync(url):
    """同期ラッパー。テキストとメタ情報を返す"""
    is_x = bool(re.match(r"https?://(?:x\.com|twitter\.com)/", url))

    if is_x and RAPIDAPI_KEY:
        return fetch_x_content(url)

    if is_youtube_url(url):
        return fetch_youtube_transcript(url)

    raise ValueError(
        "X/TwitterまたはYouTubeのURLのみ対応しています。"
    )


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://x.com/minatoku_genkai/status/2025414074969063837"
    result = fetch_sync(url)
    print(f"TITLE: {result['meta'].get('article_title', '')}")
    print(f"AUTHOR: {result['meta'].get('display_name', '')} (@{result['meta'].get('username', '')})")
    print(f"LENGTH: {len(result['text'])}")
    print("---CONTENT---")
    print(result["text"])
