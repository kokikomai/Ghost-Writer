import os
import re
import json
import uuid
import tempfile
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, g, Response, stream_with_context
from flask_cors import CORS
import anthropic
import requests as http_requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
import jwt as pyjwt
from jwt import PyJWKClient
from scraper import fetch_sync, is_youtube_url

load_dotenv()

app = Flask(__name__, static_folder="static")
CORS(app)

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

DEFAULT_MODEL = "claude-opus-4-6"

AVAILABLE_MODELS = [
    {"id": "claude-opus-4-6", "name": "Claude Opus 4.6", "desc": "最高性能 / 長文・高品質向け"},
    {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet 4", "desc": "高速・バランス型"},
    {"id": "claude-haiku-4-20250514", "name": "Claude Haiku 4", "desc": "最速・低コスト"},
]

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Sessions / Templates / CTA use local JSON files (/tmp for Vercel compatibility)
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(tempfile.gettempdir(), "ghost-writer-data"))
os.makedirs(DATA_DIR, exist_ok=True)
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
PROMPT_TEMPLATES_FILE = os.path.join(DATA_DIR, "prompt_templates.json")
CTA_TEMPLATES_FILE = os.path.join(DATA_DIR, "cta_templates.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


_jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True)


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "認証が必要です"}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                audience="authenticated",
            )
            g.user_id = payload["sub"]
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "トークンの有効期限が切れています"}), 401
        except pyjwt.InvalidTokenError:
            return jsonify({"error": "無効なトークンです"}), 401
        return f(*args, **kwargs)
    return decorated


def load_sessions():
    return load_json(SESSIONS_FILE)


def save_sessions(sessions):
    save_json(SESSIONS_FILE, sessions)


def load_prompt_templates():
    return load_json(PROMPT_TEMPLATES_FILE)


def save_prompt_templates(templates):
    save_json(PROMPT_TEMPLATES_FILE, templates)


def load_cta_templates():
    return load_json(CTA_TEMPLATES_FILE)


def save_cta_templates(templates):
    save_json(CTA_TEMPLATES_FILE, templates)


def call_claude(system_prompt, user_prompt, *, json_mode=False, model=None):
    messages = [{"role": "user", "content": user_prompt}]

    kwargs = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": messages,
    }

    response = client.messages.create(**kwargs)
    text = response.content[0].text

    if json_mode:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)

    return text


def _stream_claude(system_prompt, user_prompt, *, model=None):
    """Claude API をストリーミングで呼び出し、(chunk_text, full_text) を yield する。"""
    messages = [{"role": "user", "content": user_prompt}]
    kwargs = {
        "model": model or DEFAULT_MODEL,
        "max_tokens": 8192,
        "system": system_prompt,
        "messages": messages,
    }
    full_text = []
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            full_text.append(text)
            yield text, None
    yield "", "".join(full_text)


def build_style_analysis_prompt(references):
    ref_texts = ""
    for i, ref in enumerate(references, 1):
        ref_texts += f"\n\n=== リファレンス記事 {i} ===\n{ref}"

    system = "あなたは文章のスタイル分析の専門家です。指示に従い、JSON形式で回答してください。"

    user = f"""以下のリファレンス記事を分析し、これらの記事と同じスタイルで新しい記事を書くための「スタイルガイド」を作成してください。

分析すべき要素：
1. 構成パターン（話の組み立て方、セクションの順序）
2. 文体・トーン（一人称、語尾、文の長さ、硬さ/柔らかさ）
3. リズム（改行の使い方、短文連打、体言止めの頻度）
4. 表現技法（比喩の傾向、カギ括弧の使い方、対比構造）
5. 感情の扱い方（自己開示の深さ、弱さの見せ方）
6. 記事の入り方と締め方のパターン
7. 句読点の使い方（読点「、」の頻度と位置、句点「。」の使い方、句読点を省略して改行で代替するパターン）

以下のJSON形式で出力してください。各項目は具体的に、かつ再現可能な指示として書いてください。
JSON以外のテキストは出力しないでください。

{{
  "structure": "構成パターンの詳細な説明（第1層〜第N層の形式で）",
  "tone": "文体・トーンのルール",
  "rhythm": "リズムの作り方のルール",
  "punctuation": "句読点のルール（読点「、」の頻度・1文あたりの平均使用回数・どの位置で打つか、句点「。」の扱い、読点の代わりに改行や接続詞で繋ぐパターンがあるか、リファレンス記事の実際の傾向を数値で示す）",
  "expression": "比喩・表現技法のルール",
  "emotion": "感情の扱い方のルール",
  "opening": "冒頭の書き方のルール",
  "closing": "締めの書き方のルール",
  "formatting": "フォーマットのルール（改行、見出し、太字など）",
  "avoid": "避けるべきことのリスト"
}}
{ref_texts}"""

    return system, user


def _format_sources(sources):
    if not sources:
        return ""
    parts = []
    for i, src in enumerate(sources, 1):
        title = src.get("title", f"資料{i}")
        text = src.get("text", "")
        if text.strip():
            parts.append(f"=== 参考資料 {i}: {title} ===\n{text[:8000]}")
    if not parts:
        return ""
    return "\n\n## 参考資料（記事執筆の背景知識として活用してください）\n" + "\n\n".join(parts)


def build_interview_prompt(style_guide, title, memo, sources=None):
    system = """あなたは敏腕インタビュアーです。経営者から記事の素材を引き出し、最高の記事に仕上げるための深掘り質問をします。日本語で応答してください。"""

    sources_text = _format_sources(sources)

    user = f"""## スタイルガイド（この記事が目指すスタイル）
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## ユーザーからの指示
タイトル: {title}
内容メモ:
{memo}
{sources_text}

## あなたのタスク

まず、ユーザーのメモの中で「すでに揃っている要素」と「足りない要素」を簡潔に整理して見せてください。

その上で、以下の7つの素材チェックリストに照らして、足りない要素を補うための質問を3〜5個まとめて投げてください。

### 素材チェックリスト
1. 【原体験】テーマに関する具体的なエピソード（いつ・どこで・誰と・何が起きたか）
2. 【感情の生データ】そのとき何を感じたか（加工前の正直な感情）
3. 【失敗・弱さ】自分の未熟さや失敗の経験
4. 【転換点】考え方や行動が変わったきっかけ
5. 【構造・パターン】個人の経験から一般化できる洞察
6. 【対比要素】ビフォーアフター、二項対立の素材
7. 【着地の問い】読者に持ち帰ってほしい問いかけ

### ルール
- 質問は番号付きで、それぞれ1〜2文で簡潔に
- 質問のトーンは「聞き手」として自然に。尋問にならない"""

    return system, user


def build_followup_prompt(style_guide, title, memo, conversation_history, sources=None):
    history_text = ""
    for msg in conversation_history:
        role = "インタビュアー" if msg["role"] == "assistant" else "ユーザー"
        history_text += f"\n{role}: {msg['content']}\n"

    sources_text = _format_sources(sources)

    system = """あなたは敏腕インタビュアーです。記事の素材を引き出すためのインタビューを続けています。日本語で応答してください。"""

    user = f"""## スタイルガイド
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## 記事の指示
タイトル: {title}
内容メモ:
{memo}
{sources_text}

## これまでのやり取り
{history_text}

## あなたのタスク

ユーザーの回答を踏まえ、まだ素材として足りない要素があれば追加質問してください（1〜3個）。

十分な素材が揃ったと判断した場合は、以下の文言を含めて回答してください：
「素材が揃いました。記事を作成します。」

### ルール
- 質問は簡潔に
- すでに回答された内容を繰り返し聞かない"""

    return system, user


def build_article_prompt(style_guide, title, memo, conversation_history, sources=None):
    history_text = ""
    for msg in conversation_history:
        role = "インタビュアー" if msg["role"] == "assistant" else "ユーザー"
        history_text += f"\n{role}: {msg['content']}\n"

    sources_text = _format_sources(sources)

    system = """あなたは長文エッセイライターです。インタビューで得た素材をすべて統合し、スタイルガイドに厳密に従って記事を作成してください。日本語で執筆してください。"""

    user = f"""## スタイルガイド
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## 記事の指示
タイトル: {title}
内容メモ:
{memo}
{sources_text}

## インタビューで得た素材
{history_text}

## 出力ルール

記事はHTML形式で出力してください。X記事（Articles）のリッチテキストエディタにコピペしたときに、見出し・太字・改行がそのまま反映されるようにするためです。

使用するHTMLタグ：
- <h1> … 記事タイトル（1つだけ）
- <h2> … セクション見出し（文章の流れが切り替わるポイントに入れる。見出しは内容を端的に表す短いフレーズ）
- <hr> … セクション間の区切り線
- <b> … 太字（記事の核となるフレーズ、スクショで切り取られたときに単体で成立する文。1セクションにつき0〜2箇所）
- <p> … 段落
- <br> … 短文連打の改行

それ以外のタグは使わない。CSSやstyleタグは不要。
HTMLタグだけを出力し、```html```などのコードブロック記法で囲まないでください。

### 追加ルール
- 箇条書きメモやインタビュー回答の順番をそのまま並べない。スタイルガイドの構成に沿って再構築する
- インプレッションを意識する：冒頭2行で自分ごと化、名言的フレーズを1〜3個埋め込む、最後の一文は問いかけ
- 2000〜4000文字程度の長文
- HTMLタグのみを出力する。コードブロックで囲まない
- 句読点（特に読点「、」）はスタイルガイドのpunctuationルールに厳密に従う。LLMのデフォルトは読点が多すぎるため、リファレンス記事が読点少なめなら意識的に減らし、改行や文の区切りで代替する"""

    return system, user


@app.route("/")
def index():
    resp = send_from_directory("static", "index.html")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL})


@app.route("/api/analyze-style", methods=["POST"])
@require_auth
def analyze_style():
    data = request.json
    references = data.get("references", [])
    model = data.get("model")

    if not references:
        return jsonify({"error": "リファレンス記事を1つ以上入力してください"}), 400

    try:
        system, user = build_style_analysis_prompt(references)
        style_guide = call_claude(system, user, json_mode=True, model=model)
        return jsonify({"style_guide": style_guide})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _sse_yield(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


@app.route("/api/interview/start", methods=["POST"])
@require_auth
def start_interview():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    sources = data.get("sources", [])
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_interview_prompt(style_guide, title, memo, sources)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            yield _sse_yield({"done": True, "message": full})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        ai_message = call_claude(system, user, model=model)
        return jsonify({"message": ai_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview/continue", methods=["POST"])
@require_auth
def continue_interview():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_followup_prompt(style_guide, title, memo, conversation, sources)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            ready = "素材が揃いました" in full
                            yield _sse_yield({"done": True, "message": full, "ready_to_write": ready})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        ai_message = call_claude(system, user, model=model)
        ready = "素材が揃いました" in ai_message
        return jsonify({"message": ai_message, "ready_to_write": ready})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-article", methods=["POST"])
@require_auth
def generate_article():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_article_prompt(style_guide, title, memo, conversation, sources)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            yield _sse_yield({"done": True, "article": full})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        article_html = call_claude(system, user, model=model)
        return jsonify({"article": article_html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contexts", methods=["GET"])
@require_auth
def list_contexts():
    data = supabase.table("contexts").select("*").eq("user_id", g.user_id).order("created_at", desc=True).execute()
    # Map DB column names to frontend expected names
    contexts = []
    for row in data.data:
        row["references"] = row.pop("reference_texts", [])
        contexts.append(row)
    return jsonify({"contexts": contexts})


@app.route("/api/contexts", methods=["POST"])
@require_auth
def create_context():
    data = request.json
    name = data.get("name", "").strip()
    references = data.get("references", [])
    style_guide = data.get("style_guide", None)

    if not name:
        return jsonify({"error": "名前を入力してください"}), 400
    if not references or not any(r.strip() for r in references):
        return jsonify({"error": "リファレンス記事を1つ以上入力してください"}), 400

    result = supabase.table("contexts").insert({
        "user_id": g.user_id,
        "name": name,
        "reference_texts": references,
        "style_guide": style_guide,
    }).execute()

    ctx = result.data[0]
    ctx["references"] = ctx.pop("reference_texts", [])
    return jsonify({"context": ctx})


@app.route("/api/contexts/<context_id>", methods=["PUT"])
@require_auth
def update_context(context_id):
    data = request.json

    update_data = {}
    if "name" in data:
        update_data["name"] = data["name"]
    if "references" in data:
        update_data["reference_texts"] = data["references"]
    if "style_guide" in data:
        update_data["style_guide"] = data["style_guide"]

    result = supabase.table("contexts").update(update_data).eq("id", context_id).eq("user_id", g.user_id).execute()

    if not result.data:
        return jsonify({"error": "コンテキストが見つかりません"}), 404

    ctx = result.data[0]
    ctx["references"] = ctx.pop("reference_texts", [])
    return jsonify({"context": ctx})


@app.route("/api/contexts/<context_id>/reference/<int:ref_index>", methods=["PUT"])
@require_auth
def update_single_reference(context_id, ref_index):
    data = request.json
    new_text = data.get("text", "")

    result = supabase.table("contexts").select("*").eq("id", context_id).eq("user_id", g.user_id).execute()

    if not result.data:
        return jsonify({"error": "コンテキストが見つかりません"}), 404

    context = result.data[0]
    refs = context.get("reference_texts", [])
    while len(refs) <= ref_index:
        refs.append("")
    refs[ref_index] = new_text

    update_result = supabase.table("contexts").update({
        "reference_texts": refs,
        "style_guide": None,
    }).eq("id", context_id).eq("user_id", g.user_id).execute()

    ctx = update_result.data[0]
    ctx["references"] = ctx.pop("reference_texts", [])
    return jsonify({"context": ctx})


@app.route("/api/contexts/<context_id>", methods=["DELETE"])
@require_auth
def delete_context(context_id):
    supabase.table("contexts").delete().eq("id", context_id).eq("user_id", g.user_id).execute()
    return jsonify({"success": True})


@app.route("/api/fetch-url", methods=["POST"])
@require_auth
def fetch_url():
    """URLからAPI経由でテキストを取得する"""
    import time as _time
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URLを入力してください"}), 400

    if is_youtube_url(url):
        return jsonify({
            "text": "",
            "source": "error",
            "message": "YouTube URLはリファレンス記事には使用できません。YouTube字幕は「参考記事」のURL追加からご利用ください。",
            "meta": {},
        }), 400

    t0 = _time.time()
    print(f"[fetch-url] START url={url[:80]}", flush=True)
    try:
        result = fetch_sync(url)
        elapsed = _time.time() - t0
        text = result.get("text", "")
        meta = result.get("meta", {})
        print(f"[fetch-url] OK {len(text)} chars in {elapsed:.1f}s", flush=True)

        if text and len(text) > 50:
            return jsonify({
                "text": text[:15000],
                "source": "api",
                "message": f"記事を取得しました（{len(text)}文字）",
                "meta": meta,
            })

        return jsonify({
            "text": "",
            "source": "api_empty",
            "message": "テキストを取得できませんでした。URLを確認してください。",
            "meta": {},
        })
    except Exception as e:
        elapsed = _time.time() - t0
        print(f"[fetch-url] ERROR in {elapsed:.1f}s: {e}", flush=True)
        return jsonify({
            "text": "",
            "source": "error",
            "message": f"取得に失敗しました: {str(e)}",
            "meta": {},
        })


def build_rewrite_interview_prompt(style_guide, original_article, user_angle):
    sources_text = _format_sources([{"title": "元記事", "text": original_article}]) if original_article else ""

    system = """あなたは敏腕インタビュアーです。ユーザーが既存の記事を自分の言葉でリライトしようとしています。
元記事の情報や有益さはそのままに、ユーザー自身の一次情報・エピソードを盛り込むために深掘り質問をします。日本語で応答してください。"""

    user = f"""## スタイルガイド（この記事が目指すスタイル）
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## 元記事（リライト対象）
{original_article[:8000]}

## ユーザーの切り口・メモ
{user_angle}

## あなたのタスク

まず元記事の構成と主要なポイントを簡潔に整理してください。

次に、ユーザー自身の一次情報を引き出すための質問を3〜5個投げてください。以下の観点で質問してください：

1. 元記事のテーマに関するユーザー自身の具体的な経験・エピソード
2. 元記事の主張に対するユーザー独自の見解や補足
3. ユーザーが現場で実際に見てきた事例・データ
4. 元記事にはない、ユーザーならではの視点や気づき
5. 読者に伝えたいユーザー自身のメッセージ

### ルール
- 質問は番号付きで、それぞれ1〜2文で簡潔に
- 元記事の内容を丸写しさせるのではなく、ユーザーの一次情報を引き出す方向で"""

    return system, user


def build_rewrite_followup_prompt(style_guide, original_article, user_angle, conversation_history):
    history_text = ""
    for msg in conversation_history:
        role = "インタビュアー" if msg["role"] == "assistant" else "ユーザー"
        history_text += f"\n{role}: {msg['content']}\n"

    system = """あなたは敏腕インタビュアーです。リライト記事の素材を引き出すためのインタビューを続けています。日本語で応答してください。"""

    user = f"""## スタイルガイド
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## 元記事（リライト対象）
{original_article[:8000]}

## ユーザーの切り口・メモ
{user_angle}

## これまでのやり取り
{history_text}

## あなたのタスク

ユーザーの回答を踏まえ、まだ一次情報として足りない要素があれば追加質問してください（1〜3個）。

十分な素材が揃ったと判断した場合は、以下の文言を含めて回答してください：
「素材が揃いました。記事を作成します。」

### ルール
- 質問は簡潔に
- すでに回答された内容を繰り返し聞かない
- 元記事のコピーではなく、ユーザーの一次情報を引き出す方向で"""

    return system, user


def build_rewrite_article_prompt(style_guide, original_article, user_angle, conversation_history, sources=None):
    history_text = ""
    for msg in conversation_history:
        role = "インタビュアー" if msg["role"] == "assistant" else "ユーザー"
        history_text += f"\n{role}: {msg['content']}\n"

    sources_text = _format_sources(sources)

    system = """あなたは長文エッセイライターです。元記事の情報や有益さをベースにしつつ、インタビューで得たユーザー独自の一次情報・エピソードを織り交ぜてリライト記事を作成してください。
元記事のコピペではなく、ユーザーの言葉と経験で再構築してください。日本語で執筆してください。"""

    user = f"""## スタイルガイド
{json.dumps(style_guide, ensure_ascii=False, indent=2)}

## 元記事（リライト対象 - 情報のベースとして使うが、文章はコピーしない）
{original_article[:8000]}

## ユーザーの切り口・メモ
{user_angle}

## インタビューで得たユーザーの一次情報
{history_text}
{sources_text}

## 出力ルール

元記事の有益な情報・構成はベースにしつつ、ユーザー自身の経験・エピソード・視点を主軸に据えたリライト記事を書いてください。

記事はHTML形式で出力してください。X記事（Articles）のリッチテキストエディタにコピペしたときに、見出し・太字・改行がそのまま反映されるようにするためです。

使用するHTMLタグ：
- <h1> … 記事タイトル（1つだけ）
- <h2> … セクション見出し
- <hr> … セクション間の区切り線
- <b> … 太字（核となるフレーズ、1セクションにつき0〜2箇所）
- <p> … 段落
- <br> … 短文連打の改行

それ以外のタグは使わない。CSSやstyleタグは不要。
HTMLタグだけを出力し、コードブロック記法で囲まないでください。

### 追加ルール
- 元記事の文章をそのまま使わない。ユーザーの言葉で再構築する
- ユーザーの一次情報・エピソードを記事の主軸に据える
- 元記事の有益な情報やデータは参考にしつつ、ユーザーの経験と絡めて記述する
- スタイルガイドの構成・トーンに従う
- 2000〜4000文字程度の長文
- HTMLタグのみを出力する。コードブロックで囲まない
- 句読点（特に読点「、」）はスタイルガイドのpunctuationルールに厳密に従う。LLMのデフォルトは読点が多すぎるため、リファレンス記事が読点少なめなら意識的に減らし、改行や文の区切りで代替する"""

    return system, user


@app.route("/api/rewrite/start", methods=["POST"])
@require_auth
def rewrite_start():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_rewrite_interview_prompt(style_guide, original_article, user_angle)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            yield _sse_yield({"done": True, "message": full})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        ai_message = call_claude(system, user, model=model)
        return jsonify({"message": ai_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rewrite/continue", methods=["POST"])
@require_auth
def rewrite_continue():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    conversation = data.get("conversation", [])
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_rewrite_followup_prompt(style_guide, original_article, user_angle, conversation)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            ready = "素材が揃いました" in full
                            yield _sse_yield({"done": True, "message": full, "ready_to_write": ready})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        ai_message = call_claude(system, user, model=model)
        ready = "素材が揃いました" in ai_message
        return jsonify({"message": ai_message, "ready_to_write": ready})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rewrite/generate", methods=["POST"])
@require_auth
def rewrite_generate():
    data = request.json or {}
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")
    use_stream = request.args.get("stream") == "1"

    try:
        system, user = build_rewrite_article_prompt(style_guide, original_article, user_angle, conversation, sources)
        if use_stream:
            def gen():
                try:
                    for chunk, full in _stream_claude(system, user, model=model):
                        if full is not None:
                            yield _sse_yield({"done": True, "article": full})
                        else:
                            yield _sse_yield({"delta": chunk})
                except Exception as e:
                    yield _sse_yield({"error": str(e)})
            return Response(
                stream_with_context(gen()),
                mimetype="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        article_html = call_claude(system, user, model=model)
        return jsonify({"article": article_html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/extract-source", methods=["POST"])
@require_auth
def extract_source():
    """URLまたはPDFからテキストを抽出する汎用エンドポイント"""
    url = (request.form.get("url") or "").strip()
    pdf_file = request.files.get("file")

    if pdf_file and pdf_file.filename:
        return _extract_pdf(pdf_file)
    elif url:
        return _extract_url(url)
    else:
        return jsonify({"error": "URLまたはPDFファイルを指定してください"}), 400


def _extract_pdf(pdf_file):
    try:
        import fitz
    except ImportError:
        return jsonify({"error": "PyMuPDF が未インストールです"}), 500

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            pdf_file.save(tmp.name)
            tmp_path = tmp.name

        doc = fitz.open(tmp_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        os.unlink(tmp_path)

        text = "\n\n".join(pages).strip()
        if not text:
            return jsonify({"text": "", "title": pdf_file.filename, "message": "PDFからテキストを抽出できませんでした。"})

        return jsonify({
            "text": text[:30000],
            "title": pdf_file.filename,
            "message": f"PDFから抽出しました（{len(text)}文字）",
        })
    except Exception as e:
        return jsonify({"error": f"PDF抽出エラー: {str(e)}"}), 500


def _extract_url(url):
    is_x = bool(re.match(r"https?://(?:x\.com|twitter\.com)/", url))
    if is_x:
        try:
            result = fetch_sync(url)
            text = result.get("text", "")
            meta = result.get("meta", {})
            title = meta.get("article_title", "") or url
            return jsonify({
                "text": text[:30000],
                "title": title,
                "message": f"取得しました（{len(text)}文字）",
            })
        except Exception as e:
            return jsonify({"error": f"X記事の取得に失敗: {str(e)}"}), 500

    if is_youtube_url(url):
        try:
            result = fetch_sync(url)
            text = result.get("text", "")
            meta = result.get("meta", {})
            title = meta.get("article_title", "") or url
            return jsonify({
                "text": text[:30000],
                "title": title,
                "message": f"YouTube字幕を取得しました（{len(text)}文字）",
            })
        except Exception as e:
            return jsonify({"error": f"YouTube字幕の取得に失敗: {str(e)}"}), 500

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = http_requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        article = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"post-content|entry-content|article-body|content-body"))
        )
        body = article if article else soup.body

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]

        text = body.get_text("\n", strip=True) if body else ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if not text or len(text) < 30:
            return jsonify({"text": "", "title": title or url, "message": "テキストを抽出できませんでした。"})

        return jsonify({
            "text": text[:30000],
            "title": title or url,
            "message": f"取得しました（{len(text)}文字）",
        })
    except Exception as e:
        return jsonify({"error": f"URL取得エラー: {str(e)}"}), 500


@app.route("/api/articles", methods=["GET"])
@require_auth
def list_articles():
    data = supabase.table("articles").select("*").eq("user_id", g.user_id).order("created_at", desc=True).execute()
    # Map DB column name text_content -> text for frontend compatibility
    articles = []
    for row in data.data:
        row["text"] = row.pop("text_content", "")
        articles.append(row)
    return jsonify({"articles": articles})


@app.route("/api/articles", methods=["POST"])
@require_auth
def create_article():
    data = request.json
    title = data.get("title", "").strip()
    html = data.get("html", "")
    text = data.get("text", "")
    memo = data.get("memo", "")
    conversation = data.get("conversation", [])
    context_id = data.get("context_id", None)

    if not title:
        return jsonify({"error": "タイトルを入力してください"}), 400

    result = supabase.table("articles").insert({
        "user_id": g.user_id,
        "title": title,
        "html": html,
        "text_content": text,
        "memo": memo,
        "conversation": conversation,
        "context_id": context_id,
    }).execute()

    art = result.data[0]
    art["text"] = art.pop("text_content", "")
    return jsonify({"article": art})


@app.route("/api/articles/<article_id>", methods=["PUT"])
@require_auth
def update_article(article_id):
    data = request.json

    update_data = {}

    if "html" in data:
        update_data["html"] = data["html"]

    if "text" in data:
        update_data["text_content"] = data["text"]

    if "title" in data:
        update_data["title"] = data["title"]

    if "memo" in data:
        update_data["memo"] = data["memo"]

    if "conversation" in data:
        update_data["conversation"] = data["conversation"]

    if "context_id" in data:
        update_data["context_id"] = data["context_id"]

    if "status" in data:
        if data["status"] not in ["draft", "interviewing", "completed"]:
            return jsonify({"error": "無効なステータスです"}), 400
        update_data["status"] = data["status"]

    if not update_data:
        return jsonify({"error": "更新するデータがありません"}), 400

    result = supabase.table("articles").update(update_data).eq("id", article_id).eq("user_id", g.user_id).execute()

    if not result.data:
        return jsonify({"error": "記事が見つかりません"}), 404

    art = result.data[0]
    art["text"] = art.pop("text_content", "")
    return jsonify({"article": art})


@app.route("/api/articles/<article_id>", methods=["DELETE"])
@require_auth
def delete_article(article_id):
    supabase.table("articles").delete().eq("id", article_id).eq("user_id", g.user_id).execute()
    return jsonify({"success": True})


@app.route("/api/sessions", methods=["GET"])
@require_auth
def list_sessions():
    sessions = load_sessions()
    sessions = sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)
    return jsonify({"sessions": sessions})


@app.route("/api/sessions", methods=["POST"])
@require_auth
def create_session():
    data = request.json or {}
    session = {
        "id": str(uuid.uuid4()),
        "mode": data.get("mode", "create"),
        "step": data.get("step", 1),
        "title": data.get("title", ""),
        "memo": data.get("memo", ""),
        "conversation": data.get("conversation", []),
        "style_guide": data.get("style_guide"),
        "context_id": data.get("context_id"),
        "sources": data.get("sources", []),
        "article_html": data.get("article_html", ""),
        "original_article": data.get("original_article", ""),
        "user_angle": data.get("user_angle", ""),
        "original_title": data.get("original_title", ""),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }
    sessions = load_sessions()
    sessions.insert(0, session)
    save_sessions(sessions)
    return jsonify({"session": session})


@app.route("/api/sessions/<session_id>", methods=["GET"])
@require_auth
def get_session(session_id):
    sessions = load_sessions()
    for s in sessions:
        if s["id"] == session_id:
            return jsonify({"session": s})
    return jsonify({"error": "セッションが見つかりません"}), 404


@app.route("/api/sessions/<session_id>", methods=["PUT"])
@require_auth
def update_session(session_id):
    data = request.json or {}
    sessions = load_sessions()
    for s in sessions:
        if s["id"] == session_id:
            if "step" in data:
                s["step"] = data["step"]
            if "title" in data:
                s["title"] = data["title"]
            if "memo" in data:
                s["memo"] = data["memo"]
            if "conversation" in data:
                s["conversation"] = data["conversation"]
            if "style_guide" in data:
                s["style_guide"] = data["style_guide"]
            if "context_id" in data:
                s["context_id"] = data["context_id"]
            if "sources" in data:
                s["sources"] = data["sources"]
            if "article_html" in data:
                s["article_html"] = data["article_html"]
            if "user_angle" in data:
                s["user_angle"] = data["user_angle"]
            if "original_article" in data:
                s["original_article"] = data["original_article"]
            if "original_title" in data:
                s["original_title"] = data["original_title"]
            s["updated_at"] = datetime.now().isoformat()
            save_sessions(sessions)
            return jsonify({"session": s})
    return jsonify({"error": "セッションが見つかりません"}), 404


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@require_auth
def delete_session(session_id):
    sessions = load_sessions()
    sessions = [s for s in sessions if s["id"] != session_id]
    save_sessions(sessions)
    return jsonify({"success": True})


@app.route("/api/prompt-templates", methods=["GET"])
@require_auth
def list_prompt_templates():
    templates = load_prompt_templates()
    return jsonify({"templates": templates})


@app.route("/api/prompt-templates", methods=["POST"])
@require_auth
def create_prompt_template():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    title = data.get("title", "")
    memo = data.get("memo", "")
    if not name:
        return jsonify({"error": "テンプレート名を入力してください"}), 400
    template = {
        "id": str(uuid.uuid4()),
        "name": name,
        "title": title,
        "memo": memo,
        "created_at": datetime.now().isoformat(),
    }
    templates = load_prompt_templates()
    templates.insert(0, template)
    save_prompt_templates(templates)
    return jsonify({"template": template})


@app.route("/api/prompt-templates/<template_id>", methods=["PUT"])
@require_auth
def update_prompt_template(template_id):
    data = request.json or {}
    templates = load_prompt_templates()
    for t in templates:
        if t["id"] == template_id:
            if "name" in data:
                t["name"] = (data["name"] or "").strip()
            if "title" in data:
                t["title"] = data["title"]
            if "memo" in data:
                t["memo"] = data["memo"]
            save_prompt_templates(templates)
            return jsonify({"template": t})
    return jsonify({"error": "テンプレートが見つかりません"}), 404


@app.route("/api/prompt-templates/<template_id>", methods=["DELETE"])
@require_auth
def delete_prompt_template(template_id):
    templates = load_prompt_templates()
    templates = [t for t in templates if t["id"] != template_id]
    save_prompt_templates(templates)
    return jsonify({"success": True})


@app.route("/api/cta-templates", methods=["GET"])
@require_auth
def list_cta_templates():
    templates = load_cta_templates()
    return jsonify({"templates": templates})


@app.route("/api/cta-templates", methods=["POST"])
@require_auth
def create_cta_template():
    data = request.json or {}
    name = (data.get("name") or "").strip()
    content = data.get("content", "")
    if not name:
        return jsonify({"error": "CTA名を入力してください"}), 400
    template = {
        "id": str(uuid.uuid4()),
        "name": name,
        "content": content,
        "created_at": datetime.now().isoformat(),
    }
    templates = load_cta_templates()
    templates.insert(0, template)
    save_cta_templates(templates)
    return jsonify({"template": template})


@app.route("/api/cta-templates/<template_id>", methods=["PUT"])
@require_auth
def update_cta_template(template_id):
    data = request.json or {}
    templates = load_cta_templates()
    for t in templates:
        if t["id"] == template_id:
            if "name" in data:
                t["name"] = (data["name"] or "").strip()
            if "content" in data:
                t["content"] = data["content"]
            save_cta_templates(templates)
            return jsonify({"template": t})
    return jsonify({"error": "CTAテンプレートが見つかりません"}), 404


@app.route("/api/cta-templates/<template_id>", methods=["DELETE"])
@require_auth
def delete_cta_template(template_id):
    templates = load_cta_templates()
    templates = [t for t in templates if t["id"] != template_id]
    save_cta_templates(templates)
    return jsonify({"success": True})


@app.route("/api/article/edit-selection", methods=["POST"])
@require_auth
def edit_selection():
    data = request.json
    full_html = data.get("full_html", "")
    selected_text = data.get("selected_text", "")
    instruction = data.get("instruction", "")
    style_guide = data.get("style_guide", {})
    model = data.get("model")

    system = "あなたは長文エッセイの編集者です。記事の一部を修正してください。日本語で応答してください。"
    user = f"""## 記事全体（HTML）
{full_html}

## 修正対象のテキスト
「{selected_text}」

## 修正指示
{instruction}

## ルール
- 修正対象の部分だけを書き換えた、記事全体のHTMLを返してください
- 修正対象以外の部分は一切変えないでください
- スタイルガイドのトーン・句読点の使い方を維持してください（特に読点「、」の頻度をスタイルガイドのpunctuationルールに合わせる）
- 使用するHTMLタグ: h1, h2, hr, b, p, br のみ
- HTMLタグだけを出力し、コードブロック記法で囲まないでください"""

    if style_guide:
        user = f"## スタイルガイド\n{json.dumps(style_guide, ensure_ascii=False)}\n\n" + user

    try:
        result = call_claude(system, user, model=model)
        return jsonify({"article": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/article/edit-full", methods=["POST"])
@require_auth
def edit_full():
    data = request.json
    full_html = data.get("full_html", "")
    instruction = data.get("instruction", "")
    style_guide = data.get("style_guide", {})
    model = data.get("model")

    system = "あなたは長文エッセイの編集者です。記事全体に対して修正指示に従って書き換えてください。日本語で応答してください。"
    user = f"""## 記事全体（HTML）
{full_html}

## 修正指示
{instruction}

## ルール
- 修正指示に従って記事全体を修正してください
- 指示に関係ない部分はできるだけ維持してください
- スタイルガイドのトーン・句読点の使い方を維持してください（特に読点「、」の頻度をスタイルガイドのpunctuationルールに合わせる）
- 使用するHTMLタグ: h1, h2, hr, b, p, br のみ
- HTMLタグだけを出力し、コードブロック記法で囲まないでください"""

    if style_guide:
        user = f"## スタイルガイド\n{json.dumps(style_guide, ensure_ascii=False)}\n\n" + user

    try:
        result = call_claude(system, user, model=model)
        return jsonify({"article": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
