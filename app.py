import os
import re
import json
import uuid
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import requests as http_requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from scraper import fetch_sync

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
CONTEXTS_FILE = os.path.join(DATA_DIR, "contexts.json")
ARTICLES_FILE = os.path.join(DATA_DIR, "articles.json")


def load_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_contexts():
    return load_json(CONTEXTS_FILE)


def save_contexts(contexts):
    save_json(CONTEXTS_FILE, contexts)


def load_articles():
    return load_json(ARTICLES_FILE)


def save_articles(articles):
    save_json(ARTICLES_FILE, articles)


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
    return send_from_directory("static", "index.html")


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({"models": AVAILABLE_MODELS, "default": DEFAULT_MODEL})


@app.route("/api/analyze-style", methods=["POST"])
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


@app.route("/api/interview/start", methods=["POST"])
def start_interview():
    data = request.json
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    sources = data.get("sources", [])
    model = data.get("model")

    try:
        system, user = build_interview_prompt(style_guide, title, memo, sources)
        ai_message = call_claude(system, user, model=model)
        return jsonify({"message": ai_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/interview/continue", methods=["POST"])
def continue_interview():
    data = request.json
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")

    try:
        system, user = build_followup_prompt(style_guide, title, memo, conversation, sources)
        ai_message = call_claude(system, user, model=model)
        ready = "素材が揃いました" in ai_message
        return jsonify({"message": ai_message, "ready_to_write": ready})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-article", methods=["POST"])
def generate_article():
    data = request.json
    style_guide = data.get("style_guide", {})
    title = data.get("title", "")
    memo = data.get("memo", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")

    try:
        system, user = build_article_prompt(style_guide, title, memo, conversation, sources)
        article_html = call_claude(system, user, model=model)
        return jsonify({"article": article_html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/contexts", methods=["GET"])
def list_contexts():
    contexts = load_contexts()
    return jsonify({"contexts": contexts})


@app.route("/api/contexts", methods=["POST"])
def create_context():
    data = request.json
    name = data.get("name", "").strip()
    references = data.get("references", [])
    style_guide = data.get("style_guide", None)

    if not name:
        return jsonify({"error": "名前を入力してください"}), 400
    if not references or not any(r.strip() for r in references):
        return jsonify({"error": "リファレンス記事を1つ以上入力してください"}), 400

    context = {
        "id": str(uuid.uuid4()),
        "name": name,
        "references": references,
        "style_guide": style_guide,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    contexts = load_contexts()
    contexts.insert(0, context)
    save_contexts(contexts)

    return jsonify({"context": context})


@app.route("/api/contexts/<context_id>", methods=["PUT"])
def update_context(context_id):
    data = request.json
    contexts = load_contexts()

    target = None
    for ctx in contexts:
        if ctx["id"] == context_id:
            target = ctx
            break

    if not target:
        return jsonify({"error": "コンテキストが見つかりません"}), 404

    if "name" in data:
        target["name"] = data["name"]
    if "references" in data:
        target["references"] = data["references"]
    if "style_guide" in data:
        target["style_guide"] = data["style_guide"]

    target["updated_at"] = datetime.now().isoformat()
    save_contexts(contexts)

    return jsonify({"context": target})


@app.route("/api/contexts/<context_id>/reference/<int:ref_index>", methods=["PUT"])
def update_single_reference(context_id, ref_index):
    """個別のリファレンス記事を上書きする"""
    data = request.json
    new_text = data.get("text", "")

    contexts = load_contexts()
    target = None
    for ctx in contexts:
        if ctx["id"] == context_id:
            target = ctx
            break

    if not target:
        return jsonify({"error": "コンテキストが見つかりません"}), 404

    refs = target["references"]
    while len(refs) <= ref_index:
        refs.append("")
    refs[ref_index] = new_text

    target["updated_at"] = datetime.now().isoformat()
    target["style_guide"] = None
    save_contexts(contexts)

    return jsonify({"context": target})


@app.route("/api/contexts/<context_id>", methods=["DELETE"])
def delete_context(context_id):
    contexts = load_contexts()
    contexts = [c for c in contexts if c["id"] != context_id]
    save_contexts(contexts)
    return jsonify({"success": True})


@app.route("/api/fetch-url", methods=["POST"])
def fetch_url():
    """URLからAPI経由でテキストを取得する"""
    data = request.json
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "URLを入力してください"}), 400

    try:
        result = fetch_sync(url)
        text = result.get("text", "")
        meta = result.get("meta", {})

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
def rewrite_start():
    data = request.json
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    model = data.get("model")

    try:
        system, user = build_rewrite_interview_prompt(style_guide, original_article, user_angle)
        ai_message = call_claude(system, user, model=model)
        return jsonify({"message": ai_message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rewrite/continue", methods=["POST"])
def rewrite_continue():
    data = request.json
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    conversation = data.get("conversation", [])
    model = data.get("model")

    try:
        system, user = build_rewrite_followup_prompt(style_guide, original_article, user_angle, conversation)
        ai_message = call_claude(system, user, model=model)
        ready = "素材が揃いました" in ai_message
        return jsonify({"message": ai_message, "ready_to_write": ready})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/rewrite/generate", methods=["POST"])
def rewrite_generate():
    data = request.json
    style_guide = data.get("style_guide", {})
    original_article = data.get("original_article", "")
    user_angle = data.get("user_angle", "")
    conversation = data.get("conversation", [])
    sources = data.get("sources", [])
    model = data.get("model")

    try:
        system, user = build_rewrite_article_prompt(style_guide, original_article, user_angle, conversation, sources)
        article_html = call_claude(system, user, model=model)
        return jsonify({"article": article_html})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/extract-source", methods=["POST"])
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
def list_articles():
    articles = load_articles()
    return jsonify({"articles": articles})


@app.route("/api/articles", methods=["POST"])
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

    article = {
        "id": str(uuid.uuid4()),
        "title": title,
        "html": html,
        "text": text,
        "memo": memo,
        "conversation": conversation,
        "context_id": context_id,
        "created_at": datetime.now().isoformat(),
    }

    articles = load_articles()
    articles.insert(0, article)
    save_articles(articles)

    return jsonify({"article": article})


@app.route("/api/articles/<article_id>", methods=["PUT"])
def update_article(article_id):
    data = request.json
    articles = load_articles()
    idx = next((i for i, a in enumerate(articles) if a["id"] == article_id), None)
    if idx is None:
        return jsonify({"error": "記事が見つかりません"}), 404

    articles[idx]["html"] = data.get("html", articles[idx].get("html", ""))
    articles[idx]["text"] = data.get("text", articles[idx].get("text", ""))
    if data.get("title"):
        articles[idx]["title"] = data["title"]
    articles[idx]["updated_at"] = datetime.now().isoformat()
    save_articles(articles)
    return jsonify({"article": articles[idx]})


@app.route("/api/articles/<article_id>", methods=["DELETE"])
def delete_article(article_id):
    articles = load_articles()
    articles = [a for a in articles if a["id"] != article_id]
    save_articles(articles)
    return jsonify({"success": True})


@app.route("/api/article/edit-selection", methods=["POST"])
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
