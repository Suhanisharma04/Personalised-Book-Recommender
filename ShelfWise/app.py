import os
import json
import sqlite3
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "shelfwise-secret-key"

API_KEY = "AIzaSyBd0aX3yGoCg2yGF6ezXiq27RGEtIJntMU"
BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.db")

# shared session keeps the connection alive between calls — much faster than opening new connections
gb_session = requests.Session()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS user_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            isbn TEXT NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT NOT NULL,
            rating INTEGER NOT NULL,
            rated_at TEXT NOT NULL,
            UNIQUE(user_id, isbn)
        );
        CREATE TABLE IF NOT EXISTS user_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            isbn TEXT NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT NOT NULL,
            review TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, isbn)
        );
        CREATE TABLE IF NOT EXISTS user_already_read (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            isbn TEXT NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT NOT NULL,
            marked_at TEXT NOT NULL,
            UNIQUE(user_id, isbn)
        );
        CREATE TABLE IF NOT EXISTS search_cache (
            query TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            matched_title TEXT,
            matched_author TEXT,
            cached_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS mood_cache (
            mood TEXT PRIMARY KEY,
            results_json TEXT NOT NULL,
            cached_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


@app.before_request
def setup():
    init_db()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


def parse_book(item):
    info = item.get("volumeInfo", {})

    isbn = ""
    for id in info.get("industryIdentifiers", []):
        if id["type"] in ("ISBN_13", "ISBN_10"):
            isbn = id["identifier"]
            break
    if not isbn:
        isbn = item.get("id", "")

    cover = None
    imgs = info.get("imageLinks", {})
    for size in ("large", "medium", "thumbnail", "smallThumbnail"):
        if imgs.get(size):
            cover = imgs[size].replace("http://", "https://")
            break

    title = info.get("title", "")
    author = ", ".join(info.get("authors", ["Unknown"]))
    genre = (info.get("categories") or [""])[0]

    if isbn:
        buy_link = f"https://www.amazon.co.uk/s?k={isbn}"
    else:
        buy_link = "https://www.amazon.co.uk/s?k=" + requests.utils.quote(f"{title} {author}")

    google_link = info.get("infoLink", "")

    return {
        "Book-Title": title,
        "Book-Author": author,
        "ISBN": isbn,
        "cover_url": cover,
        "genre": genre,
        "description": info.get("description", ""),
        "buy_link": buy_link,
        "google_link": google_link,
        "ratings_mean": info.get("averageRating"),
        "published_year": (info.get("publishedDate", "") or "")[:4],
    }


def fetch_pool_from_google(query):
    # try 3 progressively simpler queries — each different so we don't retry the same thing
    words = query.strip().split()
    attempts = [
        query,                          # full query e.g. "atomic habits james clear"
        " ".join(words[:3]),            # first 3 words e.g. "atomic habits james"
        words[0] if words else query,   # just first word e.g. "atomic"
    ]
    # remove duplicates while keeping order
    seen = set()
    unique_attempts = [a for a in attempts if a not in seen and not seen.add(a)]

    for q in unique_attempts:
        try:
            r = gb_session.get(BOOKS_URL, params={
                "q": q, "maxResults": 40, "key": API_KEY
            }, timeout=20)
            items = r.json().get("items", [])
            if items:
                return items
        except Exception:
            pass
    return []


def run_tfidf(query, pool):
    df = pd.DataFrame([parse_book(i) for i in pool])
    df = df[df["Book-Title"].str.strip() != ""].drop_duplicates(subset=["Book-Title"]).reset_index(drop=True)

    if len(df) < 2:
        return None

    df["search_text"] = (
        (df["Book-Title"].fillna("") + " ") * 3 +
        (df["Book-Author"].fillna("") + " ") * 2 +
        df["description"].fillna("")
    ).str.lower()

    all_texts = [query.lower()] + df["search_text"].tolist()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(all_texts)
    scores = cosine_similarity(matrix[0], matrix[1:]).ravel()
    top = np.argsort(-scores)[:20]
    return df.iloc[top].copy().reset_index(drop=True)


def get_recommendations(query, user_id=None):
    cache_key = query.strip().lower()

    # check cache first — return instantly if already searched before
    conn = get_db()
    cached = conn.execute("SELECT results_json, matched_title, matched_author FROM search_cache WHERE query=?", (cache_key,)).fetchone()
    conn.close()

    if cached:
        results = pd.DataFrame(json.loads(cached["results_json"]))
        matched_title = cached["matched_title"]
        matched_author = cached["matched_author"]
    else:
        # not cached — fetch from google with retry
        pool = fetch_pool_from_google(query)

        if not pool:
            raise ValueError(f"Couldn't find anything for '{query}'. Please try again.")

        seed_info = pool[0].get("volumeInfo", {})
        matched_title = seed_info.get("title", query)
        matched_author = ", ".join(seed_info.get("authors", []))

        # supplement with category books — optional, short timeout
        categories = seed_info.get("categories", [])
        if categories:
            try:
                cat = categories[0].split("/")[0].strip()
                r = gb_session.get(BOOKS_URL, params={
                    "q": f"subject:{cat}", "maxResults": 20, "key": API_KEY
                }, timeout=6)
                pool += r.json().get("items", [])
            except Exception:
                pass

        results = run_tfidf(query, pool)

        if results is None:
            raise ValueError(f"Not enough books found for '{query}'.")

        # save to cache so next search is instant
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO search_cache (query, results_json, matched_title, matched_author, cached_at) VALUES (?,?,?,?,?)",
            (cache_key, json.dumps(results.to_dict("records")), matched_title, matched_author, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

    # filter already-read books — done after cache so filter stays personal per user
    if user_id:
        conn = get_db()
        rows = conn.execute("SELECT isbn, book_title FROM user_already_read WHERE user_id=?", (user_id,)).fetchall()
        conn.close()
        if rows:
            read_isbns = {r["isbn"].lower() for r in rows if r["isbn"]}
            read_titles = {r["book_title"].lower().strip() for r in rows if r["book_title"]}
            mask = (
                results["ISBN"].str.lower().isin(read_isbns) |
                results["Book-Title"].str.lower().str.strip().isin(read_titles)
            )
            results = results[~mask].reset_index(drop=True)

    return matched_title, matched_author, results


def get_user_book_data(user_id, isbns):
    if not isbns:
        return {}
    conn = get_db()
    ph = ",".join("?" * len(isbns))
    args = [user_id] + list(isbns)
    ratings = {r["isbn"]: r["rating"] for r in conn.execute(f"SELECT isbn, rating FROM user_ratings WHERE user_id=? AND isbn IN ({ph})", args).fetchall()}
    already_read = {r["isbn"] for r in conn.execute(f"SELECT isbn FROM user_already_read WHERE user_id=? AND isbn IN ({ph})", args).fetchall()}
    reviews = {r["isbn"]: r["review"] for r in conn.execute(f"SELECT isbn, review FROM user_reviews WHERE user_id=? AND isbn IN ({ph})", args).fetchall()}
    conn.close()
    return {isbn: {"rating": ratings.get(isbn, 0), "already_read": isbn in already_read, "review": reviews.get(isbn, "")} for isbn in isbns}


@app.route("/", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    errors = {}
    email = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not email:
            errors["email"] = "Email is required."
        if not password:
            errors["password"] = "Password is required."
        if not errors:
            conn = get_db()
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
            conn.close()
            if not user or not check_password_hash(user["password_hash"], password):
                errors["password"] = "Incorrect email or password."
            else:
                session["user_id"] = user["id"]
                session["user_name"] = user["name"]
                session["user_email"] = user["email"]
                return redirect(url_for("dashboard"))
    return render_template("login.html", errors=errors, email=email)


@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    errors = {}
    name = email = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not name:
            errors["name"] = "Name is required."
        if not email:
            errors["email"] = "Email is required."
        if not password:
            errors["password"] = "Password is required."
        elif len(password) < 6:
            errors["password"] = "Password must be at least 6 characters."
        if not errors:
            try:
                conn = get_db()
                conn.execute("INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
                             (name, email, generate_password_hash(password), datetime.utcnow().isoformat()))
                conn.commit()
                conn.close()
                flash("Account created! Please log in.", "success")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                errors["email"] = "An account with that email already exists."
    return render_template("register.html", errors=errors, email=email, name=name)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    user_id = session["user_id"]
    query = ""
    matched_title = matched_author = None
    recs = []
    user_data = {}
    error = None

    if request.method == "POST":
        query = request.form.get("title", "").strip()
        if not query:
            error = "Please enter a book title."
        else:
            try:
                matched_title, matched_author, df = get_recommendations(query, user_id)
                recs = df.to_dict("records")
                user_data = get_user_book_data(user_id, [r["ISBN"] for r in recs if r.get("ISBN")])
            except ValueError as e:
                error = str(e)
            except Exception as e:
                error = f"Something went wrong: {e}"

    return render_template("dashboard.html",
        user_name=session.get("user_name", "there"),
        user_email=session.get("user_email", ""),
        query=query, matched_title=matched_title,
        matched_author=matched_author, recs=recs,
        user_data=user_data, error=error)


@app.route("/favourites")
@login_required
def favourites():
    user_id = session["user_id"]
    conn = get_db()
    rated = conn.execute("SELECT isbn, book_title, book_author, rating, rated_at FROM user_ratings WHERE user_id=? ORDER BY rated_at DESC", (user_id,)).fetchall()
    reviews = {r["isbn"]: r["review"] for r in conn.execute("SELECT isbn, review FROM user_reviews WHERE user_id=?", (user_id,)).fetchall()}
    conn.close()

    books = []
    for row in rated:
        buy_link = "https://www.amazon.co.uk/s?k=" + requests.utils.quote(f"{row['book_title']} {row['book_author']}")
        books.append({
            "isbn": row["isbn"],
            "title": row["book_title"],
            "author": row["book_author"],
            "rating": row["rating"],
            "rated_at": row["rated_at"][:10],
            "review": reviews.get(row["isbn"], ""),
            "buy_link": buy_link,
        })

    return render_template("favourites.html",
        user_name=session.get("user_name", "there"),
        user_email=session.get("user_email", ""),
        books=books)


# new page showing all books marked as already read
@app.route("/already_read")
@login_required
def already_read():
    user_id = session["user_id"]
    conn = get_db()
    books = conn.execute(
        "SELECT isbn, book_title, book_author, marked_at FROM user_already_read WHERE user_id=? ORDER BY marked_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()

    book_list = []
    for row in books:
        buy_link = "https://www.amazon.co.uk/s?k=" + requests.utils.quote(f"{row['book_title']} {row['book_author']}")
        book_list.append({
            "isbn": row["isbn"],
            "title": row["book_title"],
            "author": row["book_author"],
            "marked_at": row["marked_at"][:10],
            "buy_link": buy_link,
        })

    return render_template("already_read.html",
        user_name=session.get("user_name", "there"),
        user_email=session.get("user_email", ""),
        books=book_list)


@app.route("/api/suggest")
@login_required
def api_suggest():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    try:
        res = gb_session.get(BOOKS_URL, params={"q": q, "maxResults": 6, "key": API_KEY}, timeout=4)
        titles = []
        seen = set()
        for item in res.json().get("items", []):
            t = item.get("volumeInfo", {}).get("title", "")
            if t and t not in seen:
                titles.append(t)
                seen.add(t)
        return jsonify(titles[:6])
    except Exception:
        return jsonify([])


@app.route("/api/rate", methods=["POST"])
@login_required
def api_rate():
    d = request.get_json() or {}
    isbn, title, author, rating = d.get("isbn","").strip(), d.get("title","").strip(), d.get("author","").strip(), d.get("rating")
    if not isbn or not isinstance(rating, int) or not 1 <= rating <= 5:
        return jsonify({"ok": False}), 400
    conn = get_db()
    conn.execute("INSERT INTO user_ratings (user_id,isbn,book_title,book_author,rating,rated_at) VALUES (?,?,?,?,?,?) ON CONFLICT(user_id,isbn) DO UPDATE SET rating=excluded.rating, rated_at=excluded.rated_at",
                 (session["user_id"], isbn, title, author, rating, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/review", methods=["POST"])
@login_required
def api_review():
    d = request.get_json() or {}
    isbn, title, author, review = d.get("isbn","").strip(), d.get("title","").strip(), d.get("author","").strip(), d.get("review","").strip()
    if not isbn or not review:
        return jsonify({"ok": False, "error": "Review cannot be empty."}), 400
    if len(review) > 500:
        return jsonify({"ok": False, "error": "Keep it under 500 characters."}), 400
    conn = get_db()
    conn.execute("INSERT INTO user_reviews (user_id,isbn,book_title,book_author,review,created_at) VALUES (?,?,?,?,?,?) ON CONFLICT(user_id,isbn) DO UPDATE SET review=excluded.review, created_at=excluded.created_at",
                 (session["user_id"], isbn, title, author, review, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# delete a user's own review
@app.route("/api/review/delete", methods=["POST"])
@login_required
def api_review_delete():
    d = request.get_json() or {}
    isbn = d.get("isbn", "").strip()
    if not isbn:
        return jsonify({"ok": False}), 400
    conn = get_db()
    conn.execute("DELETE FROM user_reviews WHERE user_id=? AND isbn=?", (session["user_id"], isbn))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/api/already_read", methods=["POST"])
@login_required
def api_already_read():
    d = request.get_json() or {}
    isbn, title, author = d.get("isbn","").strip(), d.get("title","").strip(), d.get("author","").strip()
    if not isbn:
        return jsonify({"ok": False}), 400
    conn = get_db()
    exists = conn.execute("SELECT id FROM user_already_read WHERE user_id=? AND isbn=?", (session["user_id"], isbn)).fetchone()
    if exists:
        conn.execute("DELETE FROM user_already_read WHERE user_id=? AND isbn=?", (session["user_id"], isbn))
        marked = False
    else:
        conn.execute("INSERT OR IGNORE INTO user_already_read (user_id,isbn,book_title,book_author,marked_at) VALUES (?,?,?,?,?)",
                     (session["user_id"], isbn, title, author, datetime.utcnow().isoformat()))
        marked = True
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "marked": marked})


@app.route("/api/book_detail")
@login_required
def api_book_detail():
    title = request.args.get("title", "").strip()
    author = request.args.get("author", "").strip()
    isbn = request.args.get("isbn", "").strip()
    if not title:
        return jsonify({"ok": False}), 400

    try:
        # try progressively looser searches until we find the book
        searches = [
            f"intitle:{title} inauthor:{author}",
            f"intitle:{title}",
            title,
        ]
        items = []
        for query in searches:
            res = gb_session.get(BOOKS_URL, params={"q": query, "maxResults": 1, "key": API_KEY}, timeout=5)
            items = res.json().get("items", [])
            if items:
                break
        if not items:
            return jsonify({"ok": False, "error": "Book not found."})

        info = items[0].get("volumeInfo", {})

        cover = None
        for size in ("large", "medium", "thumbnail", "smallThumbnail"):
            if info.get("imageLinks", {}).get(size):
                cover = info["imageLinks"][size].replace("http://", "https://")
                break

        desc = info.get("description", "")
        description = (desc[:400] + "...") if len(desc) > 400 else desc

        similar = []
        if author:
            try:
                ar = gb_session.get(BOOKS_URL, params={"q": f"inauthor:{author}", "maxResults": 5, "key": API_KEY}, timeout=4)
                for item in ar.json().get("items", []):
                    i = item.get("volumeInfo", {})
                    if i.get("title", "").lower() != title.lower():
                        c = (i.get("imageLinks") or {}).get("thumbnail", "")
                        similar.append({"title": i.get("title", ""), "cover_url": c.replace("http://", "https://") if c else None})
                similar = similar[:4]
            except Exception:
                pass

        user_review = ""
        all_reviews = []
        if isbn:
            conn = get_db()
            row = conn.execute("SELECT review FROM user_reviews WHERE user_id=? AND isbn=?", (session["user_id"], isbn)).fetchone()
            if row:
                user_review = row["review"]
            rows = conn.execute("SELECT u.name, r.review, r.created_at FROM user_reviews r JOIN users u ON u.id = r.user_id WHERE r.isbn=? ORDER BY r.created_at DESC", (isbn,)).fetchall()
            conn.close()
            all_reviews = [{"name": r["name"], "review": r["review"], "date": r["created_at"][:10]} for r in rows]

        buy_link = f"https://www.amazon.co.uk/s?k={isbn}" if isbn else "https://www.amazon.co.uk/s?k=" + requests.utils.quote(f"{title} {author}")
        google_link = info.get("infoLink", "")

        return jsonify({
            "ok": True,
            "cover_url": cover,
            "year": (info.get("publishedDate") or "")[:4] or None,
            "pages": info.get("pageCount"),
            "subjects": (info.get("categories") or [])[:8],
            "description": description,
            "similar": similar,
            "user_review": user_review,
            "all_reviews": all_reviews,
            "buy_link": buy_link,
            "google_link": google_link,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


MOODS = {
    "inspired":  "subject:inspiration motivational self-help",
    "mystery":   "subject:mystery detective",
    "learn":     "subject:nonfiction education science",
    "romance":   "subject:romance love story",
    "fantasy":   "subject:fantasy magic",
    "scifi":     "subject:science fiction space",
    "thriller":  "subject:thriller suspense crime",
    "history":   "subject:history biography",
    "horror":    "subject:horror supernatural",
    "adventure": "subject:adventure action",
    "selfhelp":  "subject:self-help personal development",
}


@app.route("/api/mood/<mood>")
@login_required
def api_mood(mood):
    query = MOODS.get(mood.lower())
    if not query:
        return jsonify({"ok": False, "error": "Unknown mood."}), 400

    # check mood cache first — return instantly if already loaded before
    conn = get_db()
    cached = conn.execute("SELECT results_json FROM mood_cache WHERE mood=?", (mood.lower(),)).fetchone()
    conn.close()

    if cached:
        recs = json.loads(cached["results_json"])
    else:
        # not cached — try up to 2 times with 20 second timeout each
        recs = []
        for attempt in range(2):
            try:
                res = gb_session.get(BOOKS_URL, params={
                    "q": query, "maxResults": 12, "orderBy": "relevance", "key": API_KEY
                }, timeout=20)
                items = res.json().get("items", [])
                if items:
                    recs = [parse_book(item) for item in items if item.get("volumeInfo", {}).get("title")]
                    break
            except Exception:
                pass

        if not recs:
            return jsonify({"ok": False, "error": "Couldn't load mood picks. Please try again."}), 502

        # save to cache so next click is instant
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO mood_cache (mood, results_json, cached_at) VALUES (?,?,?)",
            (mood.lower(), json.dumps(recs), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

    # filter already-read books
    user_id = session["user_id"]
    conn = get_db()
    rows = conn.execute("SELECT isbn, book_title FROM user_already_read WHERE user_id=?", (user_id,)).fetchall()
    conn.close()
    if rows:
        read_isbns = {r["isbn"].lower() for r in rows if r["isbn"]}
        read_titles = {r["book_title"].lower().strip() for r in rows if r["book_title"]}
        recs = [r for r in recs if
                r["ISBN"].lower() not in read_isbns and
                r["Book-Title"].lower().strip() not in read_titles]

    return jsonify({"ok": True, "recs": recs[:12]})


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)