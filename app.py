"""CodeTitans Attendance - face-recognition attendance system.

Stack: Flask + face_recognition (dlib) + SQLite. Runs fully on the local machine.

Privacy by design:
  * Nothing leaves this computer; the server binds to 127.0.0.1 only.
  * Registration requires explicit consent (timestamp stored).
  * Only 128-number face encodings are stored, NOT photos.
  * Any person can be deleted (face data + attendance) with one click.
  * Manual attendance exists as a fallback for people who opt out of face scanning.

Features:
  * Consent-based registration (and duplicate-face protection)
  * Live multi-face recognition with confirmation over consecutive frames
  * Optional blink-based liveness check (stops printed-photo spoofing)
  * "Late after HH:MM" marking, manual mark / undo
  * Dashboard stats, 7-day chart, per-person attendance %, CSV export
"""
import base64
import csv
import io
import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta

import face_recognition
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image

APP_NAME = "CodeTitans"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE_DIR, "attendance.db")

TOLERANCE = 0.5        # lower = stricter face matching
CONFIRM_FRAMES = 2     # consecutive recognitions before marking present
MAX_WIDTH = 400        # images are downscaled to this width for speed
EAR_CLOSED = 0.21      # eye-aspect-ratio below this = eyes closed
BLINK_VALID_S = 8      # a blink counts as proof of life for this many seconds

app = Flask(__name__)
streaks = {}           # person_id -> consecutive match count
liveness = {}          # person_id -> {"closed": bool, "last": timestamp of last blink}


# ---------------------------------------------------------------- database
@contextmanager
def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS people(
            id TEXT PRIMARY KEY, name TEXT NOT NULL,
            encodings TEXT NOT NULL, consent_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance(
            person_id TEXT NOT NULL, day TEXT NOT NULL, time TEXT NOT NULL,
            method TEXT DEFAULT 'face', PRIMARY KEY(person_id, day))""")
        c.execute("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT)")
        # migration for databases created by the first version (no 'method' column)
        cols = [r[1] for r in c.execute("PRAGMA table_info(attendance)")]
        if "method" not in cols:
            c.execute("ALTER TABLE attendance ADD COLUMN method TEXT DEFAULT 'face'")


def get_setting(key, default=""):
    with db() as c:
        r = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return r["value"] if r else default


def set_setting(key, value):
    with db() as c:
        c.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?, ?)", (key, value))


# ----------------------------------------------------------------- helpers
def decode_image(data_url):
    try:
        raw = base64.b64decode(data_url.split(",", 1)[1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid image data.") from exc
    if img.width > MAX_WIDTH:
        h = int(img.height * MAX_WIDTH / img.width)
        img = img.resize((MAX_WIDTH, h))
    return np.array(img)


def load_known():
    with db() as c:
        rows = c.execute("SELECT id, name, encodings FROM people").fetchall()
    ids, names, encs = [], [], []
    for r in rows:
        for e in json.loads(r["encodings"]):
            ids.append(r["id"])
            names.append(r["name"])
            encs.append(np.array(e))
    return ids, names, encs


def eye_aspect_ratio(eye):
    """Eye aspect ratio from the 6 eye landmarks (dlib order). ~0.3 open, <0.2 closed."""
    p = [np.array(pt, dtype=float) for pt in eye]
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal == 0:
        return 1.0
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horizontal)


def is_late(time_str, cutoff):
    return bool(cutoff and time_str and time_str[:5] > cutoff)


def valid_day(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return datetime.now().strftime("%Y-%m-%d")


def csv_safe(value):
    """Prevent spreadsheet formula injection from names like '=cmd|...'."""
    s = str(value)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def day_rows(day):
    cutoff = get_setting("late_after")
    with db() as c:
        rows = c.execute("""SELECT p.id, p.name, a.time, a.method FROM people p
            LEFT JOIN attendance a ON a.person_id = p.id AND a.day = ?
            ORDER BY p.name""", (day,)).fetchall()
    out = []
    for r in rows:
        t = r["time"]
        status = "absent" if not t else ("late" if is_late(t, cutoff) else "present")
        out.append(dict(id=r["id"], name=r["name"], time=t, method=r["method"], status=status))
    return out


@app.errorhandler(ValueError)
def handle_value_error(err):
    return jsonify(error=str(err)), 400


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    return render_template("index.html", app_name=APP_NAME)


@app.post("/api/register")
def register():
    d = request.get_json(silent=True) or {}
    pid, name = (d.get("id") or "").strip(), (d.get("name") or "").strip()
    if not d.get("consent"):
        return jsonify(error="Consent is required to register."), 400
    if not pid or not name:
        return jsonify(error="ID and name are required."), 400
    if len(pid) > 32 or len(name) > 80:
        return jsonify(error="ID or name is too long."), 400
    with db() as c:
        if c.execute("SELECT 1 FROM people WHERE id=?", (pid,)).fetchone():
            return jsonify(error="That ID is already registered."), 400
    encodings = []
    for url in (d.get("images") or [])[:10]:
        img = decode_image(url)
        locs = face_recognition.face_locations(img)
        if len(locs) != 1:
            continue  # need exactly one clear face
        encodings.append(face_recognition.face_encodings(img, locs)[0].tolist())
    if len(encodings) < 3:
        return jsonify(error="Could not capture a clear single face. Try better lighting."), 400

    # block registering the same face under a second ID (proxy-attendance protection)
    ids, names, known = load_known()
    if known:
        dist = face_recognition.face_distance(known, np.array(encodings[0]))
        i = int(np.argmin(dist))
        if dist[i] <= TOLERANCE:
            return jsonify(error=f"This face is already registered as {names[i]} ({ids[i]})."), 400

    with db() as c:
        c.execute("INSERT INTO people VALUES(?,?,?,?)",
                  (pid, name, json.dumps(encodings), datetime.now().isoformat(timespec="seconds")))
    return jsonify(ok=True)


@app.post("/api/recognize")
def recognize():
    d = request.get_json(silent=True) or {}
    img = decode_image(d.get("image", ""))
    need_live = bool(d.get("liveness"))

    locs = face_recognition.face_locations(img)
    encs = face_recognition.face_encodings(img, locs)
    lms = face_recognition.face_landmarks(img, locs) if (need_live and locs) else [None] * len(locs)
    ids, names, known = load_known()

    now, dt = time.time(), datetime.now()
    day, clock = dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S")
    faces, seen, new_marks = [], set(), []

    for (top, right, bottom, left), enc, lm in zip(locs, encs, lms):
        pid, name, state = None, "Unknown", "unknown"
        if known:
            dist = face_recognition.face_distance(known, enc)
            i = int(np.argmin(dist))
            if dist[i] <= TOLERANCE:
                pid, name = ids[i], names[i]

        if pid:
            seen.add(pid)
            streaks[pid] = streaks.get(pid, 0) + 1
            alive = True
            if need_live:
                st = liveness.setdefault(pid, {"closed": False, "last": 0.0})
                if lm and "left_eye" in lm and "right_eye" in lm:
                    ear = (eye_aspect_ratio(lm["left_eye"]) + eye_aspect_ratio(lm["right_eye"])) / 2
                else:
                    ear = 1.0
                if ear < EAR_CLOSED:
                    st["closed"] = True
                elif st["closed"]:          # eyes re-opened -> that was a blink
                    st["closed"] = False
                    st["last"] = now
                alive = (now - st["last"]) <= BLINK_VALID_S

            if streaks[pid] < CONFIRM_FRAMES:
                state = "detecting"
            elif not alive:
                state = "blink"             # recognised, waiting for a blink
            else:
                state = "marked"
                with db() as c:
                    cur = c.execute(
                        "INSERT OR IGNORE INTO attendance(person_id, day, time, method) "
                        "VALUES(?, ?, ?, 'face')", (pid, day, clock))
                    if cur.rowcount:
                        new_marks.append(dict(id=pid, name=name, time=clock))

        faces.append(dict(name=name, id=pid, state=state,
                          box=[left, top, right - left, bottom - top]))

    for k in list(streaks):
        if k not in seen:
            streaks[k] = 0
            if k in liveness:
                liveness[k]["closed"] = False

    return jsonify(faces=faces, new_marks=new_marks,
                   width=img.shape[1], height=img.shape[0])


@app.get("/api/people")
def people():
    with db() as c:
        total_days = c.execute("SELECT COUNT(DISTINCT day) FROM attendance").fetchone()[0]
        rows = c.execute("""SELECT p.id, p.name, p.consent_at, COUNT(a.day) AS days
            FROM people p LEFT JOIN attendance a ON a.person_id = p.id
            GROUP BY p.id ORDER BY p.name""").fetchall()
    return jsonify([dict(id=r["id"], name=r["name"], consent_at=r["consent_at"], days=r["days"],
                         percent=round(100 * r["days"] / total_days) if total_days else 0)
                    for r in rows])


@app.delete("/api/people/<pid>")
def delete_person(pid):
    with db() as c:
        c.execute("DELETE FROM attendance WHERE person_id=?", (pid,))
        c.execute("DELETE FROM people WHERE id=?", (pid,))
    streaks.pop(pid, None)
    liveness.pop(pid, None)
    return jsonify(ok=True)


@app.get("/api/records")
def records():
    return jsonify(day_rows(valid_day(request.args.get("date"))))


@app.get("/api/stats")
def stats():
    today = datetime.now().strftime("%Y-%m-%d")
    rows = day_rows(today)
    total = len(rows)
    present = sum(1 for r in rows if r["status"] != "absent")
    late = sum(1 for r in rows if r["status"] == "late")
    last7 = []
    with db() as c:
        for i in range(6, -1, -1):
            d = datetime.now() - timedelta(days=i)
            key = d.strftime("%Y-%m-%d")
            n = c.execute("""SELECT COUNT(*) FROM attendance a
                JOIN people p ON p.id = a.person_id WHERE a.day = ?""", (key,)).fetchone()[0]
            last7.append(dict(day=key, label=d.strftime("%a"), present=n))
    return jsonify(registered=total, present=present, late=late, absent=total - present,
                   percent=round(100 * present / total) if total else 0,
                   last7=last7, late_after=get_setting("late_after"))


@app.post("/api/manual")
def manual_mark():
    d = request.get_json(silent=True) or {}
    pid = (d.get("id") or "").strip()
    day = valid_day(d.get("date"))
    today = datetime.now().strftime("%Y-%m-%d")
    with db() as c:
        if not c.execute("SELECT 1 FROM people WHERE id=?", (pid,)).fetchone():
            return jsonify(error="Unknown person."), 404
        t = datetime.now().strftime("%H:%M:%S") if day == today else "00:00:00"
        c.execute("INSERT OR IGNORE INTO attendance(person_id, day, time, method) "
                  "VALUES(?, ?, ?, 'manual')", (pid, day, t))
    return jsonify(ok=True)


@app.delete("/api/attendance/<pid>")
def undo_mark(pid):
    day = valid_day(request.args.get("date"))
    with db() as c:
        c.execute("DELETE FROM attendance WHERE person_id=? AND day=?", (pid, day))
    return jsonify(ok=True)


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        value = ((request.get_json(silent=True) or {}).get("late_after") or "").strip()
        if value and not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value):
            return jsonify(error="Time must look like HH:MM."), 400
        set_setting("late_after", value)
    return jsonify(late_after=get_setting("late_after"))


@app.get("/api/export.csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "Name", "Date", "Status", "Time", "Method"])
    if request.args.get("all"):
        with db() as c:
            days = [r[0] for r in c.execute("SELECT DISTINCT day FROM attendance ORDER BY day")]
        filename = "attendance-all.csv"
    else:
        days = [valid_day(request.args.get("date"))]
        filename = f"attendance-{days[0]}.csv"
    for day in days:
        for r in day_rows(day):
            w.writerow([csv_safe(r["id"]), csv_safe(r["name"]), day, r["status"].capitalize(),
                        r["time"] or "", r["method"] or "" if r["time"] else ""])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@app.post("/api/wipe")
def wipe():
    with db() as c:
        c.execute("DELETE FROM attendance")
        c.execute("DELETE FROM people")
    streaks.clear()
    liveness.clear()
    return jsonify(ok=True)


init_db()

if __name__ == "__main__":
    print(f"{APP_NAME} Attendance running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
