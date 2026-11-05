"""SQLite persistence with parameterized queries and local image references."""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from .config import DATABASE_PATH


def now():return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self,path=DATABASE_PATH):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as connection:
            connection.executescript('''
            CREATE TABLE IF NOT EXISTS predictions(
              id TEXT PRIMARY KEY,created_at TEXT NOT NULL,image_path TEXT NOT NULL,image_sha256 TEXT NOT NULL,
              top_class TEXT,confidence REAL,needs_review INTEGER NOT NULL,result_json TEXT NOT NULL,quality_json TEXT NOT NULL,model_sha256 TEXT);
            CREATE TABLE IF NOT EXISTS feedback(
              id INTEGER PRIMARY KEY AUTOINCREMENT,prediction_id TEXT NOT NULL REFERENCES predictions(id),created_at TEXT NOT NULL,
              verdict TEXT NOT NULL,corrected_class TEXT,comment TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS reviews(
              prediction_id TEXT PRIMARY KEY REFERENCES predictions(id),status TEXT NOT NULL,reviewer_note TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS species_records(
              class_label TEXT PRIMARY KEY,record_json TEXT NOT NULL,updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS species_drafts(
              id INTEGER PRIMARY KEY AUTOINCREMENT,class_label TEXT NOT NULL,note TEXT NOT NULL,source_url TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS model_versions(
              model_sha256 TEXT PRIMARY KEY,version TEXT NOT NULL,classes_json TEXT NOT NULL,roles_json TEXT NOT NULL,registered_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS prediction_created ON predictions(created_at DESC);
            CREATE INDEX IF NOT EXISTS prediction_class ON predictions(top_class);
            CREATE INDEX IF NOT EXISTS feedback_prediction ON feedback(prediction_id,id DESC);
            ''')

    @contextmanager
    def connection(self):
        connection=sqlite3.connect(self.path,timeout=20)
        connection.row_factory=sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback();raise
        finally:connection.close()

    def sync_species(self,records):
        with self.connection() as connection:
            connection.executemany("INSERT INTO species_records VALUES(?,?,?) ON CONFLICT(class_label) DO UPDATE SET record_json=excluded.record_json,updated_at=excluded.updated_at",
                                   [(row["class_label"],json.dumps(row),now()) for row in records])

    def register_model(self,model_hash,version,classes,roles):
        """Append model provenance without modifying historical predictions or feedback."""
        with self.connection() as connection:
            existing=connection.execute("SELECT * FROM model_versions WHERE model_sha256=?",(model_hash,)).fetchone()
            values=(version,json.dumps(classes),json.dumps(roles,sort_keys=True))
            if existing and (existing["version"],existing["classes_json"],existing["roles_json"])!=values:
                raise ValueError("A model hash cannot be registered with a different mapping")
            connection.execute("INSERT OR IGNORE INTO model_versions VALUES(?,?,?,?,?)",(model_hash,*values,now()))

    def save_prediction(self,image_path,image_hash,result,quality,model_hash):
        identifier=str(uuid.uuid4())
        best=result["predictions"][0]
        with self.connection() as connection:
            connection.execute("INSERT INTO predictions VALUES(?,?,?,?,?,?,?,?,?,?)",
                               (identifier,now(),str(image_path),image_hash,best["class_label"],best["confidence"],int(result["needs_review"]),json.dumps(result),json.dumps(quality),model_hash))
            connection.execute("INSERT INTO reviews VALUES(?,?,?,?)",(identifier,"pending" if result["needs_review"] else "unreviewed","",now()))
        return identifier

    def add_feedback(self,prediction_id,verdict,corrected_class=None,comment=""):
        if verdict not in {"Looks correct","Incorrect","Unsure"}:raise ValueError("Invalid feedback verdict")
        if len(comment)>4000:raise ValueError("Feedback comment is too long")
        with self.connection() as connection:
            connection.execute("INSERT INTO feedback(prediction_id,created_at,verdict,corrected_class,comment) VALUES(?,?,?,?,?)",(prediction_id,now(),verdict,corrected_class,comment))
            if verdict!="Looks correct":
                connection.execute("UPDATE predictions SET needs_review=1 WHERE id=?",(prediction_id,))
                connection.execute("UPDATE reviews SET status='pending',updated_at=? WHERE prediction_id=?",(now(),prediction_id))

    def history(self,limit=200,flagged_only=False,search="",species=None,date_from=None,date_to=None,review_status=None,low_confidence=None):
        limit=min(max(int(limit),1),2000)
        clauses=[];parameters=[]
        if flagged_only:clauses.append("p.needs_review=1")
        if search:
            clauses.append("(p.top_class LIKE ? ESCAPE '\\' OR p.id LIKE ? ESCAPE '\\')")
            term="%"+search.replace("\\","\\\\").replace("%","\\%").replace("_","\\_")+"%"
            parameters.extend([term,term])
        if species:clauses.append("p.top_class=?");parameters.append(species)
        if date_from:clauses.append("substr(p.created_at,1,10)>=?");parameters.append(str(date_from))
        if date_to:clauses.append("substr(p.created_at,1,10)<=?");parameters.append(str(date_to))
        if review_status:clauses.append("r.status=?");parameters.append(review_status)
        if low_confidence is not None:clauses.append("p.confidence<?");parameters.append(float(low_confidence))
        where="WHERE "+" AND ".join(clauses) if clauses else ""
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(f"""SELECT p.*,r.status AS review_status,r.reviewer_note,
                (SELECT verdict FROM feedback f WHERE f.prediction_id=p.id ORDER BY id DESC LIMIT 1) AS feedback_status,
                (SELECT corrected_class FROM feedback f WHERE f.prediction_id=p.id ORDER BY id DESC LIMIT 1) AS corrected_class
                FROM predictions p LEFT JOIN reviews r ON r.prediction_id=p.id {where} ORDER BY p.created_at DESC LIMIT ?""",[*parameters,limit])]

    def get_prediction(self,identifier):
        with self.connection() as connection:
            row=connection.execute("SELECT * FROM predictions WHERE id=?",(identifier,)).fetchone()
            return dict(row) if row else None

    def stats(self):
        with self.connection() as connection:
            counts={name:connection.execute("SELECT COUNT(*) FROM "+name).fetchone()[0] for name in ["predictions","feedback","species_records","species_drafts"]}
            counts["flagged"]=connection.execute("SELECT COUNT(*) FROM predictions WHERE needs_review=1").fetchone()[0]
            counts["reviewed"]=connection.execute("SELECT COUNT(*) FROM reviews WHERE status IN ('confirmed','rejected')").fetchone()[0]
            counts["pending"]=connection.execute("SELECT COUNT(*) FROM reviews WHERE status IN ('pending','needs expert','unreviewed')").fetchone()[0]
            return counts

    def all_prediction_ids(self):
        with self.connection() as connection:return [row[0] for row in connection.execute("SELECT id FROM predictions")]

    def delete_predictions(self,identifiers,uploads=None):
        """Delete explicit records transactionally, then remove unreferenced retained photos within uploads."""
        identifiers=list(dict.fromkeys(identifiers))
        if not identifiers:return {"deleted":0,"images_removed":0,"cleanup_warnings":[]}
        paths=set();deleted=0
        with self.connection() as connection:
            for identifier in identifiers:
                row=connection.execute("SELECT image_path FROM predictions WHERE id=?",(identifier,)).fetchone()
                if not row:continue
                if row[0]:paths.add(row[0])
                connection.execute("DELETE FROM feedback WHERE prediction_id=?",(identifier,))
                connection.execute("DELETE FROM reviews WHERE prediction_id=?",(identifier,))
                deleted+=connection.execute("DELETE FROM predictions WHERE id=?",(identifier,)).rowcount
        warnings=[];removed=0
        if uploads is not None:
            root=Path(uploads).resolve()
            for value in paths:
                candidate=Path(value).resolve()
                if not candidate.is_relative_to(root) or candidate==root:continue
                with self.connection() as connection:
                    if connection.execute("SELECT 1 FROM predictions WHERE image_path=?",(value,)).fetchone():continue
                if candidate.is_file():
                    try:candidate.unlink();removed+=1
                    except OSError as error:warnings.append(str(error))
        return {"deleted":deleted,"images_removed":removed,"cleanup_warnings":warnings}

    def feedback(self):
        with self.connection() as connection:return [dict(row) for row in connection.execute("SELECT * FROM feedback ORDER BY id DESC")]

    def review(self,prediction_id,status,note):
        if status not in {"pending","confirmed","needs expert","rejected"}:raise ValueError("Invalid review status")
        with self.connection() as connection:
            connection.execute("UPDATE reviews SET status=?,reviewer_note=?,updated_at=? WHERE prediction_id=?",(status,note[:4000],now(),prediction_id))
            connection.execute("UPDATE predictions SET needs_review=? WHERE id=?",(int(status in {"pending","needs expert","rejected"}),prediction_id))

    def add_draft(self,class_label,note,source_url):
        from urllib.parse import urlparse
        parsed=urlparse(source_url)
        if parsed.scheme!="https" or not parsed.hostname:raise ValueError("Provide an HTTPS source URL")
        if not note.strip() or len(note)>4000:raise ValueError("Enter a concise source-backed review note")
        with self.connection() as connection:
            if not connection.execute("SELECT 1 FROM species_records WHERE class_label=?",(class_label,)).fetchone():raise ValueError("Unsupported species")
            connection.execute("INSERT INTO species_drafts(class_label,note,source_url,status,created_at) VALUES(?,?,?,?,?)",(class_label,note,source_url,"awaiting expert verification",now()))

    def drafts(self):
        with self.connection() as connection:return [dict(row) for row in connection.execute("SELECT * FROM species_drafts ORDER BY id DESC")]
