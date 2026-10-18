"""Text shown next to the club logo in the top-left of the annotated video: 'Video Analysis', the game's name, its date and its
description (all from the Game Logger record)."""
import datetime


def _clean(v):
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "nat", "undated") else " ".join(s.split())


def _date_text(v):
    s = _clean(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return f"{v.day} {v.strftime('%B %Y')}"
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            return f"{dt.day} {dt.strftime('%B %Y')}"
        except ValueError:
            pass
    return s


def build_header(rec, heading="Video Analysis"):
    """{'heading','title','date','description'} (empty strings when a part is missing)."""
    get = rec.get if hasattr(rec, "get") else (lambda k, d=None: d)
    ta, tb = _clean(get("team_a")), _clean(get("team_b"))
    title = _clean(get("title")) or (f"{ta} vs {tb}" if ta and tb else "")
    return {"heading": heading, "title": title, "date": _date_text(get("date")), "description": _clean(get("description"))}
