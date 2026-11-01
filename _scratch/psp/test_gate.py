import sys, os, sqlite3, json
sys.path.insert(0, os.getcwd())
exec(open("_scratch/psp/compare_report.py").read().split("import pandas")[0])
from game_logger import config, run_status as rs, identity_health as ih
conn=sqlite3.connect(f"file:{config.DB_PATH}?mode=ro",uri=True); conn.row_factory=sqlite3.Row
for gid in (25,27):
    print("== game",gid)
    ev=rs.evaluate(gid)
    for k,v in ev.items(): print(f"  {k:24s} {v['state']:11s} {v['at']} {v['reasons']}")
    for steps,tgt in ((["player_report"],"named"),(["player_positions"],None),(["match_report"],None),(["transformations"],None),(["player_report"],"team:Polis Lions")):
        print("  gate",steps,tgt,"->",json.dumps(rs.gate(gid,conn,steps,tgt),indent=None))
    print("  fingerprint", rs.identity_fingerprint(gid))
h=ih.health(conn,25)
print({k:h[k] for k in ("positions","tidy_candidates","tags")}); print(len(h["issues"]),"issues"); 
for i in h["issues"][:14]: print("  ",i["severity"],i["kind"],i.get("seek_t"),i["text"][:130])
print(h["players"][:1])
