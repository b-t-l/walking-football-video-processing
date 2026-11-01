import sys, os, sqlite3, shutil, json
sys.path.insert(0, os.getcwd())
exec(open("_scratch/psp/compare_report.py").read().split("import pandas")[0])
from game_logger import config, identity, identity_health as ih, repo
src=config.DB_PATH; cp=os.path.expanduser("~/gl_copy.db")
c0=sqlite3.connect(f"file:{src}?mode=ro",uri=True); c1=sqlite3.connect(cp); c0.backup(c1); c0.close(); c1.close()
conn=sqlite3.connect(cp); conn.row_factory=sqlite3.Row
print("tidy dry:", {k:v for k,v in ih.tidy(conn,25,True).items() if k!="rows"}, len(ih.tidy(conn,25,True)["rows"]))
# boxes: dup flag
b=identity.boxes_at_time(conn,25,10.0); print("boxes",len(b["boxes"]),[x["track_player_id"] for x in b["boxes"] if x["dup_track"]])
# find a time where a track has two boxes (track 1 early in game 25)
for t in (1,2,3,5,20,40):
    bb=identity.boxes_at_time(conn,25,float(t)); d=[x["track_player_id"] for x in bb["boxes"] if x["dup_track"]]
    if d: print("dup at",t,d); break
# assign on the copy: Alison onto some track, check warnings and tidy-on-write
g=repo.get_game(conn,25,with_detail=True)
rows=identity.list_assignments(conn,25); n0=len(rows)
a=rows[len(rows)//2]
res=identity.assign(conn,g,a["roster_player_id"],a["track_player_id"],(a["from_frame"]+50)/24.0)
res=dict(res); print("assign ok", res["id"], "rows",n0,"->",len(identity.list_assignments(conn,25)))
print("warnings:", ih.join_warnings(conn,25,res))
# tidy real on copy
print(ih.tidy(conn,25,False).get("saved_to"))
