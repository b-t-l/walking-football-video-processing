import sys, runpy, sqlite3, os, warnings
warnings.filterwarnings("ignore")
sys.argv=["x"]
# reuse the stub harness setup
exec(open("_scratch/psp/run_stubbed.py").read().split("sys.argv =")[0])
import player_positions.build as b
real=b._load_tags
RED=[int(x) for x in sys.stdin.read().split()]
b._load_tags=lambda conn,gid:[t for t in real(conn,gid) if t["id"] not in RED]
out=os.path.expanduser("~/psp_sim_delete.db")
b.build_player_positions(25,out,verbose=False)
import pandas as pd
a=pd.read_sql("select roster_player_id,frame_id,source_track_id,x_m,y_m,conflict from player_frames",sqlite3.connect(os.path.expanduser("~/psp_test_0025_players.db")))
c=pd.read_sql("select roster_player_id,frame_id,source_track_id,x_m,y_m,conflict from player_frames",sqlite3.connect(out))
m=a.merge(c,on=["roster_player_id","frame_id"],how="outer",suffixes=("_before","_after"),indicator=True)
print("rows before",len(a),"after deleting redundant tags",len(c))
print("only before:",(m._merge=="left_only").sum()," only after:",(m._merge=="right_only").sum())
both=m[m._merge=="both"]
print("same source track:",(both.source_track_id_before==both.source_track_id_after).all()," same conflict flags:",(both.conflict_before==both.conflict_after).mean().round(4))
