import os, sys, sqlite3, importlib.abc, importlib.machinery, importlib.util
from unittest import mock
STUB={"torch","torchvision","cv2","supervision","ultralytics","moviepy","shapely","inference","roboflow","PIL","sklearn","scipy","matplotlib","reportlab","numba"}
class F(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self,name,path,target=None):
        if name.split(".")[0] in STUB:
            sys.meta_path.remove(self)
            try: real=importlib.util.find_spec(name)
            except Exception: real=None
            sys.meta_path.insert(0,self)
            if real is None: return importlib.machinery.ModuleSpec(name,self,is_package=True)
    def create_module(self,spec):
        m=mock.MagicMock(name=spec.name); m.__path__=[]; m.__spec__=spec; m.__name__=spec.name; return m
    def exec_module(self,m): pass
sys.meta_path.insert(0,F()); sys.path.insert(0,os.getcwd())
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from game_logger import config, paths, record
from reports.player_report import metrics as pm
gid=int(sys.argv[1])
lc=sqlite3.connect(f"file:{config.DB_PATH}?mode=ro",uri=True); lc.row_factory=sqlite3.Row
rec=record.build_game_record(lc,gid); teams=[rec['team_a'],rec['team_b']]
dbp=paths.detections_db(gid)
M=pm.compute_player_reports(dbp,rec,teams,'named')
mine=sqlite3.connect(sys.argv[2])
P=pd.read_sql("select * from player_frames",mine)
print(f"{'player':18s} {'report frames':>13s} {'saved in_pitch':>14s} {'diff':>6s} | {'report dist m':>13s} {'saved dist m':>12s} {'diff %':>7s}")
for r in M['reports']:
    n=r['subject']['label']; g=P[(P.name==n)&(P.in_pitch==1)]
    d=g.rename(columns={'source_track_id':'pid','x_m':'x','y_m':'y','t_video_s':'t'})
    dist=pm._path_distance_m(d)
    print(f"{n:18s} {r['n_frames']:13d} {g.frame_id.nunique():14d} {g.frame_id.nunique()-r['n_frames']:6d} | {r['distance_m']:13.0f} {dist:12.0f} {100*(dist-r['distance_m'])/r['distance_m']:7.1f}")
