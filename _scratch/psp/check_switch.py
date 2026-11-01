exec(open("_scratch/psp/compare_report.py").read().split("M=pm.compute")[0])
M=pm.compute_player_reports(dbp,rec,teams,sys.argv[2] if len(sys.argv)>2 else 'named')
for r in M['reports']:
    print(f"{r['subject']['label']:14s} team={r['team']} frames={r['n_frames']:6d} dist={r['distance_m']:7.0f} vmax={r['speed_max'] or 0:5.1f} tracks={r['tracks']} cov={r['accuracy']}")
    print("   ", r['source'], "| dots in/out/unk:", len(r['dots_in']['x']), len(r['dots_out']['x']), len(r['dots_unk']['x']), "| ball spells", r['ball_spells']['count'])
