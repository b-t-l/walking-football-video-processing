import os


class GameFiles:
    """File names for the things a run makes: <4-digit game number>_<what>_<time>.<ext>, e.g. 0022_stats-report_20261010-121744.pdf"""

    @staticmethod
    def prefix(game_record):
        try:
            return f"{int(game_record['game_id']):04d}"
        except (TypeError, ValueError):
            return str(game_record['game_id'])

    @staticmethod
    def file_name(game_record, name, timestr, ext):
        return f"{GameFiles.prefix(game_record)}_{name}_{timestr}.{ext}"

    @staticmethod
    def report_file(folder, game_record, name, timestr, ext, subfolder=None):
        """Full path for a new file in `folder` (or its `subfolder`); the folder is made if needed."""
        target = os.path.join(folder, subfolder) if subfolder else folder
        os.makedirs(target, exist_ok=True)
        return os.path.join(target, GameFiles.file_name(game_record, name, timestr, ext))
