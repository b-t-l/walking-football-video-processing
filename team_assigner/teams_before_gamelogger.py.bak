# define the teams we are using including their dominant colors:
class Teams:

    def __init__(self):
        
        self.TEAMS = {
            "Polis": [
                {"rgb": (50, 94, 46), "tolerance": (10, 50, 50)}, 
                {"rgb": (132, 208, 116), "tolerance": (10, 50, 50)}
            ],
            "West Coast": [
                {"rgb": (107, 120, 182), "tolerance": (10, 50, 50)}, 
                {"rgb": (194, 199, 239), "tolerance": (10, 50, 50)}
            ],
            "Aphrodite Wanderers": [
                {"rgb": (0,0,0), "tolerance": (10, 50, 50)}, 
                {"rgb": (212,209,112), "tolerance": (10, 50, 50)}
            ],
            "Akamas": [
                {"rgb": (255, 255, 255), "tolerance": (10, 50, 50)}, 
                {"rgb": (195, 195, 195), "tolerance": (10, 50, 50)}
            ]
        }
        '''
        self.TEAMS = {
            "Polis": 
            "West Coast": [
                {"rgb": (107, 120, 182), "tolerance": (10, 50, 50)}, 
                {"rgb": (194, 199, 239), "tolerance": (10, 50, 50)}
            ],
            "Aphrodite Wanderers": [
                {"rgb": (0,0,0), "tolerance": (10, 50, 50)}, 
                {"rgb": (212,209,112), "tolerance": (10, 50, 50)}
            ],
            "Akamas": [
                {"rgb": (255, 255, 255), "tolerance": (10, 50, 50)}, 
                {"rgb": (195, 195, 195), "tolerance": (10, 50, 50)}
            ]
        }
        '''
    # PASS BACK ALL THE TEAMS:
    def get_all_teams(self):
        return self.TEAMS

    # PASS BACK TEAMS PLAYING IN THIS GAME:
    def get_playing_teams(self,team_names):
        playing_teams = {team: self.TEAMS[team] for team in team_names if team in self.TEAMS}
        return playing_teams
