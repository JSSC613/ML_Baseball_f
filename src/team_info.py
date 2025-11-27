# src/team_info.py

# 定義 MLB 架構
MLB_STRUCTURE = {
    'AL': {
        'East': ['BAL', 'BOS', 'NYA', 'TBA', 'TOR'],
        'Central': ['CHA', 'CLE', 'DET', 'MIN', 'KCA'],
        'West': ['HOU', 'ANA', 'OAK', 'SEA', 'TEX']
    },
    'NL': {
        'East': ['ATL', 'MIA', 'NYN', 'PHI', 'WAS'],
        'Central': ['CHN', 'CIN', 'MIL', 'PIT', 'SLN'],
        'West': ['ARI', 'SFN', 'LAN', 'COL', 'SDN']
    }
}

# 建立反查字典
TEAM_DISPLAY_INFO = {}
for league, divisions in MLB_STRUCTURE.items():
    for division, teams in divisions.items():
        for team in teams:
            TEAM_DISPLAY_INFO[team] = f"{league} {division}"

def get_relation(team1, team2):
    """
    判斷兩隊關係:
    DIVISION: 同分區 (模擬打 13 場)
    LEAGUE: 同聯盟不同區 (模擬打 6 場)
    INTER: 跨聯盟 (模擬打 3 場)
    """
    info1 = TEAM_DISPLAY_INFO.get(team1, "")
    info2 = TEAM_DISPLAY_INFO.get(team2, "")
    
    if not info1 or not info2: return 'INTER'
    
    L1, D1 = info1.split()
    L2, D2 = info2.split()
    
    if L1 != L2: return 'INTER'
    if D1 == D2: return 'DIVISION'
    return 'LEAGUE'