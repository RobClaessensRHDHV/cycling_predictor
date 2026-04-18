import math


CPRaceCategoryMap = {
    'classics': [
        'omloop-het-nieuwsblad',
        'kuurne-brussel-kuurne',
        'strade-bianche',
        'milano-sanremo',
        'classic-brugge-de-panne',
        'e3-harelbeke',
        'gent-wevelgem',
        'dwars-door-vlaanderen',
        'ronde-van-vlaanderen',
        'scheldeprijs',
        'paris-roubaix',
        'brabantse-pijl',
        'amstel-gold-race',
        'la-fleche-wallonne',
        'liege-bastogne-liege',
    ],
    'gts': [
        'giro-d-italia',
        'tour-de-france',
        'vuelta-a-espana'
    ],
    'giro': ['giro-d-italia'],
    'tour': ['tour-de-france'],
    'vuelta': ['vuelta-a-espana'],
    'paris-nice': ['paris-nice'],
    'tirreno-adriatico': ['tirreno-adriatico'],
}

CPTerrainTypeMap = {
    'omloop-het-nieuwsblad': [
        'cobbles'
    ],
    'kuurne-brussel-kuurne': [
        'sprint',
        'cobbles',
    ],
    'strade-bianche': [
        'hills',
    ],
    'milano-sanremo': [
        'sprint',
        'hills',
    ],
    'classic-brugge-de-panne': [
        'sprint',
    ],
    'e3-harelbeke': [
        'cobbles',
        'hills',
    ],
    'gent-wevelgem': [
        'sprint',
        'cobbles',
    ],
    'dwars-door-vlaanderen': [
        'cobbles',
    ],
    'ronde-van-vlaanderen': [
        'cobbles',
        'hills',
    ],
    'scheldeprijs': [
        'sprint',
    ],
    'paris-roubaix': [
        'cobbles',
    ],
    'brabantse-pijl': [
        'hills',
    ],
    'amstel-gold-race': [
        'hills',
    ],
    'la-fleche-wallonne': [
        'hills',
    ],
    'liege-bastogne-liege': [
        'hills',
    ],
}

CPAbbreviationMap = {
    'omloop-het-nieuwsblad': 'OHN',
    'kuurne-brussel-kuurne': 'KBK',
    'strade-bianche': 'SB',
    'milano-sanremo': 'MSR',
    'classic-brugge-de-panne': 'BDP',
    'e3-harelbeke': 'E3',
    'gent-wevelgem': 'GW',
    'dwars-door-vlaanderen': 'DDV',
    'ronde-van-vlaanderen': 'RVV',
    'scheldeprijs': 'SP',
    'paris-roubaix': 'PR',
    'brabantse-pijl': 'BP',
    'amstel-gold-race': 'AGR',
    'la-fleche-wallonne': 'LFW',
    'liege-bastogne-liege': 'LBL',
    'paris-nice': 'PN',
    'tirreno-adriatico': 'TA',
}

CPRiderInfoMap = {
    'anders-halland-johannessen': {
        'weight': 65.0
    },
    'emilien-jeanniere': {
        'height': 1.76,
        'weight': 68.0
    },
    'matheo-vercher': {
        'weight': 65.0
    },
    'thomas-gachignard': {
        'height': 1.65,
        'weight': 60.0
    },
    'filippo-fiorelli': {
        'height': 1.65,
        'weight': 62.0
    },
    'max-poole': {
        'height': 1.85,
        'weight': 64.0
    },
    'thomas-bonnet': {
        'weight': 65.0
    },
    'fabien-doubey': {
        'height': 1.74,
        'weight': 62.0
    },
    'alan-jousseaume': {
        'weight': 68.0
    },
    'manuele-tarozzi': {
        'height': 1.75,
        'weight': 66.0
    },
    'bart-lemmen': {
        'height': 1.80,
        'weight': 68.0
    },
    'johannes-kulset': {
        'height': 1.75,
        'weight': 58.0
    },
    'sandy-dujardin': {
        'height': 1.78,
        'weight': 64.0
    },
    'enzo-leijnse': {
        'height': 1.94,
        'weight': 72.0
    },
    'xabier-berasategi-garmendia': {
        'height': 1.66,
        'weight': 60.0
    },
    'xabier-isasa-larranaga': {
        'height': 1.85,
        'weight': 65.0
    },
    'casper-van-uden': {
        'height': 1.81,
        'weight': 68.0
    },
    'nickolas-zukowsky': {
        'height': 1.88,
        'weight': 70.0
    },
    'bjoern-koerdt': {
        'height': 1.65,
        'weight': 66.0
    },
    'markel-beloki': {
        'height': 1.83,
        'weight': 65.0
    },
    'lukas-nerurkar': {
        'height': 1.77,
        'weight': 64.0
    },
    'oliver-knight': {
        'height': 1.74,
        'weight': 62.0
    },
    'fabio-christen': {
        'height': 1.84,
        'weight': 66.0
    },
    'ben-healy': {
        'height': 1.75,
        'weight': 65.0
    },
    'pavel-sivakov': {
        'height': 1.88,
        'weight': 70.0
    },
    'embret-svestad-bardseng':
    {
        'height': 1.85,
        'weight': 74.0
    },
    'martin-tjotta': {
        'height': 1.76,
        'weight': 66.0
    },
    'yannis-voisard':
    {
        'height': 1.70,
        'weight': 56.0
    },
    'per-strand-hagenes':
    {
        'height': 1.86,
        'weight': 74.0
    },
    'fredrik-dversnes':
    {
        'height': 1.84,
        'weight': 72.0
    },
    'tim-torn-teutenberg':
    {
        'height': 1.83,
        'weight': 70.0
    },
    'albert-withen-philipsen':
    {
        'height': 1.84,
        'weight': 66.0
    },
}

CPClassicPointsMap = {
    # Provide top 5 with additional points to simulate captain points
    1: math.ceil(100 * 1.5),
    2: math.ceil(90 * 1.4),
    3: math.ceil(80 * 1.3),
    4: math.ceil(70 * 1.2),
    5: math.ceil(64 * 1.1),
    6: 60,
    7: 56,
    8: 52,
    9: 48,
    10: 44,
    11: 40,
    12: 36,
    13: 32,
    14: 28,
    15: 24,
    16: 20,
    17: 16,
    18: 12,
    19: 8,
    20: 4,
}

CPCOPointsMap = {
    1: 15,
    2: 10,
    3: 7,
    4: 5,
    5: 4,
    6: 3,
    7: 3,
    8: 3,
    9: 3,
    10: 3,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 1,
    17: 1,
    18: 1,
    19: 1,
    20: 1,
}

CPCOFactorMap = {
    1: 3.0,
    2: 2.5,
    3: 2.0,
    4: 1.5,
    5: 1.5,
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 1.0,
}

CPOMaxScore = sum(CPCOFactorMap.get(rank) * points for rank, points in CPCOPointsMap.items() if rank <= 10)