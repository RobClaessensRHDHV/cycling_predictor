from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cloudscraper
import requests

from cycling_predictor.maps import CPOMaxScore, CPAbbreviationMap, CPCOPointsMap, CPCOFactorMap
from cycling_predictor.predictions.prediction import CPPrediction


RACE_ALIAS_MAP = {
    'BDP': 'RVB',
    'GW': 'IFF',
}


def score_model():
    """
    Compare Robot's model predictions against CyclingOracle benchmark.
    """
    # Define CyclingOracle's top 10 predictions per race (alias: [rider_uids])
    # The user can update this dictionary with actual CO predictions.
    co_predictions = {
        'OHN': [
            'mathieu-van-der-poel',
            'tom-pidcock',
            'tim-wellens',
            'biniam-girmay',
            'jasper-philipsen',
            'valentin-madouas',
            'arnaud-de-lie',
            'toms-skujins',
            'paul-magnier',
            'florian-vermeersch'
        ],
        'KBK': [
            'jonathan-milan',
            'jasper-philipsen',
            'jordi-meeus',
            'paul-magnier',
            'biniam-girmay',
            'arnaud-de-lie',
            'laurence-pithie',
            'tobias-lund-andresen',
            'kaden-groves',
            'matthew-brennan'
        ],
        'SB': [
            'tadej-pogacar',
            'isaac-del-toro',
            'tom-pidcock',
            'matteo-jorgenson',
            'ben-healy',
            'wout-van-aert',
            'jan-christen',
            'romain-gregoire',
            'paul-seixas',
            'valentin-madouas'
        ],
        'MSR': [
            'mathieu-van-der-poel',
            'tadej-pogacar',
            'filippo-ganna',
            'wout-van-aert',
            'isaac-del-toro',
            'tom-pidcock',
            'jasper-philipsen',
            'matteo-jorgenson',
            'biniam-girmay',
            'romain-gregoire'
        ],
        'BDP': [
            'jasper-philipsen',
            'dylan-groenewegen',
            'soren-waerenskjold',
            'alexis-renard',
            'pavel-bittner',
            'emilien-jeanniere',
            'luca-mozzato',
            'juan-sebastian-molano',
            'milan-fretin',
            'stanislaw-aniolkowski'
        ],
        'E3': [
            'mathieu-van-der-poel',
            'mads-pedersen',
            'christophe-laporte',
            'florian-vermeersch',
            'gianni-vermeersch',
            'jasper-stuyven',
            'matteo-trentin',
            'jonas-abrahamsen',
            'antonio-morgado',
            'biniam-girmay',
        ],
        'GW': [
            'wout-van-aert',
            'mathieu-van-der-poel',
            'jasper-philipsen',
            'tobias-lund-andresen',
            'jordi-meeus',
            'jonathan-milan',
            'matthew-brennan',
            'florian-vermeersch',
            'matteo-trentin',
            'paul-magnier'
        ],
        'DDV': [
            'mads-pedersen',
            'wout-van-aert',
            'jasper-philipsen',
            'tobias-lund-andresen',
            'arnaud-de-lie',
            'biniam-girmay',
            'paul-magnier',
            'jonathan-milan',
            'christophe-laporte',
            'florian-vermeersch',
        ],
        'RVV': [
            'tadej-pogacar',
            'mathieu-van-der-poel',
            'wout-van-aert',
            'mads-pedersen',
            'remco-evenepoel',
            'florian-vermeersch',
            'jonas-abrahamsen',
            'christophe-laporte',
            'jasper-stuyven',
            'antonio-morgado',
        ],
        'SP': [
            'jasper-philipsen',
            'dylan-groenewegen',
            'jordi-meeus',
            'tim-merlier',
            'pavel-bittner',
            'steffen-de-schuyteneer',
            'tim-torn-teutenberg',
            'laurence-pithie',
            'pascal-ackermann',
            'stanislaw-aniolkowski'
        ],
        'PR': [
            'mathieu-van-der-poel',
            'tadej-pogacar',
            'wout-van-aert',
            'mads-pedersen',
            'filippo-ganna',
            'christophe-laporte',
            'florian-vermeersch',
            'jasper-philipsen',
            'jasper-stuyven',
            'gianni-vermeersch',
        ],
        'BP': [
            'romain-gregoire',
            'mauro-schmid',
            'florian-vermeersch',
            'clement-venturini',
            'jenno-berckmoes',
            'tibor-del-grosso',
            'tim-wellens',
            'valentin-madouas',
            'alex-baudin',
            'edoardo-zambanini'
         ],
        'AGR': [
            'remco-evenepoel',
            'kevin-vauquelin',
            'mattias-skjelmose-jensen',
            'matteo-jorgenson',
            'alex-aranburu',
            'axel-laurance',
            'romain-gregoire',
            'dorian-godon',
            'pello-bilbao',
            'christophe-laporte'
        ],
        'LFW': [
            'paul-seixas',
            'mattias-skjelmose-jensen',
            'kevin-vauquelin',
            'tobias-halland-johannessen',
            'lenny-martinez',
            'benoit-cosnefroy',
            'christian-scaroni',
            'pello-bilbao',
            'romain-gregoire',
            'guillaume-martin',
        ],
        'LBL': [
            'tadej-pogacar',
            'remco-evenepoel',
            'paul-seixas',
            'tom-pidcock',
            'mattias-skjelmose-jensen',
            'romain-gregoire',
            'ben-tulett',
            'giulio-ciccone',
            'christian-scaroni',
            'tobias-halland-johannessen',
        ],
    }

    # Define Robot prediction files
    prediction_files = [
        'prediction_omloop_het_nieuwsblad_2026.json',
        'prediction_kuurne_brussel_kuurne_2026.json',
        'prediction_strade_bianche_2026.json',
        'prediction_paris_nice_stage7_2026.json',
        'prediction_tirreno_adriatico_stage7_2026.json',
        'prediction_milano_sanremo_2026.json',
        'prediction_classic_brugge_de_panne_2026.json',
        'prediction_e3_harelbeke_2026.json',
        'prediction_gent_wevelgem_2026.json',
        'prediction_dwars_door_vlaanderen_2026.json',
        'prediction_ronde_van_vlaanderen_2026.json',
        'prediction_scheldeprijs_2026.json',
        'prediction_paris_roubaix_2026.json',
        'prediction_brabantse_pijl_2026.json',
        'prediction_amstel_gold_race_2026.json',
        'prediction_la_fleche_wallonne_2026.json',
        'prediction_liege_bastogne_liege_2026.json',
    ]

    # Monkey patch requests with cloudscraper
    scraper = cloudscraper.create_scraper()
    requests.get = scraper.get

    # Data for plotting
    race_aliases = []
    robot_scores = []
    co_scores = []
    
    enriched_data_dir = Path(__file__).resolve().parent / 'data'
    enriched_data_dir.mkdir(exist_ok=True)

    print("Comparing Robot vs. CyclingOracle scores...")
    print("-" * 40)

    for prediction_file in prediction_files:
        prediction_path = Path.cwd().parent / 'cycling_predictor' / 'predictions' / 'data' / prediction_file
        
        if not prediction_path.exists():
            print(f"Warning: File not found: {prediction_path}")
            continue

        try:
            # Load Robot's prediction
            prediction = CPPrediction.load(prediction_path.as_posix())
            
            # Get race alias
            alias = CPAbbreviationMap.get(prediction.stage.name.lower().replace(' ', '-'), prediction.stage.name)
            
            if alias not in co_predictions:
                print(f"Skipping {alias} - no CyclingOracle prediction provided.")
                continue

            # Fetch results via API if not present
            if not any(prediction.result):
                print(f"Fetching results for {prediction.stage.name}...")
                for rider in prediction.riders:
                    if rider.name == 'thomas-pidcock':
                        rider.name = 'tom-pidcock'
                    elif rider.name == 'romain-gregoire1':
                        rider.name = 'romain-gregoire'
                        print(f"Corrected rider name: {rider.name}")
                    rider.get_results(prediction.stage.year)
                
                for i, rider in enumerate(prediction.riders):
                    rank = rider.get_rank(prediction.stage)
                    prediction.result[i] = rank if rank else 1E3
                
                # Save enriched prediction locally
                prediction.dump(enriched_data_dir / prediction_file)

            # Robot score
            robot_score = prediction.co_score()
            
            # CyclingOracle score
            # We need to find the actual ranks for the CO top 10 riders
            co_score = 0.0
            co_top_10 = co_predictions[alias]
            
            # Create a helper map of rider UID to index in prediction.riders
            rider_idx_map = {rider.name: i for i, rider in enumerate(prediction.riders)}
            
            print(f"\nCyclingOracle Scoring for {alias}:")
            for rank, rider_name in enumerate(co_top_10, start=1):
                idx = rider_idx_map.get(rider_name)

                # Special case for Romain Grégoire
                if idx is None and 'romain-gregoire' in rider_name and (_idx := rider_idx_map.get('romain-gregoire1')):
                    idx = _idx
                    print(f"Matched Romain Grégoire with corrected name: {prediction.riders[idx].name}")

                if idx is not None:
                    actual_rank = prediction.result[idx]
                    if actual_rank in CPCOPointsMap:
                        factor = CPCOFactorMap.get(rank, 1.0)
                        points = CPCOPointsMap[actual_rank]
                        co_score += points * factor
                        print(f"#{rank}: {rider_name} - res.{actual_rank} ({points}x{factor})")
                    else:
                        print(f"#{rank}: {rider_name} - res.{actual_rank} (0 points)")
                else:
                    # Rider might not be in the startlist of the original prediction file
                    # In a real scenario, we might need to fetch the rider result separately
                    print(f"#{rank}: {rider_name} - Rider not found in prediction riders list.")

            race_aliases.append(alias)
            robot_scores.append(robot_score / CPOMaxScore * 100)  # Convert to percentage
            co_scores.append(co_score / CPOMaxScore * 100)  # Convert to percentage
            
            print(f"\nFinal scores for {alias}:")
            print(f"Robot: {robot_score:.1f} ({robot_score / CPOMaxScore:.1%})")
            print(f"CyclingOracle: {co_score:.1f} ({co_score / CPOMaxScore:.1%})")
            print("-" * 40)

        except Exception as e:
            print(f"Error processing {prediction_file}: {e}")

    if not race_aliases:
        print("No matches found.")
        return

    # Cumulative scores
    robot_cumulative = np.cumsum(robot_scores)
    co_cumulative = np.cumsum(co_scores)

    # Print cumulatives and averages
    print("\nCumulative and average scores:")
    print(f"Robot: {robot_cumulative[-1]:.1f} ({(robot_cumulative[-1] / len(race_aliases)):.1f}% average)")
    print(f"CyclingOracle: {co_cumulative[-1]:.1f} ({(co_cumulative[-1] / len(race_aliases)):.1f}% average)")

    # TODO: Plot of rider finishing top 20?
    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(race_aliases))
    width = 0.25

    ax1.bar(x - width/2, robot_scores, width, label='Robot', color='tab:red', alpha=0.8, edgecolor='black')
    ax1.bar(x + width/2, co_scores, width, label='CyclingOracle', color='tab:blue', alpha=0.8, edgecolor='black')
    
    ax1.set_xlabel('Race')
    ax1.set_ylabel('CyclingOracle score per race [%]')
    ax1.set_ylim(0, 100)
    ax1.set_title('Robot vs. CyclingOracle - Model performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([RACE_ALIAS_MAP.get(alias, alias) for alias in race_aliases], rotation=45)
    ax1.grid(axis='y', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, robot_cumulative, color='tab:red', marker='o', linewidth=2, label='Robot (Cumul.)')
    ax2.plot(x, co_cumulative, color='tab:blue', marker='o', linewidth=2, label='CO (Cumul.)')
    ax2.set_ylabel('Cumulative CyclingOracle score [%]')
    
    # Combine legends into one with 2 columns and 2 rows
    # Left column: individual results (bars), Right column: cumulative progress (lines)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # Create maps for safe handle retrieval
    m1 = dict(zip(l1, h1))
    m2 = dict(zip(l2, h2))
    
    # Explicitly order: [Robot_bar, CO_bar, Robot_line, CO_line]
    # For ncol=2 and column-first rendering:
    # Column 1: Robot_bar, CO_bar
    # Column 2: Robot_line, CO_line
    combined_handles = [
        m1['Robot'], m1['CyclingOracle'],
        m2['Robot (Cumul.)'], m2['CO (Cumul.)']
    ]
    combined_labels = [
        'Robot', 'CyclingOracle',
        'Robot (Cumul.)', 'CO (Cumul.)'
    ]
    
    ax1.legend(combined_handles, combined_labels, loc='upper left', ncol=2, frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    score_model()
