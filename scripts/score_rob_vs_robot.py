import numpy as np
import matplotlib.pyplot as plt


RACE_ALIAS_MAP = {
    'BDP': 'RVB',
    'GW': 'IFF',
}

def score_rob_vs_robot():
    """
    Compare Rob (manual), Robot (model), and Average (benchmark) scores.
    Data is provided directly in the script (e.g. from Scorito results).
    """
    # Predictors to compare
    predictors = ('rob', 'robot', 'avg')
    
    # Define Scorito points per race for each predictor
    # Update these dictionaries with your results.
    # Alias names from race_data.json: OHN, KBK, SB, MSR, BDP, E3, GW, DDV, RVV, SP, PR, BP, AGR, WP, LBL, EF
    race_scores = {
        'rob': {
            'OHN': 766,
            'KBK': 98,
            'SB': 730,
            'PN': 0,
            'TA': 310,
            'MSR': 598,
            'BDP': 388,
            'E3': 620,
            'GW': 360,
            'DDV': 620,
            'RVV': 799,
            'SP': 270,
            'PR': 765,
            'BP': 220,
            'AGR': 655,
            'WP': 412,
            'LBL': 588,
        },
        'robot': {
            'OHN': 396,
            'KBK': 136,
            'SB': 730,
            'PN': 225,
            'TA': 426,
            'MSR': 600,
            'BDP': 600,
            'E3': 440,
            'GW': 510,
            'DDV': 288,
            'RVV': 766,
            'SP': 226,
            'PR': 481,
            'BP': 260,
            'AGR': 530,
            'WP': 506,
            'LBL': 642,
        },
        'avg': {
            'OHN': 539,
            'KBK': 170,
            'SB': 651,
            'PN': 205,
            'TA': 245,
            'MSR': 661,
            'BDP': 330,
            'E3': 579,
            'GW': 380,
            'DDV': 415,
            'RVV': 794,
            'SP': 226,
            'PR': 674,
            'BP': 186,
            'AGR': 411,
            'WP': 348,
            'LBL': 561,
        }
    }

    # Define the order of races for plotting
    race_order = ['OHN', 'KBK', 'SB', 'PN', 'TA', 'MSR', 'BDP', 'E3', 'GW', 'DDV', 'RVV', 'SP', 'PR', 'BP', 'AGR', 'WP', 'LBL']

    # Collect data for plotting (only for races that have scores defined)
    race_labels = [alias for alias in race_order if alias in race_scores['rob']]
    num_races = len(race_labels)
    
    predictor_scores = {k: [race_scores[k].get(alias, 0) for alias in race_labels] for k in predictors}
    
    # Calculate cumulative points
    predictor_cumulative = {k: np.cumsum(predictor_scores[k]) for k in predictors}
    
    print("Scoring Results (Total score):")
    for k in predictors:
        print(f"{k.capitalize()}: {predictor_cumulative[k][-1]} points")
    print("-" * 40)

    # Plotting logic
    # Colors matching the original classics script
    colors = {'rob': 'tab:brown', 'robot': 'tab:red', 'avg': 'tab:gray'}
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    x = np.arange(num_races)
    width = 0.25 # Width of the bars

    # Plot grouped bars
    ax1.bar(x - width, predictor_scores['rob'], width, label='Rob', color=colors['rob'], alpha=0.8, edgecolor='black')
    ax1.bar(x, predictor_scores['robot'], width, label='Robot', color=colors['robot'], alpha=0.8, edgecolor='black')
    ax1.bar(x + width, predictor_scores['avg'], width, label='Average', color=colors['avg'], alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Race')
    ax1.set_ylabel('Scorito score per race [-]')
    ax1.set_ylim(0, 1000)
    ax1.set_title('Rob vs. Robot - Scorito performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels([RACE_ALIAS_MAP.get(alias, alias) for alias in race_labels], rotation=45)
    ax1.grid(axis='y', alpha=0.5)

    # Secondary axis for cumulative lines
    ax2 = ax1.twinx()
    ax2.plot(x, predictor_cumulative['rob'], color=colors['rob'], marker='o', linewidth=2, label='Rob (Cumul.)')
    ax2.plot(x, predictor_cumulative['robot'], color=colors['robot'], marker='o', linewidth=2, label='Robot (Cumul.)')
    ax2.plot(x, predictor_cumulative['avg'], color=colors['avg'], marker='o', linewidth=2, label='Average (Cumul.)')
    
    ax2.set_ylabel('Cumulative Scorito score [-]')

    # Combine legends into one with 2 columns and 3 rows
    # Left column: individual results (bars), Right column: cumulative progress (lines)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    
    # Create maps for safe handle retrieval
    m1 = dict(zip(l1, h1))
    m2 = dict(zip(l2, h2))
    
    # Explicitly order: [Bar1, Bar2, Bar3, Line1, Line2, Line3]
    # In some Matplotlib versions, ncol=2 fills column-first.
    # By listing all bars then all lines, we ensure:
    # Col 1: Rob Bar, Robot Bar, Average Bar
    # Col 2: Rob Line, Robot Line, Average Line
    combined_handles = [
        m1['Rob'], m1['Robot'], m1['Average'],
        m2['Rob (Cumul.)'], m2['Robot (Cumul.)'], m2['Average (Cumul.)']
    ]
    combined_labels = [
        'Rob', 'Robot', 'Average',
        'Rob (Cumul.)', 'Robot (Cumul.)', 'Average (Cumul.)'
    ]
    
    ax1.legend(combined_handles, combined_labels, loc='upper left', ncol=2, frameon=True, shadow=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    score_rob_vs_robot()
