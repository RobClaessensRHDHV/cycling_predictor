from datetime import date
from typing import List, Tuple, Optional

from ortools.linear_solver import pywraplp

from cycling_predictor.maps import CPClassicPointsMap, CPAbbreviationMap
from cycling_predictor.classes import CPRider
from cycling_predictor.predictions import CPPrediction


class CPSelector:
    """
    CyclingPredictor Selector class for optimizing team selection.
    """

    def __init__(self, riders: List[CPRider], predictions: List[CPPrediction]):
        self.riders = riders
        self.predictions = predictions
        self.scores = None
        self.selection = None

    def score(self, include_team_points: bool = False, include_past_races: bool = False) -> None:
        """
        Score riders based on stored predictions and Scorito logic.

        :param include_team_points: Whether to include additional points for teammates of top 3 riders.
        :param include_past_races: Whether to include points from past races.
        """
        self.scores = {rider.name: 0 for rider in self.riders}
        
        for prediction in self.predictions:
            if not prediction.riders:
                continue

            for i, rider in enumerate(prediction.riders):
                predicted_rank = int(prediction.prediction[i])
                
                # Scorito scoring: check if rank is in points map
                points = 0
                if predicted_rank in CPClassicPointsMap:
                    points = CPClassicPointsMap[predicted_rank]

                    # Exclude races that have already taken place
                    if not include_past_races and prediction.stage.start_date < date.today():
                        continue

                    # Temporarily only include top 10 for classic-brugge-de-panne, scheldeprijs and brabantse-pijl,
                    # as these only have few participants registered yet, same for stages of PN & TA
                    if predicted_rank > 10 and prediction.stage.name in (
                            'classic-brugge-de-panne', 'scheldeprijs', 'brabantse-pijl', 'paris-nice', 'tirreno-adriatico'):
                        continue

                    # Include team points for top 3 riders
                    if include_team_points and predicted_rank in (1, 2, 3):

                        # Add points for everyone in the team
                        for _team_rider in prediction.riders:
                            if rider.team == _team_rider.team and rider.name != _team_rider.name:
                                if predicted_rank == 1:
                                    self.scores[_team_rider.name] += 30
                                elif predicted_rank == 2:
                                    self.scores[_team_rider.name] += 20
                                elif predicted_rank == 3:
                                    self.scores[_team_rider.name] += 10
                
                if rider.name in self.scores:
                    self.scores[rider.name] += points
                else:
                    self.scores[rider.name] = points

    def select(self,
               budget: float,
               team_limit: int = 4,
               total_riders: int = 20,
               exclude_riders: Optional[Tuple[str, ...]] = None,
               min_riders_per_race: Optional[int] = None,
               min_riders_scoring_per_race: Optional[int] = None,
               use_full_budget: bool = False,
               verbose: bool = True) -> Tuple[float, float]:
        """
        Select optimal team using OR-Tools (Knapsack-like).

        :param budget: Total budget.
        :param team_limit: Max riders per team.
        :param total_riders: Exact number of riders to select.
        :param exclude_riders: Tuple of rider names to exclude from selection.
        :param min_riders_per_race: Minimum number of riders present in each race.
        :param min_riders_scoring_per_race: Minimum number of riders scoring points (top 20) in each race.
        :param use_full_budget: Whether to enforce using the full budget.
        :param verbose: Whether to print selection and corresponding details.
        :return: Tuple of maximum achievable score and cost of selected team.
        """
        self.selection = list()

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            raise Exception("SCIP solver not found.")

        # Variables
        x = dict()

        # Ensure we filter out riders with None/invalid costs if any
        valid_riders = [r for r in self.riders if r.cost is not None and r.name not in (exclude_riders or ())]
        
        for i, rider in enumerate(valid_riders):
            x[i] = solver.IntVar(0, 1, f"rider_{i}")

        # Constraint 1: Total Cost <= Budget (or == Budget)
        costs = [r.cost for r in valid_riders]
        if use_full_budget:
            solver.Add(solver.Sum([x[i] * costs[i] for i in range(len(valid_riders))]) == budget)
        else:
            solver.Add(solver.Sum([x[i] * costs[i] for i in range(len(valid_riders))]) <= budget)

        # Constraint 2: Total Riders == total_riders
        solver.Add(solver.Sum([x[i] for i in range(len(valid_riders))]) == total_riders)

        # Constraint 3: Team Limit
        teams = set(r.team for r in valid_riders)
        for team in teams:
            team_indices = [i for i, r in enumerate(valid_riders) if r.team == team]
            solver.Add(solver.Sum([x[i] for i in team_indices]) <= team_limit)

        # Constraint 4: Minimal riders per race
        if min_riders_per_race:
            for prediction in self.predictions:

                # Get indices of valid_riders that are in this race's startlist
                race_rider_names = [r.name for r in prediction.riders]
                race_indices = [i for i, r in enumerate(valid_riders) if r.name in race_rider_names]
                
                # Only apply if it's possible to satisfy (pool has enough riders for this race)
                if len(race_indices) >= min_riders_per_race:
                    solver.Add(solver.Sum([x[i] for i in race_indices]) >= min_riders_per_race)

        # Constraint 5: Minimal scoring riders per race (Top 20)
        if min_riders_scoring_per_race:
            for prediction in self.predictions:

                # Get names of riders predicted to finish in top 20
                scoring_rider_names = []
                for j, rider in enumerate(prediction.riders):
                    predicted_rank = int(prediction.prediction[j])
                    if predicted_rank <= 20:
                        scoring_rider_names.append(rider.name)
                
                # Get indices of valid_riders that match these scoring riders
                scoring_indices = [i for i, r in enumerate(valid_riders) if r.name in scoring_rider_names]
                
                # Only apply if it's possible to satisfy
                if len(scoring_indices) >= min_riders_scoring_per_race:
                    solver.Add(solver.Sum([x[i] for i in scoring_indices]) >= min_riders_scoring_per_race)

        # Objective: Maximize Score
        solver.Maximize(solver.Sum([x[i] * self.scores[rider.name] for i, rider in enumerate(valid_riders)]))

        # Extract solution if optimal was found
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            # Retrieve max score (and round, as it can have some floating point precision issues)
            max_score = round(solver.Objective().Value())

            # Retrieve selected riders
            for i in range(len(valid_riders)):
                if x[i].solution_value() > 0.5:
                    self.selection.append(valid_riders[i])
            self.selection.sort(key=lambda r: self.scores[r.name], reverse=True)

            # Compute total cost (generally equal to budget but could be less)
            cost = sum(r.cost for r in self.selection)

            # Print selection if verbose
            if verbose:
                print(f"\nMax score: {max_score} | Total cost: {cost} | Team:")
                print("Rider (Team)".ljust(50), "Cost  Score")
                for _i, _rider in enumerate(self.selection, start=1):
                    print(f"{_i}. {_rider.name} ({_rider.team})".ljust(50),
                          f"€{_rider.cost}, {self.scores[_rider.name]} pts")

            return max_score, cost
        else:
            print("The problem does not have an optimal solution.")
            return 0.0, 0.0

    def print_selection_table(self) -> None:
        """
        Print selected riders' participation across races in a table.
        """
        if not self.selection:
            return

        # Sort predictions by start_date if available
        sorted_predictions = sorted(self.predictions,
                                    key=lambda p: p.stage.start_date if p.stage and p.stage.start_date else date.min)

        race_labels = [CPAbbreviationMap.get(p.stage.name, p.stage.name[:5]) for p in sorted_predictions]
        header = f"{'Rider':<30} | " + " | ".join(f"{r:^5}" for r in race_labels) + " | Total"
        print(f"\n{header}\n{'-' * len(header)}")

        race_totals = [0] * len(sorted_predictions)
        for rider in self.selection:
            row_data = []
            for i, p in enumerate(sorted_predictions):
                is_competing = any(r.name == rider.name for r in p.riders)
                row_data.append("x" if is_competing else " ")
                if is_competing: race_totals[i] += 1
            print(f"{rider.name:<30} | " + " | ".join(f"{v:^5}" for v in row_data) + f" | {sum(1 for v in row_data if v == 'x'):^5}")

        print(f"{'-' * len(header)}")
        print(f"{'Total':<30} | " + " | ".join(f"{t:^5}" for t in race_totals) + f" | {sum(race_totals):^5}")

    def print_prediction_table(self) -> None:
        """
        Print selected riders' predicted rank and points across races in a table.
        """
        if not self.selection:
            return

        # Sort predictions by start_date if available
        sorted_predictions = sorted(self.predictions,
                                    key=lambda p: p.stage.start_date if p.stage and p.stage.start_date else date.min)

        race_labels = [CPAbbreviationMap.get(p.stage.name, p.stage.name[:5]) for p in sorted_predictions]
        header = f"{'Rider':<30} | " + " | ".join(f"{r:^8}" for r in race_labels) + " | Total"
        print(f"\n{header}\n{'-' * len(header)}")

        race_totals = [0] * len(sorted_predictions)
        for rider in self.selection:
            row_data = []
            rider_total_points = 0
            for i, p in enumerate(sorted_predictions):

                # Find rider in prediction
                rank = None
                for idx, r in enumerate(p.riders):
                    if r.name == rider.name:
                        rank = int(p.prediction[idx])
                        break
                
                if rank:
                    points = CPClassicPointsMap.get(rank, 0)
                    row_data.append(f"{rank} ({points})")
                    rider_total_points += points
                    race_totals[i] += points
                else:
                    row_data.append(" ")
            
            print(f"{rider.name:<30} | " + " | ".join(f"{v:^8}" for v in row_data) + f" | {rider_total_points:^5}")

        print(f"{'-' * len(header)}")
        print(f"{'Total':<30} | " + " | ".join(f"{t:^8}" for t in race_totals) + f" | {sum(race_totals):^5}")


if __name__ == "__main__":

    from cycling_predictor.processors import CPPredictor, CPEnsemblePredictor

    # Load predictors
    _sprint_predictor = CPPredictor.load(r'data\CPPredictor_classics_2026_RR_sprint.json')
    _cobbles_predictor = CPPredictor.load(r'data\CPPredictor_classics_2026_RR_cobbles.json')
    _hills_predictor = CPPredictor.load(r'data\CPPredictor_classics_2026_RR_hills.json')
    _pn_predictor = CPPredictor.load(r'data\CPPredictor_paris_nice_tirreno_adriatico_2026_RR_5.json')
    _ta_predictor = CPPredictor.load(r'data\CPPredictor_paris_nice_tirreno_adriatico_2026_RR_1.json')

    # Set up ensemble predictor
    _ensemble_predictor = CPEnsemblePredictor(
        predictors=[
            _sprint_predictor,
            _cobbles_predictor,
            _hills_predictor,
            _pn_predictor,
            _ta_predictor,
        ]
    )

    # Preprocess data for predictions
    _ensemble_predictor.preprocess(rider_feature_noise=0.1)

    # Predict
    _predictions = _ensemble_predictor.predict(n=1000, rider_feature_noise=0.1, normalize=True)

    # Create selector
    _selector = CPSelector(
        riders=_sprint_predictor.collector.riders,
        predictions=_predictions,
    )

    # Score riders
    _selector.score()

    # Select team
    _max_score, _cost = _selector.select(
        budget=45.0,
        team_limit=4,
        total_riders=20,
        exclude_riders=('remco-evenepoel', 'mads-pedersen', 'tim-merlier'),
        min_riders_per_race=3,
        min_riders_scoring_per_race=3,
        use_full_budget=True,
    )

    # Print selection table
    _selector.print_selection_table()
    _selector.print_prediction_table()
