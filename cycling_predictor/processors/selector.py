from typing import List, Tuple, Optional

from ortools.linear_solver import pywraplp

from cycling_predictor.maps import CPClassicPointsMap
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

    def score(self) -> None:
        """
        Score riders based on stored predictions and Scorito logic.
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

                # TODO: Consider including captain points and/or team points
                
                if rider.name in self.scores:
                    self.scores[rider.name] += points
                else:
                    self.scores[rider.name] = points

    def select(self, 
               budget: float, 
               team_limit: int = 4, 
               total_riders: int = 20,
               verbose: Optional[bool] = True) -> Tuple[float, float]:
        """
        Select optimal team using OR-Tools (Knapsack-like).

        :param budget: Total budget.
        :param team_limit: Max riders per team.
        :param total_riders: Exact number of riders to select.
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
        valid_riders = [r for r in self.riders if r.cost is not None]
        
        for i, rider in enumerate(valid_riders):
            x[i] = solver.IntVar(0, 1, f"rider_{i}")

        # Constraint 1: Total Cost <= Budget
        costs = [r.cost for r in valid_riders]
        solver.Add(solver.Sum([x[i] * costs[i] for i in range(len(valid_riders))]) <= budget)

        # Constraint 2: Total Riders == total_riders
        solver.Add(solver.Sum([x[i] for i in range(len(valid_riders))]) == total_riders)

        # Constraint 3: Team Limit
        teams = set(r.team for r in valid_riders)
        for team in teams:
            team_indices = [i for i, r in enumerate(valid_riders) if r.team == team]
            solver.Add(solver.Sum([x[i] for i in team_indices]) <= team_limit)

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
                for _i, _rider in enumerate(_selector.selection, start=1):
                    print(f"{_i}. {_rider.name} ({_rider.team})".ljust(50),
                          f"€{_rider.cost}, {_selector.scores[_rider.name]} pts")

            return max_score, cost
        else:
            print("The problem does not have an optimal solution.")
            return 0.0, 0.0


if __name__ == "__main__":

    from cycling_predictor.processors import CPPredictor, CPEnsemblePredictor

    # Load predictors
    _sprint_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_sprint.json')
    _cobbles_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_cobbles.json')
    _hills_predictor = CPPredictor.load(r'data\CPPredictor_classics_2022_50_2022_hills.json')

    # Set up ensemble predictor
    _ensemble_predictor = CPEnsemblePredictor(
        predictors=[
            _sprint_predictor,
            _cobbles_predictor,
            _hills_predictor
        ]
    )

    # Preprocess data for predictions
    _ensemble_predictor.preprocess(rider_feature_noise=0.1)

    # Predict
    _predictions = _ensemble_predictor.predict()

    # Create selector
    _selector = CPSelector(
        riders=_sprint_predictor.collector.riders,
        predictions=_predictions,
    )

    # Score riders
    _selector.score()

    # Select team
    _max_score, _cost = _selector.select(
        budget=48.0,
        team_limit=4,
        total_riders=20,
    )
