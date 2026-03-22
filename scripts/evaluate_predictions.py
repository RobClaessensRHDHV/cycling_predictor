from pathlib import Path

from cycling_predictor.maps import CPOMaxScore
from cycling_predictor.predictions.prediction import CPPrediction


def evaluate_race_predictions():
    # Define paths to prediction files
    # Using the 2026 predictions found in the expected package structure
    prediction_files = [
        'prediction_omloop_het_nieuwsblad_2026.json',
        'prediction_kuurne_brussel_kuurne_2026.json',
        'prediction_strade_bianche_2026.json',
        'prediction_paris_nice_stage7_2026.json',
        'prediction_tirreno_adriatico_stage7_2026.json',
        'prediction_milano_sanremo_2026.json',
    ]

    # Initiate dict to store rider results
    rider_results_dict = dict()

    # Iterate over predictions
    for prediction_file in prediction_files:

        # Retrieve path to prediction
        prediction_file_path = Path.cwd().parent / 'cycling_predictor' / 'predictions' / 'data' / prediction_file

        # Check whether path exists
        if not prediction_file_path.exists():
            print(f"Warning: File not found: {prediction_file_path}")
            continue
            
        try:

            # Load prediction
            prediction = CPPrediction.load(prediction_file_path.as_posix())

            # Retrieve results if not set yet
            if not any(prediction.result):

                for i, rider in enumerate(prediction.riders):

                    # Retrieve results
                    if rider_results := rider_results_dict.get(rider.uid):
                        rider.results = rider_results
                    else:
                        rider.get_results(prediction.stage.year)
                        rider_results_dict[rider.uid] = rider.results

                    # Get rank
                    rank = rider.get_rank(prediction.stage)

                    # Set rank
                    if rank:
                        prediction.result[i] = rank
                    else:
                        prediction.result[i] = 1E3

            # Print prediction
            prediction.print()

            # Compute score
            score = prediction.co_score()
            
            if score is not None:
                print(f"Race: {prediction.stage.name} {prediction.stage.year}")
                print(f"CO Score: {score:.1f} ({score / CPOMaxScore:.1%})")
            else:
                print(f"Race: {prediction.stage.name} {prediction.stage.year}")
                print("CO Score: Result not available yet.")
            print("-" * 40)
            
        except Exception as e:
            print(f"Error processing {prediction_file}: {e}")


if __name__ == "__main__":

    import cloudscraper
    import requests

    # Monkey patch requests with cloudscraper to bypass Cloudflare protections
    scraper = cloudscraper.create_scraper()
    requests.get = scraper.get

    # Evaluate predictions
    evaluate_race_predictions()
