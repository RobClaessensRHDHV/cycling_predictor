# CyclingPredictor

CyclingPredictor is a machine learning project by Rob Claessens, designed to predict cycling race rankings.
It leverages historical data from [ProCyclingStats](https://www.procyclingstats.com/), rider skill scores from 
[CyclingOracle](https://www.cyclingoracle.com/) and machine learning techniques to estimate the performance of riders in
various types of races (classics, grand tours, etc.).

## Architecture

The project is built around several core classes that represent the data and domain of cycling racing.

### Core classes

- **CPRider**: Represents a professional cyclist, including metadata like team, birthdate, weight, and a set of 
  performance-related skills (e.g., sprint, mountains, cobbles, time trial).
- **CPRace**: An abstract base class representing a cycling event or race. It stores general race information like
  name, start- and end dates, startlist, etc.
- **CPStage**: A subclass of `CPRace` specifically representing a stage with more detailed information, such as distance,
  vertical meters, profile score, etc. Although its name might suggest these only apply to stage races (e.g. grand tours),
  they are also used for one-day races, such as classics, to capture the required, more specific details.
- **CPEntry**: Represents the interaction between a `CPRider` and a `CPStage`. This class encapsulates data that is 
  specific to a particular event, combining rider skills, stage details and current rider form and age.
  The `CPEntry` is the primary data structure used for training and prediction.

### Collectors

Collectors are responsible for gathering data from various sources (i.e., ProCyclingStats, CyclingOracle).
- **CPRiderCollector**: Extracts rider metadata and skills.
- **CPGTEntryCollector / CPClassicEntryCollector**: Specialized entry collectors for **Grand tours** and **Classics**.

### Processors

Processors handle the data logic, training, and prediction workflows.
- **CPProcessor**: Abstract base class for processors, defining common methods for data filtering, preprocessing, dumping, loading, etc.
- **CPTrainer**: Orchestrates the training process, including data preprocessing, scaling, and model fitting.
- **CPPredictor**: Uses trained models to generate rank predictions for given stages.
- **CPSelector**: Employs optimization techniques (like a Knapsack algorithm using OR-Tools) to select an optimal team 
  based on predicted scores and budget constraints. *Note that this class isn't a descendant of `CPProcessor`*,
  but because of its functionality is still considered a processor.

### Models

Models wrap the machine learning algorithms used for prediction.
- **BaseModel**: An abstract base class defining the interface for all models, including methods for training, testing, 
  predicting, dumping, loading, etc. It allows for easy extension with different types of models, think of:
  - A simple linear regression implementation
  - A random forest regressor
  - A neural network model
  - ...
  
  *Please feel free to contribute your own model implementations!*
- **XGBModel**: An implementation using an XGBoost Ranker, of course very suitable for ranking tasks.

## License

This project is licensed under the **MIT License**, allowing for wide-ranging use and modification by others.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.
Extensions to any part of the code are welcomed, though especially model implementations would add great value.
Of course, please ensure that your code adheres to the existing style.
