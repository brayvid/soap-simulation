# Soap Simulation

This repository contains a Python-based agent-based simulation designed to model and explore the economic dynamics and user engagement incentives for the [Soap platform](https://use.soap.fyi) (though it can be adapted for other similar user-contribution systems).

The primary goal of this simulation is to help understand:

1.  **Economic Viability:** Can an internal economy with costs for actions (submitting/agreeing with words) and rewards for popular contributions be structured to be sustainable for both the platform and its users?
2.  **User Incentive & Growth:** What economic parameters (costs, rewards, platform rake) and platform characteristics (attractiveness factors) would encourage users to join, participate actively, and remain engaged over time?
3.  **Impact of Managerial Levers:** How do changes in platform-controlled settings affect the overall health and growth of the user ecosystem?
4.  **Robustness to Population Characteristics:** How do chosen managerial strategies perform under different assumptions about user behavior and market conditions (empirical cases)?

## Core Simulated Mechanics

The simulation models the following key aspects of the Soap platform's conceptual economy:

*   **Users (Agents):**
    *   Belong to different **archetypes** (e.g., "Innovator," "Follower," "Balanced") with varying behavioral probabilities.
    *   Have an internal **balance** of simulated currency.
    *   Make decisions to **join** the platform based on its perceived attractiveness.
    *   Can **churn** (leave the platform) if conditions become unfavorable (e.g., low balance, inactivity).
*   **Politicians:** Abstract entities against which users submit descriptive words.
*   **Words & Vocabulary:**
    *   A **finite dictionary** of abstract words is used.
    *   Word choice by users follows a **Zipfian distribution**, mimicking real-world language where some words are very common and many are rare.
    *   Innovator archetypes have a small chance to introduce "out-of-dictionary" words.
*   **Platform Actions & Costs:**
    *   **Submitting a "first-time" word** for a specific politician (i.e., a (politician, word) pair not yet in the system) has a defined cost.
    *   **Agreeing with an existing word** for a politician (or re-submitting a known one) has a (potentially different) defined cost.
*   **Word Popularity & Rewards:**
    *   The **popularity** of a (politician, word) pair is tracked by the number of direct user interactions (`count`).
    *   A separate `raw_popularity_score` (which can decay) influences follower choices.
    *   When a word's `count` reaches a `POPULARITY_THRESHOLD_BASE`, a **reward** is triggered.
    *   The user who **first submitted** that specific (politician, word) pair receives a significant share of the reward.
    *   Other users who "agreed" with (interacted with) the word share the remaining portion of the reward.
    *   Reward pools are funded by a portion of the action costs.
*   **Platform Economy:**
    *   The platform takes a **rake** (percentage) from each action cost, contributing to the `platform_treasury`.
    *   The `platform_treasury` is used to understand the platform's financial sustainability.
*   **Platform Attractiveness & User Growth:**
    *   The decision of *new potential users* (from a fixed `POTENTIAL_USER_POOL_SIZE`) to join is probabilistic and influenced by a calculated `platform_attractiveness_score`.
    *   This attractiveness score is a function of factors like recent reward payouts on the platform, overall activity levels, perceived fairness (Gini coefficient of existing users), and the average financial health of current users.

## Simulation Script (`sim.py`)

The main script `sim.py` includes:

*   **Class Definitions:** `User`, `WordData`.
*   **Helper Functions:** `get_gini`, `initialize_zipfian_dictionary`, `get_simulated_word_zipfian`, `calculate_platform_attractiveness`.
*   **Core Simulation Logic:** `run_full_simulation_with_dynamic_users()` which simulates the platform over a defined number of steps with a given set of parameters.
*   **Objective Function:** `calculate_dynamic_user_growth_score()` which evaluates the outcome of a simulation run based on key metrics like user growth, user financial health, platform treasury, and content dynamism. This score is used for parameter tuning.
*   **Parameter Tuning Framework:**
    *   The `if __name__ == "__main__":` block sets up a grid search.
    *   It defines `managerial_params_grid` (parameters the platform controls) and allows testing against different `empirical_cases_to_test` (assumptions about user population behavior defined in `get_empirical_params_for_case()`).
    *   It iterates through combinations, runs simulations, calculates scores, and identifies the "best" parameter set.
    *   Results (summary CSV and plots) are saved to a uniquely named directory in `simulation_results_*`.

## How to Run

1.  **Prerequisites:**
    *   Python 3.x
    *   NumPy: `pip install numpy`
    *   Pandas: `pip install pandas`
    *   Matplotlib: `pip install matplotlib`
    *   Seaborn: `pip install seaborn`
    (It's highly recommended to use a virtual environment.)

2.  **Execute the Script:**
    ```bash
    python sim.py
    ```

3.  **Configure the Grid Search (in `sim.py`):**
    *   Modify `managerial_params_grid` to define the ranges of platform-controlled parameters you want to test.
    *   Modify `empirical_cases_to_test` and the `get_empirical_params_for_case()` function to define different scenarios of user population behavior.
    *   Adjust `fixed_sim_params` to set the scale and duration of the simulation runs (e.g., `POTENTIAL_USER_POOL_SIZE`, `SIMULATION_STEPS`).
    *   For faster iterations during initial exploration, reduce `SIMULATION_STEPS`, `POTENTIAL_USER_POOL_SIZE`, `NUM_POLITICIANS`, and the number of options in `managerial_params_grid`.
    *   Set `num_runs_per_set` (inside the main loop) to `1` for quick grid searches, or higher (e.g., 3-5) for more robust averaging when validating promising parameter sets.

4.  **Analyze Results:**
    *   The script will print the best parameter configuration found and its associated metrics.
    *   A summary CSV file and plots (user growth for the best run, parameter sensitivity, score distribution) will be saved in a timestamped sub-directory within `simulation_results_*`. These are crucial for understanding the impact of different parameters.

## Interpreting Results & Next Steps

The primary goal is to find **managerial parameter settings** that lead to a high `calculate_dynamic_user_growth_score` (or your custom objective function) across various plausible **empirical case scenarios**. This indicates a robust platform design.

Look for:

*   High `final_num_active_users` relative to `POTENTIAL_USER_POOL_SIZE`.
*   Low `users_broke_percent`.
*   A healthy, non-negative `final_treasury`.
*   A moderate `final_gini_coefficient`.
*   High overall user activity (`avg_actions_per_active_user_per_step_overall`).
*   Good content dynamism (`total_popular_word_events` and `unique_popular_word_strings_count`).

Use the generated plots and the summary CSV to understand which parameters have the most significant impact and to identify optimal ranges. The process is iterative: run a broad (but fast) search, identify promising regions, then conduct more focused searches with longer/larger simulations in those regions.

## Future Development / Model Enhancements

*   More sophisticated user archetypes and decision-making logic (e.g., learning, social influence).
*   Dynamic costs or rewards based on platform state.
*   Modeling the actual content of words and their semantic similarity.
*   More detailed churn models.
*   Integration with a proper data analysis and visualization dashboard.