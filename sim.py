import random
import collections
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os # For creating directories and file naming
from datetime import datetime # For timestamp in filename

# --- Classes ---
class User:
    def __init__(self, user_id, archetype, initial_balance, join_step):
        self.id = user_id
        self.archetype = archetype
        self.balance = initial_balance
        self.words_originated = {} # {(pol_id, word_str): step_submitted} - Tracks words this user was first to submit for a pol
        self.join_step = join_step
        self.last_activity_step = join_step
        self.total_rewards_received = 0.0
        self.active_in_sim = True # True if joined and not churned

class WordData:
    def __init__(self, original_submitter_id, submission_step):
        self.count = 0 # Total direct interactions (submissions/agreements) for this (pol,word)
        self.raw_popularity_score = 0 # A score that can decay, used for follower choice
        self.original_submitter_id = original_submitter_id # User who first submitted this (pol,word)
        self.submission_step = submission_step # Step when this (pol,word) was first created
        self.agreed_by_users = set() # Set of unique user_ids who have interacted (submitted/agreed)
        self.fees_contributed_to_pool = 0.0 # Accumulates the portion of fees for rewards for this (pol,word)
        self.reward_paid_out_for_threshold = {} # {threshold_value: True}

# --- Helper Functions ---
def get_gini(balances_list):
    """Calculates the Gini coefficient for a list of balances."""
    if not isinstance(balances_list, list) or not balances_list or len(balances_list) < 2:
        return 0.0
    
    balances = np.sort(np.array(balances_list, dtype=float))
    balances = np.maximum(balances, 0) # Gini not well-defined for negative values

    n = len(balances)
    if n == 0: return 0.0
    
    index = np.arange(1, n + 1)
    sum_balances = np.sum(balances)
    if sum_balances == 0: # Avoid division by zero if all balances are zero
        return 0.0
    
    return (np.sum((2 * index - n - 1) * balances)) / (n * sum_balances)

# Global for Zipfian distribution (initialized once per tuning session if fixed, or per run if varied)
FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], []

def initialize_zipfian_dictionary(vocab_size, zipf_alpha):
    """Initializes the global finite dictionary and its Zipfian probabilities."""
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS
    if vocab_size <=0 or zipf_alpha <= 0:
        # print("Error: vocab_size and zipf_alpha must be positive for Zipfian dictionary.")
        FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], [] # Ensure they are empty
        return

    FINITE_DICTIONARY_WORDS = [f"dict_word_{i}" for i in range(vocab_size)]
    if vocab_size > 0:
        word_indices = np.arange(1, vocab_size + 1) # Ranks from 1 to N
        probabilities = 1.0 / (word_indices**zipf_alpha)
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            FINITE_DICTIONARY_PROBS = probabilities / sum_probs
        else: # Should not happen with positive alpha and vocab_size > 0
            FINITE_DICTIONARY_PROBS = np.ones(vocab_size) / vocab_size # Fallback to uniform
    else:
        FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], []
    # print(f"Initialized Zipfian dictionary with {len(FINITE_DICTIONARY_WORDS)} words.")


def get_simulated_word_zipfian(user_archetype, params):
    """Generates a simulated word string based on Zipfian distribution."""
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS # Ensure access to globals
    
    # Fallback mechanism if dictionary is not properly initialized
    if not FINITE_DICTIONARY_WORDS or FINITE_DICTIONARY_PROBS.size == 0 or len(FINITE_DICTIONARY_WORDS) != len(FINITE_DICTIONARY_PROBS):
        # This indicates an issue with initialization or parameter passing.
        # print("Warning: Zipfian dictionary not properly initialized. Using fallback word generation.")
        if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_RARE_WORD_CHANCE', 0.1): # Default if param missing
            return f"fallback_unique_word_{random.randint(1000, 2000)}"
        return f"fallback_common_word_{random.randint(1, params.get('COMMON_WORD_VOCAB_SIZE', 200))}" # Default if param missing

    # Innovators might have a small chance to generate an "out-of-dictionary" word
    if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_OUT_OF_DICTIONARY_CHANCE', 0.05):
        return f"innov_ood_word_{random.randint(5000,6000)}" # OOD = Out Of Dictionary

    # Standard sampling from the Zipfian distribution
    try:
        chosen_word = np.random.choice(FINITE_DICTIONARY_WORDS, p=FINITE_DICTIONARY_PROBS)
        return chosen_word
    except ValueError as e:
        # This can happen if probabilities don't sum to 1.0 perfectly due to float precision.
        # print(f"Warning: np.random.choice error with Zipfian probs (sum: {np.sum(FINITE_DICTIONARY_PROBS)}): {e}. Falling back to uniform dictionary choice.")
        return random.choice(FINITE_DICTIONARY_WORDS) if FINITE_DICTIONARY_WORDS else "fallback_word"


# --- Platform Attractiveness & Joining/Churn ---
def calculate_platform_attractiveness(current_users_on_platform_list, platform_rewards_this_step, actions_this_step, params, step):
    """
    Calculates a score representing how attractive the platform is to potential new users.
    Score typically between 0 and 1 (or slightly more).
    """
    if not current_users_on_platform_list: # No users yet on platform
        return params.get('BASE_ATTRACTIVENESS_NO_USERS', 0.02) # Small base chance to join an empty platform

    num_platform_users = len(current_users_on_platform_list)
    balances_on_platform = [u.balance for u in current_users_on_platform_list if u.active_in_sim] # only consider active users for attractiveness
    
    reward_signal = 0
    # Normalize rewards by number of users and maybe by cost of action
    if num_platform_users > 0 and params.get('COST_AGREE_OR_RESUBMIT', 3) > 0 : # Avoid division by zero
        # Rewards per user relative to a cost of action (e.g., if avg reward is 1x cost of action)
        reward_per_user_norm = (platform_rewards_this_step / num_platform_users) / params.get('COST_AGREE_OR_RESUBMIT', 3)
        reward_signal = np.clip(reward_per_user_norm * params.get('ATTRACT_REWARD_SENSITIVITY', 0.3), 0, 0.35) # Max 0.35 from rewards

    # Factor 2: Activity Level - Is the platform busy?
    # Actions per user (those on platform) this step
    activity_signal = 0
    if num_platform_users > 0:
        activity_per_user = actions_this_step / num_platform_users
        # Normalize: e.g. if avg user does 0.5 actions, that's good. Max 1.
        activity_signal = np.clip(activity_per_user * params.get('ATTRACT_ACTIVITY_SENSITIVITY', 0.4), 0, 0.25) # Max 0.25 from activity

    # Factor 3: Perceived "Fairness" or "Opportunity" (inverse of Gini of active users)
    gini_on_platform = get_gini(balances_on_platform) if balances_on_platform else 1.0 # Assume max inequality if no balances or all zero
    fairness_signal = np.clip((1 - gini_on_platform) * params.get('ATTRACT_FAIRNESS_SENSITIVITY', 0.15), 0, 0.15) # Max 0.15 from fairness

    # Average balance change signal (more complex, requires tracking previous balances)
    # For simplicity now, let's use current avg balance relative to initial
    avg_balance_on_platform = np.mean(balances_on_platform) if balances_on_platform else 0
    balance_health_signal = 0
    if params.get('INITIAL_USER_BALANCE_MEAN', 100) > 0:
        balance_ratio = avg_balance_on_platform / params.get('INITIAL_USER_BALANCE_MEAN', 100)
        if balance_ratio < 0.5: # Penalize if avg balance is too low
            balance_health_signal = -0.1 # Negative impact on attractiveness
        elif balance_ratio > 0.8: # Reward if avg balance is healthy
            balance_health_signal = np.clip((balance_ratio - 0.8) * params.get('ATTRACT_BALANCE_SENSITIVITY', 0.2), 0, 0.2)


    base_attractiveness = params.get('BASE_ATTRACTIVENESS_WITH_USERS', 0.01)
    total_attractiveness = base_attractiveness + reward_signal + activity_signal + fairness_signal + balance_health_signal
    
    return np.clip(total_attractiveness, 0.001, 0.95) # Ensure it's a probability between 0.1% and 95%

# --- Main Simulation Function ---
def run_full_simulation_with_dynamic_users(params, simulation_run_id="sim_dyn_users_run"):
    # Unpack parameters with .get for safety and defaults
    POTENTIAL_USER_POOL_SIZE = params.get('POTENTIAL_USER_POOL_SIZE', 1000)
    INITIAL_ACTIVE_USERS = params.get('INITIAL_ACTIVE_USERS', 50)
    NUM_POLITICIANS = params.get('NUM_POLITICIANS', 5)
    SIMULATION_STEPS = params.get('SIMULATION_STEPS', 100)

    INITIAL_USER_BALANCE_MEAN = params.get('INITIAL_USER_BALANCE_MEAN', 100)
    INITIAL_USER_BALANCE_STDDEV = params.get('INITIAL_USER_BALANCE_STDDEV', 20)
    
    COST_SUBMIT_FIRST_TIME_WORD = params.get('COST_SUBMIT_FIRST_TIME_WORD', 5)
    COST_AGREE_OR_RESUBMIT = params.get('COST_AGREE_OR_RESUBMIT', 3)

    POPULARITY_THRESHOLD_BASE = params.get('POPULARITY_THRESHOLD_BASE', 10)
    POPULARITY_DECAY_RATE = params.get('POPULARITY_DECAY_RATE', 0.035)
    
    PLATFORM_RAKE_PERCENTAGE = params.get('PLATFORM_RAKE_PERCENTAGE', 0.30)
    
    REWARD_TO_ORIGINAL_SUBMITTER_SHARE = params.get('REWARD_TO_ORIGINAL_SUBMITTER_SHARE', 0.65)
    REWARD_TO_AGREERS_SHARE = 1.0 - REWARD_TO_ORIGINAL_SUBMITTER_SHARE # Derived

    # User behavior archetypes and probabilities
    USER_ARCHETYPES_DIST = params.get('USER_ARCHETYPES_DIST', {'Innovator': 0.20, 'Follower': 0.60, 'Balanced': 0.20})
    INNOVATOR_PROB_NEW_CONCEPT = params.get('INNOVATOR_PROB_NEW_CONCEPT', 0.6)
    INNOVATOR_PROB_AGREE = params.get('INNOVATOR_PROB_AGREE', 0.3)
    FOLLOWER_PROB_NEW_CONCEPT = params.get('FOLLOWER_PROB_NEW_CONCEPT', 0.1)
    FOLLOWER_PROB_AGREE = params.get('FOLLOWER_PROB_AGREE', 0.8)
    FOLLOWER_POPULARITY_BIAS_FACTOR = params.get('FOLLOWER_POPULARITY_BIAS_FACTOR', 2.0)
    BALANCED_PROB_NEW_CONCEPT = params.get('BALANCED_PROB_NEW_CONCEPT', 0.40)
    BALANCED_PROB_AGREE = params.get('BALANCED_PROB_AGREE', 0.50)
    USER_ACTIVITY_RATE_ON_PLATFORM = params.get('USER_ACTIVITY_RATE_ON_PLATFORM', 0.7) # % of current users who are active
    
    # Churn parameters
    ENABLE_CHURN = params.get('ENABLE_CHURN', True)
    CHURN_INACTIVITY_THRESHOLD_STEPS = params.get('CHURN_INACTIVITY_THRESHOLD_STEPS', 15)
    CHURN_LOW_BALANCE_THRESHOLD = params.get('CHURN_LOW_BALANCE_THRESHOLD') # This will be derived and passed in p_full_config_combo
    CHURN_BASE_PROB_IF_CONDITIONS_MET = params.get('CHURN_BASE_PROB_IF_CONDITIONS_MET', 0.08)

    # --- Data Structures Initialization ---
    users_master_list = {} # Stores all User objects, {user_id: User_instance}
    next_user_id_counter = 0 # To generate unique user IDs
    
    politicians_dict = {f"pol_{i}": {'words': {}} for i in range(NUM_POLITICIANS)}
    platform_treasury = 0.0
    platform_total_rewards_paid_overall = 0.0
    total_word_instances_created = 0 # Unique (pol,word) pairs
    total_actions_simulation = 0 # Accumulator for all actions in the simulation
    
    archetype_keys = list(USER_ARCHETYPES_DIST.keys())
    archetype_probs = np.array(list(USER_ARCHETYPES_DIST.values()))
    if abs(np.sum(archetype_probs) - 1.0) > 1e-5 : 
        archetype_probs /= np.sum(archetype_probs)

    # Seed initial users
    for _ in range(INITIAL_ACTIVE_USERS):
        user_id = f"user_{next_user_id_counter}"
        archetype = np.random.choice(archetype_keys, p=archetype_probs)
        balance = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
        users_master_list[user_id] = User(user_id, archetype, balance, join_step=0)
        next_user_id_counter += 1
    
    # History trackers
    history_num_active_users = []
    history_platform_attractiveness = []
    history_avg_user_balance = [] # New history tracker

    # --- Simulation Loop ---
    for step in range(1, SIMULATION_STEPS + 1):
        current_platform_active_users_objs = [u for u in users_master_list.values() if u.active_in_sim]
        num_current_platform_active_users = len(current_platform_active_users_objs)
        
        actions_this_step = 0
        platform_rewards_this_step = 0 # Reset for attractiveness calculation

        # --- User Actions (for users currently on platform) ---
        if num_current_platform_active_users > 0:
            potential_actors_ids = [u.id for u in current_platform_active_users_objs]
            # Ensure sample size is not greater than population
            num_to_sample = min(len(potential_actors_ids), int(num_current_platform_active_users * USER_ACTIVITY_RATE_ON_PLATFORM))
            actors_for_this_step_ids = random.sample(potential_actors_ids, num_to_sample)

            for user_id in actors_for_this_step_ids:
                user = users_master_list[user_id] # Should exist and be active
                user.last_activity_step = step 
                action_taken_this_turn_by_user = False
                
                # Determine action probabilities based on archetype
                if user.archetype == 'Innovator': prob_try_new, prob_try_agree = INNOVATOR_PROB_NEW_CONCEPT, INNOVATOR_PROB_AGREE
                elif user.archetype == 'Follower': prob_try_new, prob_try_agree = FOLLOWER_PROB_NEW_CONCEPT, FOLLOWER_PROB_AGREE
                else: prob_try_new, prob_try_agree = BALANCED_PROB_NEW_CONCEPT, BALANCED_PROB_AGREE
                
                action_roll = random.random()
                chosen_pol_id = f"pol_{random.randint(0, NUM_POLITICIANS - 1)}"
                pol_words_dict = politicians_dict[chosen_pol_id]['words']

                # Try to submit/introduce a word
                if action_roll < prob_try_new:
                    chosen_word_str = get_simulated_word_zipfian(user.archetype, params)
                    is_first_time_for_this_pol_word = chosen_word_str not in pol_words_dict
                    current_action_cost = COST_SUBMIT_FIRST_TIME_WORD if is_first_time_for_this_pol_word else COST_AGREE_OR_RESUBMIT

                    if user.balance >= current_action_cost:
                        user.balance -= current_action_cost
                        rake_for_this_action = current_action_cost * PLATFORM_RAKE_PERCENTAGE
                        reward_pool_contribution = current_action_cost - rake_for_this_action
                        platform_treasury += rake_for_this_action
                        
                        if is_first_time_for_this_pol_word:
                            pol_words_dict[chosen_word_str] = WordData(user.id, step)
                            total_word_instances_created += 1
                            user.words_originated[(chosen_pol_id, chosen_word_str)] = step
                        
                        word_obj = pol_words_dict[chosen_word_str]
                        word_obj.count += 1
                        word_obj.raw_popularity_score +=1 # Boost raw score on interaction
                        word_obj.fees_contributed_to_pool += reward_pool_contribution
                        word_obj.agreed_by_users.add(user.id)
                        action_taken_this_turn_by_user = True
                
                # Try to agree with an existing word
                elif prob_try_new <= action_roll < prob_try_new + prob_try_agree:
                    if not pol_words_dict: continue
                    existing_word_strs = list(pol_words_dict.keys())
                    if not existing_word_strs: continue

                    word_to_agree_with_str = ""
                    if user.archetype == 'Follower':
                        word_scores = np.array([pol_words_dict[w].raw_popularity_score + 0.1 for w in existing_word_strs])
                        word_scores = np.maximum(word_scores, 0.01)
                        word_scores_biased = word_scores ** FOLLOWER_POPULARITY_BIAS_FACTOR
                        sum_biased_scores = np.sum(word_scores_biased)
                        if sum_biased_scores > 1e-6 :
                            probs = word_scores_biased / sum_biased_scores
                            if not np.isnan(probs).any() and abs(np.sum(probs) - 1.0) < 1e-5:
                                 try: word_to_agree_with_str = np.random.choice(existing_word_strs, p=probs)
                                 except ValueError: word_to_agree_with_str = random.choice(existing_word_strs)
                            else: word_to_agree_with_str = random.choice(existing_word_strs)
                        else: word_to_agree_with_str = random.choice(existing_word_strs)
                    else: word_to_agree_with_str = random.choice(existing_word_strs)

                    if word_to_agree_with_str and user.balance >= COST_AGREE_OR_RESUBMIT:
                        user.balance -= COST_AGREE_OR_RESUBMIT
                        rake_for_this_action = COST_AGREE_OR_RESUBMIT * PLATFORM_RAKE_PERCENTAGE
                        reward_pool_contribution = COST_AGREE_OR_RESUBMIT - rake_for_this_action
                        platform_treasury += rake_for_this_action
                        
                        word_obj = pol_words_dict[word_to_agree_with_str]
                        word_obj.count += 1
                        word_obj.raw_popularity_score +=1
                        word_obj.fees_contributed_to_pool += reward_pool_contribution
                        word_obj.agreed_by_users.add(user.id)
                        action_taken_this_turn_by_user = True
                
                if action_taken_this_turn_by_user:
                    actions_this_step += 1
        
        total_actions_simulation += actions_this_step

        # --- Decay raw_popularity_score and Process Rewards ---
        step_rewards_this_step_val = 0 # To use in attractiveness calculation
        for pol_id, pol_data in politicians_dict.items():
            for word_str, word_obj in pol_data['words'].items():
                if word_obj.raw_popularity_score > 0: # Decay only if positive
                    word_obj.raw_popularity_score -= word_obj.raw_popularity_score * POPULARITY_DECAY_RATE
                    if word_obj.raw_popularity_score < 0.01: word_obj.raw_popularity_score = 0
                
                current_reward_threshold = POPULARITY_THRESHOLD_BASE
                if word_obj.count >= current_reward_threshold and not word_obj.reward_paid_out_for_threshold.get(current_reward_threshold, False):
                    reward_pool = word_obj.fees_contributed_to_pool
                    original_submitter_id = word_obj.original_submitter_id
                    
                    if original_submitter_id and original_submitter_id in users_master_list and users_master_list[original_submitter_id].active_in_sim:
                        reward_for_originator = reward_pool * REWARD_TO_ORIGINAL_SUBMITTER_SHARE
                        users_master_list[original_submitter_id].balance += reward_for_originator
                        users_master_list[original_submitter_id].total_rewards_received += reward_for_originator
                        platform_total_rewards_paid_overall += reward_for_originator
                        step_rewards_this_step_val += reward_for_originator
                    
                    num_agreers = len(word_obj.agreed_by_users)
                    if num_agreers > 0:
                        reward_for_agreers_total = reward_pool * REWARD_TO_AGREERS_SHARE
                        per_agreer_reward = reward_for_agreers_total / num_agreers
                        for agreer_id in word_obj.agreed_by_users:
                            if agreer_id in users_master_list and users_master_list[agreer_id].active_in_sim:
                                users_master_list[agreer_id].balance += per_agreer_reward
                                users_master_list[agreer_id].total_rewards_received += per_agreer_reward
                                platform_total_rewards_paid_overall += per_agreer_reward
                                step_rewards_this_step_val += per_agreer_reward
                    
                    word_obj.reward_paid_out_for_threshold[current_reward_threshold] = True
                    word_obj.fees_contributed_to_pool = 0.0
        
        # --- New User Joining ---
        current_active_users_for_attract_calc = [u for u in users_master_list.values() if u.active_in_sim]
        attractiveness = calculate_platform_attractiveness(
            current_active_users_for_attract_calc,
            step_rewards_this_step_val, # Pass rewards from this step
            actions_this_step,          # Pass actions from this step
            params, step
        )
        history_platform_attractiveness.append(attractiveness)

        newly_joined_count_this_step = 0
        if len(users_master_list) < POTENTIAL_USER_POOL_SIZE: # Only try to add if pool isn't full
            # Number of potential new users considering joining this step
            num_considering_join = min(
                POTENTIAL_USER_POOL_SIZE - len(users_master_list), # Max available spots
                int(params.get('POTENTIAL_JOIN_TRIALS_PER_STEP', 50) + len(current_active_users_for_attract_calc) * 0.05) # Dynamic trials
            )
            max_actual_joins_this_step = int(params.get('MAX_NEW_JOINS_PER_STEP_SCALER', 0.02) * POTENTIAL_USER_POOL_SIZE)

            for _ in range(num_considering_join):
                if len(users_master_list) >= POTENTIAL_USER_POOL_SIZE or newly_joined_count_this_step >= max_actual_joins_this_step:
                    break # Stop if pool is full or step join cap is reached
                if random.random() < attractiveness:
                    user_id = f"user_{next_user_id_counter}"
                    archetype = np.random.choice(archetype_keys, p=archetype_probs)
                    balance = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
                    users_master_list[user_id] = User(user_id, archetype, balance, join_step=step)
                    next_user_id_counter += 1
                    newly_joined_count_this_step += 1
        
        # --- User Churn ---
        if ENABLE_CHURN and step > params.get('CHURN_GRACE_PERIOD_STEPS', 10):
            users_to_churn_ids = []
            # Iterate over a copy of keys if modifying the dictionary (though here we just mark inactive)
            for user_id_key, user_obj_val in list(users_master_list.items()): 
                if not user_obj_val.active_in_sim: continue # Already churned

                churn_prob_this_user = 0.0
                if (step - user_obj_val.last_activity_step) > CHURN_INACTIVITY_THRESHOLD_STEPS:
                    churn_prob_this_user += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                if user_obj_val.balance < CHURN_LOW_BALANCE_THRESHOLD: # CHURN_LOW_BALANCE_THRESHOLD is now in merged_params
                    churn_prob_this_user += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                # More aggressive churn if consistently not earning rewards and balance is low
                if user_obj_val.total_rewards_received == 0 and user_obj_val.balance < INITIAL_USER_BALANCE_MEAN * 0.4 and step > user_obj_val.join_step + (SIMULATION_STEPS * 0.1): # e.g. after 10% of sim
                     churn_prob_this_user += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.4 # Higher chance

                if random.random() < np.clip(churn_prob_this_user, 0, 0.75): # Max churn prob for these factors
                    users_to_churn_ids.append(user_id_key)
            
            for uid_to_churn in users_to_churn_ids:
                if uid_to_churn in users_master_list: # Double check
                    users_master_list[uid_to_churn].active_in_sim = False
        
        current_active_after_churn_and_join = len([u for u in users_master_list.values() if u.active_in_sim])
        history_num_active_users.append(current_active_after_churn_and_join)
        
        # Store average balance of active users for history
        active_balances_hist = [u.balance for u in users_master_list.values() if u.active_in_sim]
        history_avg_user_balance.append(np.mean(active_balances_hist) if active_balances_hist else 0)


    # --- Collect Final Metrics ---
    final_active_users_list = [u for u in users_master_list.values() if u.active_in_sim]
    final_balances = [u.balance for u in final_active_users_list] if final_active_users_list else [0.0] # Avoid empty list for numpy ops
    
    final_num_active_users = len(final_active_users_list)
    final_gini = get_gini(final_balances)
    final_avg_balance = np.mean(final_balances) if final_balances else 0.0 # Check if final_balances could be empty
    final_median_balance = np.median(final_balances) if final_balances else 0.0

    min_action_cost_val = min(COST_SUBMIT_FIRST_TIME_WORD, COST_AGREE_OR_RESUBMIT)
    users_broke_count = sum(1 for b in final_balances if b < min_action_cost_val)
    
    total_popular_word_events = 0
    unique_popular_words_global = set()
    for pol_data in politicians_dict.values():
        for word_str, word_obj in pol_data['words'].items():
            if word_obj.reward_paid_out_for_threshold.get(POPULARITY_THRESHOLD_BASE, False):
                total_popular_word_events +=1
                unique_popular_words_global.add(word_str)

    avg_hist_users_val = np.mean(history_num_active_users) if history_num_active_users else 0
    
    return {
        "params_config_copy": merged_params.copy(), # Important: pass the actual merged_params used
        "final_num_active_users": final_num_active_users,
        "final_treasury": platform_treasury,
        "total_rewards_paid": platform_total_rewards_paid_overall,
        "final_avg_balance": final_avg_balance,
        "final_median_balance": final_median_balance,
        "final_gini_coefficient": final_gini,
        "users_broke_percent": (users_broke_count / final_num_active_users) * 100 if final_num_active_users > 0 else (100.0 if INITIAL_ACTIVE_USERS > 0 and final_num_active_users == 0 else 0.0),
        "total_actions_in_sim": total_actions_simulation, # Use the correctly accumulated total
        "avg_actions_per_active_user_per_step_overall": total_actions_simulation / (avg_hist_users_val * SIMULATION_STEPS) if avg_hist_users_val * SIMULATION_STEPS > 0 else 0,
        "total_unique_word_instances_created": total_word_instances_created,
        "unique_popular_word_strings_count": len(unique_popular_words_global),
        "total_popular_word_events": total_popular_word_events,
        "history_num_active_users": history_num_active_users, # For plotting growth
        "history_platform_attractiveness": history_platform_attractiveness, # For plotting
        "history_avg_user_balance": history_avg_user_balance # For plotting
    }

# --- Objective Function (Refined for Dynamic Users & User Growth Maximization) ---
def calculate_dynamic_user_growth_score(results, target_gini=0.50, min_avg_balance_ratio_retained=0.80, desired_final_user_ratio_of_potential=0.75):
    """
    Calculates a score. Higher is better.
    Focuses on maximizing user growth (reaching high % of potential pool)
    while maintaining user financial health and platform viability.
    """
    params = results['params_config_copy']
    score = 0.0
    
    POTENTIAL_USER_POOL_SIZE = params.get('POTENTIAL_USER_POOL_SIZE', 1000) # Ensure this is from params

    # --- Primary Driver: User Growth ---
    final_user_ratio_of_potential = results['final_num_active_users'] / POTENTIAL_USER_POOL_SIZE if POTENTIAL_USER_POOL_SIZE > 0 else 0
    
    # Strong reward for achieving high percentage of potential pool
    growth_target_score = 0
    if final_user_ratio_of_potential < desired_final_user_ratio_of_potential * 0.5: # Significantly below target
        growth_target_score = -((desired_final_user_ratio_of_potential * 0.5 - final_user_ratio_of_potential) * 30000) # Large penalty
    else:
        # Reward proportionally for getting close, bonus for hitting/exceeding
        growth_target_score = np.clip(final_user_ratio_of_potential / desired_final_user_ratio_of_potential, 0, 1.15) * 12000.0 
    if final_user_ratio_of_potential >= desired_final_user_ratio_of_potential:
        growth_target_score += 5000 # Bonus for achieving target

    score += growth_target_score

    # --- Secondary Modifiers: Health of the ecosystem for those who joined ---
    if results['final_num_active_users'] > params.get('INITIAL_ACTIVE_USERS',10) * 0.2 : # Apply these only if a reasonable number of users are left (e.g. >20% of initial)
        
        # 1. Platform Treasury (must be solvent, avoid excessive extraction if growth is already good)
        treasury_score_component = 0
        if results['final_treasury'] < -(results['final_num_active_users'] * params['INITIAL_USER_BALANCE_MEAN'] * 0.05): 
            treasury_score_component = -20000 # Massive penalty if platform is bankrupt
        elif results['final_treasury'] < 0: 
            treasury_score_component = results['final_treasury'] * 5.0 # Significant penalty for any negative treasury
        else: 
            # Smaller positive impact compared to user growth, just ensures it's not losing money.
            # Cap the positive contribution from treasury to avoid over-optimizing for rake.
            treasury_cap = results['total_actions_in_sim'] * params.get('PLATFORM_RAKE_PERCENTAGE', 0.2) * params.get('COST_AGREE_OR_RESUBMIT',3) * 0.5 # e.g. max 50% of total possible rake
            treasury_score_component = np.clip(results['final_treasury'], 0, treasury_cap) * 0.0005
        score += treasury_score_component

        # 2. User Broke Percentage (among active users at the end)
        target_broke_active = params.get('TARGET_BROKE_AMONG_ACTIVE_PERCENT', 8.0) # Stricter: aim for <8% broke among active
        broke_diff = results['users_broke_percent'] - target_broke_active
        if broke_diff > 0: 
            score -= (broke_diff / 3.0)**2 * 800.0  # Very sharp penalty for exceeding target broke %

        # 3. Average User Balance (for active users - should not be totally depleted)
        avg_balance_ratio = results['final_avg_balance'] / params['INITIAL_USER_BALANCE_MEAN'] if params['INITIAL_USER_BALANCE_MEAN'] > 0 else 0
        if avg_balance_ratio < min_avg_balance_ratio_retained: # e.g., if avg balance falls below 80% of initial for retained users
            score -= (min_avg_balance_ratio_retained - avg_balance_ratio) * 1500.0 # Strong penalty
        elif avg_balance_ratio > 1.6: # Discourage balances becoming too inflated (might signal broken economy)
            score -= (avg_balance_ratio - 1.6) * 250.0
        else: # In a good range
            score += avg_balance_ratio * 200.0 

        # 4. Gini Coefficient (Inequality among active users)
        gini_diff = results['final_gini_coefficient'] - target_gini
        if gini_diff > 0 : # If Gini is worse (higher) than target
            score -= gini_diff * 1200.0 
        else: # If Gini is better (lower) than target
            score += (target_gini - results['final_gini_coefficient']) * 300.0 

        # 5. Activity Level (overall, normalized by avg users and steps)
        # This indicates how engaging the platform is for users who are on it.
        score += results['avg_actions_per_active_user_per_step_overall'] * 7000.0 # Very strong weight

        # 6. Ecosystem Dynamism (Popular Word Events & Diversity)
        if results['total_actions_in_sim'] > 0:
            pop_ev_per_1k_act = (results['total_popular_word_events'] / results['total_actions_in_sim']) * 1000
            score += pop_ev_per_1k_act * 80.0 
        if results['total_popular_word_events'] > 0:
            diversity = (results['unique_popular_word_strings_count'] / results['total_popular_word_events'])
            score += diversity * 700.0 # Strong reward for diversity
        elif results['total_unique_word_instances_created'] > 0 and results['total_popular_word_events'] == 0:
            score -= 800 # Penalize if words are created but nothing becomes popular
    else: # Catastrophic collapse: No (or very few) active users at the end
        score -= 100000 

    return score

# --- Main Execution for Parameter Tuning ---
if __name__ == "__main__":
    # --- Define EMPIRICAL POPULATION CASES ---
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO'

    # --- Parameters that are MANAGERIAL LEVERS (VERY FOCUSED GRID for "Accelerated Happy Medium") ---
    managerial_params_grid = {
        'POPULARITY_THRESHOLD_BASE': [8, 12, 18],      # Test a few thresholds
        'PLATFORM_RAKE_PERCENTAGE': [0.15, 0.25, 0.35], # Test a few rake levels
        # 'COST_AGREE_OR_RESUBMIT': [3],              # Keeping some fixed for this focused run
        # 'COST_SUBMIT_FIRST_TIME_WORD': [5],         # Keeping some fixed
    }
    # This grid: 3 (POP_THRESH) * 3 (RAKE) = 9 managerial combinations.
    # Total sims = 9 managerial combos * 3 empirical cases = 27 simulations.

    # --- Parameters that are FIXED for this "Accelerated Happy Medium" tuning session ---
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 15000,  # Scaled down but still significant
        'INITIAL_ACTIVE_USERS': 80,        # Seed users
        'NUM_POLITICIANS': 30,             
        'SIMULATION_STEPS': 120,          # ~4 months of daily steps

        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        
        'COST_SUBMIT_FIRST_TIME_WORD': 5, # Fixed based on previous good results/intuition
        'COST_AGREE_OR_RESUBMIT': 3,      # Fixed
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.70, # Slightly higher for originator
        'POPULARITY_DECAY_RATE': 0.03,

        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 35,
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.8,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.04,

        'FINITE_DICTIONARY_SIZE': 1200,
        'ZIPFIAN_ALPHA': 1.08, # Slightly flatter Zipf
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.04,
        'COMMON_WORD_VOCAB_SIZE': 150, # Fallback
        'INNOVATOR_RARE_WORD_CHANCE': 0.1, # Fallback

        # Fixed Attractiveness Sensitivities for this run to focus on core econ levers
        'BASE_ATTRACTIVENESS_NO_USERS': 0.035,
        'BASE_ATTRACTIVENESS_WITH_USERS': 0.025,
        'ATTRACT_REWARD_SENSITIVITY': 0.40,
        'ATTRACT_ACTIVITY_SENSITIVITY': 0.50,
        'ATTRACT_FAIRNESS_SENSITIVITY': 0.20,
        'ATTRACT_BALANCE_SENSITIVITY': 0.30,
    }

    # --- Create parameter combinations ---
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            current_combo = fixed_sim_params.copy()
            empirical_set_for_case = get_empirical_params_for_case(empirical_case_name)
            current_combo.update(empirical_set_for_case) 
            current_combo.update(managerial_combo) 
            current_combo[empirical_case_param_name] = empirical_case_name
            min_cost_val = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD'), current_combo.get('COST_AGREE_OR_RESUBMIT'))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')
            all_parameter_combinations.append(current_combo)
    
    num_total_combinations = len(all_parameter_combinations)
    print(f"Starting 'Accelerated Happy Medium (v2)' Grid Search with {num_total_combinations} total parameter combinations.")
    print(f"Each simulation run will have {fixed_sim_params['SIMULATION_STEPS']} steps.")

    all_sim_results_for_df = []
    best_overall_score = -float('inf')
    best_params_overall = None
    best_detailed_results_overall = None
    tuning_start_time = time.time()

    initialize_zipfian_dictionary(
        fixed_sim_params.get('FINITE_DICTIONARY_SIZE'),
        fixed_sim_params.get('ZIPFIAN_ALPHA')
    )

    for i, p_full_config_combo in enumerate(all_parameter_combinations):
        sim_id_str = f"accel_hm_v2_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        if (i + 1) % (num_total_combinations // 5 or 1) == 0 or i == 0 or (i+1) == num_total_combinations:
            print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 1 # Keep at 1 for this accelerated grid search phase
        current_set_scores, current_set_details_list = [], []
        
        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.65) # Target 65% of this pool
            current_set_scores.append(score)
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        rep_details = current_set_details_list[0] if current_set_details_list else {}
        
        if (i + 1) % (num_total_combinations // 5 or 1) == 0 or i == 0 or (i+1) == num_total_combinations:
             print(f"  Combo {i+1} (Case: {empirical_case_tag}) Score: {avg_score:,.2f} | EndUsers: {rep_details.get('final_num_active_users',0)}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {rep_details.get('final_treasury',0):,.0f} | Broke%: {rep_details.get('users_broke_percent',0):.1f}")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  -------> *** New Best: {best_overall_score:,.2f} (Combo {i+1}, Case: {empirical_case_tag}, EndUsers: {best_detailed_results_overall.get('final_num_active_users')}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']}, Broke%: {best_detailed_results_overall.get('users_broke_percent'):.1f}%) *** <-------")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_accel_hm_v2"
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_ahm_v2_{timestamp_str}"
    
    print(f"\n--- 'Accelerated Happy Medium (v2)' Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")

    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Overall Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Parameter Configuration Found (Managerial Levers Varied in this Run):")
        varied_managerial_keys_in_grid = [k for k,v in managerial_params_grid.items() if isinstance(v,list) and len(v)>1]
        for k_man_var in varied_managerial_keys_in_grid: print(f"  {k_man_var}: {best_params_overall.get(k_man_var)}")
        
        print("\nMetrics for Best Config (Single Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", "total_popular_word_events", "unique_popular_word_strings_count", "avg_actions_per_active_user_per_step_overall"]
        for k_met_show in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met_show)
            if isinstance(val, float): print(f"  {k_met_show}: {val:,.4f}")
            else: print(f"  {k_met_show}: {val}")
        
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        if not df_all_runs.empty:
            try:
                df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
                print(f"\nFull summary saved to: {results_filename_base}_summary.csv")
            except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")

            if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
                plt.figure(figsize=(10,6))
                history_active = best_detailed_results_overall['history_num_active_users']
                history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
                plt.plot(history_active, label=f"Active Users (Best Score: {best_overall_score:,.0f})", color="navy", linewidth=2)
                plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='orangered', linestyle='--', label="Potential Pool Size")
                ax_attract_twin = plt.gca().twinx()
                if history_attract:
                    ax_attract_twin.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="forestgreen", linestyle=':', alpha=0.8)
                    ax_attract_twin.set_ylabel("Attractiveness Score (x100)", color="forestgreen", fontsize=10); ax_attract_twin.tick_params(axis='y', labelcolor="forestgreen", labelsize=8)
                plt.gca().set_xlabel("Simulation Step", fontsize=10); plt.gca().set_ylabel("Active Users", color="navy", fontsize=10)
                plt.gca().tick_params(axis='y', labelcolor="navy", labelsize=8); plt.gca().tick_params(axis='x', labelsize=8)
                plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=12)
                lines1, labels1 = plt.gca().get_legend_handles_labels(); lines2_twin, labels2_twin = ax_attract_twin.get_legend_handles_labels()
                if lines2_twin: ax_attract_twin.legend(lines1 + lines2_twin, labels1 + labels2_twin, loc='best', fontsize=8)
                else: plt.gca().legend(loc='best', fontsize=8)
                plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
                plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
            
            if empirical_case_param_name in df_all_runs.columns:
                plt.figure(figsize=(8, 5))
                sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Spectral", hue=empirical_case_param_name, legend=False)
                plt.title("Impact of Empirical Cases (Accel. HM v2)", fontsize=12); plt.xlabel("Empirical Case", fontsize=10); plt.ylabel("Score", fontsize=10)
                plt.xticks(rotation=10, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.tight_layout()
                plt.savefig(f"{results_filename_base}_empirical_case_impact.png"); plt.show()

            managerial_params_to_plot = [k for k,v_list_plot in managerial_params_grid.items() if isinstance(v_list_plot,list) and len(v_list_plot)>1]
            if managerial_params_to_plot:
                print("\nGenerating limited sensitivity plots (faceted by empirical case)...")
                for param_name_plot in managerial_params_to_plot:
                    if param_name_plot in df_all_runs.columns and df_all_runs[param_name_plot].nunique() > 1:
                        g = sns.catplot(x=param_name_plot, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="PRGn", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)), height=4, aspect=1.1)
                        g.fig.suptitle(f"Score vs. {param_name_plot} (by Case - Accel. HM v2)", fontsize=13, y=1.03); g.set_xticklabels(rotation=20, ha='right', fontsize=8)
                        g.set_ylabels("Score", fontsize=9); g.set_xlabels(param_name_plot, fontsize=9); g.tick_params(labelsize=8)
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name_plot}_by_case.png"); plt.show()
            
            plt.figure(figsize=(8,5)); scores_hist_vals = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist_vals:
                plt.hist(scores_hist_vals, bins=max(8, num_total_combinations//2 if num_total_combinations>0 else 1), edgecolor='black', color='salmon')
                plt.title('Score Distribution (Accel. HM v2 - All Cases)', fontsize=12); plt.xlabel('Score', fontsize=10); plt.ylabel('# Combos', fontsize=10); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.yticks(fontsize=9); plt.xticks(fontsize=9); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png")
    else: print("No simulations run or no valid results.")