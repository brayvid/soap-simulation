import random
import collections
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime

# --- [Keep User, WordData classes, get_gini, initialize_zipfian_dictionary, get_simulated_word_zipfian, calculate_platform_attractiveness] ---
# (These are assumed to be the same as the previous "full file" version)
class User:
    def __init__(self, user_id, archetype, initial_balance, join_step):
        self.id = user_id; self.archetype = archetype; self.balance = initial_balance
        self.words_originated = {}; self.join_step = join_step; self.last_activity_step = join_step
        self.total_rewards_received = 0.0; self.active_in_sim = True
class WordData:
    def __init__(self, original_submitter_id, submission_step):
        self.count = 0; self.raw_popularity_score = 0; self.original_submitter_id = original_submitter_id
        self.submission_step = submission_step; self.agreed_by_users = set(); self.fees_contributed_to_pool = 0.0
        self.reward_paid_out_for_threshold = {}
def get_gini(balances_list):
    if not isinstance(balances_list, list) or not balances_list or len(balances_list) < 2: return 0.0
    balances = np.sort(np.array(balances_list, dtype=float)); balances = np.maximum(balances, 0)
    n = len(balances);
    if n == 0: return 0.0
    index = np.arange(1, n + 1); sum_balances = np.sum(balances)
    if sum_balances == 0: return 0.0
    return (np.sum((2 * index - n - 1) * balances)) / (n * sum_balances)
FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], []
def initialize_zipfian_dictionary(vocab_size, zipf_alpha):
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS
    if vocab_size <=0 or zipf_alpha <= 0: FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], []; return
    FINITE_DICTIONARY_WORDS = [f"dict_word_{i}" for i in range(vocab_size)]
    if vocab_size > 0:
        word_indices = np.arange(1, vocab_size + 1); probabilities = 1.0 / (word_indices**zipf_alpha)
        sum_probs = np.sum(probabilities)
        if sum_probs > 0: FINITE_DICTIONARY_PROBS = probabilities / sum_probs
        else: FINITE_DICTIONARY_PROBS = np.ones(vocab_size) / vocab_size
    else: FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], []
def get_simulated_word_zipfian(user_archetype, params):
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS
    if not FINITE_DICTIONARY_WORDS or FINITE_DICTIONARY_PROBS.size == 0 or len(FINITE_DICTIONARY_WORDS) != len(FINITE_DICTIONARY_PROBS):
        if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_RARE_WORD_CHANCE', 0.1): return f"fallback_unique_word_{random.randint(1000, 2000)}"
        return f"fallback_common_word_{random.randint(1, params.get('COMMON_WORD_VOCAB_SIZE', 200))}"
    if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_OUT_OF_DICTIONARY_CHANCE', 0.05): return f"innov_ood_word_{random.randint(5000,6000)}"
    try: return np.random.choice(FINITE_DICTIONARY_WORDS, p=FINITE_DICTIONARY_PROBS)
    except ValueError: return random.choice(FINITE_DICTIONARY_WORDS) if FINITE_DICTIONARY_WORDS else "fallback_word"
def calculate_platform_attractiveness(current_users_on_platform_list, platform_rewards_this_step, actions_this_step, params, step):
    if not current_users_on_platform_list: return params.get('BASE_ATTRACTIVENESS_NO_USERS', 0.02)
    num_platform_users = len(current_users_on_platform_list)
    balances_on_platform = [u.balance for u in current_users_on_platform_list if u.active_in_sim]
    reward_signal = 0
    if num_platform_users > 0 and params.get('COST_AGREE_OR_RESUBMIT', 3) > 0:
        reward_per_user_norm = (platform_rewards_this_step / num_platform_users) / params.get('COST_AGREE_OR_RESUBMIT', 3)
        reward_signal = np.clip(reward_per_user_norm * params.get('ATTRACT_REWARD_SENSITIVITY', 0.3), 0, 0.35)
    activity_signal = 0
    if num_platform_users > 0:
        activity_per_user = actions_this_step / num_platform_users
        activity_signal = np.clip(activity_per_user * params.get('ATTRACT_ACTIVITY_SENSITIVITY', 0.4), 0, 0.25)
    gini_on_platform = get_gini(balances_on_platform) if balances_on_platform else 1.0
    fairness_signal = np.clip((1 - gini_on_platform) * params.get('ATTRACT_FAIRNESS_SENSITIVITY', 0.15), 0, 0.15)
    avg_balance_on_platform = np.mean(balances_on_platform) if balances_on_platform else 0
    balance_health_signal = 0
    if params.get('INITIAL_USER_BALANCE_MEAN', 100) > 0:
        balance_ratio = avg_balance_on_platform / params.get('INITIAL_USER_BALANCE_MEAN', 100)
        if balance_ratio < 0.5: balance_health_signal = -0.1
        elif balance_ratio > 0.8: balance_health_signal = np.clip((balance_ratio - 0.8) * params.get('ATTRACT_BALANCE_SENSITIVITY', 0.2), 0, 0.2)
    base_attractiveness = params.get('BASE_ATTRACTIVENESS_WITH_USERS', 0.01)
    total_attractiveness = base_attractiveness + reward_signal + activity_signal + fairness_signal + balance_health_signal
    return np.clip(total_attractiveness, 0.001, 0.95)


# --- Function to define Empirical Parameter Sets ---
def get_empirical_params_for_case(case_name):
    """
    Returns a dictionary of empirical parameters based on the case_name.
    These are parameters representing user population tendencies.
    """
    if case_name == "BaselineEngagement":
        return {
            'USER_ARCHETYPES_DIST': {'Innovator': 0.20, 'Follower': 0.65, 'Balanced': 0.15},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.0,
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.70, # Overall activity rate for joined users
            'INNOVATOR_PROB_NEW_CONCEPT': 0.60,'INNOVATOR_PROB_AGREE': 0.30,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.10,'FOLLOWER_PROB_AGREE': 0.80,
            'BALANCED_PROB_NEW_CONCEPT': 0.35,'BALANCED_PROB_AGREE': 0.55,
            # Attractiveness parameters (how sensitive *potential new users* are)
            'BASE_ATTRACTIVENESS_NO_USERS': 0.03, 'BASE_ATTRACTIVENESS_WITH_USERS': 0.02,
            'ATTRACT_REWARD_SENSITIVITY': 0.3, 'ATTRACT_ACTIVITY_SENSITIVITY': 0.4,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.15, 'ATTRACT_BALANCE_SENSITIVITY': 0.2,
        }
    elif case_name == "HighEngagement_RewardDriven":
        return {
            'USER_ARCHETYPES_DIST': {'Innovator': 0.25, 'Follower': 0.55, 'Balanced': 0.20}, # More innovators
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 1.8, # Slightly less herd-like
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.80, # More active
            'INNOVATOR_PROB_NEW_CONCEPT': 0.65,'INNOVATOR_PROB_AGREE': 0.25,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.15,'FOLLOWER_PROB_AGREE': 0.75, # Followers also try new things a bit more
            'BALANCED_PROB_NEW_CONCEPT': 0.40,'BALANCED_PROB_AGREE': 0.50,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.05, 'BASE_ATTRACTIVENESS_WITH_USERS': 0.03,
            'ATTRACT_REWARD_SENSITIVITY': 0.5, # Highly sensitive to rewards
            'ATTRACT_ACTIVITY_SENSITIVITY': 0.5,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.2,
            'ATTRACT_BALANCE_SENSITIVITY': 0.3,
        }
    elif case_name == "CautiousFollowers_LowActivity":
        return {
            'USER_ARCHETYPES_DIST': {'Innovator': 0.10, 'Follower': 0.75, 'Balanced': 0.15}, # Fewer innovators
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.5, # Stronger herd behavior
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.55, # Less active
            'INNOVATOR_PROB_NEW_CONCEPT': 0.50,'INNOVATOR_PROB_AGREE': 0.40,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.05,'FOLLOWER_PROB_AGREE': 0.85,
            'BALANCED_PROB_NEW_CONCEPT': 0.25,'BALANCED_PROB_AGREE': 0.65,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.01, 'BASE_ATTRACTIVENESS_WITH_USERS': 0.005,
            'ATTRACT_REWARD_SENSITIVITY': 0.2, # Less sensitive to rewards
            'ATTRACT_ACTIVITY_SENSITIVITY': 0.2,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.1,
            'ATTRACT_BALANCE_SENSITIVITY': 0.1,
        }
    else: # Default to baseline if case_name is unknown
        print(f"Warning: Unknown empirical case_name '{case_name}'. Using BaselineEngagement.")
        return get_empirical_params_for_case("BaselineEngagement")


# --- Main Simulation Function (Modified to accept merged params) ---
def run_full_simulation_with_dynamic_users(merged_params, simulation_run_id="sim_dyn_users_run"):
    # Unpack ALL parameters from merged_params using .get()
    # Managerial Levers
    POTENTIAL_USER_POOL_SIZE = merged_params.get('POTENTIAL_USER_POOL_SIZE', 1000)
    INITIAL_ACTIVE_USERS = merged_params.get('INITIAL_ACTIVE_USERS', 50)
    NUM_POLITICIANS = merged_params.get('NUM_POLITICIANS', 5)
    SIMULATION_STEPS = merged_params.get('SIMULATION_STEPS', 100)
    INITIAL_USER_BALANCE_MEAN = merged_params.get('INITIAL_USER_BALANCE_MEAN', 100)
    INITIAL_USER_BALANCE_STDDEV = merged_params.get('INITIAL_USER_BALANCE_STDDEV', 20)
    COST_SUBMIT_FIRST_TIME_WORD = merged_params.get('COST_SUBMIT_FIRST_TIME_WORD', 5)
    COST_AGREE_OR_RESUBMIT = merged_params.get('COST_AGREE_OR_RESUBMIT', 3)
    POPULARITY_THRESHOLD_BASE = merged_params.get('POPULARITY_THRESHOLD_BASE', 10)
    POPULARITY_DECAY_RATE = merged_params.get('POPULARITY_DECAY_RATE', 0.035)
    PLATFORM_RAKE_PERCENTAGE = merged_params.get('PLATFORM_RAKE_PERCENTAGE', 0.30)
    REWARD_TO_ORIGINAL_SUBMITTER_SHARE = merged_params.get('REWARD_TO_ORIGINAL_SUBMITTER_SHARE', 0.65)
    ENABLE_CHURN = merged_params.get('ENABLE_CHURN', True)
    CHURN_INACTIVITY_THRESHOLD_STEPS = merged_params.get('CHURN_INACTIVITY_THRESHOLD_STEPS', 15)
    CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER = merged_params.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER', 1.5)
    CHURN_BASE_PROB_IF_CONDITIONS_MET = merged_params.get('CHURN_BASE_PROB_IF_CONDITIONS_MET', 0.08)
    
    # Empirical Parameters (now explicitly unpacked, these come from get_empirical_params_for_case)
    USER_ARCHETYPES_DIST = merged_params.get('USER_ARCHETYPES_DIST') # Should be present from empirical case
    INNOVATOR_PROB_NEW_CONCEPT = merged_params.get('INNOVATOR_PROB_NEW_CONCEPT')
    INNOVATOR_PROB_AGREE = merged_params.get('INNOVATOR_PROB_AGREE')
    FOLLOWER_PROB_NEW_CONCEPT = merged_params.get('FOLLOWER_PROB_NEW_CONCEPT')
    FOLLOWER_PROB_AGREE = merged_params.get('FOLLOWER_PROB_AGREE')
    FOLLOWER_POPULARITY_BIAS_FACTOR = merged_params.get('FOLLOWER_POPULARITY_BIAS_FACTOR')
    BALANCED_PROB_NEW_CONCEPT = merged_params.get('BALANCED_PROB_NEW_CONCEPT')
    BALANCED_PROB_AGREE = merged_params.get('BALANCED_PROB_AGREE')
    USER_ACTIVITY_RATE_ON_PLATFORM = merged_params.get('USER_ACTIVITY_RATE_ON_PLATFORM')
    # Attractiveness function parameters are also part of empirical (how potential users react)
    # and are used inside calculate_platform_attractiveness by passing `merged_params`

    # Derived parameters
    REWARD_TO_AGREERS_SHARE = 1.0 - REWARD_TO_ORIGINAL_SUBMITTER_SHARE
    CHURN_LOW_BALANCE_THRESHOLD = min(COST_SUBMIT_FIRST_TIME_WORD, COST_AGREE_OR_RESUBMIT) * CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER

    # --- Data Structures, Initialization, Simulation Loop, Metrics ---
    # (This part of the function remains largely the same as your previous "full file" version)
    # Ensure it uses the unpacked parameters above.
    # ...
    users_master_list = {}
    next_user_id_counter = 0
    politicians_dict = {f"pol_{i}": {'words': {}} for i in range(NUM_POLITICIANS)}
    platform_treasury = 0.0
    platform_total_rewards_paid_overall = 0.0
    total_word_instances_created = 0
    total_actions_simulation = 0
    
    archetype_keys = list(USER_ARCHETYPES_DIST.keys())
    archetype_probs = np.array(list(USER_ARCHETYPES_DIST.values())); archetype_probs /= np.sum(archetype_probs)

    for _ in range(INITIAL_ACTIVE_USERS):
        user_id = f"user_{next_user_id_counter}"
        archetype = np.random.choice(archetype_keys, p=archetype_probs)
        balance = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
        users_master_list[user_id] = User(user_id, archetype, balance, join_step=0)
        next_user_id_counter += 1
    
    history_num_active_users, history_platform_attractiveness = [], []

    for step in range(1, SIMULATION_STEPS + 1):
        current_platform_users_objs = [u for u in users_master_list.values() if u.active_in_sim]
        num_current_platform_users = len(current_platform_users_objs)
        actions_this_step, platform_rewards_this_step = 0, 0
        
        if num_current_platform_users > 0:
            potential_actors_ids = [u.id for u in current_platform_users_objs]
            actors_for_this_step_ids = random.sample(potential_actors_ids, int(num_current_platform_users * USER_ACTIVITY_RATE_ON_PLATFORM))
            for user_id in actors_for_this_step_ids:
                user = users_master_list[user_id]
                if not user.active_in_sim: continue
                user.last_activity_step = step; action_taken_this_turn_by_user = False
                if user.archetype == 'Innovator': prob_try_new, prob_try_agree = INNOVATOR_PROB_NEW_CONCEPT, INNOVATOR_PROB_AGREE
                elif user.archetype == 'Follower': prob_try_new, prob_try_agree = FOLLOWER_PROB_NEW_CONCEPT, FOLLOWER_PROB_AGREE
                else: prob_try_new, prob_try_agree = BALANCED_PROB_NEW_CONCEPT, BALANCED_PROB_AGREE
                action_roll = random.random(); chosen_pol_id = f"pol_{random.randint(0, NUM_POLITICIANS - 1)}"
                pol_words_dict = politicians_dict[chosen_pol_id]['words']
                if action_roll < prob_try_new:
                    chosen_word_str = get_simulated_word_zipfian(user.archetype, merged_params)
                    is_first_time = chosen_word_str not in pol_words_dict
                    cost = COST_SUBMIT_FIRST_TIME_WORD if is_first_time else COST_AGREE_OR_RESUBMIT
                    if user.balance >= cost:
                        user.balance -= cost; platform_treasury += cost * PLATFORM_RAKE_PERCENTAGE
                        reward_contrib = cost * (1 - PLATFORM_RAKE_PERCENTAGE)
                        if is_first_time:
                            pol_words_dict[chosen_word_str] = WordData(user.id, step); total_word_instances_created += 1
                            user.words_originated[(chosen_pol_id, chosen_word_str)] = step
                        word_obj = pol_words_dict[chosen_word_str]
                        word_obj.count += 1; word_obj.raw_popularity_score +=1
                        word_obj.fees_contributed_to_pool += reward_contrib
                        word_obj.agreed_by_users.add(user.id); action_taken_this_turn_by_user = True
                elif prob_try_new <= action_roll < prob_try_new + prob_try_agree:
                    if pol_words_dict:
                        existing_words = list(pol_words_dict.keys())
                        if existing_words:
                            word_to_agree = ""
                            if user.archetype == 'Follower':
                                scores = np.array([pol_words_dict[w].raw_popularity_score + 0.1 for w in existing_words])
                                scores = np.maximum(scores, 0.01); scores_b = scores ** FOLLOWER_POPULARITY_BIAS_FACTOR
                                sum_b = np.sum(scores_b)
                                if sum_b > 1e-6:
                                    probs = scores_b / sum_b
                                    if not np.isnan(probs).any() and abs(np.sum(probs)-1.0)<1e-5:
                                        try: word_to_agree = np.random.choice(existing_words, p=probs)
                                        except ValueError: word_to_agree = random.choice(existing_words)
                                    else: word_to_agree = random.choice(existing_words)
                                else: word_to_agree = random.choice(existing_words)
                            else: word_to_agree = random.choice(existing_words)
                            if word_to_agree and user.balance >= COST_AGREE_OR_RESUBMIT:
                                user.balance -= COST_AGREE_OR_RESUBMIT; platform_treasury += COST_AGREE_OR_RESUBMIT * PLATFORM_RAKE_PERCENTAGE
                                reward_contrib = COST_AGREE_OR_RESUBMIT * (1-PLATFORM_RAKE_PERCENTAGE)
                                word_obj = pol_words_dict[word_to_agree]
                                word_obj.count += 1; word_obj.raw_popularity_score +=1
                                word_obj.fees_contributed_to_pool += reward_contrib
                                word_obj.agreed_by_users.add(user.id); action_taken_this_turn_by_user = True
                if action_taken_this_turn_by_user: actions_this_step += 1
        total_actions_simulation += actions_this_step
        step_rewards_this_step = 0
        for pol_id, pol_data in politicians_dict.items():
            for word_str, word_obj in pol_data['words'].items():
                if word_obj.raw_popularity_score > 0:
                    word_obj.raw_popularity_score -= word_obj.raw_popularity_score * POPULARITY_DECAY_RATE
                    if word_obj.raw_popularity_score < 0.01: word_obj.raw_popularity_score = 0
                thresh_key = POPULARITY_THRESHOLD_BASE
                if word_obj.count >= thresh_key and not word_obj.reward_paid_out_for_threshold.get(thresh_key, False):
                    pool = word_obj.fees_contributed_to_pool; orig_sub_id = word_obj.original_submitter_id
                    if orig_sub_id and orig_sub_id in users_master_list and users_master_list[orig_sub_id].active_in_sim:
                        rew_orig = pool * REWARD_TO_ORIGINAL_SUBMITTER_SHARE
                        users_master_list[orig_sub_id].balance += rew_orig; users_master_list[orig_sub_id].total_rewards_received += rew_orig
                        platform_total_rewards_paid_overall += rew_orig; step_rewards_this_step += rew_orig
                    num_agr = len(word_obj.agreed_by_users)
                    if num_agr > 0:
                        rew_agr_tot = pool * REWARD_TO_AGREERS_SHARE; per_agr = rew_agr_tot / num_agr
                        for agr_id in word_obj.agreed_by_users:
                            if agr_id in users_master_list and users_master_list[agr_id].active_in_sim:
                                users_master_list[agr_id].balance += per_agr; users_master_list[agr_id].total_rewards_received += per_agr
                                platform_total_rewards_paid_overall += per_agr; step_rewards_this_step += per_agr
                    word_obj.reward_paid_out_for_threshold[thresh_key] = True; word_obj.fees_contributed_to_pool = 0.0
        current_active_users_for_attract = [u for u in users_master_list.values() if u.active_in_sim]
        attractiveness = calculate_platform_attractiveness(current_active_users_for_attract, step_rewards_this_step, actions_this_step, merged_params, step)
        history_platform_attractiveness.append(attractiveness)
        newly_joined_count = 0
        if len(users_master_list) < POTENTIAL_USER_POOL_SIZE:
            num_potential_join_trials = min(POTENTIAL_USER_POOL_SIZE - len(users_master_list), int(merged_params.get('POTENTIAL_JOIN_TRIALS_PER_STEP', 50) + len(current_active_users_for_attract) * 0.05))
            max_joins_this_step = int(merged_params.get('MAX_NEW_JOINS_PER_STEP_SCALER', 0.02) * POTENTIAL_USER_POOL_SIZE)
            for _ in range(num_potential_join_trials):
                if len(users_master_list) >= POTENTIAL_USER_POOL_SIZE or newly_joined_count >= max_joins_this_step: break
                if random.random() < attractiveness:
                    user_id = f"user_{next_user_id_counter}"; archetype = np.random.choice(archetype_keys, p=archetype_probs)
                    balance = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
                    users_master_list[user_id] = User(user_id, archetype, balance, join_step=step)
                    next_user_id_counter += 1; newly_joined_count += 1
        if ENABLE_CHURN and step > merged_params.get('CHURN_GRACE_PERIOD_STEPS', 10):
            users_to_churn_ids = []
            for user_id, user_obj in users_master_list.items():
                if not user_obj.active_in_sim: continue
                churn_prob = 0.0
                if (step - user_obj.last_activity_step) > CHURN_INACTIVITY_THRESHOLD_STEPS: churn_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                if user_obj.balance < CHURN_LOW_BALANCE_THRESHOLD: churn_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                if user_obj.total_rewards_received == 0 and user_obj.balance < INITIAL_USER_BALANCE_MEAN * 0.4 and step > user_obj.join_step + 5:
                     churn_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.4
                if random.random() < np.clip(churn_prob, 0, 0.6): users_to_churn_ids.append(user_id)
            for uid_churn in users_to_churn_ids: users_master_list[uid_churn].active_in_sim = False
        history_num_active_users.append(len([u for u in users_master_list.values() if u.active_in_sim]))
    # --- End of Simulation Loop ---

    final_active_users_list = [u for u in users_master_list.values() if u.active_in_sim]
    final_balances = [u.balance for u in final_active_users_list] if final_active_users_list else [0]
    final_num_active_users = len(final_active_users_list)
    final_gini = get_gini(final_balances); final_avg_balance = np.mean(final_balances); final_median_balance = np.median(final_balances)
    min_action_cost = min(COST_SUBMIT_FIRST_TIME_WORD, COST_AGREE_OR_RESUBMIT)
    users_broke_count = sum(1 for b in final_balances if b < min_action_cost)
    total_pop_events = 0; unique_pop_words = set()
    for pol_data in politicians_dict.values():
        for word_str, word_obj in pol_data['words'].items():
            if word_obj.reward_paid_out_for_threshold.get(POPULARITY_THRESHOLD_BASE, False):
                total_pop_events +=1; unique_pop_words.add(word_str)
    avg_hist_users = np.mean(history_num_active_users) if history_num_active_users else 0
    return {
        "params_config_copy": merged_params.copy(), "final_num_active_users": final_num_active_users,
        "final_treasury": platform_treasury, "total_rewards_paid": platform_total_rewards_paid_overall,
        "final_avg_balance": final_avg_balance, "final_median_balance": final_median_balance,
        "final_gini_coefficient": final_gini,
        "users_broke_percent": (users_broke_count / final_num_active_users) * 100 if final_num_active_users > 0 else (100.0 if INITIAL_ACTIVE_USERS > 0 and final_num_active_users == 0 else 0.0),
        "total_actions_in_sim": total_actions_simulation,
        "avg_actions_per_active_user_per_step_overall": total_actions_simulation / (avg_hist_users * SIMULATION_STEPS) if avg_hist_users * SIMULATION_STEPS > 0 else 0,
        "total_unique_word_instances_created": total_word_instances_created,
        "unique_popular_word_strings_count": len(unique_pop_words),
        "total_popular_word_events": total_pop_events,
        "history_num_active_users": history_num_active_users,
        "history_platform_attractiveness": history_platform_attractiveness
    }
# --- [Keep calculate_dynamic_user_growth_score objective function as defined before] ---
def calculate_dynamic_user_growth_score(results, target_gini=0.55, min_avg_balance_ratio=0.75, desired_final_user_ratio=0.80):
    params = results['params_config_copy']; score = 0.0
    POTENTIAL_USER_POOL_SIZE = params.get('POTENTIAL_USER_POOL_SIZE', 1000)
    final_user_ratio = results['final_num_active_users'] / POTENTIAL_USER_POOL_SIZE if POTENTIAL_USER_POOL_SIZE > 0 else 0
    if final_user_ratio < desired_final_user_ratio * 0.6: score -= (desired_final_user_ratio * 0.6 - final_user_ratio) * 20000
    else: score += np.clip(final_user_ratio / desired_final_user_ratio, 0, 1.1) * 8000.0
    if final_user_ratio >= desired_final_user_ratio: score += 3000
    if results['final_num_active_users'] > params.get('INITIAL_ACTIVE_USERS',10) * 0.5 :
        if results['final_treasury'] < -(results['final_num_active_users'] * params['INITIAL_USER_BALANCE_MEAN'] * 0.1): score -= 15000
        elif results['final_treasury'] < 0: score -= abs(results['final_treasury']) * 3.0
        else: score += results['final_treasury'] * 0.001
        target_broke_among_active = params.get('TARGET_BROKE_AMONG_ACTIVE_PERCENT', 10.0)
        broke_diff = results['users_broke_percent'] - target_broke_among_active
        if broke_diff > 0: score -= (broke_diff / 5.0)**2 * 600.0
        avg_balance_ratio_active = results['final_avg_balance'] / params['INITIAL_USER_BALANCE_MEAN'] if params['INITIAL_USER_BALANCE_MEAN'] > 0 else 0
        if avg_balance_ratio_active < min_avg_balance_ratio: score -= (min_avg_balance_ratio - avg_balance_ratio_active) * 1200.0
        elif avg_balance_ratio_active > 1.7: score -= (avg_balance_ratio_active - 1.7) * 200.0
        else: score += avg_balance_ratio_active * 150.0
        gini_diff_active = results['final_gini_coefficient'] - target_gini
        if gini_diff_active > 0 : score -= gini_diff_active * 1000.0
        else: score += (target_gini - results['final_gini_coefficient']) * 250.0
        score += results['avg_actions_per_active_user_per_step_overall'] * 6000.0
        if results['total_actions_in_sim'] > 0:
            pop_ev_per_1k_act = (results['total_popular_word_events'] / results['total_actions_in_sim']) * 1000
            score += pop_ev_per_1k_act * 60.0
        if results['total_popular_word_events'] > 0:
            diversity = (results['unique_popular_word_strings_count'] / results['total_popular_word_events'])
            score += diversity * 500.0
        elif results['total_unique_word_instances_created'] > 0 and results['total_popular_word_events'] == 0: score -= 400
    else: score -= 80000
    return score

# --- Main Execution for Parameter Tuning ---
if __name__ == "__main__":
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO'

    # --- MANAGERIAL LEVERS (VERY FOCUSED GRID - 2 params, 2 options each) ---
    managerial_params_grid = {
        'POPULARITY_THRESHOLD_BASE': [8, 15],          # Still a key lever
        'PLATFORM_RAKE_PERCENTAGE': [0.20, 0.35],       # Still a key lever
        # 'COST_AGREE_OR_RESUBMIT': [3],               # Fixed for this run
        # 'COST_SUBMIT_FIRST_TIME_WORD': [5],          # Fixed
        # 'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': [0.65],# Fixed
        # 'POPULARITY_DECAY_RATE': [0.025],            # Fixed
    }
    # This grid: 2*2 = 4 managerial combinations.
    # Total sims = 4 managerial combos * 3 empirical cases = 12 simulations. (Very fast)

    # --- FIXED PARAMETERS for this "Accelerated Happy Medium" ---
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 10000,  # <<< ACCELERATED: Halved from previous Happy Medium
        'INITIAL_ACTIVE_USERS': 75,        # Seed users
        'NUM_POLITICIANS': 25,             # <<< ACCELERATED
        'SIMULATION_STEPS': 90,           # <<< ACCELERATED: ~3 months of daily steps

        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        
        'COST_SUBMIT_FIRST_TIME_WORD': 5, # Fixed
        'COST_AGREE_OR_RESUBMIT': 3,      # Fixed
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.65,
        'POPULARITY_DECAY_RATE': 0.025,

        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 30,    # Adjusted for ~3 month sim
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.8,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.03,

        'FINITE_DICTIONARY_SIZE': 1000,       # <<< ACCELERATED
        'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.04,
        'COMMON_WORD_VOCAB_SIZE': 120,
        'INNOVATOR_RARE_WORD_CHANCE': 0.1,

        # --- CRITICAL FOR SPEED: Fix more empirical attractiveness sensitivities ---
        # The get_empirical_params_for_case will still set USER_ARCHETYPES and BEHAVIORAL_PROBS
        # but we can override/fix the attractiveness sensitivities here for all cases FOR THIS FAST RUN
        'BASE_ATTRACTIVENESS_NO_USERS': 0.03,       # Fixed
        'BASE_ATTRACTIVENESS_WITH_USERS': 0.02,     # Fixed
        'ATTRACT_REWARD_SENSITIVITY': 0.35,         # Fixed to a reasonable mid-value
        'ATTRACT_ACTIVITY_SENSITIVITY': 0.45,      # Fixed
        'ATTRACT_FAIRNESS_SENSITIVITY': 0.15,       # Fixed
        'ATTRACT_BALANCE_SENSITIVITY': 0.25,        # Fixed
    }

    # --- Create parameter combinations ---
    # (The logic for creating all_parameter_combinations by merging fixed, empirical, and managerial remains the same)
    # ... (paste the combination logic from your previous full script here) ...
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            current_combo = fixed_sim_params.copy()
            empirical_set_for_case = get_empirical_params_for_case(empirical_case_name) # Assumes this function is defined
            
            # Merge carefully: fixed_sim_params already has some attractiveness defaults.
            # Empirical case might override behavioral but use fixed attractiveness for this fast run.
            # The `current_combo.update(empirical_set)` will get the archetype distributions
            # and core behavioral probabilities from the empirical case.
            # The fixed attractiveness sensitivities in `fixed_sim_params` will be used.
            current_combo.update(empirical_set_for_case) 
            current_combo.update(managerial_combo) # Managerial levers are applied last
            current_combo[empirical_case_param_name] = empirical_case_name

            min_cost_val = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD'), current_combo.get('COST_AGREE_OR_RESUBMIT'))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')
            all_parameter_combinations.append(current_combo)

    num_total_combinations = len(all_parameter_combinations)
    print(f"Starting 'Accelerated Happy Medium' Grid Search with {num_total_combinations} total parameter combinations.")
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
        sim_id_str = f"accel_hm_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        if (i + 1) % (num_total_combinations // 4 or 1) == 0 or i == 0: # Print progress ~4 times
            print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 1 # ESSENTIAL FOR GRID SPEED
        current_set_scores, current_set_details_list = [], []
        # set_start_time = time.time() # Not strictly needed for these very fast runs

        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            # Adjust desired_final_user_ratio for this shorter, smaller pool sim
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.50) # Aim for 50% of 10k pool
            current_set_scores.append(score)
            # Print only for new bests to reduce log clutter
            # print(f"  Run {run_idx+1}/{num_runs_per_set} | Score: {score:,.2f} | EndUsers: {run_detail['final_num_active_users']}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {run_detail['final_treasury']:,.0f} | Broke%: {run_detail['users_broke_percent']:.1f}")
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        rep_details = current_set_details_list[0] if current_set_details_list else {}
        
        # print(f"  Combo {i+1} (Case: {empirical_case_tag}) Score: {avg_score:,.2f} (took {time.time() - set_start_time:.2f}s)")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  ---> New Best: {best_overall_score:,.2f} (Combo {i+1}, Case: {empirical_case_tag}, EndUsers: {best_detailed_results_overall.get('final_num_active_users')}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']}, Broke%: {best_detailed_results_overall.get('users_broke_percent'):.1f}%)")
    
    # --- [The rest of the __main__ block (saving results, printing best, plotting) remains the same] ---
    # Remember to use a new results_dir like "simulation_results_accel_happy_medium"
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_accel_hm" # New directory
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_ahm_{timestamp_str}"
    print(f"\n--- 'Accelerated Happy Medium' Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")
    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Overall Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Parameter Configuration Found (Managerial Levers Varied):")
        varied_managerial_keys = [k for k,v in managerial_params_grid.items() if len(v)>1]
        for k_man in varied_managerial_keys: print(f"  {k_man}: {best_params_overall.get(k_man)}")
        print("\nMetrics for Best Config (Single Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", "total_popular_word_events", "unique_popular_word_strings_count", "avg_actions_per_active_user_per_step_overall"]
        for k_met in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met)
            if isinstance(val, float): print(f"  {k_met}: {val:,.4f}")
            else: print(f"  {k_met}: {val}")
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        try:
            df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
            print(f"\nFull summary saved to: {results_filename_base}_summary.csv")
        except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")
        if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
            plt.figure(figsize=(10,6)); history_active = best_detailed_results_overall['history_num_active_users']
            history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
            plt.plot(history_active, label=f"Active Users (Best Run Score: {best_overall_score:,.0f})", color="blue", linewidth=1.5)
            plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='r', linestyle='--', label="Potential Pool Size")
            ax_attract = plt.gca().twinx()
            if history_attract:
                ax_attract.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="green", linestyle=':', alpha=0.6)
                ax_attract.set_ylabel("Attractiveness (x100)", color="green", fontsize=9); ax_attract.tick_params(axis='y', labelcolor="green", labelsize=8)
            plt.gca().set_xlabel("Simulation Step", fontsize=10); plt.gca().set_ylabel("Active Users", color="blue", fontsize=10)
            plt.gca().tick_params(axis='y', labelcolor="blue", labelsize=8); plt.gca().tick_params(axis='x', labelsize=8)
            plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=12)
            lines, labels = plt.gca().get_legend_handles_labels(); lines2, labels2 = ax_attract.get_legend_handles_labels()
            if lines2: ax_attract.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)
            else: plt.gca().legend(loc='best', fontsize=8)
            plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
        if not df_all_runs.empty:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Set3", hue=empirical_case_param_name, legend=False)
            plt.title("Impact of Empirical Cases (Accel. HM)", fontsize=12); plt.xlabel("Empirical Case", fontsize=10); plt.ylabel("Score", fontsize=10)
            plt.xticks(rotation=10, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_empirical_case_impact.png"); plt.show()
            managerial_params_to_plot_ahm = [k for k,v in managerial_params_grid.items() if len(v)>1]
            if managerial_params_to_plot_ahm:
                print("\nGenerating limited sensitivity plots...")
                for param_name in managerial_params_to_plot_ahm:
                    if param_name in df_all_runs.columns and df_all_runs[param_name].nunique() > 1:
                        g = sns.catplot(x=param_name, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="pastel", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)), height=4, aspect=1)
                        g.fig.suptitle(f"Score vs. {param_name} (by Case - Accel. HM)", fontsize=13, y=1.03); g.set_xticklabels(rotation=20, ha='right', fontsize=8)
                        g.set_ylabels("Score", fontsize=9); g.set_xlabels(param_name, fontsize=9); g.tick_params(labelsize=8)
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name}_by_case.png"); plt.show()
            plt.figure(figsize=(8,5)); scores_hist = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist:
                plt.hist(scores_hist, bins=max(8, num_total_combinations//2 if num_total_combinations>0 else 1), edgecolor='grey', color='lightcoral')
                plt.title('Score Distribution (Accel. HM - All Cases)', fontsize=12); plt.xlabel('Score', fontsize=10); plt.ylabel('# Combos', fontsize=10); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.yticks(fontsize=9); plt.xticks(fontsize=9); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png")
    else: print("No simulations run or no valid results.")
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO'

    # --- MANAGERIAL LEVERS (VERY, VERY FOCUSED GRID - e.g., 2 params, 2 options each) ---
    managerial_params_grid = {
        'POPULARITY_THRESHOLD_BASE': [7, 15],      # Key for reward frequency
        'PLATFORM_RAKE_PERCENTAGE': [0.20, 0.40],    # Key for platform vs. user income
        # 'COST_AGREE_OR_RESUBMIT': [3],           # Fixed for this super-fast grid
        # 'COST_SUBMIT_FIRST_TIME_WORD': [5],      # Fixed
    }
    # This grid: 2*2 = 4 managerial combinations.
    # Total sims = 4 managerial combos * 3 empirical cases = 12 simulations.

    # --- FIXED PARAMETERS for this "Aggressively Fast Sketch" ---
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 5000,   # <<< MUCH SMALLER POOL
        'INITIAL_ACTIVE_USERS': 50,         # <<< FEWER SEED USERS
        'NUM_POLITICIANS': 10,              # <<< FEWER POLITICIANS
        'SIMULATION_STEPS': 40,            # <<< DRASTICALLY REDUCED STEPS (just over a month)

        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        
        # Fixing more managerial levers to values that seemed okay or are central
        'COST_SUBMIT_FIRST_TIME_WORD': 5,
        'COST_AGREE_OR_RESUBMIT': 3,
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.65,
        'POPULARITY_DECAY_RATE': 0.025, # Can be a bit higher with fewer steps

        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 15,    # Shorter for fewer steps
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.5,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.05, # Slightly higher churn propensity

        'FINITE_DICTIONARY_SIZE': 500,        # <<< SMALLER DICTIONARY
        'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.05,
        'COMMON_WORD_VOCAB_SIZE': 100,
        'INNOVATOR_RARE_WORD_CHANCE': 0.1,
    }

    # --- Create parameter combinations (same logic as before) ---
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            current_combo = fixed_sim_params.copy()
            empirical_set = get_empirical_params_for_case(empirical_case_name) # Assumes this function is defined
            current_combo.update(empirical_set)
            current_combo.update(managerial_combo)
            current_combo[empirical_case_param_name] = empirical_case_name
            min_cost_val = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD'), current_combo.get('COST_AGREE_OR_RESUBMIT'))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')
            all_parameter_combinations.append(current_combo)
    
    num_total_combinations = len(all_parameter_combinations)
    print(f"Starting 'Aggressively Fast Sketch' Grid Search with {num_total_combinations} total parameter combinations.")
    print(f"Each simulation run will have {fixed_sim_params['SIMULATION_STEPS']} steps.")

    all_sim_results_for_df = [] # For pandas DataFrame later
    best_overall_score = -float('inf')
    best_params_overall = None
    best_detailed_results_overall = None
    tuning_start_time = time.time()

    initialize_zipfian_dictionary( # Initialize once
        fixed_sim_params.get('FINITE_DICTIONARY_SIZE'),
        fixed_sim_params.get('ZIPFIAN_ALPHA')
    )

    for i, p_full_config_combo in enumerate(all_parameter_combinations):
        sim_id_str = f"fast_sketch_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        # Reduce print frequency for faster overall execution
        if (i + 1) % (num_total_combinations // 5 or 1) == 0 or i == 0: # Print first and then ~5 times
            print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 1 # <<< ESSENTIAL FOR SPEED IN GRID SEARCH
        current_set_scores, current_set_details_list = [], []
        # set_start_time = time.time() # Not needed for such fast runs

        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            # Adjust desired_final_user_ratio if POTENTIAL_USER_POOL_SIZE changes significantly
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.60) # Aim for 60% of smaller pool
            current_set_scores.append(score)
            # Only print for new bests during the fast run to reduce clutter
            # print(f"  Run {run_idx+1}/{num_runs_per_set} | Score: {score:,.2f} | EndUsers: {run_detail['final_num_active_users']}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {run_detail['final_treasury']:,.0f} | Broke%: {run_detail['users_broke_percent']:.1f}")
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        rep_details = current_set_details_list[0] if current_set_details_list else {}
        
        # print(f"  Combo {i+1} (Case: {empirical_case_tag}) Score: {avg_score:,.2f} (took {time.time() - set_start_time:.2f}s)")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  ---> New Best: {best_overall_score:,.2f} (Combo {i+1}, Case: {empirical_case_tag}, EndUsers: {best_detailed_results_overall.get('final_num_active_users')}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']}, Broke%: {best_detailed_results_overall.get('users_broke_percent'):.1f}%)") # More info on best
    
    # --- [The rest of the __main__ block (saving results, printing best, plotting) remains the same] ---
    # You will need the `get_empirical_params_for_case` and `calculate_dynamic_user_growth_score`
    # functions, and the class definitions from the previous "full script".
    # The plotting part will also be the same.
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_fast_sketch" # New directory
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_fs_{timestamp_str}"
    
    print(f"\n--- 'Aggressively Fast Sketch' Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")

    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Overall Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Parameter Configuration Found (Managerial Levers Varied):")
        varied_managerial_keys = [k for k,v in managerial_params_grid.items() if len(v)>1]
        for k_man in varied_managerial_keys: print(f"  {k_man}: {best_params_overall.get(k_man)}")
        
        print("\nMetrics for Best Config (Single Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", "total_popular_word_events", "unique_popular_word_strings_count", "avg_actions_per_active_user_per_step_overall"]
        for k_met in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met)
            if isinstance(val, float): print(f"  {k_met}: {val:,.4f}")
            else: print(f"  {k_met}: {val}")
        
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        try:
            df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
            print(f"\nFull summary saved to: {results_filename_base}_summary.csv")
        except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")

        if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
            plt.figure(figsize=(10,6)) # Smaller plot for faster display
            history_active = best_detailed_results_overall['history_num_active_users']
            history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
            plt.plot(history_active, label=f"Active Users (Best Run Score: {best_overall_score:,.0f})", color="blue", linewidth=1.5)
            plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='r', linestyle='--', label="Potential Pool Size")
            ax_attract = plt.gca().twinx()
            if history_attract:
                ax_attract.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="green", linestyle=':', alpha=0.6)
                ax_attract.set_ylabel("Attractiveness (x100)", color="green", fontsize=9); ax_attract.tick_params(axis='y', labelcolor="green", labelsize=8)
            plt.gca().set_xlabel("Simulation Step", fontsize=10); plt.gca().set_ylabel("Active Users", color="blue", fontsize=10)
            plt.gca().tick_params(axis='y', labelcolor="blue", labelsize=8); plt.gca().tick_params(axis='x', labelsize=8)
            plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=12)
            lines, labels = plt.gca().get_legend_handles_labels(); lines2, labels2 = ax_attract.get_legend_handles_labels()
            if lines2: ax_attract.legend(lines + lines2, labels + labels2, loc='best', fontsize=8)
            else: plt.gca().legend(loc='best', fontsize=8)
            plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
        
        if not df_all_runs.empty:
            plt.figure(figsize=(8, 5)) # Smaller
            sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Set3", hue=empirical_case_param_name, legend=False)
            plt.title("Impact of Empirical Cases (Fast Sketch)", fontsize=12); plt.xlabel("Empirical Case", fontsize=10); plt.ylabel("Score", fontsize=10)
            plt.xticks(rotation=10, ha='right', fontsize=9); plt.yticks(fontsize=9); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_empirical_case_impact.png"); plt.show()

            managerial_params_to_plot = [k for k,v in managerial_params_grid.items() if len(v)>1]
            if managerial_params_to_plot:
                print("\nGenerating limited sensitivity plots...")
                for param_name in managerial_params_to_plot[:2]: # Only plot for first 2 varied params for speed
                    if param_name in df_all_runs.columns and df_all_runs[param_name].nunique() > 1:
                        g = sns.catplot(x=param_name, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="pastel", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)), height=4, aspect=1)
                        g.fig.suptitle(f"Score vs. {param_name} (by Case - Fast Sketch)", fontsize=13, y=1.03); g.set_xticklabels(rotation=20, ha='right', fontsize=8)
                        g.set_ylabels("Score", fontsize=9); g.set_xlabels(param_name, fontsize=9); g.tick_params(labelsize=8)
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name}_by_case.png"); plt.show()
            
            plt.figure(figsize=(8,5)); scores_hist = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist:
                plt.hist(scores_hist, bins=max(8, num_total_combinations//2 if num_total_combinations>0 else 1), edgecolor='grey', color='lightcoral')
                plt.title('Score Distribution (Fast Sketch - All Cases)', fontsize=12); plt.xlabel('Score', fontsize=10); plt.ylabel('# Combos', fontsize=10); plt.grid(axis='y', linestyle=':', alpha=0.5); plt.yticks(fontsize=9); plt.xticks(fontsize=9); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png")
    else: print("No simulations run or no valid results.")
    # --- Define EMPIRICAL POPULATION CASES ---
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO'

    # --- Parameters that are MANAGERIAL LEVERS (VERY FOCUSED GRID) ---
    managerial_params_grid = {
        # Focus on the most impactful levers based on previous intuition
        'POPULARITY_THRESHOLD_BASE': [10, 20],         # Was [15, 25]. Lowering slightly for faster reward cycles.
        'PLATFORM_RAKE_PERCENTAGE': [0.25, 0.40],       # Key lever for treasury vs. user rewards
        'COST_AGREE_OR_RESUBMIT': [2, 4],               # Impactful on user spend rate
        # 'COST_SUBMIT_FIRST_TIME_WORD': [6],           # Fixed for this medium grid
        # 'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': [0.65], # Fixed
        # 'POPULARITY_DECAY_RATE': [0.02],              # Fixed
    }
    # This grid: 2*2*2 = 8 managerial combinations.
    # Total sims = 8 managerial combos * 3 empirical cases = 24 simulations.

    # --- Parameters that are FIXED for this "Happy Medium" tuning session ---
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 20000,  # <<< HAPPY MEDIUM: Smaller than full real-world, larger than rough
        'INITIAL_ACTIVE_USERS': 100,        # Seed users
        'NUM_POLITICIANS': 50,             # <<< HAPPY MEDIUM
        'SIMULATION_STEPS': 180,          # <<< HAPPY MEDIUM: ~6 months of daily steps

        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        
        # Fixing some managerial levers that were varied in the "real-world" grid to reduce combinations here
        'COST_SUBMIT_FIRST_TIME_WORD': 6, # Fixed based on intuition or previous bests
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.65,
        'POPULARITY_DECAY_RATE': 0.02, # Slightly faster decay if steps are fewer

        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 45,    # Adjusted for ~6 month sim
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.8,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.025, # Slightly higher base churn if conditions met

        'FINITE_DICTIONARY_SIZE': 1500,       # <<< HAPPY MEDIUM
        'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.04,
        'COMMON_WORD_VOCAB_SIZE': 150,
        'INNOVATOR_RARE_WORD_CHANCE': 0.1,
    }

    # --- Create parameter combinations (same logic as before) ---
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            current_combo = fixed_sim_params.copy()
            empirical_set = get_empirical_params_for_case(empirical_case_name) # Assumes this function is defined
            current_combo.update(empirical_set)
            current_combo.update(managerial_combo)
            current_combo[empirical_case_param_name] = empirical_case_name
            min_cost_val = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD'), current_combo.get('COST_AGREE_OR_RESUBMIT'))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')
            all_parameter_combinations.append(current_combo)
    
    num_total_combinations = len(all_parameter_combinations)
    print(f"Starting 'Happy Medium' Grid Search with {num_total_combinations} total parameter combinations.")
    print(f"Each simulation run will have {fixed_sim_params['SIMULATION_STEPS']} steps.")

    all_sim_results_for_df = []
    best_overall_score = -float('inf')
    best_params_overall = None
    best_detailed_results_overall = None
    tuning_start_time = time.time()

    initialize_zipfian_dictionary( # Initialize once
        fixed_sim_params.get('FINITE_DICTIONARY_SIZE'),
        fixed_sim_params.get('ZIPFIAN_ALPHA')
    )

    for i, p_full_config_combo in enumerate(all_parameter_combinations):
        sim_id_str = f"happy_medium_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 1 # <<< KEPT AT 1 FOR GRID SEARCH SPEED
        current_set_scores, current_set_details_list = [], []
        set_start_time = time.time()

        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            # Target 70% of potential pool for this "happy medium" length
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.70) 
            current_set_scores.append(score)
            print(f"  Run {run_idx+1}/{num_runs_per_set} | Score: {score:,.2f} | EndUsers: {run_detail['final_num_active_users']}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {run_detail['final_treasury']:,.0f} | Broke%: {run_detail['users_broke_percent']:.1f}")
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        rep_details = current_set_details_list[0] if current_set_details_list else {} # Since num_runs_per_set is 1
        
        print(f"  Combo {i+1} (Case: {empirical_case_tag}) Score: {avg_score:,.2f} (took {time.time() - set_start_time:.2f}s)")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  -------> *** New Best Overall Score: {best_overall_score:,.2f} for Combo {i+1} (Case: {empirical_case_tag}) *** <-------")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_happy_medium" # New directory
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_hm_{timestamp_str}"
    
    print(f"\n--- 'Happy Medium' Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")

    # --- [The rest of the __main__ block (saving results, printing best, plotting) remains the same] ---
    # ... (ensure this part is correctly pasted from the previous full script) ...
    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Full Parameter Set for this 'Happy Medium' run:")
        for key, val in best_params_overall.items(): print(f"  {key}: {val}")
        print("\nMetrics for Best Config (Single Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", "total_popular_word_events", "unique_popular_word_strings_count", "avg_actions_per_active_user_per_step_overall"]
        for k_met in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met)
            if isinstance(val, float): print(f"  {k_met}: {val:,.4f}")
            else: print(f"  {k_met}: {val}")
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        try:
            df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
            print(f"\nFull summary saved to: {results_filename_base}_summary.csv")
        except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")
        if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
            plt.figure(figsize=(12,7)); history_active = best_detailed_results_overall['history_num_active_users']
            history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
            plt.plot(history_active, label=f"Active Users (Best Run Score: {best_overall_score:,.0f})", color="blue", linewidth=2)
            plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='r', linestyle='--', label="Potential Pool Size")
            ax_attract = plt.gca().twinx()
            if history_attract:
                ax_attract.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="green", linestyle=':', alpha=0.7)
                ax_attract.set_ylabel("Attractiveness Score (x100)", color="green", fontsize=10); ax_attract.tick_params(axis='y', labelcolor="green")
            plt.gca().set_xlabel("Simulation Step (Day)", fontsize=12); plt.gca().set_ylabel("Number of Active Users", color="blue", fontsize=12)
            plt.gca().tick_params(axis='y', labelcolor="blue"); plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=14, fontweight='bold')
            lines, labels = plt.gca().get_legend_handles_labels(); lines2, labels2 = ax_attract.get_legend_handles_labels()
            if lines2: ax_attract.legend(lines + lines2, labels + labels2, loc='center right')
            else: plt.gca().legend(loc='center right')
            plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
        if not df_all_runs.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Set2", hue=empirical_case_param_name, legend=False)
            plt.title("Impact of Empirical Case Scenarios on Score (Happy Medium)", fontsize=14); plt.xlabel("Empirical Case Scenario", fontsize=12); plt.ylabel("Score", fontsize=12)
            plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_empirical_case_impact.png"); plt.show()
            managerial_params_to_plot = [k for k,v in managerial_params_grid.items() if len(v)>1]
            if managerial_params_to_plot:
                print("\nGenerating parameter sensitivity plots (faceted by empirical case)...")
                for param_name in managerial_params_to_plot:
                    if param_name in df_all_runs.columns and df_all_runs[param_name].nunique() > 1:
                        g = sns.catplot(x=param_name, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="viridis", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)), height=4.5, aspect=1.1)
                        g.fig.suptitle(f"Score vs. {param_name} (by Empirical Case)", fontsize=15, y=1.03); g.set_xticklabels(rotation=25, ha='right')
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name}_by_case.png"); print(f"Sensitivity plot for {param_name} saved."); plt.show()
            plt.figure(figsize=(10,6)); scores_hist = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist:
                plt.hist(scores_hist, bins=max(8, num_total_combinations//2 if num_total_combinations>0 else 1), edgecolor='k', color='darkcyan')
                plt.title('Distribution of Scores (Happy Medium - All Cases)', fontsize=14); plt.xlabel('Score'); plt.ylabel('# Param Combos'); plt.grid(axis='y'); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png"); print(f"Overall score distribution plot saved.")
    else: print("No simulations run or no valid results.")
    # --- Define a few EMPIRICAL POPULATION CASES ---
    # (Using the same get_empirical_params_for_case function as before)
    # We will iterate our managerial levers against EACH of these cases.
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO' # For tagging results

    # --- Parameters that are MANAGERIAL LEVERS (what "Soap Inc." controls) ---
    # We will do a very small grid search on these.
    managerial_params_grid = {
        'COST_SUBMIT_FIRST_TIME_WORD': [5, 8],          # Cost to introduce brand new (pol,word)
        'COST_AGREE_OR_RESUBMIT': [2, 4],               # Cost to interact with existing (pol,word)
        'POPULARITY_THRESHOLD_BASE': [15, 25],         # Higher threshold for longer sim
        'PLATFORM_RAKE_PERCENTAGE': [0.20, 0.35],       # Key lever
        # 'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': [0.65], # Fixed for this small grid
        # 'POPULARITY_DECAY_RATE': [0.015],             # Slower decay for longer sim
    }
    # This grid: 2*2*2*2 = 16 managerial combinations.
    # Total sims = 16 managerial combos * 3 empirical cases = 48 simulations.
    # Each simulation will be longer, so 48 is already a decent number for a single tuning session.

    # --- Parameters that are FIXED for this entire tuning session (based on "Real-World" estimates) ---
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 100000, # Ambitious target market
        'INITIAL_ACTIVE_USERS': 200,       # Seed users
        'NUM_POLITICIANS': 150,            # Realistic number of relevant politicians
        'SIMULATION_STEPS': 365 * 1,       # Simulate for 1 year (daily steps)
        
        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.65, # Fixed for this run
        'POPULARITY_DECAY_RATE': 0.01,         # Slower decay for longer timescale

        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 60,    # 2 months of inactivity
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 2.0,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.02, # 2% chance to churn per step if conditions met

        'FINITE_DICTIONARY_SIZE': 3000,        # Larger, more realistic vocab
        'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.03, # Slightly lower chance for OOD words
        'COMMON_WORD_VOCAB_SIZE': 200, # Fallback
        'INNOVATOR_RARE_WORD_CHANCE': 0.1, # Fallback
    }

    # --- Create parameter combinations ---
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            current_combo = fixed_sim_params.copy()
            # Empirical parameters are set first, as they might contain some defaults
            # for keys also in fixed_sim_params (e.g. specific behavioral probs)
            # or attractiveness params.
            empirical_set = get_empirical_params_for_case(empirical_case_name)
            current_combo.update(empirical_set) # Load empirical set
            
            # Managerial levers override if any keys happen to overlap with empirical set
            # (though they are designed to be mostly distinct categories of params)
            current_combo.update(managerial_combo) 
            current_combo[empirical_case_param_name] = empirical_case_name

            min_cost = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD'), current_combo.get('COST_AGREE_OR_RESUBMIT'))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')
            
            all_parameter_combinations.append(current_combo)
    
    num_total_combinations = len(all_parameter_combinations)
    print(f"Starting Grid Search with {num_total_combinations} total parameter combinations (Managerial x Empirical Cases).")
    print(f"Each simulation run will have {fixed_sim_params['SIMULATION_STEPS']} steps.")

    all_sim_results_for_df = [] # For pandas DataFrame later
    best_overall_score = -float('inf')
    best_params_overall = None
    best_detailed_results_overall = None
    tuning_start_time = time.time()

    # Initialize Zipfian dictionary once
    initialize_zipfian_dictionary(
        fixed_sim_params.get('FINITE_DICTIONARY_SIZE'),
        fixed_sim_params.get('ZIPFIAN_ALPHA')
    )

    for i, p_full_config_combo in enumerate(all_parameter_combinations):
        sim_id_str = f"real_world_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 1 # <<<< CHANGED: Only 1 run per combo due to longer individual runtimes
                             # For final validation of best params, you'd increase this.
        current_set_scores, current_set_details_list = [], []
        set_start_time = time.time()

        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            # Target 60% of potential pool for "good" user growth in this longer sim
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.60) 
            current_set_scores.append(score)
            print(f"  Run {run_idx+1}/{num_runs_per_set} | Score: {score:,.2f} | EndUsers: {run_detail['final_num_active_users']}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {run_detail['final_treasury']:,.0f} | Broke%: {run_detail['users_broke_percent']:.1f}")
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        if current_set_scores:
            # For single run, avg_score is just the score, and rep_details is that run's details
            rep_details = current_set_details_list[0] 
        else: rep_details = {}
        
        print(f"  Combo {i+1} (Case: {empirical_case_tag}) Avg Score: {avg_score:,.2f} (took {time.time() - set_start_time:.2f}s)")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  -------> *** New Best Overall Avg Score: {best_overall_score:,.2f} for Combo {i+1} (Case: {empirical_case_tag}) *** <-------")
    
    # --- [The rest of the __main__ block (saving results, printing best, plotting) remains the same as your last full script] ---
    # ... (pasting it here for completeness) ...
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_real_world_params" # New directory for these results
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_rw_{timestamp_str}"
    
    print(f"\n--- Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")

    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Avg Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Full Parameter Set (Managerial Levers + Assumed Empirical Case):")
        # Print all params for the best combo, including fixed and empirical that led to it
        for key, val in best_params_overall.items(): print(f"  {key}: {val}")
        
        print("\nMetrics for Best Config (Representative Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", 
                       "final_gini_coefficient", "total_popular_word_events", 
                       "unique_popular_word_strings_count", 
                       "avg_actions_per_active_user_per_step_overall"]
        for k_met in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met)
            if isinstance(val, float): print(f"  {k_met}: {val:,.4f}")
            else: print(f"  {k_met}: {val}")
        
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        try:
            df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
            print(f"\nFull summary of all combinations saved to: {results_filename_base}_summary.csv")
        except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")

        if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
            plt.figure(figsize=(12,7))
            history_active = best_detailed_results_overall['history_num_active_users']
            history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
            plt.plot(history_active, label=f"Active Users (Best Run Score: {best_overall_score:,.0f})", color="blue", linewidth=2)
            plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='r', linestyle='--', label="Potential Pool Size")
            ax_attract = plt.gca().twinx()
            if history_attract:
                ax_attract.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="green", linestyle=':', alpha=0.7)
                ax_attract.set_ylabel("Attractiveness Score (x100)", color="green", fontsize=10); ax_attract.tick_params(axis='y', labelcolor="green")
            plt.gca().set_xlabel("Simulation Step (Day)", fontsize=12); plt.gca().set_ylabel("Number of Active Users", color="blue", fontsize=12)
            plt.gca().tick_params(axis='y', labelcolor="blue")
            plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=14, fontweight='bold')
            lines, labels = plt.gca().get_legend_handles_labels(); lines2, labels2 = ax_attract.get_legend_handles_labels()
            if lines2: # Only add legend if ax_attract was plotted on
                ax_attract.legend(lines + lines2, labels + labels2, loc='center right')
            else:
                plt.gca().legend(loc='center right')

            plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
        
        if not df_all_runs.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Set2", hue=empirical_case_param_name, legend=False)
            plt.title("Impact of Empirical Case Scenarios on Score", fontsize=14)
            plt.xlabel("Empirical Case Scenario", fontsize=12); plt.ylabel("Average Incentive Score", fontsize=12)
            plt.xticks(rotation=15, ha='right'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_empirical_case_impact.png"); plt.show()

            managerial_params_to_plot = [k for k,v in managerial_params_grid.items() if len(v)>1]
            if managerial_params_to_plot:
                print("\nGenerating parameter sensitivity plots (faceted by empirical case)...")
                for param_name in managerial_params_to_plot:
                    if param_name in df_all_runs.columns and df_all_runs[param_name].nunique() > 1:
                        g = sns.catplot(x=param_name, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="viridis", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)), height=5, aspect=1.2)
                        g.fig.suptitle(f"Score vs. {param_name} (by Empirical Case)", fontsize=16, y=1.03)
                        g.set_xticklabels(rotation=30, ha='right')
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name}_by_case.png")
                        print(f"Sensitivity plot for {param_name} saved.")
                        plt.show()
            
            plt.figure(figsize=(10,6))
            scores_hist = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist:
                plt.hist(scores_hist, bins=max(10, num_total_combinations//2 if num_total_combinations>0 else 1), edgecolor='k', color='purple')
                plt.title('Distribution of Avg Scores (All Empirical Cases)', fontsize=14); plt.xlabel('Avg Score'); plt.ylabel('# Param Combos'); plt.grid(axis='y'); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png")
                print(f"Overall score distribution plot saved to: {results_filename_base}_score_distribution_all_cases.png")
    else:
        print("No simulations run or no valid results.")
    # Parameters that are MANAGERIAL LEVERS (we control these)
    managerial_params_grid = {
        'COST_SUBMIT_FIRST_TIME_WORD': [5, 7],
        'COST_AGREE_OR_RESUBMIT': [2, 4],
        'POPULARITY_THRESHOLD_BASE': [7, 12],
        'PLATFORM_RAKE_PERCENTAGE': [0.15, 0.30], # Lower rake to see effect on user growth
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': [0.60, 0.75], # Test impact of originator reward
        'POPULARITY_DECAY_RATE': [0.025, 0.045],
    }

    # EMPIRICAL POPULATION CASES (we estimate these, try different scenarios)
    empirical_case_param_name = 'EMPIRICAL_CASE_SCENARIO'
    empirical_cases_to_test = ["BaselineEngagement", "HighEngagement_RewardDriven", "CautiousFollowers_LowActivity"]

    # Fixed parameters for this tuning run (not part of the grid search directly)
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 1500, # Fixed potential market size
        'INITIAL_ACTIVE_USERS': 25,       # Small seed group
        'NUM_POLITICIANS': 4,
        'SIMULATION_STEPS': 80,      # Slightly longer to see growth/churn
        'INITIAL_USER_BALANCE_MEAN': 100, 'INITIAL_USER_BALANCE_STDDEV': 20,
        
        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 20,
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.8, # Multiplier of min_action_cost
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.10,

        'FINITE_DICTIONARY_SIZE': 400, 'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.05,
        'COMMON_WORD_VOCAB_SIZE': 200, # Fallback if Zipfian fails
        'INNOVATOR_RARE_WORD_CHANCE': 0.1, # Fallback if Zipfian fails
    }

    # --- Create parameter combinations ---
    # 1. Get combinations of managerial params
    managerial_keys, managerial_values = zip(*managerial_params_grid.items())
    managerial_combinations = [dict(zip(managerial_keys, v)) for v in itertools.product(*managerial_values)]

    # 2. Create full combinations by merging with each empirical case
    all_parameter_combinations = []
    for managerial_combo in managerial_combinations:
        for empirical_case_name in empirical_cases_to_test:
            # Start with fixed params, then empirical, then override with managerial
            current_combo = fixed_sim_params.copy()
            current_combo.update(get_empirical_params_for_case(empirical_case_name)) # Load empirical set
            current_combo.update(managerial_combo) # Managerial levers override if keys overlap
            current_combo[empirical_case_param_name] = empirical_case_name # Tag the combo with its case

            # Derive CHURN_LOW_BALANCE_THRESHOLD based on current costs in combo
            min_cost = min(current_combo.get('COST_SUBMIT_FIRST_TIME_WORD', 5), current_combo.get('COST_AGREE_OR_RESUBMIT', 3))
            current_combo['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost * current_combo.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER', 2.0)
            
            all_parameter_combinations.append(current_combo)
    
    num_total_combinations = len(all_parameter_combinations)
    # Managerial combos = 2^6 = 64. Times 3 empirical cases = 192 combinations.
    print(f"Starting Grid Search with {num_total_combinations} total parameter combinations (Managerial x Empirical Cases).")

    all_sim_results_for_df = []
    best_overall_score = -float('inf')
    best_params_overall = None # Will store the full merged dict
    best_detailed_results_overall = None
    tuning_start_time = time.time()

    # Initialize Zipfian dictionary once (assuming FINITE_DICTIONARY_SIZE and ZIPFIAN_ALPHA are in fixed_sim_params)
    initialize_zipfian_dictionary(
        fixed_sim_params.get('FINITE_DICTIONARY_SIZE', 500),
        fixed_sim_params.get('ZIPFIAN_ALPHA', 1.1)
    )

    for i, p_full_config_combo in enumerate(all_parameter_combinations):
        sim_id_str = f"emp_case_combo_{i+1}"
        empirical_case_tag = p_full_config_combo.get(empirical_case_param_name, "UnknownCase")
        print(f"\nRunning Combo {i+1}/{num_total_combinations} (ID: {sim_id_str}, Case: {empirical_case_tag})")
        
        num_runs_per_set = 2 # Robustness
        current_set_scores, current_set_details_list = [], []
        set_start_time = time.time()

        for run_idx in range(num_runs_per_set):
            run_detail = run_full_simulation_with_dynamic_users(p_full_config_combo.copy(), f"{sim_id_str}_run{run_idx+1}")
            current_set_details_list.append(run_detail)
            # Use a consistent desired_final_user_ratio for scoring
            score = calculate_dynamic_user_growth_score(run_detail, desired_final_user_ratio=0.80) 
            current_set_scores.append(score)
            print(f"  Run {run_idx+1}/{num_runs_per_set} | Score: {score:,.2f} | EndUsers: {run_detail['final_num_active_users']}/{p_full_config_combo['POTENTIAL_USER_POOL_SIZE']} | Treas: {run_detail['final_treasury']:,.0f} | Broke%: {run_detail['users_broke_percent']:.1f}")
        
        avg_score = np.mean(current_set_scores) if current_set_scores else -float('inf')
        if current_set_scores:
            closest_run_idx = np.argmin(np.abs(np.array(current_set_scores) - avg_score))
            rep_details = current_set_details_list[closest_run_idx]
        else: rep_details = {}
        
        print(f"  Combo {i+1} (Case: {empirical_case_tag}) Avg Score: {avg_score:,.2f} (took {time.time() - set_start_time:.2f}s)")

        temp_df_res = rep_details.copy(); temp_df_res.update(p_full_config_combo); temp_df_res['avg_score_for_combo'] = avg_score
        all_sim_results_for_df.append(temp_df_res)

        if avg_score > best_overall_score:
            best_overall_score, best_params_overall, best_detailed_results_overall = avg_score, p_full_config_combo, rep_details
            print(f"  -------> *** New Best Overall Avg Score: {best_overall_score:,.2f} for Combo {i+1} (Case: {empirical_case_tag}) *** <-------")
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_empirical_cases"
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sim_results_emp_cases_{timestamp_str}"
    
    print(f"\n--- Grid Search Complete --- (Total: { (time.time() - tuning_start_time)/60:.2f}m)")

    if best_params_overall and best_detailed_results_overall:
        print(f"\nBest Avg Score: {best_overall_score:,.2f} (Achieved with Empirical Case: {best_params_overall.get(empirical_case_param_name)})")
        print("Best Full Parameter Set (Managerial Levers + Assumed Empirical Case):")
        for key, val in best_params_overall.items(): print(f"  {key}: {val}")
        
        print("\nMetrics for Best Config (Representative Run):")
        key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", 
                       "total_popular_word_events", "unique_popular_word_strings_count", 
                       "avg_actions_per_active_user_per_step_overall"]
        for k_met in key_metrics_to_show:
            val = best_detailed_results_overall.get(k_met)
            if isinstance(val, float): print(f"  {k_met}: {val:,.4f}")
            else: print(f"  {k_met}: {val}")
        
        df_all_runs = pd.DataFrame(all_sim_results_for_df)
        try:
            df_all_runs.to_csv(f"{results_filename_base}_summary.csv", index=False)
            print(f"\nFull summary of all combinations saved to: {results_filename_base}_summary.csv")
        except Exception as e_csv: print(f"Error saving summary CSV: {e_csv}")

        if 'history_num_active_users' in best_detailed_results_overall and best_detailed_results_overall['history_num_active_users']:
            plt.figure(figsize=(12,7))
            history_active = best_detailed_results_overall['history_num_active_users']
            history_attract = best_detailed_results_overall.get('history_platform_attractiveness', [])
            plt.plot(history_active, label=f"Active Users (Best Run Score: {best_overall_score:,.0f})", color="blue", linewidth=2)
            plt.axhline(y=best_params_overall['POTENTIAL_USER_POOL_SIZE'], color='r', linestyle='--', label="Potential Pool Size")
            ax_attract = plt.gca().twinx()
            if history_attract:
                ax_attract.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="green", linestyle=':', alpha=0.7)
                ax_attract.set_ylabel("Attractiveness Score (x100)", color="green", fontsize=10); ax_attract.tick_params(axis='y', labelcolor="green")
            plt.gca().set_xlabel("Simulation Step", fontsize=12); plt.gca().set_ylabel("Number of Active Users", color="blue", fontsize=12)
            plt.gca().tick_params(axis='y', labelcolor="blue")
            plt.title(f"User Growth (Best Params, Case: {best_params_overall.get(empirical_case_param_name)})", fontsize=14, fontweight='bold')
            lines, labels = plt.gca().get_legend_handles_labels(); lines2, labels2 = ax_attract.get_legend_handles_labels()
            ax_attract.legend(lines + lines2, labels + labels2, loc='center right')
            plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
            plt.savefig(f"{results_filename_base}_best_run_growth.png"); plt.show()
        
        if not df_all_runs.empty:
            # Plot avg_score_for_combo vs. EMPIRICAL_CASE_SCENARIO
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=empirical_case_param_name, y='avg_score_for_combo', data=df_all_runs, palette="Set2", hue=empirical_case_param_name, legend=False)
            plt.title("Impact of Empirical Case Scenarios on Score", fontsize=14)
            plt.xlabel("Empirical Case Scenario", fontsize=12)
            plt.ylabel("Average Incentive Score", fontsize=12)
            plt.xticks(rotation=15, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{results_filename_base}_empirical_case_impact.png")
            print(f"Empirical case impact plot saved to: {results_filename_base}_empirical_case_impact.png")
            plt.show()

            # Sensitivity plots for managerial levers, perhaps faceted by empirical case
            managerial_params_to_plot = [k for k,v in managerial_params_grid.items() if len(v)>1]
            if managerial_params_to_plot:
                print("\nGenerating parameter sensitivity plots (faceted by empirical case)...")
                for param_name in managerial_params_to_plot:
                    if param_name in df_all_runs.columns and df_all_runs[param_name].nunique() > 1:
                        plt.figure(figsize=(12, 7))
                        sns.catplot(x=param_name, y='avg_score_for_combo', data=df_all_runs, col=empirical_case_param_name, kind="box", palette="viridis", sharey=True, col_wrap=min(3, len(empirical_cases_to_test)))
                        plt.suptitle(f"Score vs. {param_name} (by Empirical Case)", fontsize=16, y=1.03)
                        # plt.tight_layout() # catplot handles layout well
                        plt.savefig(f"{results_filename_base}_sensitivity_{param_name}_by_case.png")
                        print(f"Sensitivity plot for {param_name} saved.")
                        plt.show() # Show after saving
            
            plt.figure(figsize=(10,6))
            scores_hist = [s for s in df_all_runs['avg_score_for_combo'].dropna() if np.isfinite(s)]
            if scores_hist:
                plt.hist(scores_hist, bins=max(10, num_total_combinations//4 if num_total_combinations>0 else 1), edgecolor='k', color='purple')
                plt.title('Distribution of Avg Scores (All Empirical Cases)', fontsize=14); plt.xlabel('Avg Score'); plt.ylabel('# Param Combos'); plt.grid(axis='y'); plt.show()
                plt.savefig(f"{results_filename_base}_score_distribution_all_cases.png")
                print(f"Overall score distribution plot saved to: {results_filename_base}_score_distribution_all_cases.png")
    else:
        print("No simulations run or no valid results.")