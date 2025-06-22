import random
import collections
import numpy as np
import time
# import itertools # No longer needed for grid search
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # Keep for potential future direct use, though not in this specific output
import os
from datetime import datetime

# --- Classes (User, WordData - unchanged) ---
class User:
    def __init__(self, user_id, archetype, initial_balance, join_step):
        self.id = user_id
        self.archetype = archetype
        self.balance = initial_balance
        self.words_originated_this_cycle = {} 
        self.join_step = join_step
        self.last_activity_step = join_step
        self.total_rewards_received = 0.0
        self.active_in_sim = True

class WordData:
    def __init__(self, first_ever_submitter_id, submission_step, initial_freshness_score=1.0):
        self.count_this_cycle = 0
        self.raw_popularity_score = 0
        self.first_ever_submitter_id = first_ever_submitter_id
        self.current_cycle_originator_id = first_ever_submitter_id 
        self.creation_step = submission_step
        self.last_interaction_step = submission_step
        self.last_reward_step = -1 
        self.agreed_by_users_this_cycle = set()
        self.fees_contributed_this_cycle = 0.0
        self.freshness_score = initial_freshness_score
        self.times_became_popular = 0

# --- Helper Functions (get_gini, initialize_zipfian_dictionary, get_simulated_word_zipfian - unchanged) ---
def get_gini(balances_list):
    if not isinstance(balances_list, list) or not balances_list or len(balances_list) < 2: return 0.0
    balances = np.sort(np.array(balances_list, dtype=float)); balances = np.maximum(balances, 0)
    n = len(balances)
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

# --- Platform Attractiveness & Joining/Churn (calculate_platform_attractiveness - unchanged) ---
def calculate_platform_attractiveness(current_users_on_platform_list, platform_rewards_this_step, actions_this_step, params, step):
    if not current_users_on_platform_list: return params.get('BASE_ATTRACTIVENESS_NO_USERS', 0.02)
    num_platform_users = len(current_users_on_platform_list)
    balances_on_platform = [u.balance for u in current_users_on_platform_list if u.active_in_sim]
    reward_signal = 0
    # Use COST_INTERACT_STALE_WORD as a proxy for avg_cost if COST_AGREE_OR_RESUBMIT isn't directly in this params dict
    avg_cost = params.get('COST_INTERACT_STALE_WORD', params.get('COST_AGREE_OR_RESUBMIT', 2))
    if num_platform_users > 0 and avg_cost > 0:
        reward_per_user_norm = (platform_rewards_this_step / num_platform_users) / avg_cost
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

# --- Function to define Empirical Parameter Sets (get_empirical_params_for_case - unchanged) ---
def get_empirical_params_for_case(case_name):
    default_behavioral = {
        'INNOVATOR_PROB_NEW_CONCEPT': 0.60, 'INNOVATOR_PROB_AGREE': 0.30,
        'FOLLOWER_PROB_NEW_CONCEPT': 0.10, 'FOLLOWER_PROB_AGREE': 0.80,
        'BALANCED_PROB_NEW_CONCEPT': 0.35, 'BALANCED_PROB_AGREE': 0.55,
    }
    default_attractiveness = { # These are base defaults, specific cases will override
        'BASE_ATTRACTIVENESS_NO_USERS': 0.03,
        'BASE_ATTRACTIVENESS_WITH_USERS': 0.02,
        'ATTRACT_REWARD_SENSITIVITY': 0.3,
        'ATTRACT_ACTIVITY_SENSITIVITY': 0.4,
        'ATTRACT_FAIRNESS_SENSITIVITY': 0.15,
        'ATTRACT_BALANCE_SENSITIVITY': 0.2,
    }

    if case_name == "BaselineEngagement":
        params = default_behavioral.copy()
        params.update(default_attractiveness)
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.2, 'Follower': 0.65, 'Balanced': 0.15}, # From HM CSV for Baseline
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.0, # From HM CSV for Baseline
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.7, # From HM CSV for Baseline
            # Attractiveness sensitivities from HM CSV for Baseline
            'BASE_ATTRACTIVENESS_NO_USERS': 0.03,
            'BASE_ATTRACTIVENESS_WITH_USERS': 0.02,
            'ATTRACT_REWARD_SENSITIVITY': 0.3,
            'ATTRACT_ACTIVITY_SENSITIVITY': 0.4,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.15,
            'ATTRACT_BALANCE_SENSITIVITY': 0.2,
        })
        return params
    elif case_name == "HighEngagement_RewardDriven":
        params = default_behavioral.copy()
        # Update with specific HighEngagement values from HM CSV if different, otherwise use defaults.
        # Assuming values from HM CSV for HighEngagement case
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.25, 'Follower': 0.55, 'Balanced': 0.2},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 1.8,
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.8,
            'INNOVATOR_PROB_NEW_CONCEPT': 0.65, 'INNOVATOR_PROB_AGREE': 0.25,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.15, 'FOLLOWER_PROB_AGREE': 0.75,
            'BALANCED_PROB_NEW_CONCEPT': 0.4, 'BALANCED_PROB_AGREE': 0.5,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.05,
            'BASE_ATTRACTIVENESS_WITH_USERS': 0.03,
            'ATTRACT_REWARD_SENSITIVITY': 0.5,
            'ATTRACT_ACTIVITY_SENSITIVITY': 0.5,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.2,
            'ATTRACT_BALANCE_SENSITIVITY': 0.3,
        })
        return params
    elif case_name == "CautiousFollowers_LowActivity":
        params = default_behavioral.copy()
        # Assuming values from HM CSV for CautiousFollowers case
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.1, 'Follower': 0.75, 'Balanced': 0.15},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.5,
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.55,
            'INNOVATOR_PROB_NEW_CONCEPT': 0.5, 'INNOVATOR_PROB_AGREE': 0.4,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.05, 'FOLLOWER_PROB_AGREE': 0.85,
            'BALANCED_PROB_NEW_CONCEPT': 0.25, 'BALANCED_PROB_AGREE': 0.65,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.01,
            'BASE_ATTRACTIVENESS_WITH_USERS': 0.005,
            'ATTRACT_REWARD_SENSITIVITY': 0.2,
            'ATTRACT_ACTIVITY_SENSITIVITY': 0.2,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.1,
            'ATTRACT_BALANCE_SENSITIVITY': 0.1,
        })
        return params
    else:
        print(f"Warning: Unknown empirical case_name '{case_name}'. Using BaselineEngagement.")
        return get_empirical_params_for_case("BaselineEngagement")

# --- Main Simulation Function (run_full_simulation_with_dynamic_users - largely unchanged, check param names) ---
# ... (function definition is the same as your provided code, ensure it uses the correct param names like
#      COST_SUBMIT_FRESH_WORD and COST_INTERACT_STALE_WORD internally)
def run_full_simulation_with_dynamic_users(params, simulation_run_id="sim_dyn_users_run"):
    # --- Unpack Parameters ---
    POTENTIAL_USER_POOL_SIZE = params.get('POTENTIAL_USER_POOL_SIZE', 10000)
    INITIAL_ACTIVE_USERS = params.get('INITIAL_ACTIVE_USERS', 70)
    NUM_POLITICIANS = params.get('NUM_POLITICIANS', 20)
    SIMULATION_STEPS = params.get('SIMULATION_STEPS', 120)

    INITIAL_USER_BALANCE_MEAN = params.get('INITIAL_USER_BALANCE_MEAN', 100)
    INITIAL_USER_BALANCE_STDDEV = params.get('INITIAL_USER_BALANCE_STDDEV', 20)
    
    # Using cost names consistent with this function's internal logic
    COST_SUBMIT_FRESH_WORD = params.get('COST_SUBMIT_FRESH_WORD', params.get('COST_SUBMIT_FIRST_TIME_WORD', 5))
    COST_INTERACT_STALE_WORD = params.get('COST_INTERACT_STALE_WORD', params.get('COST_AGREE_OR_RESUBMIT', 2))
    
    POPULARITY_THRESHOLD_BASE = params.get('POPULARITY_THRESHOLD_BASE', 12)
    POPULARITY_DECAY_RATE = params.get('POPULARITY_DECAY_RATE', 0.025)
    
    PLATFORM_RAKE_PERCENTAGE = params.get('PLATFORM_RAKE_PERCENTAGE', 0.25)
    
    REWARD_TO_ORIGINATOR_SHARE = params.get('REWARD_TO_ORIGINAL_SUBMITTER_SHARE', 0.65)
    REWARD_TO_AGREERS_SHARE = 1.0 - REWARD_TO_ORIGINATOR_SHARE

    FRESHNESS_DECAY_ON_INTERACT = params.get('FRESHNESS_DECAY_ON_INTERACT', 0.01)
    FRESHNESS_RECOVERY_PER_DORMANT_STEP = params.get('FRESHNESS_RECOVERY_PER_DORMANT_STEP', 0.002)
    FRESHNESS_DROP_AFTER_REWARD = params.get('FRESHNESS_DROP_AFTER_REWARD', 0.80) # This will be overridden by best_config
    MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD = params.get('MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD', 0.8)
    REWARD_CYCLE_COOLDOWN_STEPS = params.get('REWARD_CYCLE_COOLDOWN_STEPS', 45) # This will be overridden

    USER_ARCHETYPES_DIST = params.get('USER_ARCHETYPES_DIST')
    INNOVATOR_PROB_NEW_CONCEPT = params.get('INNOVATOR_PROB_NEW_CONCEPT')
    INNOVATOR_PROB_AGREE = params.get('INNOVATOR_PROB_AGREE')
    FOLLOWER_PROB_NEW_CONCEPT = params.get('FOLLOWER_PROB_NEW_CONCEPT')
    FOLLOWER_PROB_AGREE = params.get('FOLLOWER_PROB_AGREE')
    FOLLOWER_POPULARITY_BIAS_FACTOR = params.get('FOLLOWER_POPULARITY_BIAS_FACTOR')
    BALANCED_PROB_NEW_CONCEPT = params.get('BALANCED_PROB_NEW_CONCEPT')
    BALANCED_PROB_AGREE = params.get('BALANCED_PROB_AGREE')
    USER_ACTIVITY_RATE_ON_PLATFORM = params.get('USER_ACTIVITY_RATE_ON_PLATFORM')
    
    ENABLE_CHURN = params.get('ENABLE_CHURN', True)
    CHURN_INACTIVITY_THRESHOLD_STEPS = params.get('CHURN_INACTIVITY_THRESHOLD_STEPS', 30)
    CHURN_LOW_BALANCE_THRESHOLD = params.get('CHURN_LOW_BALANCE_THRESHOLD')
    CHURN_BASE_PROB_IF_CONDITIONS_MET = params.get('CHURN_BASE_PROB_IF_CONDITIONS_MET', 0.03)

    users_master_list = {}
    next_user_id_counter = 0
    politicians_dict = {f"pol_{i}": {'words': {}} for i in range(NUM_POLITICIANS)}
    platform_treasury = 0.0
    platform_total_rewards_paid_overall = 0.0
    total_word_instances_created = 0
    total_actions_simulation = 0
    
    archetype_keys = list(USER_ARCHETYPES_DIST.keys())
    archetype_probs = np.array(list(USER_ARCHETYPES_DIST.values()))
    if abs(np.sum(archetype_probs) - 1.0) > 1e-5: archetype_probs /= np.sum(archetype_probs)

    for _ in range(INITIAL_ACTIVE_USERS):
        user_id = f"user_{next_user_id_counter}"
        archetype = np.random.choice(archetype_keys, p=archetype_probs)
        balance = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
        users_master_list[user_id] = User(user_id, archetype, balance, join_step=0)
        next_user_id_counter += 1
    
    history_num_active_users, history_platform_attractiveness, history_avg_user_balance = [], [], []

    for step in range(1, SIMULATION_STEPS + 1):
        current_platform_active_users_objs = [u for u in users_master_list.values() if u.active_in_sim]
        num_current_platform_active_users = len(current_platform_active_users_objs)
        actions_this_step, step_rewards_this_step_val = 0, 0
        
        if num_current_platform_active_users > 0:
            num_to_sample_float = num_current_platform_active_users * USER_ACTIVITY_RATE_ON_PLATFORM
            num_to_sample = min(len(current_platform_active_users_objs), int(num_to_sample_float))
            
            if num_to_sample > 0 :
                actors_for_this_step_ids = random.sample([u.id for u in current_platform_active_users_objs], num_to_sample)

                for user_id in actors_for_this_step_ids:
                    user = users_master_list[user_id]
                    user.last_activity_step = step; action_taken = False
                    
                    if user.archetype == 'Innovator': prob_try_new, prob_try_agree = INNOVATOR_PROB_NEW_CONCEPT, INNOVATOR_PROB_AGREE
                    elif user.archetype == 'Follower': prob_try_new, prob_try_agree = FOLLOWER_PROB_NEW_CONCEPT, FOLLOWER_PROB_AGREE
                    else: prob_try_new, prob_try_agree = BALANCED_PROB_NEW_CONCEPT, BALANCED_PROB_AGREE
                    
                    roll = random.random(); pol_id = f"pol_{random.randint(0, NUM_POLITICIANS - 1)}"
                    pol_words = politicians_dict[pol_id]['words']

                    if roll < prob_try_new:
                        word_str = get_simulated_word_zipfian(user.archetype, params)
                        word_obj = pol_words.get(word_str)
                        is_new_instance = not word_obj

                        current_action_cost = COST_SUBMIT_FRESH_WORD if is_new_instance or \
                                                       (word_obj and word_obj.freshness_score >= MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD) \
                                                     else COST_INTERACT_STALE_WORD
                        
                        if user.balance >= current_action_cost:
                            user.balance -= current_action_cost
                            rake_for_this_action = current_action_cost * PLATFORM_RAKE_PERCENTAGE
                            reward_pool_contribution = current_action_cost - rake_for_this_action
                            platform_treasury += rake_for_this_action
                            
                            if is_new_instance:
                                word_obj = WordData(user.id, step, initial_freshness_score=1.0)
                                pol_words[word_str] = word_obj
                                total_word_instances_created += 1
                                user.words_originated_this_cycle[(pol_id, word_str)] = step
                            else:
                                word_obj = pol_words[word_str]

                            if word_obj.freshness_score >= MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD and \
                               word_obj.current_cycle_originator_id != user.id and \
                               (word_obj.current_cycle_originator_id is None or (step - word_obj.last_reward_step) > REWARD_CYCLE_COOLDOWN_STEPS) :
                                word_obj.current_cycle_originator_id = user.id
                                user.words_originated_this_cycle[(pol_id, word_str)] = step
                                word_obj.count_this_cycle = 0
                                word_obj.agreed_by_users_this_cycle.clear()
                                word_obj.fees_contributed_this_cycle = 0.0
                            
                            word_obj.count_this_cycle += 1
                            word_obj.raw_popularity_score += 1 
                            word_obj.fees_contributed_this_cycle += reward_pool_contribution
                            word_obj.agreed_by_users_this_cycle.add(user.id)
                            word_obj.last_interaction_step = step
                            word_obj.freshness_score = max(0, word_obj.freshness_score - FRESHNESS_DECAY_ON_INTERACT)
                            action_taken = True
                    
                    elif prob_try_new <= roll < prob_try_new + prob_try_agree:
                        if pol_words:
                            existing_words = list(pol_words.keys())
                            if existing_words:
                                word_to_agree_str = ""
                                if user.archetype == 'Follower':
                                    scores = np.array([pol_words[w].raw_popularity_score + 0.1 for w in existing_words])
                                    scores = np.maximum(scores, 0.01); scores_b = scores ** FOLLOWER_POPULARITY_BIAS_FACTOR
                                    sum_b = np.sum(scores_b)
                                    if sum_b > 1e-6:
                                        probs = scores_b / sum_b
                                        if not np.isnan(probs).any() and abs(np.sum(probs)-1.0)<1e-5:
                                            try: word_to_agree_str = np.random.choice(existing_words, p=probs)
                                            except ValueError: word_to_agree_str = random.choice(existing_words)
                                        else: word_to_agree_str = random.choice(existing_words)
                                    else: word_to_agree_str = random.choice(existing_words)
                                else: word_to_agree_str = random.choice(existing_words)

                                if word_to_agree_str:
                                    word_obj = pol_words[word_to_agree_str]
                                    cost = COST_INTERACT_STALE_WORD 
                                    if user.balance >= cost:
                                        user.balance -= cost; platform_treasury += cost * PLATFORM_RAKE_PERCENTAGE
                                        reward_contrib = cost * (1-PLATFORM_RAKE_PERCENTAGE)
                                        word_obj.count_this_cycle += 1; word_obj.raw_popularity_score +=1
                                        word_obj.fees_contributed_this_cycle += reward_contrib
                                        word_obj.agreed_by_users_this_cycle.add(user.id)
                                        word_obj.last_interaction_step = step
                                        word_obj.freshness_score = max(0, word_obj.freshness_score - FRESHNESS_DECAY_ON_INTERACT)
                                        action_taken = True
                    if action_taken: actions_this_step +=1
        total_actions_simulation += actions_this_step

        for pol_data in politicians_dict.values():
            for word_str, word_obj in pol_data['words'].items():
                if word_obj.raw_popularity_score > 0:
                    word_obj.raw_popularity_score -= word_obj.raw_popularity_score * POPULARITY_DECAY_RATE
                    if word_obj.raw_popularity_score < 0.01: word_obj.raw_popularity_score = 0
                
                if step > word_obj.last_interaction_step:
                    word_obj.freshness_score = min(1.0, word_obj.freshness_score + FRESHNESS_RECOVERY_PER_DORMANT_STEP)

                if word_obj.count_this_cycle >= POPULARITY_THRESHOLD_BASE and \
                   (step - word_obj.last_reward_step) > REWARD_CYCLE_COOLDOWN_STEPS:
                    
                    reward_pool = word_obj.fees_contributed_this_cycle
                    originator_id = word_obj.current_cycle_originator_id
                    actual_originator_share = REWARD_TO_ORIGINATOR_SHARE
                    
                    if originator_id and originator_id in users_master_list and users_master_list[originator_id].active_in_sim:
                        rew_orig = reward_pool * actual_originator_share
                        users_master_list[originator_id].balance += rew_orig
                        users_master_list[originator_id].total_rewards_received += rew_orig
                        platform_total_rewards_paid_overall += rew_orig; step_rewards_this_step_val += rew_orig
                    
                    num_agreers_this_cycle = len(word_obj.agreed_by_users_this_cycle)
                    if num_agreers_this_cycle > 0:
                        rew_agr_tot = reward_pool * (1.0 - actual_originator_share)
                        per_agr = rew_agr_tot / num_agreers_this_cycle
                        for agr_id in word_obj.agreed_by_users_this_cycle:
                            if agr_id in users_master_list and users_master_list[agr_id].active_in_sim:
                                users_master_list[agr_id].balance += per_agr
                                users_master_list[agr_id].total_rewards_received += per_agr
                                platform_total_rewards_paid_overall += per_agr; step_rewards_this_step_val += per_agr
                    
                    word_obj.fees_contributed_this_cycle = 0.0
                    word_obj.count_this_cycle = 0
                    word_obj.agreed_by_users_this_cycle.clear()
                    word_obj.times_became_popular += 1
                    word_obj.last_reward_step = step
                    word_obj.freshness_score *= (1.0 - FRESHNESS_DROP_AFTER_REWARD)
                    word_obj.current_cycle_originator_id = None
        
        current_active_users_for_attract_calc = [u for u in users_master_list.values() if u.active_in_sim]
        attractiveness = calculate_platform_attractiveness(current_active_users_for_attract_calc,step_rewards_this_step_val, actions_this_step,params, step)
        history_platform_attractiveness.append(attractiveness)
        newly_joined_this_step = 0
        if len(users_master_list) < POTENTIAL_USER_POOL_SIZE:
            num_potential_can_join = POTENTIAL_USER_POOL_SIZE - len(users_master_list)
            num_trials = min(num_potential_can_join, int(params.get('POTENTIAL_JOIN_TRIALS_PER_STEP', 50) + len(current_active_users_for_attract_calc) * 0.05))
            max_joins_this_step = int(params.get('MAX_NEW_JOINS_PER_STEP_SCALER', 0.02) * POTENTIAL_USER_POOL_SIZE)
            max_joins = min(num_potential_can_join, max_joins_this_step)

            for _ in range(num_trials):
                if newly_joined_this_step >= max_joins: break
                if random.random() < attractiveness:
                    uid = f"user_{next_user_id_counter}"; arch = np.random.choice(archetype_keys, p=archetype_probs)
                    bal = max(0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
                    users_master_list[uid] = User(uid, arch, bal, step); next_user_id_counter +=1; newly_joined_this_step +=1
        
        if ENABLE_CHURN and step > params.get('CHURN_GRACE_PERIOD_STEPS', 20):
            churn_ids = []
            for uid_key, u_obj_val in list(users_master_list.items()): 
                if not u_obj_val.active_in_sim: continue
                c_prob = 0.0
                if (step - u_obj_val.last_activity_step) > CHURN_INACTIVITY_THRESHOLD_STEPS: c_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                if u_obj_val.balance < CHURN_LOW_BALANCE_THRESHOLD: c_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5
                if u_obj_val.total_rewards_received == 0 and u_obj_val.balance < INITIAL_USER_BALANCE_MEAN*0.4 and step > u_obj_val.join_step + (SIMULATION_STEPS*0.15):
                    c_prob += CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.6
                if random.random() < np.clip(c_prob, 0, 0.85): churn_ids.append(uid_key)
            for cid in churn_ids: users_master_list[cid].active_in_sim = False
        
        active_count_this_step = len([u for u in users_master_list.values() if u.active_in_sim])
        history_num_active_users.append(active_count_this_step)
        active_bals = [u.balance for u in users_master_list.values() if u.active_in_sim]
        history_avg_user_balance.append(np.mean(active_bals) if active_bals else 0)

    final_active_users_list = [u for u in users_master_list.values() if u.active_in_sim]
    final_balances = [u.balance for u in final_active_users_list] if final_active_users_list else [0.0]
    final_num_active_users = len(final_active_users_list)
    final_gini = get_gini(final_balances); final_avg_balance = np.mean(final_balances); final_median_balance = np.median(final_balances)
    
    min_action_cost_val = min(COST_SUBMIT_FRESH_WORD, COST_INTERACT_STALE_WORD) 
    users_broke_count = sum(1 for b in final_balances if b < min_action_cost_val)
    
    total_popular_word_events = 0; unique_popular_words_global = set()
    for pol_data in politicians_dict.values():
        for word_s, word_o in pol_data['words'].items():
            if word_o.times_became_popular > 0:
                total_popular_word_events += word_o.times_became_popular
                unique_popular_words_global.add(word_s)
    avg_hist_u = np.mean(history_num_active_users) if history_num_active_users else 0
    
    return {
        "params_config_copy": params.copy(), 
        "final_num_active_users": final_num_active_users,
        "final_treasury": platform_treasury, 
        "total_rewards_paid": platform_total_rewards_paid_overall,
        "final_avg_balance": final_avg_balance, 
        "final_median_balance": final_median_balance,
        "final_gini_coefficient": final_gini,
        "users_broke_percent": (users_broke_count / final_num_active_users) * 100 if final_num_active_users > 0 else (100.0 if INITIAL_ACTIVE_USERS > 0 and final_num_active_users == 0 else 0.0),
        "total_actions_in_sim": total_actions_simulation,
        "avg_actions_per_active_user_per_step_overall": total_actions_simulation / (avg_hist_u * SIMULATION_STEPS) if avg_hist_u * SIMULATION_STEPS > 0 else 0,
        "total_unique_word_instances_created": total_word_instances_created,
        "unique_popular_word_strings_count": len(unique_popular_words_global),
        "total_popular_word_events": total_popular_word_events,
        "history_num_active_users": history_num_active_users,
        "history_platform_attractiveness": history_platform_attractiveness,
        "history_avg_user_balance": history_avg_user_balance
    }

# --- Objective Function (calculate_dynamic_user_growth_score - unchanged) ---
def calculate_dynamic_user_growth_score(results, target_gini=0.50, min_avg_balance_ratio_retained=0.70, desired_final_user_ratio_of_potential=0.60, target_sustainability_of_peak=0.75):
    params = results['params_config_copy']
    score = 0.0
    POTENTIAL_USER_POOL_SIZE = params.get('POTENTIAL_USER_POOL_SIZE', 10000)

    final_user_ratio = results['final_num_active_users'] / POTENTIAL_USER_POOL_SIZE if POTENTIAL_USER_POOL_SIZE > 0 else 0
    growth_target_score = 0
    if final_user_ratio < desired_final_user_ratio_of_potential * 0.4: 
        growth_target_score = -((desired_final_user_ratio_of_potential * 0.4 - final_user_ratio) * 25000)
    else:
        growth_target_score = np.clip(final_user_ratio / desired_final_user_ratio_of_potential, 0, 1.1) * 10000.0
    if final_user_ratio >= desired_final_user_ratio_of_potential: growth_target_score += 4000
    score += growth_target_score

    if results['history_num_active_users'] and max(results['history_num_active_users']) > 0:
        peak_users = max(results['history_num_active_users'])
        sustainability_of_peak = results['final_num_active_users'] / peak_users if peak_users > 0 else 0
        if sustainability_of_peak < target_sustainability_of_peak * 0.8:
            score -= (target_sustainability_of_peak * 0.8 - sustainability_of_peak) * 8000
        else:
            score += sustainability_of_peak * 1500
    elif results['final_num_active_users'] == 0 and params.get('INITIAL_ACTIVE_USERS',0) > 0 :
             score -= 25000 

    if results['final_num_active_users'] > params.get('INITIAL_ACTIVE_USERS',10) * 0.15 :
        treasury_score_comp = 0
        if results['final_treasury'] < 0: treasury_score_comp = results['final_treasury'] * 4.0
        else: treasury_score_comp = results['final_treasury'] * 0.0008
        score += treasury_score_comp
        target_broke = params.get('TARGET_BROKE_AMONG_ACTIVE_PERCENT', 5.0)
        broke_diff = results['users_broke_percent'] - target_broke
        if broke_diff > 0: score -= (broke_diff / 2.0)**2 * 1000.0
        avg_bal_ratio = results['final_avg_balance'] / params['INITIAL_USER_BALANCE_MEAN'] if params['INITIAL_USER_BALANCE_MEAN'] > 0 else 0
        if avg_bal_ratio < min_avg_balance_ratio_retained: score -= (min_avg_balance_ratio_retained - avg_bal_ratio) * 1800.0
        elif avg_bal_ratio > 1.5: score -= (avg_bal_ratio - 1.5) * 300.0
        else: score += avg_bal_ratio * 250.0
        gini_diff = results['final_gini_coefficient'] - target_gini
        if gini_diff > 0 : score -= gini_diff * 1500.0
        else: score += (target_gini - results['final_gini_coefficient']) * 400.0
        score += results['avg_actions_per_active_user_per_step_overall'] * 8000.0
        if results['total_actions_in_sim'] > 0:
            pop_ev_per_1k_act = (results['total_popular_word_events'] / results['total_actions_in_sim']) * 1000
            score += pop_ev_per_1k_act * 100.0
        if results['total_popular_word_events'] > 0:
            diversity = (results['unique_popular_word_strings_count'] / results['total_popular_word_events'])
            score += diversity * 800.0
        elif results['total_unique_word_instances_created'] > 0 and results['total_popular_word_events'] == 0: score -= 1000
    else: score -= 120000
    return score

# --- Main Execution for Single Best Run ---
if __name__ == "__main__":
    
    # --- Best Parameters Identified from HM CSV + Smoothness Tweaks ---
    best_empirical_case_name = 'BaselineEngagement' # From HM CSV best score
    
    # Managerial levers from HM CSV's best run & smoothness tweaks
    best_hm_managerial_config = {
        'COST_SUBMIT_FRESH_WORD': 6,          # From HM: COST_SUBMIT_FIRST_TIME_WORD
        'COST_INTERACT_STALE_WORD': 2,        # From HM: COST_AGREE_OR_RESUBMIT
        'POPULARITY_THRESHOLD_BASE': 10,      # From HM
        'PLATFORM_RAKE_PERCENTAGE': 0.25,   # From HM
        
        # Smoothness Tweaks (overriding potential defaults from HM run)
        'FRESHNESS_DROP_AFTER_REWARD': 0.15,
        'REWARD_CYCLE_COOLDOWN_STEPS': 10,
        
        # Freshness interaction/recovery (using sensible defaults or values that performed well in freshness sketch)
        'FRESHNESS_DECAY_ON_INTERACT': 0.005, 
        'FRESHNESS_RECOVERY_PER_DORMANT_STEP': 0.005,
    }

    # Base fixed parameters (some values from HM CSV's best run, others are general defaults)
    fixed_sim_params = {
        'POTENTIAL_USER_POOL_SIZE': 20000,    # From HM CSV
        'INITIAL_ACTIVE_USERS': 100,          # From HM CSV
        'NUM_POLITICIANS': 50,                # From HM CSV
        'SIMULATION_STEPS': 240,              # Increased for longer view
        'INITIAL_USER_BALANCE_MEAN': 100,     # From HM CSV
        'INITIAL_USER_BALANCE_STDDEV': 20,    # From HM CSV
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.65, # Consistent value
        'MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD': 0.8, # Consistent value
        'POPULARITY_DECAY_RATE': 0.02,        # From HM CSV (was 0.025 in freshness)
        'ENABLE_CHURN': True,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 45, # From HM CSV (was 30 in freshness)
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 1.8,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.025, # From HM CSV (was 0.03 in freshness)
        'FINITE_DICTIONARY_SIZE': 1500,       # From HM CSV
        'ZIPFIAN_ALPHA': 1.1,                 # Consistent
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.04, # Consistent
        'COMMON_WORD_VOCAB_SIZE': 150,        # From HM CSV
        'INNOVATOR_RARE_WORD_CHANCE': 0.1,    # Consistent
    }

    # --- Construct the single best parameter configuration ---
    single_best_config = fixed_sim_params.copy()
    
    empirical_params = get_empirical_params_for_case(best_empirical_case_name)
    single_best_config.update(empirical_params) # Apply empirical case defaults first
    
    # Override with specific values from HM's best BaselineEngagement if they differ from get_empirical_params_for_case defaults
    # (The get_empirical_params_for_case for BaselineEngagement already reflects HM's Baseline values, so this is fine)
    
    single_best_config.update(best_hm_managerial_config) # Apply managerial levers
    single_best_config['EMPIRICAL_CASE_SCENARIO'] = best_empirical_case_name
    
    min_cost_val = min(single_best_config.get('COST_SUBMIT_FRESH_WORD'), single_best_config.get('COST_INTERACT_STALE_WORD'))
    single_best_config['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * single_best_config.get('CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER')

    print(f"--- Running Simulation with Best Parameters from HM CSV (and Smoothness Tweaks) ---")
    print(f"Empirical Case: {single_best_config['EMPIRICAL_CASE_SCENARIO']}")
    print(f"Selected Managerial & Key Fixed Levers:")
    for k, v in best_hm_managerial_config.items():
        print(f"  {k}: {v}")
    print(f"  POTENTIAL_USER_POOL_SIZE: {single_best_config['POTENTIAL_USER_POOL_SIZE']}")
    print(f"  SIMULATION_STEPS: {single_best_config['SIMULATION_STEPS']}")
    
    initialize_zipfian_dictionary(
        single_best_config.get('FINITE_DICTIONARY_SIZE'),
        single_best_config.get('ZIPFIAN_ALPHA')
    )

    run_start_time = time.time()
    num_runs_for_avg = 3
    all_scores_for_best_config = []
    all_details_for_best_config = []

    for i in range(num_runs_for_avg):
        print(f"\nRunning simulation {i+1}/{num_runs_for_avg} for the best configuration...")
        detailed_results = run_full_simulation_with_dynamic_users(single_best_config.copy(), f"best_hm_config_run_{i+1}")
        score = calculate_dynamic_user_growth_score(detailed_results, 
                                                    desired_final_user_ratio_of_potential=0.60, 
                                                    target_sustainability_of_peak=0.70)
        all_scores_for_best_config.append(score)
        all_details_for_best_config.append(detailed_results)
        print(f"  Run {i+1} Score: {score:,.2f} | EndUsers: {detailed_results.get('final_num_active_users',0)}/{single_best_config['POTENTIAL_USER_POOL_SIZE']} | Treas: {detailed_results.get('final_treasury',0):,.0f} | Broke%: {detailed_results.get('users_broke_percent',0):.1f}")

    avg_score_best_config = np.mean(all_scores_for_best_config)
    closest_run_idx = np.argmin(np.abs(np.array(all_scores_for_best_config) - avg_score_best_config))
    representative_run_details = all_details_for_best_config[closest_run_idx]
    
    run_duration_minutes = (time.time() - run_start_time) / 60
    print(f"\n--- Simulation with Best Parameters (HM CSV Based) Complete --- (Total time: {run_duration_minutes:.2f}m)")
    print(f"Average Score over {num_runs_for_avg} runs: {avg_score_best_config:,.2f}")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_best_hm_run"
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/best_hm_run_results_{timestamp_str}"

    print("\nMetrics for Representative Run (Best HM Config with Smoothness):")
    key_metrics_to_show = ["final_num_active_users", "final_treasury", "users_broke_percent", "final_gini_coefficient", "total_popular_word_events", "unique_popular_word_strings_count", "avg_actions_per_active_user_per_step_overall"]
    for k_met_s in key_metrics_to_show:
        val = representative_run_details.get(k_met_s)
        if isinstance(val, float): print(f"  {k_met_s}: {val:,.4f}")
        else: print(f"  {k_met_s}: {val}")

    df_best_run_summary = pd.DataFrame([representative_run_details])
    for p_key, p_val in single_best_config.items():
        if p_key not in df_best_run_summary.columns:
             df_best_run_summary[p_key] = [p_val] 
    try:
        df_best_run_summary.to_csv(f"{results_filename_base}_summary.csv", index=False)
        print(f"\nRepresentative run summary saved to: {results_filename_base}_summary.csv")
    except Exception as e_csv: 
        print(f"Error saving summary CSV: {e_csv}")

    if 'history_num_active_users' in representative_run_details and representative_run_details['history_num_active_users']:
        plt.figure(figsize=(12, 7))
        history_active = representative_run_details['history_num_active_users']
        history_attract = representative_run_details.get('history_platform_attractiveness', [])
        history_avg_bal = representative_run_details.get('history_avg_user_balance', [])
        
        main_ax = plt.gca()
        p1, = main_ax.plot(history_active, label=f"Active Users (Avg Score: {avg_score_best_config:,.0f})", color="navy", linewidth=2)
        main_ax.axhline(y=single_best_config['POTENTIAL_USER_POOL_SIZE'], color='orangered', linestyle='--', label="Potential Pool")
        main_ax.set_xlabel("Simulation Step", fontsize=10)
        main_ax.set_ylabel("Active Users", color="navy", fontsize=10)
        main_ax.tick_params(axis='y', labelcolor="navy", labelsize=8)
        main_ax.tick_params(axis='x', labelsize=8)

        ax_attract_twin = main_ax.twinx()
        p2 = None
        if history_attract:
            p2, = ax_attract_twin.plot(np.array(history_attract) * 100, label=f"Attractiveness (x100)", color="forestgreen", linestyle=':', alpha=0.8)
            ax_attract_twin.set_ylabel("Attractiveness (x100)", color="forestgreen", fontsize=9)
            ax_attract_twin.tick_params(axis='y', labelcolor="forestgreen", labelsize=8)
            
        ax_avg_bal_twin = main_ax.twinx()
        ax_avg_bal_twin.spines["right"].set_position(("outward", 60)) 
        p3 = None
        if history_avg_bal:
            p3, = ax_avg_bal_twin.plot(history_avg_bal, label=f"Avg User Balance", color="purple", linestyle='-.', alpha=0.7)
            ax_avg_bal_twin.set_ylabel("Avg User Balance", color="purple", fontsize=9)
            ax_avg_bal_twin.tick_params(axis='y', labelcolor="purple", labelsize=8)

        plt.title(f"User Growth & Dynamics (Best HM Params: {single_best_config['EMPIRICAL_CASE_SCENARIO']}, Smoothed Rewards)", fontsize=12)
        
        lines = [p1]
        labels = [p1.get_label()]
        if p2: 
            lines.append(p2)
            labels.append(p2.get_label())
        if p3:
            lines.append(p3)
            labels.append(p3.get_label())
        
        main_ax.legend(lines, labels, loc='best', fontsize=8)
        main_ax.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{results_filename_base}_growth_dynamics.png")
        print(f"Growth dynamics plot saved to: {results_filename_base}_growth_dynamics.png")
        plt.show()
    else:
        print("No history data to plot for the best run.")