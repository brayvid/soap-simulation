import random
import collections
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
from datetime import datetime
import cProfile # For profiling
import pstats   # For profiling
import numba    # For JIT compilation

# --- Classes ---
class User:
    def __init__(self, user_id, archetype, initial_balance, join_step):
        self.id = user_id
        self.archetype = archetype
        self.initial_balance = initial_balance # Store initial balance
        self.balance = initial_balance
        self.words_originated_this_cycle = {}
        self.join_step = join_step
        self.last_activity_step = join_step
        self.total_rewards_received = 0.0
        self.active_in_sim = True
        self.departure_step = -1 # For tracking user lifespan

class WordData:
    def __init__(self, first_ever_submitter_id, submission_step, initial_freshness_score=1.0):
        self.count_this_cycle = 0
        self.raw_popularity_score = 0.0 # Ensure it's float for Numba
        self.first_ever_submitter_id = first_ever_submitter_id # Keep as string or int ID
        self.current_cycle_originator_id = first_ever_submitter_id # Keep as string or int ID
        self.creation_step = submission_step
        self.last_interaction_step = submission_step
        self.last_reward_step = -1
        self.agreed_by_users_this_cycle = set() # Numba might struggle with sets directly in nopython mode for complex objects
        self.fees_contributed_this_cycle = 0.0
        self.freshness_score = initial_freshness_score
        self.times_became_popular = 0


# --- Helper Functions ---
@numba.jit(nopython=True)
def get_gini_numba(balances_np_array): # Expect a NumPy array
    # Assumes balances_np_array is already sorted and non-negative
    n = len(balances_np_array)
    if n < 2: return 0.0
    index = np.arange(1, n + 1).astype(np.float64)
    sum_balances = np.sum(balances_np_array)
    if sum_balances == 0: return 0.0
    return (np.sum((2 * index - n - 1) * balances_np_array)) / (n * sum_balances)

def get_gini(balances_list):
    if not isinstance(balances_list, list) or not balances_list or len(balances_list) < 2: return 0.0
    balances = np.sort(np.array(balances_list, dtype=np.float64))
    balances = np.maximum(balances, 0)
    return get_gini_numba(balances)


FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], np.array([], dtype=np.float64)
def initialize_zipfian_dictionary(vocab_size, zipf_alpha):
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS
    if vocab_size <=0 or zipf_alpha <= 0:
        FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], np.array([], dtype=np.float64)
        return
    FINITE_DICTIONARY_WORDS = [f"dict_word_{i}" for i in range(vocab_size)]
    if vocab_size > 0:
        word_indices = np.arange(1, vocab_size + 1, dtype=np.float64)
        probabilities = 1.0 / (word_indices**zipf_alpha)
        sum_probs = np.sum(probabilities)
        if sum_probs > 0:
            FINITE_DICTIONARY_PROBS = probabilities / sum_probs
        else:
            FINITE_DICTIONARY_PROBS = np.ones(vocab_size, dtype=np.float64) / vocab_size
    else:
        FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS = [], np.array([], dtype=np.float64)

def get_simulated_word_zipfian(user_archetype, params):
    global FINITE_DICTIONARY_WORDS, FINITE_DICTIONARY_PROBS
    if not FINITE_DICTIONARY_WORDS or FINITE_DICTIONARY_PROBS.size == 0 or len(FINITE_DICTIONARY_WORDS) != len(FINITE_DICTIONARY_PROBS):
        if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_RARE_WORD_CHANCE', 0.1):
            return f"fallback_unique_word_{random.randint(1000, 2000)}"
        return f"fallback_common_word_{random.randint(1, params.get('COMMON_WORD_VOCAB_SIZE', 200))}"

    if user_archetype == 'Innovator' and random.random() < params.get('INNOVATOR_OUT_OF_DICTIONARY_CHANCE', 0.05):
        return f"innov_ood_word_{random.randint(5000,6000)}"
    try:
        return np.random.choice(FINITE_DICTIONARY_WORDS, p=FINITE_DICTIONARY_PROBS)
    except ValueError:
        return random.choice(FINITE_DICTIONARY_WORDS) if FINITE_DICTIONARY_WORDS else "fallback_word"


def calculate_platform_attractiveness(current_users_on_platform_list, platform_rewards_this_step, actions_this_step, params, step):
    if not current_users_on_platform_list: return params.get('BASE_ATTRACTIVENESS_NO_USERS', 0.02)
    num_platform_users = len(current_users_on_platform_list)
    balances_on_platform = [u.balance for u in current_users_on_platform_list if u.active_in_sim]

    reward_signal = 0.0
    avg_cost = params.get('COST_INTERACT_STALE_WORD', params.get('COST_SUBMIT_FRESH_WORD', 2))
    if num_platform_users > 0 and avg_cost > 0:
        reward_per_user_norm = (platform_rewards_this_step / num_platform_users) / avg_cost
        reward_signal = np.clip(reward_per_user_norm * params.get('ATTRACT_REWARD_SENSITIVITY', 0.3), 0.0, 0.35)

    activity_signal = 0.0
    if num_platform_users > 0:
        activity_per_user = actions_this_step / num_platform_users
        activity_signal = np.clip(activity_per_user * params.get('ATTRACT_ACTIVITY_SENSITIVITY', 0.4), 0.0, 0.25)

    gini_on_platform = get_gini(balances_on_platform)
    fairness_signal = np.clip((1.0 - gini_on_platform) * params.get('ATTRACT_FAIRNESS_SENSITIVITY', 0.15), 0.0, 0.15)

    avg_balance_on_platform = np.mean(balances_on_platform) if balances_on_platform else 0.0
    balance_health_signal = 0.0
    initial_mean_bal = params.get('INITIAL_USER_BALANCE_MEAN', 100.0)
    if initial_mean_bal > 0:
        balance_ratio = avg_balance_on_platform / initial_mean_bal
        if balance_ratio < 0.5: balance_health_signal = -0.1
        elif balance_ratio > 0.8: balance_health_signal = np.clip((balance_ratio - 0.8) * params.get('ATTRACT_BALANCE_SENSITIVITY', 0.2), 0.0, 0.2)

    base_attractiveness = params.get('BASE_ATTRACTIVENESS_WITH_USERS', 0.01)
    total_attractiveness = base_attractiveness + reward_signal + activity_signal + fairness_signal + balance_health_signal
    return np.clip(total_attractiveness, 0.001, 0.95)


def get_empirical_params_for_case(case_name):
    default_behavioral = {
        'INNOVATOR_PROB_NEW_CONCEPT': 0.60, 'INNOVATOR_PROB_AGREE': 0.30,
        'FOLLOWER_PROB_NEW_CONCEPT': 0.10, 'FOLLOWER_PROB_AGREE': 0.80,
        'BALANCED_PROB_NEW_CONCEPT': 0.35, 'BALANCED_PROB_AGREE': 0.55,
    }
    default_attractiveness = {
        'BASE_ATTRACTIVENESS_NO_USERS': 0.03,
        'BASE_ATTRACTIVENESS_WITH_USERS': 0.02,
        'ATTRACT_REWARD_SENSITIVITY': 0.3,
        'ATTRACT_ACTIVITY_SENSITIVITY': 0.4,
        'ATTRACT_FAIRNESS_SENSITIVITY': 0.15,
        'ATTRACT_BALANCE_SENSITIVITY': 0.2,
    }

    if case_name == "BaselineEngagement":
        params = default_behavioral.copy(); params.update(default_attractiveness)
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.2, 'Follower': 0.65, 'Balanced': 0.15},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.0,
            'USER_ACTIVITY_RATE_ON_PLATFORM': 0.7,
        })
        return params
    elif case_name == "HighEngagement_RewardDriven":
        params = default_behavioral.copy(); params.update(default_attractiveness)
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.25, 'Follower': 0.55, 'Balanced': 0.2},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 1.8, 'USER_ACTIVITY_RATE_ON_PLATFORM': 0.8,
            'INNOVATOR_PROB_NEW_CONCEPT': 0.65, 'INNOVATOR_PROB_AGREE': 0.25,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.15, 'FOLLOWER_PROB_AGREE': 0.75,
            'BALANCED_PROB_NEW_CONCEPT': 0.4, 'BALANCED_PROB_AGREE': 0.5,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.05, 'BASE_ATTRACTIVENESS_WITH_USERS': 0.03,
            'ATTRACT_REWARD_SENSITIVITY': 0.5, 'ATTRACT_ACTIVITY_SENSITIVITY': 0.5,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.2, 'ATTRACT_BALANCE_SENSITIVITY': 0.3,
        })
        return params
    elif case_name == "CautiousFollowers_LowActivity":
        params = default_behavioral.copy(); params.update(default_attractiveness)
        params.update({
            'USER_ARCHETYPES_DIST': {'Innovator': 0.1, 'Follower': 0.75, 'Balanced': 0.15},
            'FOLLOWER_POPULARITY_BIAS_FACTOR': 2.5, 'USER_ACTIVITY_RATE_ON_PLATFORM': 0.55,
            'INNOVATOR_PROB_NEW_CONCEPT': 0.5, 'INNOVATOR_PROB_AGREE': 0.4,
            'FOLLOWER_PROB_NEW_CONCEPT': 0.05, 'FOLLOWER_PROB_AGREE': 0.85,
            'BALANCED_PROB_NEW_CONCEPT': 0.25, 'BALANCED_PROB_AGREE': 0.65,
            'BASE_ATTRACTIVENESS_NO_USERS': 0.01, 'BASE_ATTRACTIVENESS_WITH_USERS': 0.005,
            'ATTRACT_REWARD_SENSITIVITY': 0.2, 'ATTRACT_ACTIVITY_SENSITIVITY': 0.2,
            'ATTRACT_FAIRNESS_SENSITIVITY': 0.1, 'ATTRACT_BALANCE_SENSITIVITY': 0.1,
        })
        return params
    else:
        print(f"Warning: Unknown empirical case_name '{case_name}'. Using BaselineEngagement.")
        return get_empirical_params_for_case("BaselineEngagement")

# --- Main Simulation Function ---
def run_full_simulation_with_dynamic_users(params, simulation_run_id="sim_dyn_users_run"):
    # --- Parameter Unpacking ---
    POTENTIAL_USER_POOL_SIZE = int(params.get('POTENTIAL_USER_POOL_SIZE', 10000))
    INITIAL_ACTIVE_USERS = int(params.get('INITIAL_ACTIVE_USERS', 70))
    NUM_POLITICIANS = int(params.get('NUM_POLITICIANS', 20))
    SIMULATION_STEPS = int(params.get('SIMULATION_STEPS', 120))
    INITIAL_USER_BALANCE_MEAN = float(params.get('INITIAL_USER_BALANCE_MEAN', 100.0))
    INITIAL_USER_BALANCE_STDDEV = float(params.get('INITIAL_USER_BALANCE_STDDEV', 20.0))
    COST_SUBMIT_FRESH_WORD = float(params.get('COST_SUBMIT_FRESH_WORD', 5.0))
    COST_INTERACT_STALE_WORD = float(params.get('COST_INTERACT_STALE_WORD', 2.0))
    POPULARITY_THRESHOLD_BASE = int(params.get('POPULARITY_THRESHOLD_BASE', 12))
    POPULARITY_DECAY_RATE = float(params.get('POPULARITY_DECAY_RATE', 0.025))
    PLATFORM_RAKE_PERCENTAGE = float(params.get('PLATFORM_RAKE_PERCENTAGE', 0.25))
    REWARD_TO_ORIGINAL_SUBMITTER_SHARE = float(params.get('REWARD_TO_ORIGINAL_SUBMITTER_SHARE', 0.65))
    FRESHNESS_DECAY_ON_INTERACT = float(params.get('FRESHNESS_DECAY_ON_INTERACT', 0.01))
    FRESHNESS_RECOVERY_PER_DORMANT_STEP = float(params.get('FRESHNESS_RECOVERY_PER_DORMANT_STEP', 0.002))
    FRESHNESS_DROP_AFTER_REWARD = float(params.get('FRESHNESS_DROP_AFTER_REWARD', 0.80))
    MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD = float(params.get('MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD', 0.8))
    REWARD_CYCLE_COOLDOWN_STEPS = int(params.get('REWARD_CYCLE_COOLDOWN_STEPS', 45))
    USER_ARCHETYPES_DIST = params.get('USER_ARCHETYPES_DIST')
    INNOVATOR_PROB_NEW_CONCEPT = float(params.get('INNOVATOR_PROB_NEW_CONCEPT'))
    INNOVATOR_PROB_AGREE = float(params.get('INNOVATOR_PROB_AGREE'))
    FOLLOWER_PROB_NEW_CONCEPT = float(params.get('FOLLOWER_PROB_NEW_CONCEPT'))
    FOLLOWER_PROB_AGREE = float(params.get('FOLLOWER_PROB_AGREE'))
    FOLLOWER_POPULARITY_BIAS_FACTOR = float(params.get('FOLLOWER_POPULARITY_BIAS_FACTOR'))
    BALANCED_PROB_NEW_CONCEPT = float(params.get('BALANCED_PROB_NEW_CONCEPT'))
    BALANCED_PROB_AGREE = float(params.get('BALANCED_PROB_AGREE'))
    USER_ACTIVITY_RATE_ON_PLATFORM = float(params.get('USER_ACTIVITY_RATE_ON_PLATFORM'))
    ENABLE_CHURN = params.get('ENABLE_CHURN', True)
    CHURN_INACTIVITY_THRESHOLD_STEPS = int(params.get('CHURN_INACTIVITY_THRESHOLD_STEPS', 30))
    CHURN_LOW_BALANCE_THRESHOLD = float(params.get('CHURN_LOW_BALANCE_THRESHOLD'))
    CHURN_BASE_PROB_IF_CONDITIONS_MET = float(params.get('CHURN_BASE_PROB_IF_CONDITIONS_MET', 0.03))
    CHURN_GRACE_PERIOD_STEPS = int(params.get('CHURN_GRACE_PERIOD_STEPS', 20))
    POTENTIAL_JOIN_TRIALS_PER_STEP = int(params.get('POTENTIAL_JOIN_TRIALS_PER_STEP', 50))
    MAX_NEW_JOINS_PER_STEP_SCALER = float(params.get('MAX_NEW_JOINS_PER_STEP_SCALER', 0.02))
    # --- NEW --- Parameter for platform money injection
    PLATFORM_REWARD_INJECTION_ON_SUCCESS = float(params.get('PLATFORM_REWARD_INJECTION_ON_SUCCESS', 0.0))

    # --- Initialization ---
    users_master_list = {}
    next_user_id_counter = 0
    politicians_dict = {f"pol_{i}": {'words': {}} for i in range(NUM_POLITICIANS)}
    platform_treasury = 0.0
    platform_total_rewards_paid_overall = 0.0
    # --- NEW --- Track total injected money for analysis
    platform_total_injections = 0.0
    total_word_instances_created = 0
    total_actions_simulation = 0
    archetype_keys = list(USER_ARCHETYPES_DIST.keys())
    archetype_probs_list = list(USER_ARCHETYPES_DIST.values())
    archetype_probs_np = np.array(archetype_probs_list, dtype=np.float64)
    if abs(np.sum(archetype_probs_np) - 1.0) > 1e-5:
        archetype_probs_np /= np.sum(archetype_probs_np)

    for i in range(INITIAL_ACTIVE_USERS):
        user_id_str = f"user_{next_user_id_counter}"
        archetype = np.random.choice(archetype_keys, p=archetype_probs_np)
        balance = max(0.0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
        users_master_list[user_id_str] = User(user_id_str, archetype, balance, 0)
        next_user_id_counter += 1

    history_num_active_users, history_platform_attractiveness = [], []
    history_avg_user_balance, history_avg_top1_percent_balance, history_avg_top10_percent_balance = [], [], []

    # --- Main Simulation Loop ---
    for step in range(1, SIMULATION_STEPS + 1):
        current_active_user_ids = [uid for uid, u in users_master_list.items() if u.active_in_sim]
        num_current_platform_active_users = len(current_active_user_ids)
        actions_this_step = 0
        step_rewards_this_step_val = 0.0

        if num_current_platform_active_users > 0:
            num_to_sample = min(num_current_platform_active_users, int(round(num_current_platform_active_users * USER_ACTIVITY_RATE_ON_PLATFORM)))
            if num_to_sample > 0:
                actors_for_this_step_ids = random.sample(current_active_user_ids, num_to_sample)
                for user_id_str in actors_for_this_step_ids:
                    user = users_master_list[user_id_str]
                    user.last_activity_step = step
                    action_taken = False
                    if user.archetype == 'Innovator': prob_try_new, prob_try_agree = INNOVATOR_PROB_NEW_CONCEPT, INNOVATOR_PROB_AGREE
                    elif user.archetype == 'Follower': prob_try_new, prob_try_agree = FOLLOWER_PROB_NEW_CONCEPT, FOLLOWER_PROB_AGREE
                    else: prob_try_new, prob_try_agree = BALANCED_PROB_NEW_CONCEPT, BALANCED_PROB_AGREE
                    
                    roll = random.random()
                    pol_idx = random.randint(0, NUM_POLITICIANS - 1); pol_id_str = f"pol_{pol_idx}"
                    pol_words = politicians_dict[pol_id_str]['words']

                    if roll < prob_try_new: # Try to submit/originate a word
                        word_str = get_simulated_word_zipfian(user.archetype, params)
                        word_obj = pol_words.get(word_str)
                        is_new_instance_globally = not word_obj
                        
                        current_action_cost = COST_SUBMIT_FRESH_WORD
                        if word_obj and word_obj.freshness_score < MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD:
                             current_action_cost = COST_INTERACT_STALE_WORD

                        if user.balance >= current_action_cost:
                            user.balance -= current_action_cost
                            rake_for_this_action = current_action_cost * PLATFORM_RAKE_PERCENTAGE
                            reward_pool_contribution = current_action_cost - rake_for_this_action
                            platform_treasury += rake_for_this_action

                            if is_new_instance_globally:
                                word_obj = WordData(user.id, step, initial_freshness_score=1.0)
                                pol_words[word_str] = word_obj
                                total_word_instances_created += 1
                                word_obj.current_cycle_originator_id = user.id
                                word_obj.count_this_cycle = 0
                                word_obj.agreed_by_users_this_cycle.clear()
                                word_obj.fees_contributed_this_cycle = 0.0 # Starts at 0 for a new word
                            else:
                                can_originate_new_cycle = (word_obj.freshness_score >= MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD and
                                   word_obj.current_cycle_originator_id != user.id and
                                   (word_obj.current_cycle_originator_id is None or (step - word_obj.last_reward_step) > REWARD_CYCLE_COOLDOWN_STEPS))

                                if can_originate_new_cycle:
                                    word_obj.current_cycle_originator_id = user.id
                                    word_obj.count_this_cycle = 0
                                    word_obj.agreed_by_users_this_cycle.clear()
                                    # --- FIX --- Do not reset the fee pool. Let it roll over to the new cycle.
                                    # This is the fix for the "lost pool" leak. The line below is now removed:
                                    # word_obj.fees_contributed_this_cycle = 0.0
                            
                            word_obj.count_this_cycle += 1
                            word_obj.raw_popularity_score += 1.0
                            word_obj.fees_contributed_this_cycle += reward_pool_contribution
                            word_obj.agreed_by_users_this_cycle.add(user.id)
                            word_obj.last_interaction_step = step
                            word_obj.freshness_score = max(0.0, word_obj.freshness_score - FRESHNESS_DECAY_ON_INTERACT)
                            action_taken = True

                    elif prob_try_new <= roll < prob_try_new + prob_try_agree: # Try to agree
                        if pol_words:
                            existing_words_keys = list(pol_words.keys())
                            if existing_words_keys:
                                word_to_agree_str = ""
                                if user.archetype == 'Follower' and FOLLOWER_POPULARITY_BIAS_FACTOR > 0:
                                    scores_list = [pol_words[w].raw_popularity_score + 0.1 for w in existing_words_keys]
                                    scores_np = np.power(np.array(scores_list, dtype=np.float64), FOLLOWER_POPULARITY_BIAS_FACTOR)
                                    sum_scores = np.sum(scores_np)
                                    if sum_scores > 1e-9:
                                        probs_np = scores_np / sum_scores
                                        try: word_to_agree_str = np.random.choice(existing_words_keys, p=probs_np)
                                        except (ValueError, FloatingPointError): word_to_agree_str = random.choice(existing_words_keys)
                                    else: word_to_agree_str = random.choice(existing_words_keys)
                                else: word_to_agree_str = random.choice(existing_words_keys)

                                if word_to_agree_str:
                                    word_obj = pol_words[word_to_agree_str]; cost = COST_INTERACT_STALE_WORD
                                    if user.balance >= cost:
                                        user.balance -= cost
                                        rake_for_this_action = cost * PLATFORM_RAKE_PERCENTAGE
                                        reward_contrib = cost - rake_for_this_action
                                        platform_treasury += rake_for_this_action
                                        
                                        word_obj.count_this_cycle += 1
                                        word_obj.raw_popularity_score +=1.0
                                        word_obj.fees_contributed_this_cycle += reward_contrib
                                        word_obj.agreed_by_users_this_cycle.add(user.id)
                                        word_obj.last_interaction_step = step
                                        word_obj.freshness_score = max(0.0, word_obj.freshness_score - FRESHNESS_DECAY_ON_INTERACT)
                                        action_taken = True
                    if action_taken: actions_this_step +=1
        
        total_actions_simulation += actions_this_step

        # --- Decay and Reward Distribution ---
        for pol_id, pol_data in politicians_dict.items():
            for word_str, word_obj in pol_data['words'].items():
                if word_obj.raw_popularity_score > 0:
                    word_obj.raw_popularity_score = max(0.0, word_obj.raw_popularity_score * (1 - POPULARITY_DECAY_RATE))

                if step > word_obj.last_interaction_step:
                    word_obj.freshness_score = min(1.0, word_obj.freshness_score + FRESHNESS_RECOVERY_PER_DORMANT_STEP)

                if word_obj.count_this_cycle >= POPULARITY_THRESHOLD_BASE and \
                   (word_obj.last_reward_step == -1 or (step - word_obj.last_reward_step) > REWARD_CYCLE_COOLDOWN_STEPS):
                    
                    reward_pool = word_obj.fees_contributed_this_cycle
                    
                    # --- NEW: "Money Faucet" - Inject fresh money from the platform into the pool ---
                    reward_pool += PLATFORM_REWARD_INJECTION_ON_SUCCESS
                    platform_total_injections += PLATFORM_REWARD_INJECTION_ON_SUCCESS
                    
                    originator_id_str = word_obj.current_cycle_originator_id
                    if originator_id_str and originator_id_str in users_master_list and users_master_list[originator_id_str].active_in_sim:
                        originator_user = users_master_list[originator_id_str]
                        rew_orig = reward_pool * REWARD_TO_ORIGINAL_SUBMITTER_SHARE
                        originator_user.balance += rew_orig
                        originator_user.total_rewards_received += rew_orig
                        platform_total_rewards_paid_overall += rew_orig; step_rewards_this_step_val += rew_orig
                    
                    agreers_for_reward = word_obj.agreed_by_users_this_cycle
                    num_agreers_this_cycle = len(agreers_for_reward)
                    if num_agreers_this_cycle > 0:
                        rew_agr_tot = reward_pool * (1.0 - REWARD_TO_ORIGINAL_SUBMITTER_SHARE)
                        per_agr = rew_agr_tot / num_agreers_this_cycle
                        for agr_id_str in agreers_for_reward:
                            if agr_id_str in users_master_list and users_master_list[agr_id_str].active_in_sim:
                                agreer_user = users_master_list[agr_id_str]
                                agreer_user.balance += per_agr
                                agreer_user.total_rewards_received += per_agr
                                platform_total_rewards_paid_overall += per_agr; step_rewards_this_step_val += per_agr
                    
                    # Reset cycle data after successful payout
                    word_obj.fees_contributed_this_cycle = 0.0
                    word_obj.count_this_cycle = 0
                    word_obj.agreed_by_users_this_cycle.clear()
                    word_obj.times_became_popular += 1
                    word_obj.last_reward_step = step
                    word_obj.freshness_score *= (1.0 - FRESHNESS_DROP_AFTER_REWARD)
                    word_obj.current_cycle_originator_id = None

        # --- User Growth and Churn ---
        current_active_users_for_attract_calc = [u for u in users_master_list.values() if u.active_in_sim]
        attractiveness = calculate_platform_attractiveness(current_active_users_for_attract_calc, step_rewards_this_step_val, actions_this_step, params, step)
        history_platform_attractiveness.append(attractiveness)
        
        newly_joined_this_step = 0
        if len(users_master_list) < POTENTIAL_USER_POOL_SIZE:
            num_potential_can_join = POTENTIAL_USER_POOL_SIZE - len(users_master_list)
            num_trials = min(num_potential_can_join, int(POTENTIAL_JOIN_TRIALS_PER_STEP + len(current_active_users_for_attract_calc) * 0.05 * attractiveness))
            max_joins_this_step = min(num_potential_can_join, int(MAX_NEW_JOINS_PER_STEP_SCALER * POTENTIAL_USER_POOL_SIZE * attractiveness))
            
            for _ in range(num_trials):
                if newly_joined_this_step >= max_joins_this_step: break
                if random.random() < attractiveness:
                    uid_str = f"user_{next_user_id_counter}"
                    arch = np.random.choice(archetype_keys, p=archetype_probs_np)
                    bal = max(0.0, np.random.normal(INITIAL_USER_BALANCE_MEAN, INITIAL_USER_BALANCE_STDDEV))
                    users_master_list[uid_str] = User(uid_str, arch, bal, step)
                    next_user_id_counter +=1; newly_joined_this_step +=1
        
        if ENABLE_CHURN and step > CHURN_GRACE_PERIOD_STEPS:
            churn_ids = [uid for uid, u in users_master_list.items() if u.active_in_sim and random.random() < np.clip(
                (CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5 if (step - u.last_activity_step) > CHURN_INACTIVITY_THRESHOLD_STEPS else 0) +
                (CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.5 if u.balance < CHURN_LOW_BALANCE_THRESHOLD else 0) +
                (CHURN_BASE_PROB_IF_CONDITIONS_MET * 0.6 if u.total_rewards_received == 0 and u.balance < INITIAL_USER_BALANCE_MEAN * 0.4 and step > (u.join_step + (SIMULATION_STEPS * 0.15)) else 0),
                0.0, 0.85)]
            for cid_str in churn_ids:
                users_master_list[cid_str].active_in_sim = False
                users_master_list[cid_str].departure_step = step
        
        # --- History Tracking ---
        active_bals = [u.balance for u in users_master_list.values() if u.active_in_sim]
        history_num_active_users.append(len(active_bals))
        history_avg_user_balance.append(np.mean(active_bals) if active_bals else 0.0)
        if active_bals:
            sorted_active_bals = np.sort(active_bals)
            k1 = max(1, int(len(sorted_active_bals) * 0.01))
            history_avg_top1_percent_balance.append(np.mean(sorted_active_bals[-k1:]))
            k10 = max(1, int(len(sorted_active_bals) * 0.10))
            history_avg_top10_percent_balance.append(np.mean(sorted_active_bals[-k10:]))
        else:
            history_avg_top1_percent_balance.append(0.0); history_avg_top10_percent_balance.append(0.0)

    # --- Post-simulation calculations ---
    final_active_users_list = [u for u in users_master_list.values() if u.active_in_sim]
    final_balances = [u.balance for u in final_active_users_list] if final_active_users_list else [0.0]
    final_num_active_users = len(final_active_users_list)
    min_action_cost_val = min(COST_SUBMIT_FRESH_WORD, COST_INTERACT_STALE_WORD)
    
    profit_making_users_count, loss_making_users_count = 0, 0
    profit_making_users_total_gains, loss_making_users_total_losses = 0.0, 0.0
    user_join_steps_for_scatter, user_final_balances_for_scatter, user_net_change_for_color = [], [], []

    for user_obj in users_master_list.values():
        net_change = user_obj.balance - user_obj.initial_balance
        if net_change > 0:
            profit_making_users_count += 1
            profit_making_users_total_gains += net_change
        elif net_change < 0:
            loss_making_users_count += 1
            loss_making_users_total_losses += abs(net_change)
        user_join_steps_for_scatter.append(user_obj.join_step)
        user_final_balances_for_scatter.append(user_obj.balance)
        user_net_change_for_color.append(net_change)

    return {
        "params_config_copy": params.copy(),
        "final_num_active_users": final_num_active_users,
        "final_treasury": platform_treasury,
        "total_rewards_paid": platform_total_rewards_paid_overall,
        "platform_total_injections": platform_total_injections, # --- NEW ---
        "final_avg_balance": np.mean(final_balances) if final_balances else 0.0,
        "final_median_balance": np.median(final_balances) if final_balances else 0.0,
        "final_gini_coefficient": get_gini(final_balances),
        "users_broke_percent": (sum(1 for b in final_balances if b < min_action_cost_val) / final_num_active_users * 100) if final_num_active_users > 0 else 0.0,
        "total_actions_in_sim": total_actions_simulation,
        "history_num_active_users": history_num_active_users,
        "history_platform_attractiveness": history_platform_attractiveness,
        "history_avg_user_balance": history_avg_user_balance,
        "history_avg_top1_percent_balance": history_avg_top1_percent_balance,
        "history_avg_top10_percent_balance": history_avg_top10_percent_balance,
        "profit_making_users_count": profit_making_users_count,
        "profit_making_users_total_gains": profit_making_users_total_gains,
        "loss_making_users_count": loss_making_users_count,
        "loss_making_users_total_losses": loss_making_users_total_losses,
        "total_users_ever_joined": len(users_master_list),
        "user_join_steps_for_scatter": user_join_steps_for_scatter,
        "user_final_balances_for_scatter": user_final_balances_for_scatter,
        "user_net_change_for_color_scatter": user_net_change_for_color,
    }

# --- Main Execution ---
if __name__ == "__main__":
    best_empirical_case_name = 'BaselineEngagement'

    # --- FIX --- Parameters for a more sustainable, generative economy
    fixed_params = {
        # --- FIX #1: Eliminate the primary money sink
        'PLATFORM_RAKE_PERCENTAGE': 0.0,
        # --- FIX #3: Introduce a "money faucet"
        'PLATFORM_REWARD_INJECTION_ON_SUCCESS': 15.0, # Platform adds 15 currency units to each successful reward pool
        
        'REWARD_TO_ORIGINAL_SUBMITTER_SHARE': 0.5, # Balance rewards more between originators and agreers
        'POPULARITY_THRESHOLD_BASE': 6,          # Lower threshold to make rewards more frequent
        'COST_SUBMIT_FRESH_WORD': 2.0,
        'COST_INTERACT_STALE_WORD': 1.0,
        'FRESHNESS_DROP_AFTER_REWARD': 0.1,
        'REWARD_CYCLE_COOLDOWN_STEPS': 5,
        'USER_ACTIVITY_RATE_ON_PLATFORM': 0.75,
        'MIN_FRESHNESS_FOR_FULL_ORIGINATOR_COST_AND_REWARD': 0.7,
        'POTENTIAL_USER_POOL_SIZE': 20000,
        'INITIAL_ACTIVE_USERS': 100,
        'NUM_POLITICIANS': 50,
        'SIMULATION_STEPS': 240,
        'INITIAL_USER_BALANCE_MEAN': 100,
        'INITIAL_USER_BALANCE_STDDEV': 20,
        'POPULARITY_DECAY_RATE': 0.02,
        'ENABLE_CHURN': True,
        'FINITE_DICTIONARY_SIZE': 1500, 'ZIPFIAN_ALPHA': 1.1,
        'INNOVATOR_OUT_OF_DICTIONARY_CHANCE': 0.04,
        'COMMON_WORD_VOCAB_SIZE': 150, 'INNOVATOR_RARE_WORD_CHANCE': 0.1,
        'CHURN_INACTIVITY_THRESHOLD_STEPS': 75,
        'CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER': 2.0,
        'CHURN_BASE_PROB_IF_CONDITIONS_MET': 0.01,
        'POTENTIAL_JOIN_TRIALS_PER_STEP': 50,
        'MAX_NEW_JOINS_PER_STEP_SCALER': 0.02,
        'CHURN_GRACE_PERIOD_STEPS': 20,
        'TARGET_BROKE_AMONG_ACTIVE_PERCENT': 5.0,
        'EMPIRICAL_CASE_SCENARIO': best_empirical_case_name
    }

    empirical_params = get_empirical_params_for_case(best_empirical_case_name)
    final_config = empirical_params.copy()
    final_config.update(fixed_params)

    min_cost_val = min(final_config['COST_SUBMIT_FRESH_WORD'], final_config['COST_INTERACT_STALE_WORD'])
    final_config['CHURN_LOW_BALANCE_THRESHOLD'] = min_cost_val * final_config['CHURN_LOW_BALANCE_THRESHOLD_MULTIPLIER']
    if final_config['CHURN_LOW_BALANCE_THRESHOLD'] <=0:
        final_config['CHURN_LOW_BALANCE_THRESHOLD'] = 0.1

    print(f"--- Running Simulation with Sustainable Economy Parameters ---")
    print(f"  PLATFORM_RAKE_PERCENTAGE: {final_config['PLATFORM_RAKE_PERCENTAGE']} (FIX: Set to 0)")
    print(f"  PLATFORM_REWARD_INJECTION_ON_SUCCESS: {final_config['PLATFORM_REWARD_INJECTION_ON_SUCCESS']} (FIX: New money enters system)")
    print(f"  'Lost Pool' Leak: Plugged by rolling over funds (see code logic).")

    initialize_zipfian_dictionary(final_config.get('FINITE_DICTIONARY_SIZE'), final_config.get('ZIPFIAN_ALPHA'))

    run_start_time = time.time()
    detailed_results = run_full_simulation_with_dynamic_users(final_config.copy(), f"fixed_params_run_1")
    run_duration_minutes = (time.time() - run_start_time) / 60
    print(f"\n--- Simulation Complete --- (Total time: {run_duration_minutes:.2f}m)")

    profit_makers_count = detailed_results.get("profit_making_users_count", 0)
    total_users_ever_joined = detailed_results.get("total_users_ever_joined", 1)
    profit_percentage = (profit_makers_count / total_users_ever_joined) * 100 if total_users_ever_joined > 0 else 0

    print(f"\n--- Key Metrics for Sustainable Configuration ---")
    print(f"  Profit Making Users: {profit_makers_count} / {total_users_ever_joined} ({profit_percentage:.2f}%)")
    
    # --- Analysis of money flow ---
    total_paid_by_users = abs(detailed_results.get("loss_making_users_total_losses", 0)) + detailed_results.get("profit_making_users_total_gains", 0)
    total_injections = detailed_results.get("platform_total_injections", 0)
    print(f"  Total Money Paid by Users for Actions (approx): {total_paid_by_users:,.2f}")
    print(f"  Total New Money Injected by Platform: {total_injections:,.2f}")
    print(f"  Total Rewards Distributed to Users: {detailed_results.get('total_rewards_paid', 0):,.2f}")


    print("\n--- Full Metrics ---")
    key_metrics_to_show = [
        "final_num_active_users", "final_treasury", "total_rewards_paid", "platform_total_injections",
        "final_avg_balance", "final_median_balance", "users_broke_percent",
        "final_gini_coefficient", "profit_making_users_count", "profit_making_users_total_gains",
        "loss_making_users_count", "loss_making_users_total_losses", "total_users_ever_joined"
    ]
    for k_met_s in key_metrics_to_show:
        val = detailed_results.get(k_met_s)
        if isinstance(val, (float, np.float64)): print(f"  {k_met_s}: {val:,.2f}")
        else: print(f"  {k_met_s}: {val}")
    
    # --- Plotting ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "simulation_results_sustainable_economy"
    os.makedirs(results_dir, exist_ok=True)
    results_filename_base = f"{results_dir}/sustainable_economy_run_{timestamp_str}"
    
    # Growth Dynamics Plot
    if 'history_num_active_users' in detailed_results and detailed_results['history_num_active_users']:
        fig, ax_active = plt.subplots(figsize=(14, 8))
        color_active = 'navy'; ax_active.set_xlabel('Simulation Step', fontsize=12)
        ax_active.set_ylabel('Active Users', color=color_active, fontsize=12)
        p1, = ax_active.plot(detailed_results['history_num_active_users'], color=color_active, label='Active Users', linewidth=2.5)
        
        ax_balances = ax_active.twinx(); ax_balances.spines["right"].set_position(("outward", 60));
        color_avg_bal = 'darkgreen'; color_top1 = 'goldenrod';
        p3, = ax_balances.plot(detailed_results['history_avg_user_balance'], color=color_avg_bal, linestyle='-.', alpha=0.9, label='Avg Balance (All Active)')
        p4, = ax_balances.plot(detailed_results['history_avg_top1_percent_balance'], color=color_top1, linestyle='--', linewidth=2, alpha=0.85, label='Avg Balance (Top 1% Active)')
        h_line_initial_bal = ax_balances.axhline(y=final_config['INITIAL_USER_BALANCE_MEAN'], color='red', linestyle=':', linewidth=1.5, label=f'Initial Avg Balance ({final_config["INITIAL_USER_BALANCE_MEAN"]})')

        ax_balances.set_ylabel('User Balance', color='black', fontsize=12)
        ax_balances.tick_params(axis='y', labelcolor='black', labelsize=10)
        max_y_bal = max(np.max(detailed_results['history_avg_user_balance']) * 1.2 if detailed_results['history_avg_user_balance'] else 0, final_config['INITIAL_USER_BALANCE_MEAN']*1.5)
        ax_balances.set_ylim(0, max_y_bal)
        
        fig.suptitle(f"User Growth & Dynamics (Sustainable Economy)\nProfit Makers: {profit_percentage:.1f}% | Injection: {final_config['PLATFORM_REWARD_INJECTION_ON_SUCCESS']}", fontsize=14, fontweight='bold')
        lines = [p1, p3, p4, h_line_initial_bal]
        labels = [l.get_label() for l in lines]
        ax_active.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.01, 0.9), fontsize=10)
        ax_active.grid(True, linestyle=':', alpha=0.6); fig.tight_layout(rect=[0, 0.03, 1, 0.90])
        plt.savefig(f"{results_filename_base}_growth_dynamics.png")
        print(f"\nGrowth dynamics plot saved to: {results_filename_base}_growth_dynamics.png")
        plt.show()

    # Net Balance Change Plot
    net_balance_changes_data = detailed_results.get("net_balance_changes", [])
    if net_balance_changes_data:
        plt.figure(figsize=(10, 6))
        plt.hist(net_balance_changes_data, bins=60, color='lightgreen', edgecolor='black')
        plt.title('Distribution of Net Balance Change (Final - Initial)', fontsize=14)
        plt.xlabel('Net Balance Change', fontsize=12); plt.ylabel('Number of Users', fontsize=12)
        plt.axvline(0, color='black', linestyle='solid', linewidth=1.5)
        plt.axvline(np.mean(net_balance_changes_data), color='blue', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(net_balance_changes_data):.2f}')
        plt.legend(); plt.grid(axis='y', alpha=0.7); plt.tight_layout()
        plt.savefig(f"{results_filename_base}_net_balance_changes.png")
        print(f"Net balance change distribution plot saved to: {results_filename_base}_net_balance_changes.png")
        plt.show()
    
    # Final Balance vs Join Time Plot
    user_join_steps_scatter = detailed_results.get('user_join_steps_for_scatter', [])
    user_final_balances_scatter = detailed_results.get('user_final_balances_for_scatter', [])
    user_net_color_scatter = detailed_results.get('user_net_change_for_color_scatter', [])
    if all([user_join_steps_scatter, user_final_balances_scatter, user_net_color_scatter]):
        plt.figure(figsize=(12, 7))
        colors_scatter = ['limegreen' if nc > 0 else ('red' if nc < 0 else 'blue') for nc in user_net_color_scatter]
        plt.scatter(user_join_steps_scatter, user_final_balances_scatter, c=colors_scatter, alpha=0.6, s=15, edgecolors='w', linewidths=0.2)
        plt.title('Final User Balance vs. User Join Timestep', fontsize=14)
        plt.xlabel('Join Timestep', fontsize=12); plt.ylabel('Final Balance', fontsize=12)
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Profit (Final > Initial)', markerfacecolor='limegreen', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Loss (Final < Initial)', markerfacecolor='red', markersize=8)]
        h_line_initial = plt.axhline(y=final_config['INITIAL_USER_BALANCE_MEAN'], color='gray', linestyle=':', linewidth=1.5, label=f'Avg Initial Balance')
        legend_elements.append(h_line_initial)
        plt.legend(handles=legend_elements, loc='upper right', title="Balance Change Key")
        plt.grid(True, linestyle=':', alpha=0.6); plt.tight_layout()
        plt.savefig(f"{results_filename_base}_balance_vs_join_step_colored.png")
        print(f"Balance vs Join Timestep plot saved to: {results_filename_base}_balance_vs_join_step_colored.png")
        plt.show()

    print("\nAnalysis script finished.")