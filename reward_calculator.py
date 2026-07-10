from __future__ import annotations

from data import (
    agent_has_potion,
    can_apply_status,
    get_agent_front,
    get_agent_potion_heal,
    get_enemy_front,
    get_move_mode,
    get_move_power,
    get_move_statistics,
    get_pokemon_attacks,
    get_pokemon_hp,
    get_pokemon_max_hp,
    get_pokemon_name,
    get_pokemon_speed,
    get_pokemon_status,
    get_stat_stage,
    modifies_stat_stage,
    move_for_action_id,
    move_status_effect,
    move_target,
    would_stage_move_have_effect,
)


status_base_values = {
    "asleep": 0.34,
    "freeze": 0.34,
    "paralyzed": 0.23,
    "burned": 0.22,
    "badlyPoisoned": 0.24,
    "poisoned": 0.18,
    "confused": 0.12,
    "attracted": 0.12,
    "cursed": 0.14,
}

status_projection_turns = 3.0
status_reward_cap = 0.35
setup_reward_cap = 0.25
bad_utility_penalty_cap = 0.15
utility_stage_weight = 0.08
immediate_ko_damage_ratio = 0.9
irrelevant_stat_threshold = 0.5
minimum_setup_turns_alive = 3.0
fast_ko_turn_threshold = 2.0

potion_low_hp_threshold = 0.5
potion_tempo_reward_cap = 0.25
potion_bad_use_penalty = 0.12
potion_no_threat_penalty = 0.03

potion_faster_tempo_reward = 0.11
potion_residual_status_reward = 0.07
potion_slow_survival_reward = 0.12
potion_setup_preservation_reward = 0.07


class RewardCalculator:
    def __init__(
        self,
        matchup_evaluator,
        invalid_action_penalty: float,
        switch_action: int,
        item_action: int,
        enemy_damage_reward_weight: float,
        agent_damage_penalty_weight: float,
        step_reward_offset: float,
        enemy_ko_reward: float,
        agent_ko_penalty: float,
        switch_offense_weight: float,
        switch_threat_weight: float,
        switch_action_cost: float,
        bad_switch_penalty: float,
    ):
        self.matchup_evaluator = matchup_evaluator
        self.invalid_action_penalty_value = invalid_action_penalty
        self.switch_action = switch_action
        self.item_action = item_action
        self.enemy_damage_reward_weight = enemy_damage_reward_weight
        self.agent_damage_penalty_weight = agent_damage_penalty_weight
        self.step_reward_offset = step_reward_offset
        self.enemy_ko_reward = enemy_ko_reward
        self.agent_ko_penalty = agent_ko_penalty
        self.switch_offense_weight = switch_offense_weight
        self.switch_threat_weight = switch_threat_weight
        self.switch_action_cost = switch_action_cost
        self.bad_switch_penalty = bad_switch_penalty

    def invalid_action_penalty(self, msg: dict) -> float:
        feedback = msg.get("action_feedback", {})
        return self.invalid_action_penalty_value if feedback.get("opponent_invalid", False) else 0.0

    def compute_step_reward(
        self,
        previous_enemy_front: dict,
        previous_agent_front: dict,
        current_enemy_front: dict,
        current_agent_front: dict,
        current_msg: dict,
        prev_msg: dict | None = None,
        action: int | None = None,
    ) -> float:
        invalid_action_penalty = self.invalid_action_penalty(current_msg)
        reward = 0.0
        reward += self._hp_exchange_reward(
            previous_enemy_front,
            previous_agent_front,
            current_enemy_front,
            current_agent_front,
        )
        reward += self._ko_reward(current_enemy_front, current_agent_front)
        reward -= invalid_action_penalty

        if prev_msg is not None and action is not None:
            reward += self.compute_switch_reward(prev_msg, current_msg, action)

            if invalid_action_penalty == 0.0:
                if action == self.item_action:
                    reward += self.compute_item_tempo_reward(
                        prev_msg,
                        previous_enemy_front,
                        previous_agent_front,
                    )
                else:
                    reward += self.compute_utility_reward(
                        prev_msg,
                        action,
                        previous_enemy_front,
                        previous_agent_front,
                        current_enemy_front,
                        current_agent_front,
                    )

        return reward

    def _hp_exchange_reward(
        self,
        previous_enemy_front: dict,
        previous_agent_front: dict,
        current_enemy_front: dict,
        current_agent_front: dict,
    ) -> float:
        enemy_max_hp = max(1.0, get_pokemon_max_hp(current_enemy_front))
        agent_max_hp = max(1.0, get_pokemon_max_hp(current_agent_front))

        damage_dealt = (get_pokemon_hp(previous_enemy_front) - get_pokemon_hp(current_enemy_front)) / enemy_max_hp
        raw_damage_taken = get_pokemon_hp(previous_agent_front) - get_pokemon_hp(current_agent_front)
        damage_taken = max(0.0, raw_damage_taken) / agent_max_hp

        return (
            self.enemy_damage_reward_weight * damage_dealt
            - self.agent_damage_penalty_weight * damage_taken
            - self.step_reward_offset
        )

    def _ko_reward(self, current_enemy_front: dict, current_agent_front: dict) -> float:
        reward = 0.0
        if get_pokemon_status(current_enemy_front) == "KO":
            reward += self.enemy_ko_reward
        if get_pokemon_status(current_agent_front) == "KO":
            reward -= self.agent_ko_penalty
        return reward

    def compute_utility_reward(
        self,
        previous_msg: dict,
        action: int,
        previous_enemy_front: dict,
        previous_agent_front: dict,
        current_enemy_front: dict,
        current_agent_front: dict,
    ) -> float:
        if action >= self.switch_action:
            return 0.0

        move = move_for_action_id(previous_msg, action)
        if move is None:
            return 0.0

        reward = 0.0
        reward += self.compute_status_reward(
            move,
            previous_enemy_front,
            current_agent_front,
            current_enemy_front,
        )
        reward += self.compute_setup_reward(
            move,
            previous_agent_front,
            previous_enemy_front,
            current_agent_front,
            current_enemy_front,
        )
        reward -= self.compute_bad_choice_penalty(
            move,
            previous_agent_front,
            previous_enemy_front,
            current_agent_front,
            current_enemy_front,
        )
        return reward

    def compute_status_reward(
        self,
        move: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> float:
        if not can_apply_status(move):
            return 0.0

        expected_status = move_status_effect(move)
        previous_status = get_pokemon_status(previous_enemy_front)
        current_status = get_pokemon_status(current_enemy_front)

        if expected_status is None or expected_status != current_status:
            return 0.0
        if previous_status != "normal":
            return 0.0
        if current_status in ("normal", "KO"):
            return 0.0

        base_status_value = status_base_values.get(current_status, 0.12)
        impact_multiplier = self.compute_status_impact_multiplier(
            current_status,
            current_agent_front,
            current_enemy_front,
        )
        projected_effect_value = self.compute_projected_residual_status_value(
            current_status,
            current_agent_front,
            current_enemy_front,
        )

        reward = base_status_value * impact_multiplier + projected_effect_value
        return min(status_reward_cap, max(0.0, reward))

    def compute_setup_reward(
        self,
        move: dict,
        previous_agent_front: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> float:
        if not modifies_stat_stage(move):
            return 0.0

        relevance = self.compute_setup_move_relevance(
            move,
            previous_agent_front,
            previous_enemy_front,
        )
        if relevance <= 0.0:
            return 0.0

        beneficial_stage_change = self.get_beneficial_setup_stage_change(
            move,
            previous_agent_front,
            previous_enemy_front,
            current_agent_front,
            current_enemy_front,
        )
        if beneficial_stage_change <= 0.0:
            return 0.0

        reward = beneficial_stage_change * utility_stage_weight * relevance
        return min(setup_reward_cap, max(0.0, reward))

    def get_beneficial_setup_stage_change(
        self,
        move: dict,
        previous_agent_front: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> float:
        target = move_target(move)
        beneficial_stage_change = 0.0

        for statistic, requested_delta in get_move_statistics(move):
            actual_stage_change = self.get_actual_stage_change_for_statistic(
                statistic,
                target,
                previous_agent_front,
                previous_enemy_front,
                current_agent_front,
                current_enemy_front,
            )

            if target == "self" and requested_delta > 0 and actual_stage_change > 0:
                beneficial_stage_change += float(actual_stage_change)
            elif target == "opponent" and requested_delta < 0 and actual_stage_change < 0:
                beneficial_stage_change += float(abs(actual_stage_change))

        return beneficial_stage_change

    def get_actual_stage_change(
        self,
        move: dict,
        previous_agent_front: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> int:
        changes = get_move_statistics(move)
        if not changes:
            return 0

        statistic = changes[0][0]
        target = move_target(move)
        return self.get_actual_stage_change_for_statistic(
            statistic,
            target,
            previous_agent_front,
            previous_enemy_front,
            current_agent_front,
            current_enemy_front,
        )

    def get_actual_stage_change_for_statistic(
        self,
        statistic: str,
        target: str,
        previous_agent_front: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> int:
        if target == "self":
            previous_stage = get_stat_stage(previous_agent_front, statistic)
            current_stage = get_stat_stage(current_agent_front, statistic)
            return current_stage - previous_stage

        if target == "opponent":
            previous_stage = get_stat_stage(previous_enemy_front, statistic)
            current_stage = get_stat_stage(current_enemy_front, statistic)
            return current_stage - previous_stage

        return 0

    def estimated_turn_alive(self, enemy_front: dict, agent_front: dict) -> float:
        incoming_damage = self.matchup_evaluator.incoming_threat_score(enemy_front, agent_front)
        if incoming_damage <= 0.0:
            return float("inf")
        agent_hp = max(0.0, get_pokemon_hp(agent_front))
        return agent_hp / incoming_damage

    def estimated_turns_to_ko(self, agent_front: dict, enemy_front: dict) -> float:
        best_damage = self.matchup_evaluator.best_attack_score(agent_front, enemy_front)
        if best_damage <= 0.0:
            return float("inf")
        enemy_hp = max(0.0, get_pokemon_hp(enemy_front))
        return enemy_hp / best_damage

    def estimate_projected_status_turns(
        self,
        agent_front: dict,
        enemy_front: dict,
    ) -> float:
        enemy_hp = max(0.0, get_pokemon_hp(enemy_front))
        estimated_damage_per_turn = self.matchup_evaluator.best_attack_score(agent_front, enemy_front)

        if estimated_damage_per_turn <= 0.0:
            projected_turns = status_projection_turns
        else:
            turns_to_ko = enemy_hp / estimated_damage_per_turn
            projected_turns = min(status_projection_turns, turns_to_ko)

        return max(0.0, projected_turns)

    def compute_status_impact_multiplier(
        self,
        status_name: str,
        agent_front: dict,
        enemy_front: dict,
    ) -> float:
        if status_name == "paralyzed":
            agent_speed = get_pokemon_speed(agent_front)
            enemy_speed = get_pokemon_speed(enemy_front)
            enemy_speed_after_paralysis = enemy_speed / 2.0

            if enemy_speed >= agent_speed and enemy_speed_after_paralysis < agent_speed:
                return 1.5
            if enemy_speed_after_paralysis < agent_speed:
                return 1.15
            return 0.75

        if status_name == "burned":
            return 1.2 if self.pokemon_has_damaging_moves(enemy_front, "physical") else 0.85

        if status_name in ("asleep", "freeze"):
            return 1.15

        if status_name in ("confused", "attracted", "cursed"):
            return 0.9

        return 1.0

    def compute_projected_residual_status_value(
        self,
        status_name: str,
        agent_front: dict,
        enemy_front: dict,
    ) -> float:
        projected_turns = self.estimate_projected_status_turns(agent_front, enemy_front)

        if status_name == "burned":
            return min(0.30, projected_turns * (1.0 / 8.0))
        if status_name == "poisoned":
            return min(0.30, projected_turns * (1.0 / 8.0))
        if status_name == "badlyPoisoned":
            return min(0.35, projected_turns * (1.5 / 16.0))
        return 0.0

    def compute_bad_choice_penalty(
        self,
        move: dict,
        previous_agent_front: dict,
        previous_enemy_front: dict,
        current_agent_front: dict,
        current_enemy_front: dict,
    ) -> float:
        penalty = 0.0
        likely_immediate_ko = self.enemy_could_likely_be_koed(previous_agent_front, previous_enemy_front)

        if self.is_pure_status_move(move):
            previous_enemy_status = get_pokemon_status(previous_enemy_front)
            current_enemy_status = get_pokemon_status(current_enemy_front)
            expected_status = move_status_effect(move)

            if previous_enemy_status not in ("normal", "KO"):
                penalty += 0.08

            if expected_status is not None and current_enemy_status != expected_status:
                penalty += 0.06

            if likely_immediate_ko:
                penalty += 0.07

        if modifies_stat_stage(move):
            relevance = self.compute_setup_move_relevance(
                move,
                previous_agent_front,
                previous_enemy_front,
            )
            beneficial_stage_change = self.get_beneficial_setup_stage_change(
                move,
                previous_agent_front,
                previous_enemy_front,
                current_agent_front,
                current_enemy_front,
            )

            if not would_stage_move_have_effect(move, previous_agent_front, previous_enemy_front):
                penalty += 0.08

            if beneficial_stage_change == 0:
                penalty += 0.05

            if relevance <= 0.0:
                penalty += 0.08
            elif relevance < irrelevant_stat_threshold:
                penalty += 0.06

        return min(bad_utility_penalty_cap, penalty)

    def compute_setup_move_relevance(
        self,
        move: dict,
        agent_front: dict,
        enemy_front: dict,
    ) -> float:
        if not modifies_stat_stage(move):
            return 0.0

        turns_alive = self.estimated_turn_alive(enemy_front, agent_front)
        if turns_alive < minimum_setup_turns_alive:
            return 0.0

        turns_to_ko = self.estimated_turns_to_ko(agent_front, enemy_front)
        if turns_to_ko <= fast_ko_turn_threshold:
            return 0.0

        if not would_stage_move_have_effect(move, agent_front, enemy_front):
            return 0.0

        target = move_target(move)
        relevances = []

        for statistic, _ in get_move_statistics(move):
            relevances.append(
                self.compute_single_setup_stat_relevance(
                    statistic,
                    target,
                    agent_front,
                    enemy_front,
                )
            )

        if not relevances:
            return 0.0

        return sum(relevances) / len(relevances)

    def compute_single_setup_stat_relevance(
        self,
        statistic: str,
        target: str,
        agent_front: dict,
        enemy_front: dict,
    ) -> float:
        if statistic == "speed":
            return 0.85

        if target == "self":
            if statistic == "atk":
                return 1.0 if self.pokemon_has_damaging_moves(agent_front, "physical") else 0.35
            if statistic == "atkSpe":
                return 1.0 if self.pokemon_has_damaging_moves(agent_front, "special") else 0.35
            if statistic == "def":
                return 1.0 if self.pokemon_has_damaging_moves(enemy_front, "physical") else 0.35
            if statistic == "defSpe":
                return 1.0 if self.pokemon_has_damaging_moves(enemy_front, "special") else 0.35

        if target == "opponent":
            if statistic == "atk":
                return 1.0 if self.pokemon_has_damaging_moves(enemy_front, "physical") else 0.35
            if statistic == "atkSpe":
                return 1.0 if self.pokemon_has_damaging_moves(enemy_front, "special") else 0.35
            if statistic == "def":
                return 1.0 if self.pokemon_has_damaging_moves(agent_front, "physical") else 0.35
            if statistic == "defSpe":
                return 1.0 if self.pokemon_has_damaging_moves(agent_front, "special") else 0.35

        return 0.5

    def pokemon_has_damaging_moves(self, pokemon: dict, mode: str | None = None) -> bool:
        expected_mode = None if mode is None else str(mode).lower()

        for move in get_pokemon_attacks(pokemon):
            move_mode = str(get_move_mode(move)).lower()
            if get_move_power(move) <= 0:
                continue
            if expected_mode is not None and move_mode != expected_mode:
                continue
            return True

        return False

    def enemy_could_likely_be_koed(self, agent_front: dict, enemy_front: dict) -> bool:
        enemy_hp = max(1.0, get_pokemon_hp(enemy_front))
        best_damage = self.matchup_evaluator.best_attack_score(agent_front, enemy_front)
        return best_damage >= enemy_hp * immediate_ko_damage_ratio

    def is_pure_status_move(self, move: dict) -> bool:
        return (
            str(get_move_mode(move)).lower() == "status"
            and get_move_power(move) <= 0.0
            and can_apply_status(move)
            and not modifies_stat_stage(move)
        )

    def stat_stage_multiplier(self, stage: int) -> float:
        stage = max(-6, min(6, int(stage)))

        if stage >= 0:
            return (2.0 + stage) / 2.0

        return 2.0 / (2.0 - stage)

    def effective_speed(self, pokemon: dict) -> float:
        return get_pokemon_speed(pokemon) * self.stat_stage_multiplier(
            get_stat_stage(pokemon, "speed")
        )

    def has_positive_setup_stage(self, pokemon: dict) -> bool:
        return any(
            get_stat_stage(pokemon, statistic) > 0
            for statistic in ("atk", "def", "atkSpe", "defSpe", "speed")
        )

    def has_residual_damage_status(self, pokemon: dict) -> bool:
        return get_pokemon_status(pokemon) in ("burned", "poisoned", "badlyPoisoned")

    def compute_item_tempo_reward(
        self,
        previous_msg: dict,
        previous_enemy_front: dict,
        previous_agent_front: dict,
    ) -> float:
        if not agent_has_potion(previous_msg):
            return -potion_bad_use_penalty

        hp = get_pokemon_hp(previous_agent_front)
        max_hp = get_pokemon_max_hp(previous_agent_front)

        if max_hp <= 0.0 or hp <= 0.0:
            return 0.0

        hp_ratio = hp / max_hp

        if hp_ratio >= potion_low_hp_threshold:
            return -potion_bad_use_penalty

        potion_heal = get_agent_potion_heal(previous_msg)

        if potion_heal <= 0.0:
            return -potion_bad_use_penalty

        hp_after_potion = min(max_hp, hp + potion_heal)

        incoming_threat = self.matchup_evaluator.incoming_threat_score(
            previous_enemy_front,
            previous_agent_front,
        )

        if incoming_threat <= 0.0:
            return -potion_no_threat_penalty

        survives_one_hit = hp_after_potion > incoming_threat
        survives_two_hits = hp_after_potion > incoming_threat * 2.0

        agent_is_faster = (
            self.effective_speed(previous_agent_front)
            > self.effective_speed(previous_enemy_front)
        )

        enemy_has_residual_status = self.has_residual_damage_status(previous_enemy_front)
        agent_has_setup = self.has_positive_setup_stage(previous_agent_front)

        reward = 0.0

        if agent_is_faster and survives_one_hit:
            reward += potion_faster_tempo_reward

        if enemy_has_residual_status and survives_one_hit:
            reward += potion_residual_status_reward

        if not agent_is_faster and survives_two_hits:
            reward += potion_slow_survival_reward

        if agent_has_setup and survives_one_hit:
            reward += potion_setup_preservation_reward

        if reward <= 0.0:
            return -potion_bad_use_penalty

        return min(potion_tempo_reward_cap, reward)

    def compute_switch_reward(self, previous_msg: dict, current_msg: dict, action: int) -> float:
        if action != self.switch_action:
            return 0.0

        previous_enemy_front = get_enemy_front(previous_msg)
        previous_agent_front = get_agent_front(previous_msg)
        current_agent_front = get_agent_front(current_msg)

        if get_pokemon_name(previous_agent_front) == get_pokemon_name(current_agent_front):
            return 0.0

        old_offense = self.matchup_evaluator.best_attack_score(previous_agent_front, previous_enemy_front)
        new_offense = self.matchup_evaluator.best_attack_score(current_agent_front, previous_enemy_front)

        old_threat = self.matchup_evaluator.incoming_threat_score(previous_enemy_front, previous_agent_front)
        new_threat = self.matchup_evaluator.incoming_threat_score(previous_enemy_front, current_agent_front)

        reward = 0.0
        reward += self.switch_offense_weight * (new_offense - old_offense)
        reward += self.switch_threat_weight * (old_threat - new_threat)
        reward -= self.switch_action_cost

        if new_offense <= old_offense and new_threat >= old_threat:
            reward -= self.bad_switch_penalty
        return reward
