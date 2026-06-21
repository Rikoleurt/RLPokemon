from __future__ import annotations

from data import get_agent_front, get_enemy_front, get_pokemon_name


class RewardCalculator:
    def __init__(
        self,
        matchup_evaluator,
        invalid_action_penalty: float,
        switch_action: int,
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
        reward = 0.0
        reward += self._hp_exchange_reward(
            previous_enemy_front,
            previous_agent_front,
            current_enemy_front,
            current_agent_front,
        )
        reward += self._ko_reward(current_enemy_front, current_agent_front)
        reward -= self.invalid_action_penalty(current_msg)

        if prev_msg is not None and action is not None:
            reward += self.compute_switch_reward(prev_msg, current_msg, action)

        return reward

    def _hp_exchange_reward(
        self,
        previous_enemy_front: dict,
        previous_agent_front: dict,
        current_enemy_front: dict,
        current_agent_front: dict,
    ) -> float:
        enemy_max_hp = max(1, current_enemy_front["maxHP"])
        agent_max_hp = max(1, current_agent_front["maxHP"])

        damage_dealt = (previous_enemy_front["HP"] - current_enemy_front["HP"]) / enemy_max_hp
        damage_taken = (previous_agent_front["HP"] - current_agent_front["HP"]) / agent_max_hp

        return (
            self.enemy_damage_reward_weight * damage_dealt
            - self.agent_damage_penalty_weight * damage_taken
            - self.step_reward_offset
        )

    def _ko_reward(self, current_enemy_front: dict, current_agent_front: dict) -> float:
        reward = 0.0
        if current_enemy_front["status"] == "KO":
            reward += self.enemy_ko_reward
        if current_agent_front["status"] == "KO":
            reward -= self.agent_ko_penalty
        return reward

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
