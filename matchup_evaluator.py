def get_stab_multiplier(move: dict) -> float:
    return 1.5 if bool(move.get("isSTAB", False)) else 1.0

def offensive_stat(attacker: dict, move: dict) -> float:
    stats = attacker.get("stats", {})
    mode = str(move.get("Mode", "status")).lower()
    if mode == "physical":
        return float(stats.get("atk", 1.0))
    if mode == "special":
        return float(stats.get("atkSpe", 1.0))
    return 1.0


def defensive_stat(defender: dict, move: dict) -> float:
    stats = defender.get("stats", {})
    mode = str(move.get("Mode", "status")).lower()
    if mode == "physical":
        return float(stats.get("def", 1.0))
    if mode == "special":
        return float(stats.get("defSpe", 1.0))
    return 1.0

class MatchupEvaluator:
    def __init__(self, type_chart: dict[str, dict[str, float]]):
        self.type_chart = type_chart

    def effectiveness_multiplier(
        self,
        move_type: str,
        defender_type1: str,
        defender_type2: str | None = None,
    ) -> float:
        move_type = str(move_type).lower()
        defender_type1 = str(defender_type1).lower()

        multiplier = self.type_chart.get(move_type, {}).get(defender_type1, 1.0)
        if defender_type2 is not None:
            defender_type2 = str(defender_type2).lower()
            multiplier *= self.type_chart.get(move_type, {}).get(defender_type2, 1.0)

        return multiplier

    def effectiveness_to_string(
        self,
        move_type: str,
        defender_type1: str,
        defender_type2: str | None = None,
    ) -> str:
        multiplier = self.effectiveness_multiplier(move_type, defender_type1, defender_type2)
        if multiplier > 1.0:
            return "super"
        if multiplier < 1.0:
            return "not_very"
        return "neutral"

    def estimated_move_score(self, move: dict, attacker: dict, defender: dict) -> float:
        mode = str(move.get("Mode", "status")).lower()
        if mode == "status":
            return 0.0

        power = float(move.get("Power", 0.0))
        if power <= 0:
            return 0.0

        level = float(attacker.get("level", 50))
        attack_value = max(1.0, offensive_stat(attacker, move))
        defense_value = max(1.0, defensive_stat(defender, move))

        multiplier = self.effectiveness_multiplier(
            move.get("type", "normal"),
            defender.get("type", "normal"),
            defender.get("type2"),
        )
        if multiplier == 0.0:
            return 0.0

        effective_power = power * get_stab_multiplier(move)
        raw_damage = (
            (((level * 0.4 + 2.0) * attack_value * effective_power) / defense_value) / 50.0
        ) + 2.0
        return raw_damage * multiplier

    def best_attack_score(self, attacker: dict, defender: dict) -> float:
        best_score = 0.0
        for move in attacker.get("attacks", []):
            if float(move.get("PP", 0.0)) <= 0:
                continue
            best_score = max(best_score, self.estimated_move_score(move, attacker, defender))
        return best_score

    def incoming_threat_score(self, enemy: dict, agent_pokemon: dict) -> float:
        return self.best_attack_score(enemy, agent_pokemon)
