import os
import re
from collections import Counter
from math import ceil
from pathlib import Path

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from sb3_contrib import MaskablePPO
from env import PokemonEnv

MODEL_PATH = "/Users/condreajason/Repositories/RLPokemon/models/switch_new_ppo"
TOTAL_TIMESTEPS = 100_000
TB_LOG_NAME = "pokemon_switch_run_1"
PLOT_DIR = "/Users/condreajason/Repositories/RLPokemon/plots"

#region helpers
def next_plot_path(plot_dir: str, base_name: str, ext: str = ".png") -> str:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    first_candidate = plot_dir / f"{base_name}{ext}"
    if not first_candidate.exists():
        return str(first_candidate)

    pattern = re.compile(rf"^{re.escape(base_name)}(\d+){re.escape(ext)}$")
    max_id = 0

    for path in plot_dir.glob(f"{base_name}*{ext}"):
        match = pattern.match(path.name)
        if match:
            max_id = max(max_id, int(match.group(1)))
        elif path.name == f"{base_name}{ext}":
            max_id = max(max_id, 0)

    return str(plot_dir / f"{base_name}{max_id + 1}{ext}")


def cumulative_mean(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return np.cumsum(values) / np.arange(1, values.size + 1)


def moving_pokemon_move_usage(env: PokemonEnv) -> dict[str, dict[str, list[float]]]:
    histories = getattr(env, "episode_pokemon_move_name_counts_history", [])
    if not histories:
        return {}

    pokemon_names = list(getattr(env, "pokemon_move_name_counts", {}).keys())
    for episode_counts in histories:
        for pokemon_name in episode_counts:
            if pokemon_name not in pokemon_names:
                pokemon_names.append(pokemon_name)

    usage_by_pokemon = {}
    for pokemon_name in pokemon_names:
        move_names = sorted(
            {
                move_name
                for episode_counts in histories
                for move_name in episode_counts.get(pokemon_name, {}).keys()
            }
        )
        if not move_names:
            continue

        move_usage = {move_name: [] for move_name in move_names}
        for episode_idx in range(len(histories)):
            start_idx = max(0, episode_idx + 1 - env.window_size)
            window_counter = Counter()
            for episode_counts in histories[start_idx:episode_idx + 1]:
                window_counter.update(episode_counts.get(pokemon_name, {}))

            total = sum(window_counter.values())
            for move_name in move_names:
                if total == 0:
                    move_usage[move_name].append(np.nan)
                else:
                    move_usage[move_name].append(100.0 * window_counter.get(move_name, 0) / total)

        usage_by_pokemon[pokemon_name] = move_usage

    return usage_by_pokemon


def annotate_bar(ax, bar, percentage: float, count: float) -> None:
    if count <= 0:
        return

    label = f"{percentage:.1f}%\n{int(count)}"
    x = bar.get_x() + bar.get_width() / 2

    if percentage >= 12:
        ax.text(x, percentage - 3, label, ha="center", va="top", color="white", fontsize=8)
    else:
        ax.text(x, percentage + 2, label, ha="center", va="bottom", color="#222222", fontsize=8)
#endregion

#region Plotting functions
def plot_global_performance(env: PokemonEnv, plot_dir: str) -> None:
    episodes = np.arange(1, len(env.winrate_history) + 1)
    if episodes.size == 0:
        return

    path = next_plot_path(plot_dir, "performance_globale")
    pokemon_usage = moving_pokemon_move_usage(env)
    usage_rows = max(1, len(pokemon_usage))
    total_rows = 2 + usage_rows

    fig, axes = plt.subplots(
        total_rows,
        1,
        figsize=(12, max(10, 3.4 * total_rows)),
        squeeze=False,
        sharex=True,
    )
    axes = axes.ravel()

    axes[0].plot(episodes, env.winrate_history, label="Winrate cumulé", linewidth=1.8)
    axes[0].plot(
        episodes,
        env.winrate_moving_history,
        label=f"Winrate glissant ({env.window_size})",
        linewidth=1.8,
    )
    axes[0].set_title("Taux de victoire")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 105)
    axes[0].set_yticks(np.arange(0, 101, 20))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    fight_lengths = np.asarray(env.fight_length_history, dtype=float)
    if fight_lengths.size:
        fight_episodes = episodes[:fight_lengths.size]
        axes[1].scatter(
            fight_episodes,
            fight_lengths,
            s=10,
            alpha=0.25,
            label="Longueur observée",
        )
        axes[1].plot(
            fight_episodes,
            env.fight_length_moving_history[:fight_lengths.size],
            label=f"Moyenne glissante ({env.window_size})",
            linewidth=2.0,
            color="#f28e2b",
        )
        axes[1].plot(
            fight_episodes,
            cumulative_mean(fight_lengths),
            label="Moyenne cumulée",
            linewidth=1.8,
            color="#59a14f",
        )
        y_min = float(np.min(fight_lengths))
        y_max = float(np.max(fight_lengths))
        padding = max(1.0, (y_max - y_min) * 0.08)
        axes[1].set_ylim(max(0.0, y_min - padding), y_max + padding)
        axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        axes[1].text(0.5, 0.5, "Aucune donnée", ha="center", va="center")

    axes[1].set_title("Longueur des combats")
    axes[1].set_ylabel("Tours")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    usage_axes = axes[2:]
    if pokemon_usage:
        for ax, (pokemon_name, move_usage) in zip(usage_axes, pokemon_usage.items()):
            for move_name, usage in move_usage.items():
                ax.plot(
                    episodes[:len(usage)],
                    usage,
                    label=move_name,
                    linewidth=1.8,
                )
            ax.set_title(f"Usage glissant des attaques - {pokemon_name}")
            ax.set_ylabel("% des actions")
            ax.set_ylim(0, 105)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=min(2, len(move_usage)), fontsize="small")
    else:
        usage_axes[0].set_title("Usage glissant des attaques par Pokémon")
        usage_axes[0].text(0.5, 0.5, "Aucune donnée par Pokémon", ha="center", va="center")
        usage_axes[0].set_ylabel("%")
        usage_axes[0].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episodes")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved: {path}")


def plot_pokemon_behavior(env: PokemonEnv, plot_dir: str) -> None:
    if not env.pokemon_move_name_counts:
        return

    path = next_plot_path(plot_dir, "comportement_par_pokemon")
    pokemon_names = list(env.pokemon_move_name_counts.keys())
    all_move_names = sorted({
        move_name
        for counter in env.pokemon_move_name_counts.values()
        for move_name in counter.keys()
    })
    color_map = plt.get_cmap("tab10")
    move_colors = {
        move_name: color_map(i % color_map.N)
        for i, move_name in enumerate(all_move_names)
    }

    cols = min(2, len(pokemon_names))
    rows = ceil(len(pokemon_names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows), squeeze=False)
    axes = axes.ravel()

    for ax, pokemon_name in zip(axes, pokemon_names):
        counter = env.pokemon_move_name_counts[pokemon_name]
        move_names = sorted(counter.keys())
        counts = np.asarray([counter[move_name] for move_name in move_names], dtype=float)
        total = float(np.sum(counts))
        percentages = 100.0 * counts / max(1.0, total)
        x = np.arange(len(move_names))

        bars = ax.bar(
            x,
            percentages,
            color=[move_colors[move_name] for move_name in move_names],
        )

        for bar, percentage, count in zip(bars, percentages, counts):
            annotate_bar(ax, bar, percentage, count)

        ax.set_title(f"{pokemon_name} ({int(total)} actions)")
        ax.set_ylabel("% d'utilisation")
        ax.set_ylim(0, 105)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xticks(x)
        ax.set_xticklabels(move_names, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[len(pokemon_names):]:
        ax.axis("off")

    fig.suptitle("Fréquence d’utilisation des attaques par Pokémon", y=0.995)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved: {path}")


def plot_tactical_understanding(env: PokemonEnv, plot_dir: str) -> None:
    path = next_plot_path(plot_dir, "comprehension_tactique")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # =========================
    # 1) Efficiency of the attacks
    # =========================
    labels = ["super efficace", "neutre", "peu efficace"]
    keys = ["super", "neutral", "not_very"]
    values = [env.effectiveness_counts[k] for k in keys]
    effectiveness_colors = ["#4e9f3d", "#8f8f8f", "#d95f02"]

    bars = axes[0].bar(labels, values, color=effectiveness_colors)
    axes[0].set_title("Fréquence super efficace / neutre / pas très efficace")
    axes[0].set_ylabel("Nombre d'actions")
    axes[0].grid(True, axis="y", alpha=0.3)

    total_effectiveness = sum(values)
    for bar, value in zip(bars, values):
        if value <= 0:
            continue
        pct = 100.0 * value / max(1, total_effectiveness)
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, total_effectiveness * 0.01),
            f"{int(value)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =========================
    # 2) Usage of switch / item / invalid actions
    # =========================
    total_attack_actions = sum(env.effectiveness_counts.values())
    switch_count = getattr(env, "switch_count", 0)
    item_count = getattr(env, "item_count", 0)
    invalid_count = getattr(env, "invalid_action_count", 0)

    usage_labels = ["attaques", "switch", "items", "actions invalides"]
    usage_values = [total_attack_actions, switch_count, item_count, invalid_count]
    usage_colors = ["#4c78a8", "#72b7b2", "#f28e2b", "#e15759"]

    bars = axes[1].bar(usage_labels, usage_values, color=usage_colors)
    axes[1].set_title("Usage des mécaniques de jeu")
    axes[1].set_ylabel("Nombre d'actions")
    axes[1].grid(True, axis="y", alpha=0.3)

    total_usage = sum(usage_values)
    for bar, value in zip(bars, usage_values):
        if value <= 0:
            continue
        pct = 100.0 * value / max(1, total_usage)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + max(0.5, total_usage * 0.01),
            f"{int(value)} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # =========================
    # 3) Heatmap of attacks per matchup
    # =========================
    if env.matchup_move_name_counts:
        matchup_names = list(env.matchup_move_name_counts.keys())
        all_move_names = sorted({
            move_name
            for counter in env.matchup_move_name_counts.values()
            for move_name in counter.keys()
        })

        matrix = np.array(
            [
                [env.matchup_move_name_counts[matchup].get(move_name, 0) for move_name in all_move_names]
                for matchup in matchup_names
            ],
            dtype=float,
        )

        usage_cmap = mcolors.LinearSegmentedColormap.from_list(
            "usage_blue_orange_red",
            ["#2166ac", "#fdae61", "#b2182b"],
        )
        norm = mcolors.Normalize(vmin=0.0, vmax=max(1.0, float(np.max(matrix))))

        im = axes[2].imshow(matrix, aspect="auto", cmap=usage_cmap, norm=norm)
        axes[2].set_title("Utilisation des attaques par matchup")
        axes[2].set_xlabel("Attaque")
        axes[2].set_ylabel("Matchup")
        axes[2].set_xticks(np.arange(len(all_move_names)))
        axes[2].set_xticklabels(all_move_names, rotation=45, ha="right")
        axes[2].set_yticks(np.arange(len(matchup_names)))
        axes[2].set_yticklabels(matchup_names)

        if matrix.size <= 80:
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    value = matrix[row_idx, col_idx]
                    normalized = norm(value)
                    text_color = "white" if normalized < 0.25 or normalized > 0.72 else "#222222"
                    axes[2].text(
                        col_idx,
                        row_idx,
                        f"{int(value)}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

        colorbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        colorbar.set_label("Nombre d'utilisations")
    else:
        axes[2].set_title("Utilisation des attaques par matchup")
        axes[2].text(0.5, 0.5, "Aucune donnée", ha="center", va="center")
        axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved: {path}")

def plot_terminal_state_summary(env: PokemonEnv, plot_dir: str) -> None:
    history = getattr(env, "terminal_state_history", [])
    if not history:
        return

    path = next_plot_path(plot_dir, "etat_fin_combat")

    rows_data = []

    for side_label, infos_key, team_key in [
        ("player", "player_infos", "player_team"),
        ("opponent", "opponent_infos", "opponent_team"),
    ]:
        grouped = {}

        for msg in history:
            team = msg.get(infos_key, {}).get(team_key, [])
            for idx, pokemon in enumerate(team):
                if pokemon is None:
                    continue

                name = pokemon.get("name", f"slot_{idx+1}")
                key = (side_label, idx + 1, name)

                if key not in grouped:
                    grouped[key] = {
                        "episodes_seen": 0,
                        "alive_count": 0,
                        "ko_count": 0,
                        "hp_ratio_sum": 0.0,
                        "hp_ratio_alive_sum": 0.0,
                        "alive_hp_samples": 0,
                        "front_end_count": 0,
                        "status_counter": Counter(),
                    }

                g = grouped[key]
                g["episodes_seen"] += 1

                status = pokemon.get("status", "normal")
                hp_ratio = float(pokemon.get("hp_ratio", 0.0))

                g["status_counter"][status] += 1
                g["hp_ratio_sum"] += hp_ratio

                if idx == 0:
                    g["front_end_count"] += 1

                if status == "KO":
                    g["ko_count"] += 1
                else:
                    g["alive_count"] += 1
                    g["hp_ratio_alive_sum"] += hp_ratio
                    g["alive_hp_samples"] += 1

        for (side, slot, name), g in grouped.items():
            episodes_seen = max(1, g["episodes_seen"])
            alive_pct = 100.0 * g["alive_count"] / episodes_seen
            ko_pct = 100.0 * g["ko_count"] / episodes_seen
            avg_hp_pct = 100.0 * g["hp_ratio_sum"] / episodes_seen
            avg_alive_hp_pct = (
                100.0 * g["hp_ratio_alive_sum"] / g["alive_hp_samples"]
                if g["alive_hp_samples"] > 0 else 0.0
            )
            front_end_pct = 100.0 * g["front_end_count"] / episodes_seen
            most_common_status = g["status_counter"].most_common(1)[0][0]

            rows_data.append([
                side,
                str(slot),
                name,
                f"{alive_pct:.1f}%",
                f"{ko_pct:.1f}%",
                f"{avg_hp_pct:.1f}%",
                f"{avg_alive_hp_pct:.1f}%",
                most_common_status,
                f"{front_end_pct:.1f}%",
            ])

    rows_data.sort(key=lambda r: (r[0], int(r[1]), r[2]))

    col_labels = [
        "Camp",
        "Slot",
        "Pokémon",
        "% vivant fin",
        "% KO fin",
        "HP moyen fin",
        "HP moyen si vivant",
        "Statut final dominant",
        "% en front fin",
    ]

    fig_height = max(3.0, 0.45 * (len(rows_data) + 2))
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title("Résumé des états en fin de combat", pad=14)

    table = ax.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9eaf7")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Graph saved: {path}")
#endregion

def plot(env: PokemonEnv, plot_dir: str = PLOT_DIR) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    plot_global_performance(env, plot_dir)
    plot_pokemon_behavior(env, plot_dir)
    plot_tactical_understanding(env, plot_dir)
    plot_terminal_state_summary(env, plot_dir)


def main() -> None:
    gym.register(
        id="gymnasium_env/Pokemon-v0",
        entry_point="env:PokemonEnv",
        max_episode_steps=300,
    )

    env = PokemonEnv()
    try:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=TB_LOG_NAME)
        model.save(MODEL_PATH)
        plot(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()
