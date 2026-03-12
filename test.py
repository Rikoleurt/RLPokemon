from gymnasium.utils.env_checker import check_env

from env import PokemonEnv

env = PokemonEnv(host="localhost", port=5001)
# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")