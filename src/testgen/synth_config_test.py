from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer, Evolution
from deepeval.synthesizer.config import (
    FiltrationConfig,
    EvolutionConfig,
    StylingConfig,
)

load_dotenv()

# Quality filter: drop weak synthetic inputs
filtration_config = FiltrationConfig(
    synthetic_input_quality_threshold=0.6,   # 0 to 1, higher = stricter
    max_quality_retries=3,                   # retries before giving up on an input
)

# Evolution: make questions harder
evolution_config = EvolutionConfig(
    num_evolutions=2,                        # how many times each input is rewritten
    evolutions={
        Evolution.REASONING: 0.4,
        Evolution.COMPARATIVE: 0.3,
        Evolution.CONCRETIZING: 0.3,         # weights should sum to 1
    },
)

# Styling: steer the persona, task, and format
styling_config = StylingConfig(
    scenario="Customers chatting with a shoe store's support assistant.",
    task="Answer questions about returns, shipping, sizing, and orders.",
    input_format="Short, natural questions typed into a chat box.",
    expected_output_format="A concise answer in 1 to 3 sentences.",
)

synthesizer = Synthesizer(
    filtration_config=filtration_config,
    evolution_config=evolution_config,
    styling_config=styling_config,
)

goldens = synthesizer.generate_goldens_from_scratch(num_goldens=5)

for g in goldens:
    print("INPUT:   ", g.input)
    print("EXPECTED:", g.expected_output)
    print("-" * 40)