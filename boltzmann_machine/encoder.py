import numpy as np
import numpy.random
from typing import Self

SEED = 42
STEP_SIZE = 2
ANNEALING_SEQUENCE = [20, 20, 15, 15, 12, 12, 10, 10, 10, 10]
COLLECTION_STEPS = 10
ENSEMBLE_RUNS = 2


class EncoderState:
    def __init__(self, visible_nodes, hidden_nodes):
        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes
        self.v1_state = np.zeros((visible_nodes), dtype=np.int64)
        self.v2_state = np.zeros((visible_nodes), dtype=np.int64)
        self.hidden_state = np.zeros((hidden_nodes), dtype=np.int64)


class ProbStats:
    def __init__(self, visible_nodes, hidden_nodes):
        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes
        self.v1_internal_prob = np.zeros((visible_nodes, visible_nodes))
        self.v2_internal_prob = np.zeros((visible_nodes, visible_nodes))
        self.v1_hidden_prob = np.zeros((visible_nodes, hidden_nodes))
        self.v2_hidden_prob = np.zeros((visible_nodes, hidden_nodes))

        self.v1_marginal_prob = np.zeros((visible_nodes))
        self.v2_marginal_prob = np.zeros((visible_nodes))
        self.hidden_marginal_prob = np.zeros((hidden_nodes))

    def add(self, prob: Self):
        self.v1_internal_prob += prob.v1_internal_prob
        self.v2_internal_prob += prob.v2_internal_prob
        self.v1_hidden_prob += prob.v1_hidden_prob
        self.v2_hidden_prob += prob.v2_hidden_prob
        self.v1_marginal_prob += prob.v1_marginal_prob
        self.v2_marginal_prob += prob.v2_marginal_prob
        self.hidden_marginal_prob += prob.hidden_marginal_prob

    def divide(self, factor):
        self.v1_internal_prob /= factor
        self.v2_internal_prob /= factor
        self.v1_hidden_prob /= factor
        self.v2_hidden_prob /= factor
        self.v1_marginal_prob /= factor
        self.v2_marginal_prob /= factor
        self.hidden_marginal_prob /= factor


def collect_state_stats(state: EncoderState) -> ProbStats:
    stats = ProbStats(state.visible_nodes, state.hidden_nodes)
    for i in range(state.visible_nodes):
        for j in range(state.visible_nodes):
            if state.v1_state[i] == 1 and state.v1_state[j] == 1:
                stats.v1_internal_prob[i][j] = 1
            if state.v2_state[i] == 1 and state.v2_state[j] == 1:
                stats.v2_internal_prob[i][j] = 1

    for i in range(state.visible_nodes):
        for j in range(state.hidden_nodes):
            if state.v1_state[i] == 1 and state.hidden_state[j] == 1:
                stats.v1_hidden_prob[i][j] = 1
            if state.v2_state[i] == 1 and state.hidden_state[j] == 1:
                stats.v2_hidden_prob[i][j] = 1

    for i in range(state.visible_nodes):
        if state.v1_state[i] == 1:
            stats.v1_marginal_prob[i] = 1
        if state.v2_state[i] == 1:
            stats.v2_marginal_prob[i] = 1

    for i in range(state.hidden_nodes):
        if state.hidden_state[i] == 1:
            stats.hidden_marginal_prob[i] = 1

    return stats


def create_random_state(
    visible_nodes, hidden_nodes, rng: numpy.random.Generator
) -> EncoderState:
    state = EncoderState(visible_nodes, hidden_nodes)
    state.v1_state = rng.binomial(1, [0.5], visible_nodes)
    state.v2_state = rng.binomial(1, [0.5], visible_nodes)
    state.hidden_state = rng.binomial(1, [0.5], hidden_nodes)
    return state


def create_clamped_state(
    state_number,
    visible_nodes,
    hidden_nodes,
    on_to_off_prob,
    off_to_on_prob,
    rng: numpy.random.Generator,
) -> EncoderState:
    state = EncoderState(visible_nodes, hidden_nodes)
    state.v1_state = rng.binomial(1, [off_to_on_prob], visible_nodes)
    state.v2_state = rng.binomial(1, [off_to_on_prob], visible_nodes)
    state.hidden_state = rng.binomial(1, [0.5], hidden_nodes)

    state.v1_state[state_number] = rng.binomial(1, [1 - on_to_off_prob])
    state.v2_state[state_number] = rng.binomial(1, [1 - on_to_off_prob])

    return state


class Encoder:
    def __init__(self, visible_nodes, hidden_nodes, seed=SEED):
        self.visible_nodes = visible_nodes
        self.hidden_nodes = hidden_nodes

        self.v1_internal_weights = np.zeros((visible_nodes, visible_nodes))
        self.v2_internal_weights = np.zeros((visible_nodes, visible_nodes))
        self.v1_hidden_weights = np.zeros((visible_nodes, hidden_nodes))
        self.v2_hidden_weights = np.zeros((visible_nodes, hidden_nodes))

        self.v1_bias = np.zeros((visible_nodes))
        self.v2_bias = np.zeros((visible_nodes))
        self.hidden_bias = np.zeros((hidden_nodes))

        self.rng = numpy.random.default_rng(seed)

    def train(
        self,
        epochs=1000,
        step_size=STEP_SIZE,
        annealing_sequence=ANNEALING_SEQUENCE,
        collection_steps=COLLECTION_STEPS,
        ensemble_runs=ENSEMBLE_RUNS,
    ):
        for _ in range(epochs):
            clamped_stats = self.compute_clamped_stats(
                collection_steps, ensemble_runs, annealing_sequence
            )
            free_stats = self.compute_free_stats(
                collection_steps, ensemble_runs, annealing_sequence
            )
            self.update_weights(clamped_stats, free_stats, step_size)

    def update_weights(
        self, clamped_stats: ProbStats, free_stats: ProbStats, step_size
    ):
        for i in range(self.visible_nodes):
            for j in range(self.visible_nodes):
                self.v1_internal_weights[i][j] += step_size * np.sign(
                    clamped_stats.v1_internal_prob[i][j]
                    - free_stats.v1_internal_prob[i][j]
                )
                self.v2_internal_weights[i][j] += step_size * np.sign(
                    clamped_stats.v2_internal_prob[i][j]
                    - free_stats.v2_internal_prob[i][j]
                )

        for i in range(self.visible_nodes):
            for j in range(self.hidden_nodes):
                self.v1_hidden_weights[i][j] += step_size * np.sign(
                    clamped_stats.v1_hidden_prob[i][j] - free_stats.v1_hidden_prob[i][j]
                )
                self.v2_hidden_weights[i][j] += step_size * np.sign(
                    clamped_stats.v2_hidden_prob[i][j] - free_stats.v2_hidden_prob[i][j]
                )

        for i in range(self.visible_nodes):
            self.v1_bias[i] += step_size * np.sign(
                clamped_stats.v1_marginal_prob[i] - free_stats.v1_marginal_prob[i]
            )
            self.v2_bias[i] += step_size * np.sign(
                clamped_stats.v2_marginal_prob[i] - free_stats.v2_marginal_prob[i]
            )
        for i in range(self.hidden_nodes):
            self.hidden_bias[i] += step_size * np.sign(
                clamped_stats.hidden_marginal_prob[i]
                - free_stats.hidden_marginal_prob[i]
            )

    # TODO refactor this function and compute_clamped_stats to extract common code
    def compute_free_stats(
        self, collection_steps, ensemble_runs, annealing_sequence
    ) -> ProbStats:
        stats = ProbStats(self.visible_nodes, self.hidden_nodes)
        for _ in range(ensemble_runs):
            state = create_random_state(
                self.visible_nodes,
                self.hidden_nodes,
                rng=self.rng,
            )

            for temperature in annealing_sequence:
                self.free_step(state, temperature)

            temperature = annealing_sequence[-1]
            for _ in range(collection_steps):
                self.free_step(state, temperature)
                cur_stats = collect_state_stats(state)

                stats.add(cur_stats)
        stats.divide(ensemble_runs * collection_steps)

        return stats

    def compute_clamped_stats(
        self, collection_steps, ensemble_runs, annealing_sequence
    ) -> ProbStats:
        stats = ProbStats(self.visible_nodes, self.hidden_nodes)
        for state_number in range(self.visible_nodes):
            for _ in range(ensemble_runs):
                state = create_clamped_state(
                    state_number,
                    self.visible_nodes,
                    self.hidden_nodes,
                    on_to_off_prob=0.15,
                    off_to_on_prob=0.05,
                    rng=self.rng,
                )

                for temperature in annealing_sequence:
                    self.clamped_step(state, temperature)

                temperature = annealing_sequence[-1]
                for _ in range(collection_steps):
                    self.clamped_step(state, temperature)
                    cur_stats = collect_state_stats(state)

                    stats.add(cur_stats)
        stats.divide(self.visible_nodes * ensemble_runs * collection_steps)
        return stats

    def compute_v1_delta_energy(self, node, state: EncoderState):
        result = 0.0
        for i in range(self.visible_nodes):
            if state.v1_state[i] == 1 and i != node:
                result += self.v1_internal_weights[i][node]

        for i in range(self.hidden_nodes):
            if state.hidden_state[i] == 1:
                result += self.v1_hidden_weights[node][i]

        result += self.v1_bias[node]
        return result

    def compute_v2_delta_energy(self, node, state: EncoderState):
        result = 0.0
        for i in range(self.visible_nodes):
            if state.v2_state[i] == 1 and i != node:
                result += self.v2_internal_weights[i][node]

        for i in range(self.hidden_nodes):
            if state.hidden_state[i] == 1:
                result += self.v2_hidden_weights[node][i]

        result += self.v2_bias[node]
        return result

    def compute_hidden_delta_energy(self, node, state: EncoderState):
        result = 0.0
        for i in range(self.visible_nodes):
            if state.v1_state[i] == 1:
                result += self.v1_hidden_weights[i][node]
            if state.v2_state[i] == 1:
                result += self.v2_hidden_weights[i][node]

        result += self.hidden_bias[node]
        return result

    @staticmethod
    def boltzmann_prob(energy, temperature):
        return 1 / (1 + np.exp(-energy / temperature))

    def free_step(self, state: EncoderState, temperature):
        perm = self.rng.permutation(2 * self.visible_nodes + self.hidden_nodes)

        for i in perm:
            if i < self.visible_nodes:
                energy = self.compute_v1_delta_energy(i, state)
                prob = self.boltzmann_prob(energy, temperature)
                state.v1_state[i] = self.rng.binomial(1, prob)

            elif i < 2 * self.visible_nodes:
                index = i - self.visible_nodes
                energy = self.compute_v2_delta_energy(index, state)
                prob = self.boltzmann_prob(energy, temperature)
                state.v2_state[index] = self.rng.binomial(1, prob)
            else:
                index = i - 2 * self.visible_nodes
                energy = self.compute_hidden_delta_energy(index, state)
                prob = self.boltzmann_prob(energy, temperature)
                state.hidden_state[index] = self.rng.binomial(1, prob)

    def clamped_step(self, state: EncoderState, temperature):
        perm = self.rng.permutation(self.hidden_nodes)
        for i in perm:
            energy = self.compute_hidden_delta_energy(i, state)
            prob = self.boltzmann_prob(energy, temperature)
            state.hidden_state[i] = self.rng.binomial(1, prob)

    def simulate_input_step(self, state: EncoderState, temperature):
        perm = self.rng.permutation(self.visible_nodes + self.hidden_nodes)

        for i in perm:
            if i < self.visible_nodes:
                energy = self.compute_v2_delta_energy(i, state)
                prob = self.boltzmann_prob(energy, temperature)
                state.v2_state[i] = self.rng.binomial(1, prob)
            else:
                index = i - self.visible_nodes
                energy = self.compute_hidden_delta_energy(index, state)
                prob = self.boltzmann_prob(energy, temperature)
                state.hidden_state[index] = self.rng.binomial(1, prob)

    def simulate_input(
        self, state_index, annealing_sequence=ANNEALING_SEQUENCE
    ) -> EncoderState:
        state = EncoderState(self.visible_nodes, self.hidden_nodes)
        state.v1_state[state_index] = 1
        state.v2_state = self.rng.binomial(1, [0.5], self.visible_nodes)
        state.hidden_state = self.rng.binomial(1, [0.5], self.hidden_nodes)
        for temperature in annealing_sequence:
            self.simulate_input_step(state, temperature)

        return state
