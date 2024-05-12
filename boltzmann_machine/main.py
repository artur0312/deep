from encoder import Encoder

# Schedule based on the algorithm implementation of https://github.com/amiaty/Boltzmann-Machine-Matlab/
schedule_gentle_sequence = [
    [40, 20],
    [40, 15],
    [40, 12],
    [40, 10],
    [40, 8],
    [40, 6],
    [40, 5],
]
schedule_gentle = []

for step in schedule_gentle_sequence:
    for i in range(step[0]):
        schedule_gentle.append(step[1])

visible_nodes = 4
hidden_nodes = 2

encoder = Encoder(visible_nodes, hidden_nodes)
encoder.train(epochs=1000)

states = []

for i in range(visible_nodes):
    state = encoder.simulate_input(i, schedule_gentle)
    states.append(state)

for i in range(visible_nodes):
    print(states[i].v2_state)
