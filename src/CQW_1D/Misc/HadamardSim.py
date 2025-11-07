import numpy as np
import matplotlib.pyplot as plt

#Outlining Variables:
t = 120 # Number of max time steps
num_positions = 160 # Number of positions on the line
initial_position = num_positions // 2 # Start at Middle
times = [20,40,60,80] #time frames to plot

#Initial state (Eqn. 5) :
initial_state = np.zeros ((num_positions, 2), dtype=complex) #Empty Array
initial_state[initial_position] = [1 / np.sqrt(2), 1j / np.sqrt (2)] #Initial Pauli Y Gate Eigenvectors
print (initial_state)
hadamard_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]]) #coin (This time is Hadamard) operator (Eqn. 3)

#Shift operator (Eqn. 4):
def apply_shift_operator(state):
    new_state = np.zeros_like(state)
    # Shift based on state: right - > |up», left -> | down»
    new_state[1:, 0] = state[: -1, 0] # down state shift
    new_state[:-1, 1] = state[1:, 1] # up state shift
    return new_state

probability_distributions = []
state = initial_state #Start at Unbiased eigenvector thing

for step in range(t):
    state = np.dot(state, hadamard_gate) #Coin Operator
    state = apply_shift_operator(state) #Shift Operator
    
    if step + 1 in times:
        probability_distribution = np. sum(np. abs (state)**2, axis=1) #Eqn.1
        probability_distributions.append (probability_distribution) #Prob Distribution for each time

# Plotting:
positions = np.arange(-initial_position, initial_position)

for i, distribution in enumerate (probability_distributions):
    plt.plot (positions, distribution, label=f"t = {times[i]}")

plt.xlabel ("Position (x)")
plt.ylabel ( "Probability")
plt.title("Probability Distribution for CQW's (Hadamard)")
plt.legend()
plt.show()

