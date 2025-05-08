import random
import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt

# Data
pouplation_size = 250
customers = ["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K10"]
tour_size = 6

# Generating random coordinates of the customers
x = np.array([(random.randint(0, 10)) for _ in range(len(customers))])
y = np.array([(random.randint(0, 10)) for _ in range(len(customers))])
cust_coords = dict(zip(customers, zip(x, y)))


def initial_population(cust_list, poup_size):
    population_perms = []
    possible_perms = list(permutations(cust_list))
    random_ids = random.sample(range(0, len(possible_perms)), poup_size)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))

    return population_perms


def dist_two_customers(c1, c2):
    c1_coords = cust_coords[c1]
    c2_coords = cust_coords[c2]
    # Euclidean
    return np.sqrt(np.sum((np.array(c1_coords) - np.array(c2_coords)) ** 2))


def total_distance(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if i == len(individual) - 1:
            total_dist += dist_two_customers(individual[i], individual[0])
        else:
            total_dist += dist_two_customers(individual[i], individual[i + 1])
    return total_dist


def evaluate_fitness(population):
    total_dist_all_individuals = []
    for i in range(len(population)):
        total_dist_all_individuals.append(total_distance(population[i]))
    return total_dist_all_individuals




# Order1 Point Crossover
def crossover(parent_1, parent_2):
    crossover_point = random.randint(0, len(parent_1) - 1)
    child = parent_1[:crossover_point] + [gene for gene in parent_2 if gene not in parent_1[:crossover_point]]
    return child


def mutation(offspring):
    n_cut = len(offspring) - 1
    index_1 = random.randint(0, n_cut)
    index_2 = random.randint(0, n_cut)

    temp = offspring[index_1]
    offspring[index_1] = offspring[index_2]
    offspring[index_2] = temp
    return offspring


def fitness_prob(population):
    fitness = evaluate_fitness(population)
    fitness_prob = [1 / dist for dist in fitness]
    total_prob = sum(fitness_prob)
    fitness_prob = [prob / total_prob for prob in fitness_prob]
    return fitness_prob


def roulette_wheel(population, fitness_probs):
    r = random.random()
    cumulative = 0
    for individual, prob in zip(population, fitness_probs):
        cumulative += prob
        if cumulative >= r:
            return individual
    # Fallback to return the last individual in case of rounding errors
    return population[-1]


def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)

    for generation in range(n_generations):
        fitness_probs = fitness_prob(population)

        # Selecting parents for crossover using roulette wheel selection
        parents_list = []
        for _ in range(int(crossover_per * n_population)):
            parents_list.append(roulette_wheel(population, fitness_probs))

        # Generating offspring using crossover and mutation
        offspring_list = []
        for i in range(0, len(parents_list), 2):
            offspring_1 = crossover(parents_list[i], parents_list[(i + 1) % len(parents_list)])
            offspring_2 = crossover(parents_list[(i + 1) % len(parents_list)], parents_list[i])

            if random.random() > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)

            if random.random() > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        # Combine parents and offspring for the next generation
        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)

        # Select the best individuals to form the next generation
        sorted_fitness_indices = np.argsort([total_distance(ind) for ind in mixed_offspring])
        population = [mixed_offspring[i] for i in sorted_fitness_indices[:n_population]]

        # Output the results for the current generation
        generation_distances = [total_distance(ind) for ind in population]
        best_distance = min(generation_distances)
        best_path = population[np.argmin(generation_distances)]

        print(f"Generation {generation + 1}: Minimum Distance = {best_distance}, Best Path = {best_path}")

    # Return the final population and best path
    return population


# Running the GA

n_population = pouplation_size
n_generations = 100
crossover_per = 0.8
mutation_per = 0.2

best_population = run_ga(customers, n_population, n_generations, crossover_per, mutation_per)

total_dist_all_individuals = [total_distance(ind) for ind in best_population]
index_minimum = np.argmin(total_dist_all_individuals)
minimum_distance = min(total_dist_all_individuals)
shortest_path = best_population[index_minimum]

print("Shortest Path:", shortest_path)
print("Minimum Distance:", minimum_distance)


#Plotting
x_shortest = []
y_shortest = []
for city in shortest_path:       #shortest_path array (from GA) that identify each city with its name or index
    #retrieving the coordinates of each city in the optimal route
    x_value, y_value = cust_coords[city]
    x_shortest.append(x_value)
    y_shortest.append(y_value)




fig, ax = plt.subplots()

# Plot the best route
ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)

# Highlight the starting point
ax.plot(x_shortest[0], y_shortest[0], 'ro', markersize=10, label='Start (City 1)')

# Highlight the ending point
ax.plot(x_shortest[-1], y_shortest[-1], 'bs', markersize=10, label='End (City N)')

# Add legend
plt.legend()

# Background lines connecting all pairs of cities
for i in range(len(x)):
    for j in range(i + 1, len(x)):
        ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)

# Title and subtitles
plt.title(label="TSP Best Route Using GA", fontsize=25, color="k")
str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
plt.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)} {str_params}", fontsize=18, y=1.047)

# Annotate each city in the shortest path
for i, txt in enumerate(shortest_path):
    ax.annotate(f"{i + 1}- {txt}", (x_shortest[i], y_shortest[i]), fontsize=20, textcoords="offset points", xytext=(0, 10), ha='center')

# Adjust figure size and save
fig.set_size_inches(16, 12)
plt.savefig('solution.png')
plt.show()