import copy
from argparse import ArgumentParser, BooleanOptionalAction
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
plt.ion()
GRID_SIZE = (100, 100)
FRACTION_EMPTY = 0.1
NUM_STEPS = 50
STEPS_PER_PLOT = 1
DELAY_PER_PLOT = 0.5
# Similarity threshold is D in the brief.
SIMILARITY_THRESHOLD = 0.2
# Segregation threshold is T_segregation in the brief
SEGREGATION_THRESHOLD = 0.5
# Consensus threshold is T_consensus in the brief
CONSENSUS_THRESHOLD = np.clip(2 * SIMILARITY_THRESHOLD, 0, 1)
# Consensus weight is "alpha" in the brief
CONSENSUS_WEIGHT = 0.1
# empty point represented by value of 0 exactly
EMPTY = 0
PARAMS = dict()
"""Dictionary to store DEFAULT parameters — these may later be replaced by values read from the command line"""
PARAMS["num_steps"] = NUM_STEPS
PARAMS["grid_size"] = GRID_SIZE
PARAMS["fraction_empty"] = FRACTION_EMPTY
PARAMS["steps_per_plot"] = STEPS_PER_PLOT
PARAMS["delay_per_plot"] = DELAY_PER_PLOT
PARAMS["use_consensus"] = True
PARAMS["use_segregation"] = True
PARAMS["use_floodfill"] = True
PARAMS["similarity_threshold"] = SIMILARITY_THRESHOLD
PARAMS["segregation_threshold"] = SEGREGATION_THRESHOLD
PARAMS["consensus_threshold"] = CONSENSUS_THRESHOLD
PARAMS["consensus_weight"] = CONSENSUS_WEIGHT
PARAMS["filename"] = None
PARAMS["outfile"] = None
def load_grid_from_file(filename):
    """
This function loads a grid of values from a file.
:param filename: string representing the file path of the grid to be loaded.
    """
    grid = []
    with open(filename, "r") as f:
        for line_count, line in enumerate(f):
            values = line.split(",")
            grid.append([float(val) for val in values])
    return np.array(grid)
def write_grid_to_file(grid, filename):
    np.savetxt(filename, grid, delimiter=",", fmt="%.2f")
def plot_grid(ax, grid, delay=0):
    cmap = plt.get_cmap("coolwarm") # sets colour map
    cmap.set_under("Black") # sets the colour for empty cells
    ax.clear() # clears the previous plot
    ax.imshow(grid, cmap=cmap, vmin = 0, vmax = 1) # shows the grid
    # removs the axis numbers 
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    # waits for the next step
    if delay > 0:
        plt.pause(delay)
    else:
        plt.show()

def make_random_grid(params):
    # grid sizes from parameters
    rows = params["grid_size"][0]
    columns = params["grid_size"][1]
    # getting fraction of empry cells
    fraction_empty = params["fraction_empty"]
    # creating grid with random values
    grid = np.random.rand(rows, columns)
    # calculates the number of cells should be empty
    total_cells = rows * columns
    empty_cells = int(total_cells * fraction_empty)
    # chooses random positions to make it empty
    indices = np.random.choice(total_cells, empty_cells, replace = False)
    # converts index to row and column
    for i in indices:
        row = i // columns
        column = i % columns
        # sets this position to emprt
        grid[row, column] = EMPTY
    return grid
def consensus_update(grid, params):
    # getting grid size
    rows = grid.shape[0]
    columns = grid.shape[1]
    # getting parameters
    threshold = params["consensus_threshold"]
    alpha = params["consensus_weight"]
    # makes a copy in order to not lose the original one
    updated_grid = grid.copy()
    for r in range(rows):
        for c in range(columns):
            if grid[r, c] == EMPTY: # skips if the cell is empty
                continue
            # checks the neighbour cells
            for delta_row in [-1, 0, 1]:
                for delta_column in [-1, 0, 1]:
                    # skips the cell itself
                    if delta_row == 0 and delta_column == 0:
                        continue
                    # finds neighbour cells 
                    neighbour_row = (r + delta_row) % rows
                    neighbour_column = (c + delta_column) % columns
                    # skips the neigbour if its empty
                    if grid[neighbour_row, neighbour_column] == EMPTY:
                        continue
                    # calculates the differance between opinions
                    differance = abs(updated_grid[r, c] - updated_grid[neighbour_row, neighbour_column])
                    # updates the opinion if the difference is beloq thrsgold
                    if differance < threshold:
                        opinion1 = updated_grid[r, c]
                        opinion2 = updated_grid[neighbour_row, neighbour_column]
                        # mobes opinions closer together
                        updated_grid[r, c] = opinion1 + alpha * (opinion2 - opinion1)
                        updated_grid[neighbour_row, neighbour_column] = opinion2 + alpha * (opinion1 - opinion2)
    return updated_grid
def segregation_update(grid, params):
    # gets the grid size
    rows = grid.shape[0]
    columns = grid.shape[1]
    # gets the parameters
    D = params["similarity_threshold"]

    T = params["segregation_threshold"]
    # copies the grid in order to not lose the original
    updated_grid = grid.copy()
    # finds all the empty cells
    empty_cells = list(zip(*np.where(grid == EMPTY)))
    # goes through every cell 
    for r in range(rows):
        for c in range(columns):
            # skips empty cells 
            if grid[r, c] == EMPTY:
                continue
            similar_neighbours = 0
            different_neighbours = 0
            # checks neighbour cells
            for delta_row in [-1, 0, 1]:
                for delta_column in [-1, 0, 1]:
                    # skips the cell itself
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbour_row = (r + delta_row) % rows
                    neighbour_column = (c + delta_column) % columns
                    # skips empty neighbpurs
                    if grid[neighbour_row, neighbour_column] == EMPTY:
                        continue
                    # cheks if the neighbour is similar or not
                    if abs(grid[r, c] - grid[neighbour_row, neighbour_column]) <= D:
                        similar_neighbours += 1
                    else:
                        different_neighbours += 1
            # skips if there are no neighbours
            if similar_neighbours + different_neighbours == 0:
                continue
            # moves the agent if similarity is too low
            S = similar_neighbours / (similar_neighbours + different_neighbours)
            if S < T and len(empty_cells) > 0:
                # gets to a free position
                new_row, new_column = empty_cells.pop(0)
                # moves the agent
                updated_grid[new_row, new_column] = grid [r, c]
                updated_grid[r, c] = EMPTY
    return updated_grid
def update(grid, params):
    """Apply one timesteps worth of updates to the grid.
:param grid: A NumPy ndarray representing a grid of values in the range (0, 1).
:param params: A dictionary storing all model parameters.
    """
    if params["use_consensus"]:
        grid = consensus_update(grid, params)
    if params["use_segregation"]:
        grid = segregation_update(grid, params)
    return grid
def calculate_mean_similarity(grid, params):
    # we all know the meaning of this now
    rows = grid.shape[0]
    columns = grid.shape[1]
    # similarity parameters
    D = params["similarity_threshold"]
    # list to store similarity valies
    similarities = []
    # goes through every cell in the gird
    for r in range(rows):

        for c in range(columns):
            # this one also skips empty cells
            if grid[r, c] == EMPTY:
                continue
            similar_neighbours = 0 
            differet_neighbours = 0
            # checks neighbouir cells 
            for delta_row in [-1, 0, 1]:
                for delta_column in [-1, 0, 1]:
                    # skips the cell itself
                    if delta_row == 0 and delta_column == 0:
                        continue
                    neighbour_row = (r + delta_row) % rows
                    neighbour_column = (c + delta_column) % columns
                    # skips empty neighbours
                    if grid[neighbour_row, neighbour_column] == EMPTY:
                        continue
                    # cheks if neighbour is similar or not
                    if abs(grid[r, c] - grid[neighbour_row, neighbour_column]) <= D:
                        similar_neighbours += 1
                    else:
                        differet_neighbours += 1
            # calculates the similarity of this cell
            if similar_neighbours + differet_neighbours > 0:
                S = similar_neighbours / (similar_neighbours + differet_neighbours)
                similarities.append(S)
    # returns 0 if there are no similarity
    if len(similarities) == 0:
        return 0.0
    return np.mean(similarities)
def flood_fill(grid, params):
    # gets the grid seizes
    rows = grid.shape[0]
    columns = grid.shape[1]
    # this keeps tracks of the visited cells
    visited = np.zeros((rows, columns), dtype=bool)
    # grid that is going to store the cluster IDs
    clustered_grid = np.zeros((rows, columns))
    cluster_id = 1
    cluster_sizes = [] # a list to track cljster sizes
    # goes through every cell 
    for r in range (rows):
        for c in range (columns):
            # skips empty cells 
            if grid[r, c] == EMPTY or visited[r, c]:
                continue
            # starts the floodfill 
            stack = [(r, c)]
            visited[r, c] = True
            cells = []
            while stack:
                current_row, current_column = stack.pop()
                cells.append((current_row, current_column))
                # checks for neighbours 
                for delta_row in [1, 0, -1]:
                    for delta_column in [1, 0, -1]:
                        if delta_row == 0 and delta_column == 0:
                            continue
                        neighbour_row = (current_row + delta_row) % rows
                        neighbour_column = (current_column + delta_column) % columns
                        # continues floodfill is there are valid neighbours
                        if grid [neighbour_row, neighbour_column] != EMPTY and not visited [neighbour_row, neighbour_column]:
                            visited[neighbour_row, neighbour_column] = True
                            stack.append((neighbour_row, neighbour_column))

            # assigns ID to all the founded cells
            for x, y in cells:
                clustered_grid[x, y] = cluster_id
            # stores the cluster size 
            cluster_sizes.append(len(cells))
            cluster_id += 1
    # counting clusters larger than size 1
    count = sum(1 for size in cluster_sizes if size > 1)
    print ("Clusters with sizer > 1:", count)
    return clustered_grid
# TEST FUNCTIONS — BELOW #####################################################
def test_make_random_grid():
    params = copy.deepcopy(PARAMS)
    params["grid_size"] = (3, 3)
    params["fraction_empty"] = 0
    grid = make_random_grid(params)
    assert isinstance(grid, (np.ndarray)), "Error: grid is not numpy array"
    assert np.shape(grid) == params["grid_size"], "Grid shape doesn't match"
    params["grid_size"] = (20, 10)
    params["fraction_empty"] = 0
    grid = make_random_grid(params)
    assert isinstance(grid, (np.ndarray)), "Error: grid is not numpy array"
    assert np.shape(grid) == params["grid_size"], "Grid shape doesn't match"
    params["grid_size"] = (10, 10)
    params["fraction_empty"] = 0
    grid = make_random_grid(params)
    any_are_zero = np.equal(grid, 0).any()
    assert (
        not any_are_zero
    ), "Grid with fraction_empty of 0 should have no zeros"
    params["grid_size"] = (10, 10)
    params["fraction_empty"] = 1
    grid = make_random_grid(params)
    all_are_zero = np.equal(grid, 0).all()
    assert all_are_zero, "Grid with fraction_empty of 1 should be all zeros"
    params["grid_size"] = (10, 10)
    params["fraction_empty"] = 0.5
    grid = make_random_grid(params)
    zero_count = np.equal(grid, 0).sum()
    expected_zero_count = params["fraction_empty"] * np.size(grid)
    assert (
        abs(zero_count - expected_zero_count) < 5
    ), f"Grid with fraction_empty = 0.5 should have close to 50 zeros, got zero_count = {zero_count}."
def test_consensus_update():
    params = copy.deepcopy(PARAMS)
    shape_3x3 = (3, 3)
    params["grid_size"] = shape_3x3
    grid = 0.4 * np.ones(shape_3x3)
    updated_grid = consensus_update(grid, params)
    zero_count = np.equal(updated_grid, 0).sum()
    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Consensus update should not change grid size"
    assert np.array_equal(
        grid, updated_grid
    ), "Grid with all values the same should not be changed"
    assert (
        zero_count == 0
    ), "Consensus update should not change number of zero cells"
    params["consensus_weight"] = 0.5
    params["grid_size"] = shape_3x3
    grid = np.array([[0.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.6]])
    updated_grid = consensus_update(grid, params)
    zero_count = np.equal(updated_grid, 0).sum()

    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Consensus update should not change grid size"
    assert (
        zero_count == 7
    ), "Consensus update should not change number of zero cells"
    assert (
        updated_grid[0, 0] == 0.5
    ), "Cell (0, 0) should equal 0.5 after update with weight of 0.5 — do you have circular boundary conditions?"
    assert (
        updated_grid[2, 2] == 0.5
    ), "Cell (2, 2) should equal 0.5 after update with weight of 0.5 — do you have circular boundary conditions?"
def test_segregation_update():
    params = copy.deepcopy(PARAMS)
    shape_3x3 = (3, 3)
    params["grid_size"] = shape_3x3
    grid = np.array([[0.4, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
    updated_grid = segregation_update(grid, params)
    zero_count = np.equal(updated_grid, 0).sum()
    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Segregation update should not change grid size"
    assert (
        zero_count == 7
    ), "Segregation update should not change number of zero cells"
    assert (
        updated_grid[0, 0] == 0.4
    ), "Cell (0, 0) should equal 0.4 after segregation update"
    assert (
        updated_grid[1, 1] == 0.4
    ), "Cell (1, 1) should equal 0.4 after segregation update"
    params["segregation_threshold"] = 1.0
    grid = np.array([[0.1, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.0]])
    updated_grid = segregation_update(grid, params)
    zero_count = np.equal(updated_grid, 0).sum()
    assert np.array_equal(
        np.shape(updated_grid), shape_3x3
    ), "Segregation update should not change grid size"
    assert (
        zero_count == 7
    ), "Segregation update should not change number of zero cells"
    assert (
        updated_grid[0, 0] != 0.1
    ), "Cell (0, 0) should not equal 0.1 after a segregation update"
    assert updated_grid[1, 1] != 0.9, "Cell (1, 1) should not equal 0.9"
    # You may wish to add additional tests to confirm your segregation update is working.
def test_calculate_mean_similarity():
    params = copy.deepcopy(PARAMS)
    grid = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mean_similarity = calculate_mean_similarity(grid, params)
    assert (
        calculate_mean_similarity(grid, params) == 1.0
    ), "Mean similarity should be 1.0, but is actually {mean_similarity}"
    grid = np.array([[0.0, 0.1, 0.0], [0.1, 0.1, 0.1], [0.0, 0.1, 0.0]])
    mean_similarity = calculate_mean_similarity(grid, params)
    assert (
        calculate_mean_similarity(grid, params) == 1.0
    ), "Mean similarity should be 1.0, but is actually {mean_similarity}"
    params["similarity_threshold"] = 0.5
    grid = np.array([[0.0, 0.1, 0.0], [0.1, 0.9, 0.1], [0.0, 0.1, 0.0]])
    mean_similarity = calculate_mean_similarity(grid, params)
    assert (
        mean_similarity == 0.6
    ), "Mean similarity should be 0.5625, but is actually {mean_similarity}"
    params["similarity_threshold"] = 0.01
    grid = np.array([[0.0, 0.2, 0.0], [0.3, 0.4, 0.5], [0.0, 0.6, 0.0]])
    mean_similarity = calculate_mean_similarity(grid, params)
    assert (
        mean_similarity == 0.0
    ), "Mean similarity should be 0.0, but is actually {mean_similarity}"

if __name__ == "__main__":
    params = copy.deepcopy(PARAMS)
    grid = make_random_grid(params)

    fig, ax = plt.subplots()
    plot_grid(ax, grid, params["delay_per_plot"])
    ax.set_title("Initial grid")

    for step in range(params["num_steps"]):
        grid = update(grid, params)

        if step % params["steps_per_plot"] == 0:
            ax.set_title(f"Step {step + 1}")
            plot_grid(ax, grid, params["delay_per_plot"])

    print("Mean similarity:", calculate_mean_similarity(grid, params))

    plt.ioff()
    plt.show(block=True)