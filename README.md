# Opinion Dynamics Simulation

A Python grid simulation exploring opinion consensus and segregation, with visualisation using NumPy and Matplotlib.

## Features

- Random grids of continuous opinions with configurable empty cells.
- Consensus updates that bring sufficiently similar neighbouring opinions closer together.
- Segregation updates that move agents whose neighbours are insufficiently similar.
- Circular boundary conditions, so opposite grid edges are neighbours.
- Mean similarity measurement and flood-fill analysis of connected occupied cells.
- Helpers for loading and saving comma-separated grids.

## Setup

Requires Python 3.9 or newer.

```sh
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```sh
source .venv/bin/activate
```

Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

## Run

```sh
python "py project.py"
```

The program displays the initial grid and successive simulation updates, then prints the mean similarity. A graphical desktop is needed for the interactive plot.

Change the constants and `PARAMS` dictionary near the top of the script to adjust grid size, iteration count, empty-cell fraction, thresholds and update settings. The current entry point uses these defaults; it does not parse command-line arguments. Flood-fill and file input/output helpers are available as functions and are not called by the default run.

## Existing checks

The script includes four test functions. Install pytest and run them with:

```sh
python -m pip install pytest
python -m pytest "py project.py"
```

The original project script is included unchanged.
