
# Octopus Locomotion Simulation with PyElastica

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

This project showcases a series of simulations built with the `PyElastica` library, progressively developing a soft, multi-armed octopus model capable of locomotion. The simulations start from a single flexible rod and evolve into a complete octopus with a central body, eight arms, and a neural-inspired gait controller that enables crawling.

---

## Simulation Showcase

The final simulation (`week3_locomotion.py`) produces the following crawling motion, saved as an animated GIF. This demonstrates the implementation of a gait controller, anisotropic friction with the ground, and the dynamic connection between multiple flexible bodies.

![Octopus Crawling Simulation](https://github.com/rayhuang2006/octopus-sim/blob/main/octopus_crawling.gif?raw=true)

---

## Key Features

* **Cosserat Rod Dynamics**: Utilizes the Cosserat rod model in PyElastica to accurately simulate the bending, twisting, shearing, and stretching of soft, flexible arms.
* **Complex System Assembly**: Demonstrates how to build and connect multiple rods to create a complex articulated body.
* **Custom Force Implementation**: Features custom-written force classes for:
    * **Ground Contact & Anisotropic Friction** (`GroundPlaneWithFriction`): Simulates a repulsive ground force and different friction coefficients for forward and backward sliding, essential for locomotion.
    * **Muscle Actuation** (`GaitController`): Applies rhythmic torques at the base of each arm with phase shifts to generate coordinated crawling patterns.
    * **Inter-body Connections** (`RodConnector`): Uses spring-like forces to connect the arms to a central body.
* **Data Visualization**: Employs `Matplotlib` for static 2D/3D plots and `PyVista` for advanced 3D rendering and animation generation.
* **Real-time Simulation**: Includes a script (`week3_locomotion_demo.py`) that uses a custom callback to provide a live 3D visualization of the simulation as it runs.

---

## Project Structure

The project is structured as a weekly progression, with each script building upon the concepts of the previous one.

* `week1_simulation.py`: **Introduction to PyElastica.**
    * Simulates a single cantilever rod (beam) bending under gravity.
    * Establishes the basic workflow: create a simulator, define a rod, apply forces and boundary conditions, and integrate over time.
    * Visualizes the initial and final states using `Matplotlib`.

* `week2_assembly.py`: **Building a Multi-Rod System.**
    * Assembles a "lifeless" octopus skeleton by creating eight arms radiating from a single fixed point.
    * Demonstrates looping to construct multiple identical systems and applying forces to each.

* `week3_locomotion.py`: **Full Octopus Locomotion.**
    * Builds the complete model with a central body and eight attached arms.
    * Implements all custom forces: `GroundPlaneWithFriction`, `GaitController`, and `RodConnector`.
    * Uses a `PyVistaCallback` to collect simulation frames and generates `octopus_crawling.gif` as the final output.

* `week3_locomotion_demo.py`: **Live Real-time Visualization.**
    * An alternative version of the Week 3 simulation.
    * Introduces the `LiveVisualizer` callback class, which opens a `PyVista` window and updates the 3D model in real-time as the simulation progresses.

* `requirements.txt`: Contains all the necessary Python packages for the project.

---

## Getting Started

Follow these steps to set up the environment and run the simulations on your local machine.

### 1. Clone the Repository

```bash
git clone https://github.com/rayhuang2006/octopus-sim.git
cd octopus-sim
````

### 2\. Create a Virtual Environment (Recommended)

It's good practice to use a virtual environment to manage project-specific dependencies.

```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

-----

## ⚠️⚠️⚠️ Important Note on Week 3 Scripts ⚠️⚠️⚠️

**Please be aware:** While the GIF showcase proves that `week3_locomotion.py` can work correctly, the Week 3 scripts may exhibit instability on different systems.

  * `week3_locomotion.py`
  * `week3_locomotion_demo.py`

Their successful execution can be sensitive to specific versions of packages (like `pyvista`, `numpy`, or `pyelastica`) and the underlying operating system. If you encounter errors, you may need to experiment with different dependency versions. The `week1` and `week2` scripts are stable and should run without issues.

-----

## How to Run the Simulations

Execute the Python scripts from your terminal. Each script corresponds to a stage of the project.

  * **Week 1: Single Bending Rod**

      * This will run the simulation and display a `Matplotlib` plot showing the rod's initial and final positions.

    <!-- end list -->

    ```bash
    python week1_simulation.py
    ```

  * **Week 2: Octopus Assembly**

      * This will simulate the 8-armed skeleton falling under gravity and display the result with `Matplotlib`.

    <!-- end list -->

    ```bash
    python week2_assembly.py
    ```

  * **Week 3: Locomotion & GIF Generation**

      * This is the main simulation. It creates the `octopus_crawling.gif` file. *(Note: See stability warning above).*

    <!-- end list -->

    ```bash
    python week3_locomotion.py
    ```

  * **Week 3: Live Demo**

      * This will open a `PyVista` window to show the octopus moving in real-time. *(Note: See stability warning above).*

    <!-- end list -->

    ```bash
    python week3_locomotion_demo.py
    ```

