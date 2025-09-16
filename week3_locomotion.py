import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import time

from elastica import *
from elastica.modules.callbacks import CallBackBaseClass
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces, NoForces
from elastica.dissipation import AnalyticalLinearDamper
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate


# --------------------------
# Simulator Class
class OctopusSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass


# --------------------------
class PostProcessingCallback(CallBackBaseClass):
    def __init__(self, step_skip, callback_params, rods):
        super().__init__()
        self.every = step_skip
        self.rods = rods
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            total_mass = 0.0
            com_position = np.zeros(3)
            for rod in self.rods:
                total_mass += np.sum(rod.mass)
                com_position += np.sum(rod.position_collection * rod.mass, axis=1)

            if total_mass > 0:
                self.callback_params["com_history"].append(com_position / total_mass)


# --------------------------
# PyVista Visualization Callback
# MODIFIED to use the callback_params pattern
class PyVistaCallback(CallBackBaseClass):
    def __init__(self, step_skip, rods, callback_params):
        super().__init__()
        self.every = step_skip
        self.rods = rods
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            frame = []
            for rod in self.rods:
                frame.append(rod.position_collection.copy().T)  # shape (n_nodes, 3)
            # Append frames to the list inside the dictionary
            self.callback_params["frames"].append(frame)


# --------------------------
# Ground + friction
class GroundPlaneWithFriction(NoForces):
    def __init__(self, k, nu_forward, nu_backward):
        super().__init__()
        self.k = k
        self.nu_forward = nu_forward
        self.nu_backward = nu_backward

    def apply_forces(self, system, time: np.float64 = 0.0):
        node_indices = np.where(system.position_collection[1, ...] < 0.0)[0]
        if node_indices.size == 0:
            return

        system.external_forces[1, node_indices] += -self.k * system.position_collection[
            1, node_indices
        ]

        element_indices = np.minimum(node_indices, system.n_elems - 1)
        tangent_vectors = system.tangents[..., element_indices]
        velocities = system.velocity_collection[..., node_indices]

        vel_along_tangent = np.einsum("ji,ji->i", velocities, tangent_vectors)
        nu = np.where(vel_along_tangent > 0, self.nu_forward, self.nu_backward)
        
        friction_force_tangential = -nu * vel_along_tangent * tangent_vectors
        friction_force_tangential[1,:] = 0
        
        system.external_forces[..., node_indices] += friction_force_tangential


# --------------------------
# Gait controller
class GaitController(NoForces):
    def __init__(self, arms, amplitude, frequency, phase_shift):
        super().__init__()
        self.arms = arms
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase_shift = phase_shift
        self.torque_axis_1 = np.array([0., 0., 1.])
        self.torque_axis_2 = np.array([0., 1., 0.])

    def apply_forces(self, system, time: np.float64 = 0.0):
        for i, arm in enumerate(self.arms):
            director_matrix = arm.director_collection[..., 0]
            
            torque1 = self.amplitude * np.sin(
                2 * np.pi * self.frequency * time + self.phase_shift * i
            )
            torque2 = self.amplitude * np.cos(
                2 * np.pi * self.frequency * time + self.phase_shift * i
            )
            
            torque_vec = torque1 * director_matrix @ self.torque_axis_1 + torque2 * self.torque_axis_2
            arm.external_torques[..., 0] += torque_vec


# --------------------------
# Connector
class RodConnector(NoForces):
    def __init__(self, connections, k_connection):
        super().__init__()
        self.connections = connections
        self.k_connection = k_connection

    def apply_forces(self, system, time: np.float64 = 0.0):
        for (rod_a, idx_a, rod_b, idx_b) in self.connections:
            pa = rod_a.position_collection[:, idx_a]
            pb = rod_b.position_collection[:, idx_b]
            delta = pb - pa
            dist = np.linalg.norm(delta)
            
            force_magnitude = self.k_connection * dist
            force_vec = force_magnitude * delta / (dist + 1e-12)
            
            rod_a.external_forces[:, idx_a] += force_vec
            rod_b.external_forces[:, idx_b] -= force_vec


# --------------------------
def build_octopus_with_body():
    n_elem = 20
    arm_length = 1.0
    arm_radius = 0.025
    density = 1000.0
    youngs_modulus = 1e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    num_arms = 8

    central_n_elem = 4
    central_length = 0.2
    central_radius = 0.08
    central_start = np.array([0.0, central_radius + 0.01, 0.0])
    central_direction = np.array([0.0, 1.0, 0.0])
    central_normal = np.array([1.0, 0.0, 0.0])

    central_body = CosseratRod.straight_rod(
        n_elements=central_n_elem,
        start=central_start,
        direction=central_direction,
        normal=central_normal,
        base_length=central_length,
        base_radius=central_radius,
        density=density,
        youngs_modulus=youngs_modulus * 5.0,
        shear_modulus=shear_modulus * 5.0,
    )

    rods_list = [central_body]
    attachment_indices = [0] * 4 + [1] * 4
    
    for i in range(num_arms):
        angle = i * (2 * np.pi / num_arms)
        attach_idx = attachment_indices[i]
        start_pt = central_body.position_collection[:, attach_idx]
        start_dir = np.array([np.cos(angle), 0.0, np.sin(angle)])
        normal = np.array([0.0, 1.0, 0.0])
        
        rod = CosseratRod.straight_rod(
            n_elements=n_elem,
            start=start_pt,
            direction=start_dir,
            normal=normal,
            base_length=arm_length,
            base_radius=arm_radius,
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
        )
        rods_list.append(rod)

    return rods_list


# --------------------------
def main():
    rods = build_octopus_with_body()
    central_body = rods[0]
    arms = rods[1:]

    sim = OctopusSimulator()

    for rod in rods:
        sim.append(rod)
        sim.add_forcing_to(rod).using(
            GravityForces, acc_gravity=np.array([0.0, -9.81, 0.0])
        )
        sim.add_forcing_to(rod).using(
            GroundPlaneWithFriction, k=1e4, nu_forward=0.8, nu_backward=1.0
        )
        sim.dampen(rod).using(
            AnalyticalLinearDamper, damping_constant=0.5, time_step=1e-5
        )

    connections = []
    attachment_indices = [0] * 4 + [1] * 4
    for i, arm in enumerate(arms):
        central_attach_idx = attachment_indices[i]
        arm_root_idx = 0
        connections.append((central_body, central_attach_idx, arm, arm_root_idx))
    
    sim.add_forcing_to(central_body).using(
        RodConnector, connections=connections, k_connection=5e3
    )
    sim.add_forcing_to(central_body).using(
        GaitController, arms=arms, amplitude=1.0, frequency=0.5, phase_shift=np.pi
    )

    # Post-processing callback
    post_processing_data = {"com_history": []}
    sim.collect_diagnostics(central_body).using(
        PostProcessingCallback, step_skip=2000, callback_params=post_processing_data, rods=rods
    )

    # PyVista callback using the same robust pattern
    pyvista_data = {"frames": []} # Create a dictionary to hold the frames
    sim.collect_diagnostics(central_body).using(
        PyVistaCallback, step_skip=1000, rods=rods, callback_params=pyvista_data # Pass it in
    )

    sim.finalize()

    timestepper = PositionVerlet()
    final_time = 5.0
    dt = 1e-5
    total_steps = int(final_time / dt)

    print("Running locomotion simulation...")
    integrate(timestepper, sim, final_time, total_steps)
    print("Simulation finished.")

    # --------------------------
    # PyVista Animation
    
    # Get the frames directly from our dictionary. No need to search for the instance.
    frames = pyvista_data["frames"]

    if not frames:
        print("Warning: No frames recorded for animation. Check step_skip and simulation time.")
        return

    print("Generating animation... this might take a moment.")
    plot_start_time = time.time()
    plotter = pv.Plotter(off_screen=True)
    plotter.set_background("white")
    
    first_frame_points = np.vstack(frames[0])
    plotter.add_mesh(pv.PolyData(first_frame_points), color='blue', render_points_as_spheres=True, point_size=5)
    plotter.camera_position = 'xy'
    plotter.camera.elevation = 30
    plotter.camera.zoom(1.2)

    plotter.open_gif("octopus_crawling.gif")

    for frame_data in frames:
        plotter.clear()
        for rod_pos in frame_data:
            plotter.add_mesh(pv.Spline(rod_pos).tube(radius=0.02), color="#0047AB")
        plotter.write_frame()

    plotter.close()
    
    plot_end_time = time.time()
    print(f"Animation saved to octopus_crawling.gif (took {plot_end_time - plot_start_time:.2f} seconds)")


if __name__ == "__main__":
    main()