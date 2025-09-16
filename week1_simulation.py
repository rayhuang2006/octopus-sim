import numpy as np
from elastica import *

from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces
from elastica.boundary_conditions import OneEndFixedBC
from elastica.dissipation import AnalyticalLinearDamper
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

class BendingRodSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass

def main():
    sim = BendingRodSimulator()

    # parameters
    n_elem = 50
    start_point = np.zeros((3,))
    direction = np.array([1.0, 0.0, 0.0])  # rod沿 x 軸
    normal = np.array([0.0, 1.0, 0.0])
    base_length = 1.0
    base_radius = 0.025
    density = 1000.0
    youngs_modulus = 1e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))

    # create rod
    rod = CosseratRod.straight_rod(
        n_elements=n_elem,
        start=start_point,
        direction=direction,
        normal=normal,
        base_length=base_length,
        base_radius=base_radius,
        density=density,
        youngs_modulus=youngs_modulus,
        shear_modulus=shear_modulus,
    )
    sim.append(rod)

    # gravity
    sim.add_forcing_to(rod).using(GravityForces, acc_gravity=np.array([0.0, -9.81, 0.0]))

    # fix one end: 使用 index-based 的保守寫法
    sim.constrain(rod).using(
        OneEndFixedBC,
        constrained_position_idx=(0,),    # 固定節點 0 的位置
        constrained_director_idx=(0,),    # 固定節點 0 的方向
    )

    # add damping (optional but often stabilizes)
    dt = 1e-5
    try:
        sim.dampen(rod).using(
            AnalyticalLinearDamper, damping_constant=0.2, time_step=dt
        )
    except TypeError:
        # 如果你的版本 API 不接受 _system 或 time_step，嘗試這個 simpler 呼叫
        try:
            sim.dampen(rod).using(AnalyticalLinearDamper, damping_constant=0.2, time_step=dt)
        except Exception:
            # 如果仍失敗，略過阻尼（阻尼不是必要的）
            print("Warning: could not attach AnalyticalLinearDamper with current API; continuing without it.")

    # finalize
    sim.finalize()

    # integrator 設定
    timestepper = PositionVerlet()
    final_time = 5.0
    total_steps = int(final_time / dt)

    print("Running simulation: final_time =", final_time, " dt =", dt, " steps =", total_steps)
    integrate(timestepper, sim, final_time, total_steps)
    print("Simulation finished.")

    # visualization: initial and final state
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8.0, 6.0), dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        # initial positions along rod (generalized by direction)
        positions_along_length = np.linspace(0.0, base_length, n_elem + 1)
        initial_position = np.outer(direction, positions_along_length) + start_point.reshape(3, 1)

        ax.plot(
            initial_position[0, :], initial_position[1, :], initial_position[2, :],
            "r-", label="Initial State",
        )

        final_position = rod.position_collection  # shape (3, n_elem+1)
        ax.plot(
            final_position[0, :], final_position[1, :], final_position[2, :],
            "b-", label="Final State",
        )

        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.legend()
        ax.set_title("Rod Bending Under Gravity (final snapshot)")

        # try to set equal aspect ratio for 3D (works on newer matplotlib)
        try:
            # Normalize box aspect by ranges to get visually balanced axes
            x_range = np.ptp(final_position[0, :]) + 1e-8
            y_range = np.ptp(final_position[1, :]) + 1e-8
            z_range = np.ptp(final_position[2, :]) + 1e-8
            # choose an aspect vector
            aspect = [x_range, y_range, max(z_range, 0.1 * max(x_range, y_range))]
            ax.set_box_aspect([1.0, y_range / x_range if x_range != 0 else 1.0, aspect[2] / x_range])
        except Exception:
            # fallback: do nothing if set_box_aspect isn't supported
            pass

        plt.show()
    except ImportError:
        print("Matplotlib not found; skipping visualization.")

if __name__ == "__main__":
    main()
