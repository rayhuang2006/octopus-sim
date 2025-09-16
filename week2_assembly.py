# week2_assembly.py
# Final version with the correct, albeit unusual, constraint syntax.

import numpy as np
from elastica import *

# 匯入我們已知的、所有需要的核心工具
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import GravityForces
from elastica.boundary_conditions import OneEndFixedBC
from elastica.dissipation import AnalyticalLinearDamper
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate

# --------------------------
# 1. 定義模擬「世界」藍圖
class OctopusSimulator(BaseSystemCollection, Constraints, Forcing, Damping, CallBacks):
    pass

# --------------------------
def main():
    sim = OctopusSimulator()
    rods = build_octopus(sim)

    sim.finalize()
    timestepper = PositionVerlet()
    
    final_time = 2.0
    dt = 1e-5
    total_steps = int(final_time / dt)
    
    print("Running final core simulation...")
    integrate(timestepper, sim, final_time, total_steps)
    print("Simulation finished.")
    
    plot_octopus(rods)


def build_octopus(sim):
    n_elem = 20
    arm_length = 1.0
    arm_radius = 0.025
    density = 1000.0
    youngs_modulus = 1e6
    poisson_ratio = 0.5
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    num_arms = 8
    
    print(f"Building an octopus with {num_arms} arms...")
    rods_list = []

    for i in range(num_arms):
        angle = i * (2 * np.pi / num_arms)
        start_direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
        start_point = np.zeros((3,))
        normal = np.array([0.0, 1.0, 0.0])

        rod = CosseratRod.straight_rod(
            n_elements=n_elem, start=start_point, direction=start_direction,
            normal=normal, base_length=arm_length, base_radius=arm_radius,
            density=density, youngs_modulus=youngs_modulus, shear_modulus=shear_modulus,
        )
        sim.append(rod)
        rods_list.append(rod)
        
        sim.add_forcing_to(rod).using(GravityForces, acc_gravity=np.array([0.0, -9.81, 0.0]))
        sim.dampen(rod).using(AnalyticalLinearDamper, damping_constant=0.5, time_step=1e-5)
        
        # 【最終修正】使用同時包含位置參數和關鍵字參數的語法
        # 1. 先取得要固定的數值
        fixed_position = rod.position_collection[..., 0].copy()
        fixed_directors = rod.director_collection[..., 0].copy()
        
        # 2. 在 .using() 中，同時傳入 (類別, 位置參數...) 和 (關鍵字參數=...)
        sim.constrain(rod).using(
            OneEndFixedBC,           # 類別
            fixed_position,          # 位置參數 1
            fixed_directors,         # 位置參數 2
            positions=(0,),          # 關鍵字參數 1
            directors=(0,),          # 關鍵字參數 2
        )
    
    print("Octopus assembly complete.")
    return rods_list


def plot_octopus(rods):
    try:
        import matplotlib.pyplot as plt
        print("Visualizing results...")
        fig = plt.figure(figsize=(8.0, 6.0), frameon=True, dpi=150)
        ax = fig.add_subplot(111, projection="3d")

        for i, rod in enumerate(rods):
            initial_label = "Initial State" if i == 0 else None
            final_label = "Final State" if i == 0 else None
            
            angle = i * (2 * np.pi / len(rods))
            direction = np.array([np.cos(angle), 0.0, np.sin(angle)])
            length = rod.rest_lengths.sum()
            n_elem = rod.n_elems
            start_point = np.array([0.0, 0.0, 0.0])
            positions_along_length = np.linspace(0.0, length, n_elem + 1)
            initial_pos = start_point[:, np.newaxis] + np.outer(direction, positions_along_length)
            
            ax.plot(initial_pos[0, :], initial_pos[1, :], initial_pos[2, :], "r-", label=initial_label)

            final_pos = rod.position_collection
            ax.plot(final_pos[0, :], final_pos[1, :], final_pos[2, :], "b-", label=final_label)

        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.legend()
        ax.set_title("Lifeless Octopus Skeleton Under Gravity")
        
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])
        ax.set_box_aspect((1, 1, 1))
        
        plt.show()

    except ImportError:
        print("Matplotlib not found.")

if __name__ == "__main__":
    main()