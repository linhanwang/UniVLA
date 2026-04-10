# # FROM https://github.com/simpler-env/SimplerEnv?tab=readme-ov-file#installation
import site
site.main()
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
import mediapy
import sapien.core as sapien

task_name = "google_robot_pick_coke_can"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

if "env" in locals():
    print("Closing existing env")
    env.close()
    del env
sapien.render_config.camera_shader_dir = "ibl"
sapien.render_config.rt_use_denoiser = False
env = simpler_env.make(task_name)
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

frames = []
done, truncated = False, False
step_idx = 0
while not (done or truncated):
    print(f"step {step_idx}: get_image", flush=True)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    print(f"step {step_idx}: sample action", flush=True)
    action = env.action_space.sample()
    print(f"step {step_idx}: env.step", flush=True)
    obs, reward, done, truncated, info = env.step(action)
    frames.append(image)
    step_idx += 1
print(f"loop exited after {step_idx} steps, done={done}, truncated={truncated}", flush=True)

episode_stats = info.get("episode_stats", {})
print("Episode stats", episode_stats)
import os
out_path = os.path.abspath("env_test.mp4")
print(f"Writing {len(frames)} frames to {out_path}")
mediapy.write_video(out_path, frames, fps=10)
print("Done")
