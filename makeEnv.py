from isaacgym import gymapi
import random

gym = gymapi.acquire_gym()


sim_params = gymapi.SimParams()
# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0


sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)#set the second variable to -1 if you don't want viewer mode


# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

asset_root = "."
wall_asset_file = "urdf/wall.urdf"
wall_asset = gym.load_asset(sim, asset_root, wall_asset_file)
box_asset_file = "urdf/box.urdf"
box_asset = gym.load_asset(sim, asset_root, box_asset_file)

# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 3
num_obstacles = 10
env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    x = random.uniform(-2.0 , 2.0)
    y = random.uniform(-2.0 , 2.0)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, -2, 0.5)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.)
    actor_handle = gym.create_actor(env, wall_asset, pose, "Wall1_" + str(i), i, 1)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 2, 0.5)
    pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.)
    actor_handle = gym.create_actor(env, wall_asset, pose, "Wall2_" + str(i), i, 1)


    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, 0.5)
    pose.p = gymapi.Vec3(-2, 0, 0.5)
    pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    actor_handle = gym.create_actor(env, wall_asset, pose, "Wall3_" + str(i), i, 1)
    #
    #
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(x, y, 0.5)
    pose.p = gymapi.Vec3(2, 0, 0.5)
    pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
    actor_handle = gym.create_actor(env, wall_asset, pose, "Wall4_" + str(i), i, 1)

    for j in range(num_obstacles):
        pose = gymapi.Transform()
        x = random.uniform(-1.8, 1.8)
        y = random.uniform(-1.8, 1.8)
        pose.p = gymapi.Vec3(x, y, 0.1)
        pose.r = gymapi.Quat(0.0, 0.0, 0, 1.0)
        actor_handle = gym.create_actor(env, box_asset, pose, "box" + str(j) + "_" + str(i), i, 1)


    actor_handles.append(actor_handle)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

gym.step_graphics(sim)
gym.draw_viewer(viewer, sim, True)

gym.sync_frame_time(sim)


while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)