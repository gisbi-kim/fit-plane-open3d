import numpy as np 
import open3d as o3d 

# load file 
filename = '000094' # kitti odometry dataset seq 00's 94th frame 
fileformat = '.ply'
pcd = o3d.io.read_point_cloud(filename + fileformat)
print(pcd)

# downsize, 0.05 is proper for an outdoor scan 
print("Downsample the point cloud with a voxel")
downpcd = pcd.voxel_down_sample(voxel_size=0.01)
o3d.visualization.draw_geometries([downpcd])

# calc normal (optional for vivid visualization) 
print("Recompute the normal of the downsampled point cloud")
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.1, max_nn=30))

# denoise the point cloud 
print("showing noise removed points")
cl, denoised_ind = downpcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=2.0)
denoised_cloud = downpcd.select_by_index(denoised_ind)
noise_cloud = downpcd.select_by_index(denoised_ind, invert=True)
noise_cloud.paint_uniform_color([1.0, 0, 0])
o3d.visualization.draw_geometries([denoised_cloud, noise_cloud])

# fit plane 
pcd = denoised_cloud
plane_model, inliers = pcd.segment_plane(distance_threshold=0.2,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# viz 
plane_cloud = pcd.select_by_index(inliers)
plane_cloud.paint_uniform_color([1.0, 0, 0])

noneplane_cloud = pcd.select_by_index(inliers, invert=True)
noneplane_cloud.paint_uniform_color([0, 0, 1.0])

o3d.visualization.draw_geometries([plane_cloud, noneplane_cloud])

# save the plane/non-plane points 
o3d.io.write_point_cloud(filename + "_plane" + fileformat, plane_cloud)
o3d.io.write_point_cloud(filename + "_nonplane" + fileformat, noneplane_cloud)

