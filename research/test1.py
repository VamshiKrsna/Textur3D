# this script leverages open3d and midas
import torch
import cv2
import numpy as np
import open3d as o3d
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import matplotlib.pyplot as plt

# loading midas models
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
model.eval()

image = cv2.imread("naruto_nobg.png")
inputs = feature_extractor(images=image, return_tensors="pt")

# depth estimation 
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth[0].numpy()

# point cloud
height, width = depth.shape
xx, yy = np.meshgrid(np.arange(width), np.arange(height))
x = xx.flatten()
y = yy.flatten()
z = depth.flatten()
points = np.vstack((x, y, z)).T

# open3d point cloud conversion
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# mesh creation 
pcd.estimate_normals()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

o3d.io.write_triangle_mesh("naruto_mesh.obj", mesh)

# plotting in 3d : 
vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                triangles=faces, cmap='viridis', edgecolor='none')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()