# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
import sys
import os
import subprocess
from pathlib import Path

from trimesh.caching import TrackedArray
from .recon import reconWrapper

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.render.mesh import compute_normal
from lib.render.camera import Camera
from lib.render.gl.color_render import ColorRender
import trimesh

import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str, required=True)
parser.add_argument('-s', '--size', type=int, default=512)
parser.add_argument('-r', '--resolution', type=int, default=512)
parser.add_argument('-ls', '--loadSize', type=int, default=1024)
parser.add_argument('-c', '--ckpt_path', type=str, default='./checkpoints/pifuhd.pt')
parser.add_argument('-op', '--openpose_dir', type=str, default='../openpose')
parser.add_argument('-f', '--frames', type=int, default=None)
parser.add_argument('-tr', '--transpose', type=int, default=1)
parser.add_argument('-cr', '--crop', type=str)

args = parser.parse_args()

source_path = os.path.normpath(args.video)
source_file = os.path.basename(source_path)
root_dir = os.path.dirname(source_path)

width = args.size
height = args.size
resolution = str(args.resolution)
loadSize = str(args.loadSize)
frames = args.frames

img_dir = os.path.join(root_dir, "img")
mesh_dir = os.path.join(root_dir, "meshes")
depth_front_dir = os.path.join(root_dir, "depth_front")
depth_back_dir = os.path.join(root_dir, "depth_back")
audio_file = os.path.join(root_dir, "audio.wav")
img_file_template = os.path.join(img_dir, "frame_%04d.jpg")

print("AWS Speaker Asset pipeline")
print("\tVideo source: {}".format(source_path))
print("\tRoot dir: {}".format(root_dir))
print("\tImage folder: {}".format(img_dir))
print("\tMesh folder: {}".format(mesh_dir))
print("\tDepthmap front folder: {}".format(depth_front_dir))
print("\tDepthmap back folder: {}".format(depth_back_dir))

if not os.path.exists(source_path):
    print("Video source not found, aborting")
    exit()

os.makedirs(img_dir, exist_ok=True)
os.makedirs(mesh_dir, exist_ok=True)
os.makedirs(depth_front_dir, exist_ok=True)
os.makedirs(depth_back_dir, exist_ok=True)

    

# Steps
# 1. Break video into frame images (ffmpeg)
# 2. Extract audio from video as wav (ffmpeg)
# 3. Run openpose across all video images to generate pose json (see batch_openpose)
# 4. Run pifuhd across all image/json files to generate meshes (see simple_test)
# 5. Load each mesh and work out the vertex bounds for scaling
# 6. Load each mesh and work out the depth range across all (front and back)
# 7. Load each mesh and project depth map using range (front and back) 

# 0. Count video frames to allow skipping over completed steps
if frames == None:
    print("Counting video frames")
    count_resp = subprocess.check_output("ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {0}".format(source_path))
    frame_count = int(count_resp)
    print("Total video frames: {}".format(frame_count))
else:
    print("Using specified frame count {}".format(frames))
    frame_count = frames


def find_files(dir, ext):
    ret = []
    for (root,dirs,files) in os.walk(dir, topdown=True): 
        for file in files:
            if ext in file:
                ret.append(os.path.join(root, file))
    return ret

# This flag forces all subsequent steps to be done
invalid = False

frame_limit = "" if frames == None else " -vframes {}".format(frames)

# 1. Break video into images (ffmpeg)
img_files = find_files(img_dir, ".jpg")
if len(img_files) != frame_count:
    invalid = True
    print("\n1.Generating images from source video")

    video_filters = ""
    transpose = str(args.transpose)
    if transpose != "" and transpose != "0":
        video_filters += "transpose={}".format(transpose) + ","
    
    if args.crop != None:
        video_filters += "crop={}".format(args.crop)

    if video_filters != "":
        video_filters = '-vf "{2}"'.format(video_filters)

    cmd = 'ffmpeg -y -i "{0}"{1} {2} -q:v 2 {3}'.format(source_path, frame_limit, video_filters, img_file_template )
    print(cmd)
    os.system(cmd)
else:
    print("\n1.{} images found, skipping image creation".format(frame_count))


# 2. Extract audio from video as wav (ffmpeg)
if invalid or not os.path.exists(audio_file):
    print("\n2.Extracting audio from video as wav")
    cmd = 'ffmpeg -y -i "{0}"{1} -q:a 0 -map a {2}'.format(source_path, frame_limit, audio_file)
    print(cmd)
    os.system(cmd)
else:
    print("\n2.Audio found, skipping extraction")

# 3. Run openpose across all video images to generate pose json (see batch_openpose)
json_files = find_files(img_dir, ".json")
if invalid or len(json_files) != frame_count:
    invalid = True
    print("\n3.Running Openpose to extract pose information from video frames")
    openpose_dir = Path(args.openpose_dir).resolve()
    
    cmd = '"{0}/bin/OpenPoseDemo.exe" --image_dir {1} --write_json {2} --render_pose 2 --face --face_render 2 --hand --hand_render 2'.format(openpose_dir, img_dir, img_dir)
    print(cmd)

    subprocess.run(cmd, cwd=openpose_dir)
else:
    print("\n3.Pose data found, skipping openpose")

# 4. Run pifuhd across all image/json files to generate meshes (see simple_test)
obj_files = find_files(mesh_dir, ".obj")
if invalid or len(obj_files) != frame_count:
    invalid = True
    print("\n4.Running pifuhd to generate meshes for video frames")
    start_id = -1
    end_id = -1
    cmd = ['--dataroot', img_dir, '--results_path', mesh_dir, '--short_paths', '1', \
        '--loadSize', loadSize, '--resolution', resolution, '--load_netMR_checkpoint_path', \
        args.ckpt_path,\
        '--start_id', '%d' % start_id, '--end_id', '%d' % end_id]
    print(' '.join(cmd))
    reconWrapper(cmd, False)
else:
    print("\n4.{} meshes found, skipping pifuhd".format(frame_count))

obj_files = find_files(mesh_dir, ".obj")
print("Found Meshes: {}".format(len(obj_files)))


renderer = ColorRender(width, height)
# cam = Camera(width=1.0, height/width)
cam = Camera(1.0, height/width)
cam.ortho_ratio = 1
cam.near = -100
cam.far = 10


angles = [0, math.radians(180)]
dirs = [depth_front_dir, depth_back_dir]
near = 0
far = 1
all_verts = None
meshes = []


# 5. Load each mesh and work out the vertex bounds for scaling
print("\n5. Finding vertice bounds for all models")
for i, obj_path in enumerate(obj_files):
    if not os.path.exists(obj_path):
        continue

    sys.stdout.write("\r\tChecking model bounds ({0}/{1}): {2}".format(i, len(obj_files), obj_path))
    sys.stdout.flush() 

    mesh = trimesh.load(obj_path)
    meshes.append(mesh)

    if i == 0:
        all_verts = mesh.vertices
    else:
        all_verts = np.concatenate((all_verts, mesh.vertices))

print('')

# 6. Load each mesh and work out the depth range across all (front and back)
bbox_max = all_verts.max(0)
bbox_min = all_verts.min(0)
print("All model bounds: {0} - {1}".format(bbox_min, bbox_max))

print("\n6. Finding depth bounds for all models")

for i, obj_path in enumerate(obj_files):
    mesh = meshes[i]

    sys.stdout.write("\r\tChecking model depth ({0}/{1}): {2}".format(i, len(obj_files), obj_path))
    sys.stdout.flush() 

    faces = mesh.faces

    vertices = mesh.vertices
    
    vertices -= 0.5 * (bbox_max + bbox_min)[None,:]
    vertices /= bbox_max[1] - bbox_min[1]
    normals = compute_normal(vertices, faces)
    
    renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces) 

    for j in range(len(angles)):
        rot = angles[j]
        cam.center = np.array([0, 0, 0])
        cam.eye = np.array([2.0*math.sin(rot), 0, 2.0*math.cos(rot)]) + cam.center

        renderer.set_camera(cam)
        renderer.display()
        
        # img = renderer.get_color(0)
        z_range = renderer.get_z_range()
        if (i == 0 and j == 0):
            near = z_range[0]
            far = z_range[1]
        else:
            if near > z_range[0]:
                near = z_range[0]
                
            if far < z_range[1]:
                far = z_range[1]
                
print('')
        
print('Near: {}'.format(near))
print('Far: {}'.format(far))

print("Creating depthmaps for all {} models".format(len(obj_files)))

# 7. Load each mesh and project depth map using range (front and back) 
count = 0
total = len(obj_files) * len(angles)
for i, obj_path in enumerate(obj_files):

    obj_file = os.path.basename(obj_path)
    obj_name = obj_file[:-4]
    mesh = meshes[i]

    vertices = mesh.vertices
    faces = mesh.faces

    normals = compute_normal(vertices, faces)
    
    renderer.set_mesh(vertices, faces, 0.5*normals+0.5, faces)

    for j in range(len(angles)):
        rot = angles[j]
        cam.center = np.array([0, 0, 0])
        cam.eye = np.array([2.0*math.sin(rot), 0, 2.0*math.cos(rot)]) + cam.center

        renderer.set_camera(cam)
        renderer.display()
        
        # img = renderer.get_color(0)
        img = renderer.get_z_value(near, far)
        # img = cv2.cvtColor(img, cv2.COLORMAP_RAINBOW)

        output_path = os.path.join(dirs[j], obj_name + ".png")

        sys.stdout.write("\rWriting {}/{} depthmap: {}".format(count, total, output_path))
        sys.stdout.flush() 

        cv2.imwrite(output_path, 255*img)
        count += 1

print('')

print("\nAll done!!\n")
    