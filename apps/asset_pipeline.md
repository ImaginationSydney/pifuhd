# AWS Asset Pipeline

The `asset_pipeline` app is intended to prep incoming speaker videos for use in Unreal.

As input it takes a video file of a speaker, shot in portrait at 4K (3840 x 2160), with the speaker's head at the left of the frame.

The output is:
```
<Root Directory>
  ├ SourceVideo.mov     - Input video
  ├ audio.wav           - Audio track gets stripped out of input video
  ├ img                 - Frame images generated from input video
  │  ├ frame_0001.jpg
  │  ├ <...>
  │  └ frame_0600.jpg
  ├ meshes              - 3D meshes generated from frame images
  │  ├ frame_0001.obj
  │  ├ <...>
  │  └ frame_0600.obj
  ├ depth_front         - Depth maps of speaker from front (from 3D meshes)
  │  ├ frame_0001.obj
  │  ├ <...>
  │  └ frame_0600.obj
  └ depth_back          - Depth maps of speaker from back (from 3D meshes)
     ├ frame_0001.obj
     ├ <...>
     └ frame_0600.obj
```

To run the pipeline, create an empty folder, copy the input video into the folder, and run the following command from the root of this repository.

```sh
conda activate pifuhd
python -m apps.asset_pipeline --video <abs path to input video> --size 1200 --resolution 256
```

## Options

`-v / --video` (required) - Path to the input video.

`-s / --size` (default = 512) - Determines the width & height of the resulting depth-maps, and it must be smaller than the screen size of the monitor attached (in both directions). If it exceeds this size the results depth maps will be distorted.

`-r / --resolution` (default = 512) - This argument determines the detail used when generating the 3D meshes. It dramatically affects the time it takes the pipeline to run, and the size of the resulting meshes. It is recommended to reduce this to 256 for longer exports. Increasing this doesn't appear to result in better output.

`-c / --ckpt_path` (default = './checkpoints/pifuhd.pt') - Path to the pifuhd checkpoint (see main readme for more info).

`-op / --openpose_dir` (default = '../openpose') - Path to the openpose library (see main readme for more info).

`-f / --frames` - Allows limiting the amount of frames are taken from the input video, mostly useful for testing. The whole video will be processed if not provided. Note that the generated audio file doesn't respect this limit currently.

`-tr / --transpose` (default = 1) - Rotates input video, see ffmpeg docs for [more info](http://underpop.online.fr/f/ffmpeg/help/transpose.htm.gz).

`-cr / --crop` (default = 1) - Crops input video (after rotate), can be passed in `w:h:x:y` format, for example `--crop 1570:3166:272:296`. See ffmpeg docs for [more info](http://underpop.online.fr/f/ffmpeg/help/crop.htm.gz).

## Steps
The pipeline goes through the following steps:
1. Break video into frame images (ffmpeg)
2. Extract audio from video as wav (ffmpeg)
3. Run openpose across all video images to generate pose json (see batch_openpose)
4. Run pifuhd across all image/json files to generate meshes (see simple_test)
5. Load each mesh and work out the vertex bounds for scaling
6. Load each mesh and work out the depth range across all (front and back)
7. Load each mesh and project depth map using range (front and back) 

Due to this being a very lengthy process, the pipeline will attempt to skip steps that appear to already be completed. It does this mainly by checking if the expected output files of a step already exist in disk. If a step is run, all subsequent steps will also be run.

This means that if the input video is replaced, you should also delete all of the other files in the folder before re-running the pipeline.

## Installing

Follow the instructions in the main repo.

This video might also help:
https://www.youtube.com/watch?v=zzgCoyYyuN0

Openpose should be added as a sibling directory to pifuhd (or override `openpose_dir`).

Installing `ffmpeg` on the path is required.

Installing nvidia Cuda, and then cuDNN will dramatically improve openpose performance. 