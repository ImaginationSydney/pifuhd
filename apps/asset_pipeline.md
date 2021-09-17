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
python -m apps.asset_pipeline --video <abs path to input video> --size 1200
```

## Options

`-v / --video` (required) - Path to the input video.

`-s / --size` (default = 512) - Determines the width & height of the resulting depth-maps, and it must be smaller than the screen size of the monitor attached (in both directions). If it exceeds this size the results depth maps will be distorted.

`-r / --resolution` (default = 512) - This argument determines the detail used when generating the 3D meshes. It dramatically affects the time it takes the pipeline to run, and the size of the resulting meshes. It is recommended to reduce this to 256 for longer exports. Increasing this doesn't appear to result in better output.

`-c / --ckpt_path` (default = './checkpoints/pifuhd.pt') - Path to the pifuhd checkpoint (see main readme for more info).

`-op / --openpose_dir` (default = '../openpose') - Path to the openpose library (see main readme for more info).

`-f / --frames` - Allows limiting the amount of frames are taken from the input video, mostly useful for testing. The whole video will be processed if not provided. Note that the generated audio file doesn't respect this limit currently.


## Installing

Follow the instructions in the main repo.

This video might also help:
https://www.youtube.com/watch?v=zzgCoyYyuN0

Openpose should be added as a sibling directory to pifuhd (or override `openpose_dir`).

Installing `ffmpeg` on the path is required.

Installing nvidia Cuda, and then cuDNN will dramatically improve openpose performance. 