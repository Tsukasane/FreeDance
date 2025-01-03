## SMPL-to-FBX Installation

### Download and Install [fbx-SDK](https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3)
```
cd SMPL-to-FBX
mkdir -p fbx-sdk/install
tar -zxvf fbx202032_fbxpythonsdk_linux.tar.gz -C fbx-sdk
fbx-sdk/fbx202032_fbxpythonsdk_linux fbx-sdk/install 
```

### Create Environment
The archived SDK file only supports python==3.7, so create a new env.

```
conda create -n fbx_env python=3.7 -c conda-forge
conda activate fbx_env
pip install numpy tqdm scipy
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
```

Then copy the SDK dependencies to the new env.
```
cp fbx-sdk/install/lib/Python37_x64/* <your_path>/miniconda3/envs/fbx_env/lib/python3.7/site-packages/
```

### Pick a Character 
* You can use the default ``SMPL-to-FBX/characters/ybot.fbx`` provided by [EDGE](https://github.com/Stanford-TML/EDGE), or the official [SMPL fbx files](https://smpl.is.tue.mpg.de/) (download and put it under ``SMPL-to-FBX/characters/``).

* If you want to customize the character, it is a good choice to visit [Mixamo](https://www.mixamo.com/#/?page=1&type=Character) website, then use [rokoko blender add on](https://www.rokoko.com/integrations/blender) to retarget the animation, following this [tutorial](https://support.rokoko.com/hc/en-us/articles/4410463481489-Retarget-an-animation-in-Blender).

### Run the Converter
```
python SMPL-to-FBX/Convert.py --input_dir fk_out --output_dir SMPL-to-FBX/fbx_out
```

As the output ``.fbx`` file is in ASCII format, which is not well supported in Blender, use [FBXConverterUI](https://aps.autodesk.com/developer/overview/fbx-converter-archives) to convert it into Binary format, then you can directly import it to Blender and do all the rendering.


### Blender Rendering
* Adjust camera position.
* Set all the output properties.
* Render to images first, then render images to a video sequence.

<image src="./blender_workflow.jpg" width="720px" />