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
You can visit the awesome [Mixamo](https://www.mixamo.com/#/?page=1&type=Character) website and find the character you like, or instead use the default ``SMPL-to-FBX/ybot.fbx``

### Run the Converter.
```
python SMPL-to-FBX/Convert.py --input_dir fk_out --output_dir SMPL-to-FBX/fbx_out
```

As the output ``.fbx`` file is in ASCII format, which is not well supported in Blender, use [FBXConverterUI](https://aps.autodesk.com/developer/overview/fbx-converter-archives) to convert it into Binary format, then you can directly import it to Blender and do all the rendering.


### Blender Rendering
* Adjust camera/main obj position.
* Set all the output properties.
* Render to images first, then render images to a video sequence.

<image src="./blender_workflow.jpg" width="720px" />