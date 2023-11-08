#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 05:15:46 2023

@author: Shiva_roshanravan
"""

from pygltflib import GLTF2
from pygltflib.utils import ImageFormat, Image, Texture, Material

filename = '/home/Shiva_roshanravan/Documents/Flame_test/flame_tools_H/data/flame2020_ebrahimi.glb'
gltf = GLTF2().load(filename)

# Step 1: Find the index of the existing texture you want to replace
# Let's assume the texture you want to replace is at index 1 (you need to replace 1 with the actual index)
existing_texture_index = 0

# Remove the image associated with the texture
# existing_image_index = gltf.materials[0].pbrMetallicRoughness.baseColorTexture["index"]
gltf.images.pop(0)

# Step 3: Add the new image and texture to the GLB
# Create and add a new image to the glTF (same as before)
new_image = Image()
new_image.uri = '/home/Shiva_roshanravan/Documents/FDECA/Test/Ehsan/Ehsan.png'
gltf.images.append(new_image)

# Create a new texture and associate it with the added image
new_texture = Texture()
new_texture.source = len(gltf.images) - 1  # Index of the newly added image
gltf.textures.append(new_texture)

# Step 4: Assign the new texture to the appropriate material(s) or mesh(es)
# Assuming you have a mesh/primitive that was using the old texture and you want to apply the new texture to it, you need to set the material index for that mesh/primitive.
# Replace 0 with the actual index of the mesh/primitive you want to update.
gltf.meshes[0].primitives[0].material = len(gltf.materials) - 1

# Now you can convert the images to data URIs and save the updated GLB
gltf.convert_images(ImageFormat.DATAURI)


#%%
import numpy as np

shape_path = '/home/Shiva_roshanravan/Documents/FDECA/Test/Ehsan/identity.npy'
flame_params = np.squeeze(np.load(shape_path))
flame_params = flame_params.astype(np.float64)
mesh = gltf.meshes[0]
for idx, key_block in enumerate(mesh.extras['targetNames']):
    # if key_block.name[:5] == "Shape":
    if key_block[:5] == "Shape":
        print(key_block)
        if idx < 100:
           gltf.meshes[0].weights[idx] = flame_params[idx]

filename2 = '/home/Shiva_roshanravan/Documents/Flame_test/flame_tools_H/data/flame2020_rr.glb'
gltf.save(filename2)
