#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 05:15:46 2023

@author: Shiva_roshanravan
"""

from pygltflib import GLTF2
from pygltflib.utils import ImageFormat, Image, Texture, Material
import numpy as np


def load_glb(path):
    gltf = GLTF2().load(path)
    return gltf

def change_texture(gltf, path):
    # Step 1: Find the index of the existing texture you want to replace
    # Let's assume the texture you want to replace is at index 1 (you need to replace 1 with the actual index)
    existing_texture_index = 0
    
    # Remove the image associated with the texture
    # existing_image_index = gltf.materials[0].pbrMetallicRoughness.baseColorTexture["index"]
    gltf.images.pop(0)
    
    # Step 3: Add the new image and texture to the GLB
    # Create and add a new image to the glTF (same as before)
    new_image = Image()
    new_image.uri = path
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
    return gltf

def load_flame_shape(path):
    flame_params = np.load(path)
    flame_params = flame_params.astype(np.float64)
    return flame_params
    
def add_flame_shape(gltf, flame_params):
    mesh = gltf.meshes[0]
    for idx, key_block in enumerate(mesh.extras['targetNames']):
        if key_block[:5] == "Shape":
            if idx < 100:
               gltf.meshes[0].weights[idx] = flame_params[idx]   
    return gltf

def save_gltf(gltf, path):
    gltf.save(path)            
    print(f'glb saved at: {path}')

def create_glb(glb_path, uv_path, flame_params, save_path):
    gltf = load_glb(glb_path)
    gltf = change_texture(gltf,uv_path)
    gltf = add_flame_shape(gltf, flame_params)
    save_gltf(gltf,save_path)
