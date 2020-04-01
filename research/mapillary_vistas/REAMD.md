# Mapillary Vistas Dataset 

A diverse street-level imagery dataset with pixel‑accurate and instance‑specific human annotations for understanding street scenes around the world.

## Features
- 25,000 high-resolution images
- 152 object categories
- 100 instance-specifically annotated categories
- Global reach, covering 6 continents
- Variety of weather, season, time of day, camera, and viewpoint

## Dataset description

The public set comprises `20,000 images`, out of which `18,000` shall be used for `training` and the remaining `2,000` for `validation`.

The official test set now contains 5,000 RGB images. We provide pixel-wise labels based on polygon annotations for `66 object classes`, where 37 are annotated in an instance-specific manner (i.e. individual instances are labeled separately).

The folder structures contain:
- raw RGB images ({training,validation}/images)
- class-specific labels for semantic segmentation (8-bit with color-palette) ({training,validation}/labels)
- instance-specific annotations (16-bit) ({training,validation}/instances) and panoptic annotations (24-bit RGB images, {training,validation}/panoptic)

**Note**: Please run 'python demo.py' from the extracted folder to get an idea about how to access label information and for mappings between label IDs and category names.

Please cite the following paper if you find `Mapillary Vistas` helpful for your work:

    @InProceedings{MVD2017,
    title=    {The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes},
    author=   {Neuhold, Gerhard and Ollmann, Tobias and Rota Bul\`o, Samuel and Kontschieder, Peter},
    booktitle={International Conference on Computer Vision (ICCV)},
    year=     {2017},
    url=      {https://www.mapillary.com/dataset/vistas}
    }


## References

[Mapillary Vistas Dataset](https://www.mapillary.com/dataset/vistas)