# hssss: Hackable Structured-Light Scanning System Software

*hssss* produces 3D point clouds from a system with two cameras 
and a projector. It is designed to be flexible and hackable.

## Working Principle

The 3D position of points on a surface are calculated by 
triangulation through two cameras. A projector is used to
identify points on the surface for this triangulation, which 
avoids the need to rely on identifying consistent image features. 

## Steps

1) **Encode** positional ID into images to project
2) **Acquire** camera images
3) **Assign** position ID to pixels images
4) **Segment** images into good regions
5) **Triangulate** the locations for each ID and produce a point cloud

Further steps can be applied to merge multiple point clouds.


## Structure

* The `Encoder` class is an interface specifying methods 
for generating images for the projector, and for assigning
regions. This class has some implementations in the package,
but it can be subclassed however one wants.

* The `AcquisitionSystem` comprises multiple cameras and 
its subclasses provide interactions with hardware, or
in the case of `Simulated` is a model.

* Various dataclasses in `acquisition_system` decribe the 
inputs/output of different processing steps, and are fairly 
self-explanatory.

* The `Camera` class provides geometric data for cameras
and the projector (though the projector is only needed
for )

* A series of images for a run can be written out using
`write_acquisition_results`.

See examples for usage.


## Simulations

*hssss* contains a subclass of `AcquisitionSystem` than can load
`.stl` files and render a simulation as if it was taken by a
real system. Different shaders can be set to test different
conditions.


## Debugging / Visualisation

A lot of hooks for messing around with systems can be found in 
the `if __name__ == "__main__"` section of individual python files.

### Acquisition/Segmentation


Colourised images for visualising the segmention part can be  
`AcquisitionResult.camera_[1|2].colourised(segmentation_parameters)`
and the raw data can be plotted easily with `AcquisitionResult.camera_[1|2].show_raw()` 

### Simulations

There are a number of different shaders that can be used for 
debugging or visualising different aspects of a simulated SLS scan.

You can try out different shaders by passing parameters to 
`Simulated.show_model()`,
`Simulated.show_with_projection_shader(encoder, index)` and
`Simulated.run_scan(encoder, show_process=True)`.

### Encoders

Encoders can be easily visualised using `Encoder.image_reel()`,
and the individual imaged can be obtained with `Encoder.get_image(index)`


