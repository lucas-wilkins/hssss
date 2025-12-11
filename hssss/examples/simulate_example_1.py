""" Basic Example """
from acquisition.acquisition_system import write_acquisition_results, SegmentationParameters
from acquisition.binary_encoder import BinaryEncoder
from acquisition.grid_binary_encoder import GridBinaryEncoder
from acquisition.simulated import SimulatedAcquisitionSystem
from acquisition.triangulation import triangulate
from camera import Camera

if __name__ == "__main__":
    camera_1 = Camera(
        position=(-3, 1, 0),
        look_at=(0,0,0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=45)

    camera_2 = Camera(
        position=(-3, -1, 0),
        look_at=(0, 0, 0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=45)

    projector = Camera(
        position=(-10, 0, 0),
        look_at=(0,0,0),
        up=(0, 0, 1),
        horizontal_pixels=800,
        vertical_pixels=600,
        fov_deg=8)

    system = SimulatedAcquisitionSystem(camera_1, camera_2, projector,"../monkey.stl")


    encoder = GridBinaryEncoder(800, 600, 64, 64, 2)

    # data = system.run_scan(encoder, show_process=True)
    data = system.run_scan(encoder)

    assignment = encoder.assign_locations(data)

    # Segmentation
    segmentation_parameters = SegmentationParameters(
                                minimal_contrast=10,
                                minimal_quality=0.99)

    segmented = assignment.segment(segmentation_parameters)

    # Triangulation

    triangulation_data = triangulate(segmented, camera_1, camera_2)

    # Write the data

    write_acquisition_results(data, "simulate_example_1_data")



    # Plot some of the details

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(assignment.camera_1.colourised(segmentation_parameters))

    plt.subplot(1, 2, 2)
    plt.imshow(assignment.camera_2.colourised(segmentation_parameters))

    plt.figure()
    assignment.camera_1.show_raw()

    plt.figure()
    triangulation_data.plot_points()

    plt.show()


