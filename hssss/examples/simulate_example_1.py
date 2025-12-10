""" Basic Example """
from acquisition.acquisition_system import write_acquisition_results, SegmentationParameters
from acquisition.binary_encoder import BinaryEncoder
from acquisition.simulated import SimulatedAcquisitionSystem
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
        fov_deg=5)

    system = SimulatedAcquisitionSystem(camera_1, camera_2, projector,"../monkey.stl")


    encoder = BinaryEncoder(800, 600, 64, 64)

    # data = system.run_scan(encoder, show_process=True)
    data = system.run_scan(encoder)

    write_acquisition_results(data, "simulate_example_1_data")

    assignment = encoder.assign_locations(data)


    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(assignment.camera_1.colourised(SegmentationParameters()))

    plt.figure()
    assignment.camera_1.show_raw()


    plt.show()


