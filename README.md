# Plant-health-analysis-in-a-simulation-env
This project performs leaf masking on an image added to the simulation environment using a previously fine-tuned YOLOv8 model. Along with this masking process, hyperspectral band integration is used to enable plant health analysis, and a user-friendly interface is designed to manage these tasks.

To run the project, ROS2 Humble, Ignition Fortress, and the Clearpath Husky robot must be installed.

Installation Links:
ROS2 Humble:
[(https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
Ignition Fortress and Clearpath Husky:[(https://github.com/clearpathrobotics/clearpath_simulator/tree/humble?tab=readme-ov-file)](https://github.com/clearpathrobotics/clearpath_simulator/tree/humble?tab=readme-ov-file)

Running the Project:First, inside the clearpath_ws workspace you installed, go to the clearpath_gz folder and locate the World directory containing the available environments. Add the world/warehouse2.sdf file and the images/myimg.png file from this project into that directory and save.

Launching the Environment and System Integration:
In the terminal, run:

`ros2 launch clearpath_gz simulation.launch.py world: <your world path>` 

(Note: When adding the SDF file path, only the file name should be written—there is no need to include the .sdf extension.)

After the system starts running, open VS Code and start the project with:

`   python3 main.py   `

The project will then start, and the system will be fully operational.
