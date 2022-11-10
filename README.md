# PerceptionProject

### Project goal
The context of this project is a quality control scenario under an Industry 4.0 perspective. Quality control is performed with a stereo camera fixed in the environment overlooking at a conveyor. The objects on the conveyor need to be detected, tracked in 3D, even if occlusion happen and need to be classified in their object class. 

### Methods

First, background subtraction is used to detect moving parts of the image. Next, the biggest possible contour is found on the processed image that should correspond to the moving object. After that a rectangle can be drawn around the contour. Next, the disparity map is calculated using stereo block matching. The x, y and disparity of the centre of the contour are converted to 3D using a perspective transform and can now be used as the measurement in the Kalman filter update. The classification of the moving objects is done using SIFT feature detection combined with a Bag of Visual Words (BOVW) method.

### Short demo 
![Alt](/demo.gif "Demo")
