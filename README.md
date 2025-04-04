# sonar-render
Software to calculate and visualize the echo response of objects to acoustic waves. This software ins intended for engineering calculations for acoustic environmental management and similar applications.

This project applies the Kirchhoff's diffraction formula for the rendering of echo's from objects. Acoustic rendering has greater computational intensity than light rendering as the wave frequency is far lower. For acoustic waves frequencies of interest are mostly between 10Hz and 150Khz pressure waves require a medium and the medium attenuation is proportional to the frequency. Very high frequencies simply do not transit very far.
As the frequencies are low phase needs to be included in the calculations and the waves spread spherically, approximating the wave as rays simply does not work. In graphics rendering light is approximated are rays of three different frequencies reb, blue and green our human perception of sound includes all frequencies from 10 Hz to 16 KHz. 

There are similarities between acoustics and graphics rendering:
* The calculations require a lot of vector mechanics in 3D space. 
* Objects are rendered using triangular meshes.
* Rays are traced as the distance for the source to the receiver is required    for the phase and attenuation calculations.
* Collision detection between rays and facets (1 triangle in the mesh) are required, in graphics this is for shadows and light intensity on the facet. It is similar for acoustic rendering accept the waves diffuse more.

Main software features:
* The target platform is a Linux or Windows workstation with a Nvidia graphics card that is CUDA capable.
* Object geometry is stored the 3D CAD format STL.
* Vector calculations are performed by a NVIDIA graphics acceleration card's Graphics Processing Unit (GPU) utilizing CUDA technology.
* The computationally intensive functions are implemented in C++ with CUDA extensions. The computational modules are joined together using python that provides methods to load models, visualize results in graphs and renders of pressure on geometry.


