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

# Some Theory
Of primary interest is the scattering of pressure waves by object, The diffraction of pressure waves can be approximated by point pressure sources provided the points are close enough together to maintain phase relationships.

This model uses the dispersion of acoustic power over small areas or pixels. The pixels have acoustic power transferred to them by the acoustic source. In turn they broadcast this power as an echo. Intensity $I(W/m^2)$ is proportional to the surface area of the wave front it can be related to pressure
$$
I = p^2 /( \rho \cdot cp )
$$
pressure amplitude $p(F/A)$, density $\rho(kg/m^3)$ and compressional wave speed $cp (m/s)$. The model attempts to instantaneously sample the steady state response of the system to a constant tone. As such power become energy over a very small $dt$. So there an input quantum of energy and that gets dispersed and amounts of energy are relayed through the system. 

The output values of the model are pressure ratios there can be standard input spherical pressure source of $P_i = 1$ Pa @ 1 meter. If the primary medium is fresh water
$$
I = 1^2 / (1000 \cdot 1480) = 1.480e-{-6}
$$
Over an area
$$
P = I \cdot A = 1.480e^{-6} \cdot 4 \cdot \pi \cdot 1^2 = 1.859e^{-05} W = 18.59 mW
$$
where micro watts are used for numerical stability. This power is proportional to fixed amount of energy over $dt$ and is the total input energy. This energy is also proportional to a pressure amplitude over an area.

So the pressure amplitude is determined by wave energy this also includes the attenuation by spreading. The phase and attenuation in the medium is determined by wave equations.

1. The surface area of the incident wave front is determined. (This may be a pixel sphere).
2. The surface area of the transmitted wave front is determined and the ratio determined this sets the attenuation by spreading as the transmitted area will be larger that the incident due.
3. Convert $I$ to $p$ at the incident location.
4. Determine propagation of $p$ using the wave equation and the distance traveled.
5. The ratio scales $I$ then the transmitted area determines $P$ i.e. $E$

Given a starting surface pressure amplitude at $i$ can be defined as
$$
p_i = I_i \cdot \rho \cdot cp
$$
and the intensity at surface $j$, 
$$
I_j = I_i \cdot (A_i/A_j)
$$
Where $A_i$ is the area of surface $i$ and $A_j$ at surface $j$. Leading to
$$
p_j = I_j \cdot \rho \cdot cp = I_i \cdot (A_i/A_j) \cdot \rho \cdot cp
$$
So the pressure rations of 
$$
p_j / p_i = A_i/A_j
$$
so the pressure amplitudes due to spreading is proportion the ratio of surface areas.

The distance between a  point on surface $A$ and $B$,
$$
R(\vec{r}_A, \vec{r}_B) = \left| \vec{r}_A - \vec{r}_B \right| = \sqrt{(x_A - x_B)^2 + (y_A - y_B)^2 + (z_A - z_B)^2}
$$
The double integral 
$$
I = \iint_{A, B} f(\vec{r}_A) \, g(\vec{r}_B) \cdot \frac{e^{i k R(\vec{r}_A, \vec{r}_B)}}{R(\vec{r}_A, \vec{r}_B)} \, dS_A \, dS_B
$$
If the areas are small compared to the wavelength and $R(\vec{r}_A, \vec{r}_B)$ which is attenuation due to spherical spreading is replace by the area ratio. Also the incident pressure is constant for a small area $f(\vec{r}_A) = p_i$ and for simplicity assume fully reflective objects $g(\vec{r}_B) = 1$ the equation becomes
$$
I_j = p_i \cdot \frac{A_i \cdot e^{i k R(\vec{r}_A, \vec{r}_B)}}{A_j}
$$
and
$$
p_j = I_j \cdot \rho \cdot cp
$$

