import modules.api as api
import modules.mh as mh
import modules.geo as geo

import numpy as np 
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math


def render_cube(frequency):
    """Render a cube with the given frequency."""
    cp = 1480.0
    #for frequency in range(1e3, 30e3, 500):
    file_name = f"cube_{int(frequency)}_Hz"

    test = False

    target = geo.make_cube()
    target = geo.translate_stl_object(target, [0, 0.1, 0])
    #target = geo.translate_stl_object(target, [-4, 0, -10])

    # Many facets increases the computation speed.
    for i in range(4):
        target = geo.halve_facets(target)

    ref_length = 50

    #field_surface = geo.make_rectangle(10,10, False)
    field_surface, viewSettings,  = mh.setup_iso_scene(ref_length)


    # Sometimes smaller facets are required as very large facets do not render.
    for i in range(3):
        field_surface = geo.halve_facets(field_surface)




    t = 15
    source_pnts=[[4*t,3*t,9*t]]
    angles = np.linspace(-180, 180, 361, endpoint=False)
    field_pnts= geo.generate_field_points(t, angles)

    api.load_points_to_cuda(source_pnts, isSource=True)
    api.load_points_to_cuda(field_pnts, isSource=False)

    # Maintain a minimum delta for scatter calculation.
    # The feild surface delta has large effect on computation time although
    # aliasing can occur if it is too large. The only effects the image if you zoom in.
    delta = (cp / frequency) / 7.5
    delta_s = min(delta, 50e-3)  # Ensure a minimum distance for source points.
    print("delta = ", delta_s)

    att = mh.seawater_attenuation_db_per_m(frequency/1000)

    api.set_initial_conditions(cp, frequency, att)
    api.load_stl_mesh_to_cuda(target, 0, delta_s) # 0 is for target object.
    api.load_stl_mesh_to_cuda(field_surface, 2, delta_s) # 1 is for field surface.

    mh.run_rendering(0, test=test)

    # field_vals = api.GetFieldPoints(len(field_pnts))
    # modeled_TES = mh.results_to_TES(field_vals, t)

    # index = np.where(angles == 0)[0][0] 
    # print("Modelled with Attenuation = ", db_values[index])
    # print('Analytical TES = ', TES)
    # print('Modelled TES = ', db_values[index] + 2* atten)


    # db_values  + 40*np.log10(Radius)    
    # plt.figure()
    # plt.plot(angles, modeled_TES, label="Field Values (dB)")
    # #plt.plot(angles, bistatic_TES, label="Analytic (dB)")
    # plt.xlabel("Angle (degrees)")
    # plt.ylabel("Field Value (dB)")
    # plt.title("Field Values vs. Angle")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    mh.render_to_file(viewSettings, file_name=file_name, test=test)
    api.TearDownCuda()


for frequency in range(1678, 30005, 1):
    print(f"Rendering cube at frequency: {frequency} Hz")
    render_cube(float(frequency))