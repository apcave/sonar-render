import numpy as np

a = 3.0
b = 2.0
A = a*b
cp = 1480.0
frequency = 2.0e3
Radius = 50.0

wavelength = cp / frequency
A = a*b
TES = 10*np.log10((4*np.pi*A**2)/(wavelength**2))
TL = 2 * 20 * np.log10(Radius)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print('Target Echo Strength = ', TES)
print('Transmission Loss = ', TL)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")

# Using the Cross-Sectional Area
CrossSection = ((4*np.pi)*(A**2))/(wavelength**2)
print('Cross Section = ', CrossSection)
PressureRatio = (CrossSection/ (4*np.pi*Radius**2))**0.5
print('Pressure Ratio = ', PressureRatio)
PressureRatio_db = 20*np.log10(PressureRatio)
print('Pressure Ratio (dB) = ', PressureRatio_db)
TransmissionLoss_db = 20*np.log10((1/(4*np.pi*Radius**2)))
print('Transmission Loss = ', TransmissionLoss_db)
EchoRatio_db = PressureRatio_db + TransmissionLoss_db
EchoRatio = PressureRatio/(4*np.pi*Radius**2)
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")
print( "This is close to the modelled value.")
print('Echo Ratio = ', 20*np.log10(EchoRatio))
print("<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>")

wavelength = cp / frequency
A = a*b
TES = 10*np.log10(4*np.pi*A**2/wavelength**2)
print('Target Strength = ', TES)
EchoLevel = TES - 2 * 20 * np.log10(Radius)
print('Echo Level = ', EchoLevel)

modelled_TES = -104.4 + 40*np.log10(Radius)

modelled_TES = 20*np.log10(EchoRatio) + 20*np.log10(4*np.pi*Radius**2)
print('Modelled Target Strength = ', modelled_TES)

denstity = 1000
cp = 1480

# Power remains constant.
P_1 = 1
I_1 = P_1**2/(denstity*cp)
A_1 = (4*np.pi*1**2)
Pow_1 = A_1*I_1
# Power remains constant.
print("Power @ 1 meter: ", Pow_1)

P_50 = 1/50