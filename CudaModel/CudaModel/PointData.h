#ifndef _PointData
#define _PointData

#define ANSI

#include <stdio.h>
#include <math.h>

#ifndef _DCOMPLEX
#define _DCOMPLEX
typedef struct DCOMPLEX {float r,i;} dcomplex;
#endif

class PointData
{
public:
	unsigned int ObjectNum;
	float3 PointLocation;
	dcomplex Pressure;
	bool isSourcePoint;


	PointData(rapidxml::xml_node<char>* PointNode,bool t_isSourcePoint, unsigned int t_ObjectNum)
	{
		ObjectNum = t_ObjectNum;
		isSourcePoint = t_isSourcePoint;

		PointLocation.x = std::stof(PointNode->first_attribute("X")->value());
		PointLocation.y = std::stof(PointNode->first_attribute("Y")->value());
		PointLocation.z = std::stof(PointNode->first_attribute("Z")->value());

		if( isSourcePoint )
		{
			Pressure.r = std::stof(PointNode->first_attribute("PressureReal")->value());
			Pressure.i = std::stof(PointNode->first_attribute("PressureImag")->value());
		}
		else
		{
			Pressure.r = 0;
			Pressure.i = 0;
		}
	}
};



#endif