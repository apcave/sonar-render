#include "Facet.hpp"
#include <iostream>
#include <iomanip>
using namespace std;

void Facet::GenerateFacetLimits()
{
	/* Split the facet into two by projecting perpendicular to the longest edge through the peak.
	 * The two new triangles have the same height, a reference point along the base at the cut.
	 */

	float Lengths[3];
	Lengths[0] = GeoMath::LengthBetweenPoints(v1, v2);
	Lengths[1] = GeoMath::LengthBetweenPoints(v1, v3);
	Lengths[2] = GeoMath::LengthBetweenPoints(v2, v3);

	unsigned int MaxIndx = GeoMath::GetMaxIndex(Lengths, 3);

	float3 vLongX;
	float3 vLongY;
	switch (MaxIndx)
	{
	case 0:
		vLongX = GeoMath::MakeVectorSc(v1, v2);
		vLongY = GeoMath::MakeVectorThroughPointPerpToVector(v1, vLongX, v3, pointOnBase);
		BaseLength = Lengths[0];
		Height = GeoMath::GetVectorLength(vLongY);
		BaseLengthNeg = GeoMath::LengthBetweenPoints(v1, pointOnBase);
		BaseLengthPos = GeoMath::LengthBetweenPoints(pointOnBase, v2);
		break;

	case 1:
		vLongX = GeoMath::MakeVectorSc(v1, v3);
		vLongY = GeoMath::MakeVectorThroughPointPerpToVector(v1, vLongX, v2, pointOnBase);
		BaseLength = Lengths[1];
		Height = GeoMath::GetVectorLength(vLongY);
		BaseLengthNeg = GeoMath::LengthBetweenPoints(v1, pointOnBase);
		BaseLengthPos = GeoMath::LengthBetweenPoints(pointOnBase, v3);
		break;

	case 2:
		vLongX = GeoMath::MakeVectorSc(v2, v3);
		vLongY = GeoMath::MakeVectorThroughPointPerpToVector(v2, vLongX, v1, pointOnBase);
		BaseLength = Lengths[2];
		Height = GeoMath::GetVectorLength(vLongY);
		BaseLengthNeg = GeoMath::LengthBetweenPoints(v2, pointOnBase);
		BaseLengthPos = GeoMath::LengthBetweenPoints(pointOnBase, v3);
		break;
	}
	xAxis = GeoMath::MakeUnitVectorSc(vLongX);
	yAxis = GeoMath::MakeUnitVectorSc(vLongY);
	Area = (BaseLength * Height) / 2.0f;

	cout << "Height: " << Height << endl;
	cout << "BaseLength: " << BaseLength << endl;
	cout << "BaseLengthNeg: " << BaseLengthNeg << endl;
	cout << "BaseLengthPos: " << BaseLengthPos << endl;
	cout << "Point on base: " << pointOnBase.x << ", " << pointOnBase.y << ", " << pointOnBase.z << endl;
	cout << "xAxis: " << xAxis.x << ", " << xAxis.y << ", " << xAxis.z << endl;
	cout << "yAxis: " << yAxis.x << ", " << yAxis.y << ", " << yAxis.z << endl;
	cout << "Normal: " << Normal.x << ", " << Normal.y << ", " << Normal.z << endl;
	cout << "v1: " << v1.x << ", " << v1.y << ", " << v1.z << endl;
	cout << "v2: " << v2.x << ", " << v2.y << ", " << v2.z << endl;
	cout << "v3: " << v3.x << ", " << v3.y << ", " << v3.z << endl;
	cout << "vLongX: " << vLongX.x << ", " << vLongX.y << ", " << vLongX.z << endl;
	cout << "vLongY: " << vLongY.x << ", " << vLongY.y << ", " << vLongY.z << endl;
	cout << "-------------------------------------\n";
}

void Facet::CalculateCentroid()
{
	Centroid.x = (v1.x + v2.x + v3.x) / 3.0f;
	Centroid.y = (v1.y + v2.y + v3.y) / 3.0f;
	Centroid.z = (v1.z + v2.z + v3.z) / 3.0f;
}

Facet::Facet(float3 t_v1, float3 t_v2, float3 t_v3)
{
	v1 = t_v1;
	v2 = t_v2;
	v3 = t_v3;
	auto a = GeoMath::MakeVectorSc(v2, v1);
	auto b = GeoMath::MakeVectorSc(v3, v1);
	auto n = GeoMath::CrossProductSc(a, b);
	Normal = GeoMath::MakeUnitVectorSc(n);

	CalculateCentroid();
	GenerateFacetLimits();
}

void Facet::MakePixelData(float pixel_length)
{
	/**
	 * A matrix of pixel areas is created.
	 * Given the two triangles from the GenerateFacetLimits, this matrix in close
	 * twice as large as needed.
	 * Lines are drawn from the bottom left corner of the matrix to the top row where the
	 * apex is and back down to the bottom right corner.
	 * The area of the analytic triangle is overlayed on the matrix where pixels need not be
	 * filled. As not every pixel is full a matrix is needed to hold the area of each pixel.
	 * The pixels are in plane and organized with offsets so later the distance from a pixel
	 * centroid to a point or another pixel can be calculated.
	 **/

	delta = pixel_length;
	NumXpntsNegative = ceil(abs(BaseLengthNeg) / delta);
	int numPos = ceil(abs(BaseLengthPos) / delta);
	NumXpnts = NumXpntsNegative + numPos;
	NumYpnts = ceil(Height / delta);

	// This is a x mid point of the row of pixels
	float *vX = new float[NumXpnts];
	for (int i = 0; NumXpnts > i; i++)
	{
		if (i < NumXpntsNegative)
		{
			// 0 to numNeg-1,
			vX[i] = ((-NumXpntsNegative + i) * delta) + delta / 2;
		}
		else
		{
			// numNeg to numXpnts-1
			vX[i] = ((i - NumXpntsNegative) * delta) + delta / 2;
		}
	}

	// This is a y mid point of the column of pixels
	float *vY = new float[NumYpnts];
	for (int i = 0; NumYpnts > i; i++)
	{
		vY[i] = (i * delta) + delta / 2;
	}

	bool *yxL = new bool[NumYpnts + 1];
	bool *yxR = new bool[NumYpnts + 1];

	PixelArea = new float[NumXpnts * NumYpnts];
	PressureValues = new dcomplex[NumXpnts * NumYpnts];

	for (int i = 0; NumXpnts * NumYpnts > i; i++)
	{
		PressureValues[i].r = 0;
		PressureValues[i].i = 0;
		PixelArea[i] = 0;
	}

	float m1 = Height / (-BaseLengthPos);
	float m2 = Height / (BaseLengthNeg);

	cout << "m1: " << m1 << endl;
	cout << "m2: " << m2 << endl;

	for (int i = 0; NumXpnts > i; i++)
	{
		float x = vX[i];			// Mid x point
		float xL = x - (delta / 2); // Left x point (pixel limits)
		float xR = x + (delta / 2); // Right x point (pixel limits)

		if (x > 0)
		{
			// x is further than the apex. The gradient is negative.

			// For the x - position find the y position at the
			float yL = m1 * xL + Height; // left of the column
			float yR = m1 * xR + Height; // right of the column

			// Mark the position in the column that are below (true)
			// and above (false) the for the left and right of the columns.
			for (int j = 0; NumYpnts + 1 > j; j++)
			{
				yxL[j] = delta * j <= yL;
				yxR[j] = delta * j <= yR;
			}

			// Traverse the column from the baseline to the top.
			for (int j = 0; NumYpnts > j; j++)
			{
				int Bottom = j;
				int Top = j + 1;
				float yB = delta * j;		// Bottom of the pixel
				float yT = delta * (j + 1); // Top of the pixel

				// lower left, lower right, upper left, upper right
				if ((!yxL[Bottom]) && (!yxR[Bottom]) && (!yxL[Top]) && (!yxR[Top]))
				{
					// All Corners outside the boundary.
					// Area = 0
					break;
				}

				if (yxL[Bottom] && yxR[Bottom] && yxL[Top] && yxR[Top])
				{
					// All Corners are with the boundary
					PixelArea[j * NumXpnts + i] = delta * delta; // Area
					continue;
				}

				if (yxL[j] && yxR[j] && yxL[j + 1] && (!yxR[j + 1]))
				{
					// Upper right out
					float x_intersect_pixel_top = (yT - Height) / m1;
					float pixel_area = delta * delta;
					float pixel_area_sub = (yT - yR) * (xR - x_intersect_pixel_top) / 2;

					if (x_intersect_pixel_top > xR && x_intersect_pixel_top < xL)
					{
						cout << "Error : Pixel Top is greater than the right intersection" << endl;
					}

					if (pixel_area_sub < 0 || pixel_area_sub > pixel_area)
					{
						cout << "Error : Pixel area sub is less than 0 or greater than the pixel area" << endl;
					}

					PixelArea[j * NumXpnts + i] = pixel_area - pixel_area_sub;

					if (PixelArea[j * NumXpnts + i] < 0 || PixelArea[j * NumXpnts + i] > pixel_area)
					{
						cout << "Error : Pixel area is less than 0 or greater than the pixel area" << endl;
					}

					continue;
				}

				if (yxL[Bottom] && yxR[Bottom] && (!yxL[Top]) && (!yxR[Top]))
				{
					// Upper right and upper left out

					PixelArea[j * NumXpnts + i] = ((yL - yR) * delta / 2) + (yR - yB) * delta;
					continue;
				}

				if (yxL[j] && (!yxR[j]) && yxL[j + 1] && (!yxR[j + 1]))
				{
					// Upper right and lower right out
					// Line intersecting the pixel top and the pixel bottom.
					float pixel_area = delta * delta;
					float x_intersect_pixel_top = ((yT - Height) / m1) - xL;
					float x_intersect_pixel_bottom = ((yB - Height) / m1) - xL;
					PixelArea[j * NumXpnts + i] = (x_intersect_pixel_top * delta) + (x_intersect_pixel_bottom - x_intersect_pixel_top) * delta / 2;

					if (PixelArea[j * NumXpnts + i] < 0 || PixelArea[j * NumXpnts + i] > pixel_area)
					{
						cout << "xL: " << xL << endl;
						cout << "xR: " << xR << endl;
						cout << "x_intersect_pixel_top: " << x_intersect_pixel_top << endl;
						cout << "x_intersect_pixel_bottom: " << x_intersect_pixel_bottom << endl;
						cout << "Error : Pixel area is less than 0 or greater than the pixel area" << endl;
					}
					continue;
				}

				if (yxL[j] && (!yxR[j]) && (!yxL[j + 1]) && (!yxR[j + 1]))
				{
					// Only Lower Left in.
					PixelArea[j * NumXpnts + i] = (yL - yB) * (((yB - Height) / m1) - xL) / 2;
					continue;
				}
			}
		}
		else
		{
			// The gradient is positive and the x is less than the apex.
			// Note x is negative.
			float yL = m2 * xL + Height;
			float yR = m2 * xR + Height;

			for (int j = 0; NumYpnts + 1 > j; j++)
			{
				yxL[j] = delta * j <= yL;
				yxR[j] = delta * j <= yR;
			}

			// printf("Pixels along y axis ******************\n");
			for (int j = 0; NumYpnts > j; j++)
			{
				// if(!yxL[j])
				//	printf("Lower Left Out\n");
				// if(!yxL[j+1])
				//	printf("Upper Left Out\n");
				// if(!yxR[j])
				//	printf("Lower Right Out\n");
				// if(!yxR[j+1])
				//	printf("Upper Right Out\n");

				// lower left, lower right, upper left, upper right
				if ((!yxL[j]) && (!yxR[j]) && (!yxL[j + 1]) && (!yxR[j + 1]))
				{
					// All Corners outside the boundary
					break;
				}

				if (yxL[j] && yxR[j] && yxL[j + 1] && yxR[j + 1])
				{
					// All Corners are with the boundary
					PixelArea[j * NumXpnts + i] = delta * delta;
					continue;
				}

				if (yxL[j] && yxR[j] && (!yxL[j + 1]) && yxR[j + 1])
				{
					// Upper left out
					PixelArea[j * NumXpnts + i] = delta * delta - (vY[j] + delta / 2 - yL) * (-(xL - ((vY[j] + (delta / 2) - Height) / m2)) / 2);
					continue;
				}

				if (yxL[j] && yxR[j] && (!yxL[j + 1]) && (!yxR[j + 1]))
				{
					// Upper right and upper left out
					PixelArea[j * NumXpnts + i] = (yR - yL) * delta / 2 + ((yL - (vY[j] - delta / 2)) * delta);
					continue;
				}

				if ((!yxL[j]) && yxR[j] && (!yxL[j + 1]) && yxR[j + 1])
				{
					// Upper left and lower left out
					PixelArea[j * NumXpnts + i] = -((vY[j] - (delta / 2) - Height) / m2 - (vY[j] + (delta / 2) - Height) / m2) * delta / 2 - (((vY[j] + (delta / 2) - Height) / m2) - xR) * delta;
					continue;
				}

				if ((!yxL[j]) && (yxR[j]) && (!yxL[j + 1]) && (!yxR[j + 1]))
				{
					// Only Right in.
					PixelArea[j * NumXpnts + i] = (yR - (vY[j] - delta / 2)) * (-((((vY[j] - delta / 2) - Height) / m2) - xR)) / 2;
					continue;
				}
			}
		}
	}
	delete[] yxL;
	delete[] yxR;
}

void Facet::PrintMatrix()
{
	std::cout << std::fixed << std::setprecision(4); // Set 2 decimal places

	cout << "Limits x: " << NumXpnts << ", y: " << NumYpnts << endl;

	float maxPixel = delta * delta;
	float total = 0;
	for (int j = NumYpnts - 1; 0 <= j; j--)
	{
		for (int i = 0; NumXpnts > i; i++)
		{
			cout << PixelArea[j * NumXpnts + i] / maxPixel << " ";
			total += PixelArea[j * NumXpnts + i];
		}
		std::cout << "\n";
	}
	std::cout << std::defaultfloat;
	cout << "Pixel Total Area: " << total << endl;
	cout << "Facet Area: " << Area << endl;
}
