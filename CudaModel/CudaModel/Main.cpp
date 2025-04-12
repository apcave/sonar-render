
#include "rapidxml.hpp"
#include "rapidxml_utils.hpp"
#include "GeoMath.h"
#include "FacetData.h"
#include "PointData.h"
#include "TraceObject.h"

#include "stdio.h"
#include <iostream>
#include <fstream>
#include <tchar.h>
#include <omp.h>

using namespace rapidxml;

extern int StartCuda();
extern int StopCuda();
extern int ProjectPointToFacet( PointData* SrcPoint, FacetData* Facet, TraceObject* TraceOb);
extern int ScanProjectFacetToPoint( PointData** DestPoint, FacetData* Facet, TraceObject* TraceOb,
						bool** PathMatrix, unsigned int NumSourcePoints, unsigned int NumFeildPoints);
extern int MakeFacetsOnGPU( FacetData** Facets, unsigned int NumFacets, dcomplex k_wave);
extern int DeleteTraceObject(TraceObject* TraceObject);
extern void PrintComplexVector( dcomplex* dev_Vector, unsigned int NumPoints);
extern void PrintVector( float* dev_Vector, unsigned int NumPoints);
extern TraceObject* MakeTraceObject();
extern TraceObject* ScanProjectFacetToFacet(FacetData* Facet_i,FacetData* Facet_j, TraceObject* TraceObj_i);
extern bool** MakeCollisionDectionMatrix( PointData** SourcePoints, PointData** FeildPoints, FacetData** Facets,
								unsigned int NumSourcePoints, unsigned int NumFieldPoints, unsigned int NumFacets,float MaxTheata);




void ProjectFacetToFacet(unsigned int NumFacets,unsigned int NumFieldPoints,unsigned int NumSourcePoints, unsigned int i,TraceObject* TraceObjIn,
									FacetData** Facets, PointData** FeildPoints, bool** PathMatrix, unsigned int depth, unsigned int MaxDepth)
{
	if(depth >= MaxDepth)
		return;
	depth++;

	printf("Facet i=%d, Order %d\n",i,depth);

	TraceObject* TraceObjOut;
	for(unsigned int j = 0; NumFacets > j; j++)
	{
		if(PathMatrix[i+NumSourcePoints+NumFieldPoints][j+NumSourcePoints+NumFieldPoints])
		{
			if( i == j )
				printf("***********Collision Detection Error.********************\n");

			TraceObjOut = ScanProjectFacetToFacet(Facets[i],Facets[j], TraceObjIn);
			ScanProjectFacetToPoint( FeildPoints, Facets[j], TraceObjOut, PathMatrix, NumSourcePoints, NumFieldPoints);
			ProjectFacetToFacet(NumFacets,NumFieldPoints,NumSourcePoints, j, TraceObjOut,Facets, FeildPoints,PathMatrix, depth, MaxDepth);
			DeleteTraceObject(TraceObjOut);
		}
	}
}




int _tmain(int argc, _TCHAR* argv[])
{
	dcomplex Cp;
	float Resolution;
	dcomplex k_wave;
	unsigned int ObjectCnt = 0;


	

	file<char> XMLfile("Cuda_model.xml");
	xml_document<char> doc;    // character type defaults to char
	doc.parse<0>(XMLfile.data());    // 0 means default parse flags

	
	xml_node<char>* RootNode = doc.first_node("Model");

	xml_node<char>* FrequencyNode = RootNode->first_node("Frequency");
	float freq = std::stof(FrequencyNode->value());

	xml_node<char>* CpNode = RootNode->first_node("Cp");
	Cp.r = std::stof(CpNode->first_attribute("Real")->value());
	Cp.i = std::stof(CpNode->first_attribute("Imag")->value());

	xml_node<char>* ResolutionNode = RootNode->first_node("Resolution");
	Resolution = std::stof(ResolutionNode->value());

	xml_node<char>* MaxTheataNode = RootNode->first_node("MaxTheata");
	float MaxTheata = std::stof(MaxTheataNode->value());
	MaxTheata = MaxTheata*PI/180;

	xml_node<char>* OrderNode = RootNode->first_node("MaxOrder");
	unsigned int Order = std::stoi(OrderNode->value());
	printf("Bounce Max Order %d\n",Order);

	xml_node<char>* NumFacetsNode = RootNode->first_node("Number_of_Facets");
	unsigned int NumFacets = std::stoi(NumFacetsNode->value());

	NumFacetsNode = RootNode->first_node("Number_of_SourcePoints");
	unsigned int NumSourcePoints = std::stoi(NumFacetsNode->value());

	NumFacetsNode = RootNode->first_node("Number_of_FieldPoints");
	unsigned int NumFieldPoints = std::stoi(NumFacetsNode->value());

	printf("Number of, Source Points :%d, Facets :%d, Feild Points :%d\n", NumSourcePoints, NumFacets, NumFieldPoints);

	xml_node<char>* FacetsNode = RootNode->first_node("Facets"); 

	xml_node<char>* FacetNode = FacetsNode->first_node("Facet");

	FacetData** Facets = (FacetData**)malloc(sizeof(FacetData*)*NumFacets);
	for(unsigned int i = 0; NumFacets > i; i++)
	{
		Facets[i] = new FacetData(FacetNode,ObjectCnt,i);
		FacetNode = FacetNode->next_sibling();
		ObjectCnt++;
	}

	xml_node<char>* SourcePointsNode = RootNode->first_node("SourcePoints"); 
	xml_node<char>* SourcePointNode = SourcePointsNode->first_node("SrcPoint");


	PointData** SourcePoints = (PointData**)malloc(sizeof(PointData*)*NumSourcePoints);
	for(unsigned int i = 0; NumSourcePoints > i; i++)
	{
		SourcePoints[i] = new PointData(SourcePointNode,true,ObjectCnt);
		SourcePointNode = SourcePointNode->next_sibling();
		ObjectCnt++;
	}

	xml_node<char>* FeildPointsNode = RootNode->first_node("FieldPoints"); 
	xml_node<char>* FeildPointNode = FeildPointsNode->first_node("FieldPoint");
	PointData** FeildPoints = (PointData**)malloc(sizeof(PointData*)*NumFieldPoints);
	for(unsigned int i = 0; NumFieldPoints > i; i++)
	{
		FeildPoints[i] = new PointData(FeildPointNode,false,ObjectCnt);
		FeildPointNode = FeildPointNode->next_sibling();
		ObjectCnt++;
	}


	

	int cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.

    cudaStatus = StartCuda();
    if (cudaStatus != 0) {
		return 1;
    }

	printf("Attemping Frequency %f kHz\n",freq/1e3f);
	float omega = freq*2.0f*PI;
	float lambda = Cp.r / freq;
	k_wave.r = omega / Cp.r;
	k_wave.i = 0;
	float delta = lambda/Resolution;
	printf("K Wave Number Real:%f, Imag:%f\n",k_wave.r,k_wave.i);

	float MaxY = 0;
	float MaxHalfBase = 0;
	for(unsigned int i = 0; NumFacets > i; i++)
	{
		if( Facets[i]->Height > MaxY)
		{
			MaxY = Facets[i]->Height;
		}

		if( Facets[i]->BaseLengthNeg > MaxHalfBase)
		{
			MaxHalfBase = Facets[i]->BaseLengthNeg;
		}

		if( Facets[i]->BaseLengthPos > MaxHalfBase)
		{
			MaxHalfBase = Facets[i]->BaseLengthPos;
		}
	}
	unsigned int NumXpnts = (unsigned int)(2*(1+floor((MaxHalfBase-delta/2)/delta)))+2;
	unsigned int NumYpnts = (unsigned int)(ceil(MaxY/delta));

	float MinL = -((float)NumXpnts/2)*delta + delta/2;
	float* vX = new float[NumXpnts];
    for(unsigned int i = 0; NumXpnts > i; i++)
	{
        vX[i] = (MinL + delta*i);
	}

	float* vY = new float[NumYpnts];
	for(unsigned int i = 0; NumYpnts > i; i++)
	{
        vY[i] = delta/2 + delta*i;
	}

	for(unsigned int i = 0; NumFacets > i; i++)
	{
		Facets[i]->MakePixelData(NumXpnts,NumYpnts,delta,vX,vY);
		Facets[i]->VectorizeZcorrect(vX,vY);
	}
	delete vX;
	delete vY;


	std::cout << "Starting to Make Facets On GPU.\n";
	cudaStatus = MakeFacetsOnGPU( Facets, NumFacets, k_wave);
    if (cudaStatus != 0) {
        printf("make facets on GPU failed!");
        return 1;
    }

	std::cout << "Starting Collision Detection On GPU.\n";
	bool** PathMatrix = MakeCollisionDectionMatrix( SourcePoints, FeildPoints, Facets, NumSourcePoints, NumFieldPoints, NumFacets, MaxTheata);

	std::cout << "Starting Projections.\n";
	
	//omp_set_num_threads(16);

	unsigned int MaxDepth = Order;
	for (unsigned int i = 0; NumSourcePoints > i; i++)
	{  
		//#pragma omp parallel for
		for(unsigned int j = 0; NumFacets > j; j++)
		{
			if(PathMatrix[i][j+NumSourcePoints+NumFieldPoints])
			{
				TraceObject* TraceObj = MakeTraceObject();
				ProjectPointToFacet( SourcePoints[i], Facets[j], TraceObj );
				//PrintComplexVector( TraceObj->dev_LastPixelPressure, Facets[j]->NumPositions );
				//PrintVector( TraceObj->dev_CosInc, Facets[j]->NumPositions);
				ScanProjectFacetToPoint( FeildPoints, Facets[j], TraceObj, PathMatrix, NumSourcePoints, NumFieldPoints);
				printf("First Order Reflections Completed.\n");
				ProjectFacetToFacet(NumFacets,NumFieldPoints,NumSourcePoints,j, TraceObj,Facets, FeildPoints,PathMatrix, 0, MaxDepth);
				DeleteTraceObject(TraceObj);
			}
		}
	}
		

  
	if (cudaStatus != 0) {
		printf("ProjectPointToFacet failed!");
		return 1;
	}
	
	std::ofstream ResultsFile("Cuda_model.dat");
	//ResultsFile << "Writing this to a file.\n";
	for(unsigned int i = 0; NumFieldPoints > i; i++ )
	{
		ResultsFile << FeildPoints[i]->Pressure.r << " " << FeildPoints[i]->Pressure.i << "\n";
	}
	ResultsFile.close();


    cudaStatus = StopCuda();
    if (cudaStatus != 0) {
        printf("cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

