//
//  Clustering.hpp
//  projectface
//
//  Created by Abdullah Tellioğlu on 11/03/16.
//  Copyright © 2016 Abdullah Tellioğlu. All rights reserved.
//

#ifndef Clustering_hpp
#define Clustering_hpp

#include <stdio.h>
#include <stdlib.h>
double findMinCentroidDistance(double* clusters,int clusterCount,int number);
double ** cluster(int* numbers,int numberCount,int clusterCount);
#endif

