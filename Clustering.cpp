//
//  Clustering.cpp
//  projectface
//
//  Created by Abdullah Tellioğlu on 11/03/16.
//  Copyright © 2016 Abdullah Tellioğlu. All rights reserved.
//

#include "Clustering.hpp"

double findMinCentroidDistance(double* clusters,int clusterCount,int number){
    int min = 99999;
    double minIndex = -1;
    int i;
    for(i = 0;i<clusterCount;i++){
        if(abs(clusters[i]-number)<min){
            min = abs(clusters[i]-number);
            minIndex = i;
        }
    }
    
    return minIndex;
}
double ** cluster(int* numbers ,int numberCount,int clusterCount){
    int i;
    double * clusterNumbers = (double*)malloc(sizeof(double)*clusterCount);
    int * clusterWeightNumbers = (int*)malloc(sizeof(int)*clusterCount);
    for(i=0;i<clusterCount;i++){
        clusterNumbers[i] = rand()%256;
        clusterWeightNumbers[i] = 1;
    }
    for(i=0;i<clusterCount;i++){
        printf("%f -  %d\n", clusterNumbers[i],clusterWeightNumbers[i]);
    }
    for(i=0;i<numberCount;i++){
        int index = findMinCentroidDistance(clusterNumbers,clusterCount,numbers[i]);
        double tmp = clusterNumbers[index]*clusterWeightNumbers[index]+numbers[i];
        clusterWeightNumbers[index]++;
        tmp = tmp/clusterWeightNumbers[index];
        clusterNumbers[index] = tmp;
    }
    double ** clusterWithWeight = (double**)malloc(sizeof(double*)*2);
    for(i =0;i<2;i++){
        clusterWithWeight[i] = (double*)malloc(sizeof(double)*clusterCount);
    }
    for(i=0;i<clusterCount;i++){
        clusterWithWeight[0][i] = clusterNumbers[i];
        clusterWithWeight[1][i] = clusterWeightNumbers[i];
    }
    for(i=0;i<clusterCount;i++){
        printf("%f -  %d\n", clusterNumbers[i],clusterWeightNumbers[i]);
    }
    printf("\n");
    printf("Clustering Ends!\n");
    return clusterWithWeight;
}