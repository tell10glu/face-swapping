#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "Clustering.hpp"
using namespace cv;
using namespace std;

void showHistogram(Mat );
void writeHistogramValues(Mat ,int *);
void clearHistArray(int *,int);
Mat rgb2binary(Mat);
void edgeDetect(Mat);
void ortaBul(Mat);
void showGrayHistogram(Mat);
Mat applyCascade(CascadeClassifier ,Scalar ,Mat ,String );
String face_cascade_name = "cascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "cascades/haarcascade_eye.xml";
String mouth_cascade_name = "cascades/haarcascade_mouth.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier mouth_cascade;
int main( int /*argc*/, char** /*argv*/ )
{
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !mouth_cascade.load( mouth_cascade_name ) ){ printf("--(!)Error loading mouth cascade\n"); return -1; };
    Mat image = imread("images/kariyuzu.jpg");
    Mat face = applyCascade(face_cascade, Scalar(10,20,30), image, "Surat");
    Mat newFace = face(Rect(Point(0,face.cols/2),Size(face.rows,face.cols/2)));
    Mat mouth = applyCascade(mouth_cascade, Scalar(100,100,100), newFace, "dudak");
    Mat grayMouth;
    Mat grayFace;
    cvtColor(face, grayFace, CV_RGB2GRAY);
    cvtColor(mouth, grayMouth, CV_RGB2GRAY);
    int * pixelNumbers = (int *)malloc(sizeof(int)* grayFace.rows * grayFace.cols);
    for(int i =0;i<grayFace.rows;i++){
        for(int j=0;j<grayFace.cols;j++){
            printf("%d ",(int)grayFace.at<uchar>(i, j));
            pixelNumbers[i * grayFace.rows + j] = (int)grayFace.at<uchar>(i, j);
        }
    }
    double ** clusters = cluster(pixelNumbers, grayFace.rows * grayFace.cols, 2);
    for(int i =0;i<grayFace.rows;i++){
        for(int j=0;j<grayFace.cols;j++){
            int num = findMinCentroidDistance(clusters[0], 3, grayFace.at<uchar>(i, j));
            grayFace.at<uchar>(i, j) = clusters[0][num];
        }
    }
    imshow("after clustering", grayFace);
    waitKey(0);
    
    return 0;
}


Mat applyCascade(CascadeClassifier classifier,Scalar scalar,Mat face,String windowName){
    vector<Rect> objects;
    Size minSize(50,50);
    Size maxSize(2000,2000);
    classifier.detectMultiScale(face, objects, 1.1, 3,CV_HAAR_SCALE_IMAGE,minSize,maxSize);
    for(int i =0;i<objects.size();i++){
        rectangle(face,Point(objects[i].x,objects[i].y),Point(objects[i].x+objects[i].width,objects[i].y+objects[i].height),scalar);
    }
    imshow(windowName,face);
    printf("%lu\n",objects.size());
    if(objects.size()>0){
        //edgeDetect(face(objects[0]));
        Rect newRect = objects[0];
        newRect.x += 20;
        newRect.y += 20;
        return face(newRect);
    }else{
        return face;
    }
}
Mat dst,src_gray,detected_edges;
int edgeThresh = 1;
int lowThreshold=10;
int const max_lowThreshold = 100;

int kernel_size = 3;

void ortaBul(Mat src)
{
    int *arr = new int[src.cols];
    
    clearHistArray(arr,src.cols);
    
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            arr[j] += src.at<uchar>(i,j);
        }
    }
    
    ofstream cikti("cik.txt");
    for(int i=0;i<src.cols;i++)
    {
        cikti<<arr[i]<<endl;
    }
    
}

void CannyThreshold(int, void*,Mat src)
{
    /// Reduce noise with a kernel 3x3
    blur( src_gray, detected_edges, Size(3,3) );
    int ratio = 3;
    /// Canny detector
    Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    
    /// Using Canny's output as a mask, we display our result
    dst = Scalar::all(0);
    
    src.copyTo( dst, detected_edges);
    
    imshow( "Edge Detection", dst );
    rgb2binary(dst);
    
    
}

void edgeDetect(Mat src)
{
    dst.create( src.size(), src.type() );
    
    /// Convert the image to grayscale
    cvtColor( src, src_gray, CV_BGR2GRAY );
    
    /// Create a window
    namedWindow( "Detect", CV_WINDOW_AUTOSIZE );
    
    /// Show the image
    CannyThreshold(0, 0,src);
    
    /// Wait until user exit program by pressing a key
    waitKey(0);
}

Mat rgb2binary(Mat src)
{
    Mat dst,src_gray;
    cvtColor( src, src_gray, CV_BGR2GRAY );
    threshold(src_gray,dst,110,255,0);
    imshow("binaryImage",dst);
    return dst;
}

void clearHistArray(int *histArray,int size)
{
    for(int i=0;i<size;i++)
    {
        histArray[i]=0;
    }
}

void writeHistogramValues(Mat src,int *histArray)
{
    for(int i=0;i<src.rows;i++)
    {
        for(int j=0;j<src.cols;j++)
        {
            histArray[(int)src.at<uchar>(i,j)]++;
        }
    }
    
    ofstream outputHist("Histogram.txt");
    for(int i=0;i<src.cols;i++)
    {
        outputHist<<histArray[i]<<endl;
    }
    outputHist.close();
}
void showGrayHistogram(Mat src){
    Mat gray = src
    ;
    if(src.type()!=3 || src.type()!=4){
        //cvtColor(src, gray, CV_RGB2GRAY);
    }
    
    namedWindow( "Gray", 1 );    imshow( "Gray", gray );
    
    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };
    
    // Calculate histogram
    MatND hist;
    calcHist( &gray, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
    
    // Show the calculated histogram in command window
    double total;
    total = gray.rows * gray.cols;
    for( int h = 0; h < histSize; h++ )
    {
        float binVal = hist.at<float>(h);
        cout<<" "<<binVal;
    }
    
    // Plot the histogram
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
    normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
    }
    
    namedWindow("grayHistogram", 1 );
    imshow( "grayHistogram", histImage );
}
void showHistogram(Mat src)
{
    Mat dst;
    
    if( !src.data )
    {
        return ;
    }
    
    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );
    
    /// Establish the number of bins
    int histSize = 256;
    
    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };
    
    bool uniform = true;
    bool accumulate = false;
    
    Mat b_hist, g_hist, r_hist;
    
    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );
    
    // Draw the histograms for B, G and R
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );
    
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
    
    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    
    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
             Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
             Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
             Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
             Scalar( 0, 0, 255), 2, 8, 0  );
    }
    
    /// Display
    namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
    imshow("calcHist Demo", histImage );
    
    imwrite("Calculeted_Histogram_Image.jpg",histImage);
    
    waitKey(0);
    
}
