//
//  OpenCVWrapper.m
//  OpenCVProject
//
//  Created by Apprentice on 9/4/16.
//  Copyright © 2016 Apprentice. All rights reserved.
//

#import "OpenCVWrapper.h"
#import <opencv2/opencv.hpp>
#import <opencv2/highgui/ios.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"
#include <vector>


using namespace cv;
using namespace std;

int minHessian = 400;

@implementation OpenCVWrapper

+(NSString *) openCVVersionString
{
    return [NSString stringWithFormat:@"OpenCV Version %s", CV_VERSION];
}

+(UIImage *) makeGrayscale:(UIImage *) image{
    
    cv::Mat imageMat;
    UIImageToMat(image, imageMat);
    
    if (imageMat.channels() == 1) return image;
    
    cv::Mat grayMat;
    cv::cvtColor(imageMat, grayMat, CV_BGR2GRAY);
    return MatToUIImage(grayMat);

}

+(UIImage *) matchFeatures:(UIImage *)image thatMatch:(UIImage *)scene{
    
    //get image to find points in, and make it MAT format
    cv::Mat imageMat;
    UIImageToMat([self makeGrayscale:image], imageMat);
    cv::Mat sceneMat;
    UIImageToMat([self makeGrayscale:scene], sceneMat);
    
    //create the feature detector, with its thresholding variable
    SurfFeatureDetector detector( minHessian );
    
    //detect and store keypoints
    std::vector<KeyPoint> keypoints, scenepoints;
    detector.detect(imageMat, keypoints);
    detector.detect(sceneMat, scenepoints);
    
    //draw the keypoints - commented out for matching
//    Mat imageMatWithPoints;
//    Mat sceneMatWithPoints;
//    
//    drawKeypoints( imageMat, keypoints, imageMatWithPoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
//    drawKeypoints( sceneMat, scenepoints, sceneMatWithPoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    
    //calculate descriptors rather than drawing
    SurfDescriptorExtractor extractor;
    
    Mat imageDescriptors, sceneDescriptors;
    
    extractor.compute(imageMat, keypoints, imageDescriptors);
    extractor.compute(sceneMat, scenepoints, sceneDescriptors);
    int keypoints_length;
    keypoints_length = keypoints.size();
    printf("keypoints length %d \n", keypoints_length);
    printf("scenepoints length %d \n", scenepoints.size());
    
    
    
    // matching vectors with brute force matcher
    BFMatcher matcher(NORM_L1, true);
    std::vector<DMatch> matches;
    matcher.match(imageDescriptors, sceneDescriptors, matches);
    
    printf("matches found %d", matches.size());
    
    // draw the matches into a composite image
    Mat combo;
    drawMatches( imageMat, keypoints, sceneMat, scenepoints, matches, combo);
    
    return MatToUIImage(combo);
}

+(UIImage *)matchFeaturesFLANN:(UIImage *)image thatMatch:(UIImage *)scene
{
    //get image to find points in, and make it MAT format
    cv::Mat imageMat;
    UIImageToMat([self makeGrayscale:image], imageMat);
    cv::Mat sceneMat;
    UIImageToMat([self makeGrayscale:scene], sceneMat);
    
    //create the feature detector, with its thresholding variable
    SurfFeatureDetector detector( minHessian );
    
    //detect and store keypoints
    std::vector<KeyPoint> keypoints, scenepoints;
    detector.detect(imageMat, keypoints);
    detector.detect(sceneMat, scenepoints);
    
    //calculate descriptors rather than drawing
    SurfDescriptorExtractor extractor;
    
    Mat imageDescriptors, sceneDescriptors;
    
    extractor.compute(imageMat, keypoints, imageDescriptors);
    extractor.compute(sceneMat, scenepoints, sceneDescriptors);
    int keypoints_length;
    keypoints_length = keypoints.size();
    printf("keypoints length %d \n", keypoints_length);
    printf("scenepoints length %d \n", scenepoints.size());
    
    //match descripors with a FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch> matches;
    matcher.match(imageDescriptors, sceneDescriptors, matches);
    
    double max_dist = 0;
    double min_dist = 100;
    
    //quick calculation of max and min distance between keypoints
    for ( int i = 0; i < imageDescriptors.rows; i++)
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );
    
    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.}
    
    std::vector< DMatch > good_matches;
    
    for( int i = 0; i < imageDescriptors.rows; i++ )
    {
        if( matches[i].distance <= max(2*min_dist, 0.2) )
        {
            good_matches.push_back( matches[i]);
        }
    }
    
    //-- Draw only "good" matches
    Mat combo;
    drawMatches( imageMat, keypoints, sceneMat, scenepoints,
                good_matches, combo, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    //print out lots of matches
    for( int i = 0; i < (int)good_matches.size(); i++ )
    { printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }
    
    return MatToUIImage(combo);


}

+(UIImage *) complexMatchFeaturesFLANN:(UIImage *) image thatMatch:(UIImage *) scene
{
    //run the stuff from main
    
    //here is where the find-flann-pairs go
    
    
    return image;
}
@end
