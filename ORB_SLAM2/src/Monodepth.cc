#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>
#include "Monodepth.h"
using namespace cv;
using namespace dnn;
using namespace std;

Mat baseDepth::depth(Mat &frame)
{
    int ori_h = frame.size[0];
    int ori_w = frame.size[1];
    cout << "ori: " << ori_h << " , " << ori_w << endl;
    Mat blobImage = blobFromImage(frame, 1.0 / 255.0, Size(this->inWidth, this->inHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blobImage);
    cout << "read model" << endl;
    vector<Mat> scores;
    this->net.forward(scores, this->net.getUnconnectedOutLayersNames());
    int channel = scores[0].size[1];
    int h = scores[0].size[2];
    int w = scores[0].size[3];
    cout << "c: " << channel << " , h: " << h << " , w: " << w << endl;
    Mat depthMap(scores[0].size[2], scores[0].size[3], CV_32F, scores[0].ptr<float>(0, 0));
    cout << depthMap.size() << endl;
    depthMap *= 255.0;
    depthMap.convertTo(depthMap, CV_8UC1);
    resize(depthMap, depthMap, Size(ori_w, ori_h));
    applyColorMap(depthMap, depthMap, COLORMAP_MAGMA);
    return depthMap;
}

Mat baseDepth::viewer(vector<Mat> imgs, double alpha)
{
    Size imgOriSize = imgs[0].size();
    Size imgStdSize(imgOriSize.width * alpha, imgOriSize.height * alpha);

    Mat imgStd;
    int delta_h = 2, delta_w = 2;
    Mat imgWindow(imgStdSize.height + 2 * delta_h, imgStdSize.width * 2 + 3 * delta_w, imgs[0].type());
    resize(imgs[0], imgStd, imgStdSize, alpha, alpha, INTER_LINEAR);
    imgStd.copyTo(imgWindow(Rect(Point2i(delta_w, delta_h), imgStdSize)));
    resize(imgs[1], imgStd, imgStdSize, alpha, alpha, INTER_LINEAR);
    imgStd.copyTo(imgWindow(Rect(Point2i(imgStdSize.width + 2 * delta_w, delta_h), imgStdSize)));
    return imgWindow;
}