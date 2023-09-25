#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

class baseDepth
{
public:
    baseDepth(int h, int w, const string &model_path = "model/mono.onnx")
    {
        this->inHeight = h;
        this->inWidth = w;
        cout << "start" << endl;
        this->net = readNetFromONNX(model_path);
        cout << "end" << endl;
    };
    void depth(Mat &frame, Mat &res);
    Mat viewer(vector<Mat> imgs, double alpha = 0.80);

private:
    Net net;
    int inWidth;
    int inHeight;
};