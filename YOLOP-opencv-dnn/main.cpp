#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "yolop.h"
using namespace cv;
using namespace dnn;
using namespace std;

int main()
{
	YOLO yolo_model("yolop.onnx", 0.25, 0.45, 0.5);
	string imgpath = "images/0ace96c3-48481887.jpg";
	Mat srcimg = imread(imgpath);
	Mat outimg = yolo_model.detect(srcimg);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, outimg);
	waitKey(0);
	destroyAllWindows();
}