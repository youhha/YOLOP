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
    Mat depth(Mat &frame);
    Mat viewer(vector<Mat> imgs, double alpha = 0.80);

private:
    Net net;
    int inWidth;
    int inHeight;
};

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
    imwrite("inference/depth_color.png", depthMap);
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

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <video_file_path> <model_path>" << std::endl;
        return 1;
    }

    VideoCapture capture(0); // 打开视频文件
    if (!capture.isOpened())
    {
        std::cerr << "Error: Could not open video file." << std::endl;
        return 1;
    }

    int h = 192, w = 640;
    baseDepth model(h, w, argv[2]);

    static const std::string kWinName = "Deep learning Mono depth estimation in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    Mat frame;
    while (capture.read(frame)) // 从视频中读取每一帧
    {
        Mat depthMap = model.depth(frame);

        Mat res = model.viewer({frame, depthMap}, 0.90);
        imshow(kWinName, res);

        char key = waitKey(1);
        if (key == 27) // 按下ESC键退出循环
            break;
    }

    destroyAllWindows();
    return 0;
}