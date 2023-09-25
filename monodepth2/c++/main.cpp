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
    // depthMap *= 255.0;
    // depthMap.convertTo(depthMap, CV_8UC1);
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

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_folder_path> <model_path>" << std::endl;
        return 1;
    }

    std::string imageFolder = argv[1];
    std::string modelPath = argv[2];

    int h = 192, w = 640;
    baseDepth model(h, w, modelPath);

    static const std::string kWinName = "Deep learning Mono depth estimation in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);

    // List all image files in the specified folder
    std::vector<std::string> imageFiles;
    cv::glob(imageFolder, imageFiles);

    for (const std::string &imageFile : imageFiles)
    {
        Mat frame = imread(imageFile);

        if (frame.empty())
        {
            std::cerr << "Error: Could not open image file " << imageFile << std::endl;
            continue;
        }

        Mat depthMap = model.depth(frame);
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        Mat res = model.viewer({frame, depthMap}, 0.90);
        imshow(kWinName, res);

        char key = waitKey(1); // Wait indefinitely for a key press
        if (key == 27)         // Press ESC to exit
            break;
    }

    destroyAllWindows();
    return 0;
}