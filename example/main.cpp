#include "CodeRecognizer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // 创建CodeRecognizer实例
    CodeRecognizer recognizer;

    // 打开视频文件
    VideoCapture cap("公司手机摄测试.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }
    int frameCount = 1;
    Mat frame;
    while (cap.read(frame)) {
        // 处理每一帧
        
        Rect newRoi = recognizer.processFrame(frame);
        
        // 获取最右侧直线的特征点坐标
        Point2f featurePoint = recognizer.getFeaturePoint();
        if (featurePoint.x >= 0 && featurePoint.y >= 0) {
            cout << "Frame:"<<frameCount<<"Feature Point: (" << featurePoint.x << ", " << featurePoint.y << ")" << endl;
        }
        std::string rec_code = recognizer.getRecCode();
        cout << "Recognized code: " << rec_code.substr(0, 8) << endl;
        int location =recognizer.getLocation();

        cout << "Location: " << (location < 0 ? "None" : to_string(location)) << endl;
        //waitKey(5000);
        // 按ESC退出
        char key = static_cast<char>(waitKey(20));
        if (key == 27) break;
        frameCount ++;

    }

    cap.release();
    destroyAllWindows();
    return 0;
}
