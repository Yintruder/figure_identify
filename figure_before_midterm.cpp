#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;

// 结构体用于替代Python中的字典
struct Line {
    int centx, centy, xl, xr;
};

// 预处理函数
cv::Mat pre_proc(const Rect& roi, const Mat& src) {
    // 缩放图像到固定尺寸
    /*
    Mat resize;
    cv::resize(src, resize, Size(640, 480), 0, 0, INTER_LINEAR);
    
    // 复制resize到pre_img
    
    */
    Mat pre_img = src.clone();
    // 画出ROI区域
    rectangle(pre_img, roi, Scalar(0, 255, 0), 2);
    
    // 反色处理
    bitwise_not(pre_img, pre_img);
    
    // 分离通道，只处理蓝色通道
    vector<Mat> bgr_channels;
    split(pre_img, bgr_channels);
    Mat blue_channel = bgr_channels[0];
    
    // 裁剪ROI并缩放
    Mat img_roi = blue_channel(roi);
    Mat code;
    //cv::resize(img_roi, code, Size(src.cols, 640), 0, 0, INTER_LINEAR);
    
    // 二值化
    threshold(img_roi, code, 150, 255, THRESH_BINARY);
    
    // 降噪处理
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat h_kernel = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1));
    Mat rec;
    erode(code, rec, h_kernel);
    erode(rec, rec, kernel);
    
    return rec;
}
void put_code(const Rect& roi, const string& num, const Point& loc, Mat& src1, Mat& src2) {
    // 计算在src1上的文本位置
    Point p1(loc.x - 10, loc.y);
    putText(src1, num, p1, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0), 1, LINE_AA);

    // 计算在src2上的文本位置
    Point p2(static_cast<int>(roi.x + loc.x * (roi.width / 400.0) - 20), loc.y + roi.y);
    putText(src2, num, p2, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0), 1, LINE_AA);

}
void detect_and_draw_orb_features(Mat& img) {
    // 创建ORB特征检测器
    Ptr<ORB> orb = ORB::create();

    // 检测ORB特征
    vector<KeyPoint> keypoints;
    orb->detect(img, keypoints);

    // 在图像上绘制特征点
    drawKeypoints(img, keypoints, img, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
}

// 比较函数，用于sort
bool compareByCentY(const Line& a, const Line& b) {
    return a.centy > b.centy;
}

pair<vector<Line>, int> cluster(const vector<Vec4i>& lines) {
    vector<Line> a, b;
    int t = 0;

    // 转换所有直线到自定义的Line结构体
    for (const auto& line : lines) {
        Line l;
        l.centx = static_cast<int>((line[0] + line[2]) / 2.0);
        l.centy = static_cast<int>((line[1] + line[3]) / 2.0);
        l.xl = line[0];
        l.xr = line[2];
        a.push_back(l);
    }

    // 根据centy坐标排序
    sort(a.begin(), a.end(), compareByCentY);

    b.push_back(a[0]);

    for (const auto& line : a) {
        if (abs(b[t].centy - line.centy) < 10) {
            b[t].centy = static_cast<int>((b[t].centy + line.centy) / 2.0);
            b[t].xl = min(b[t].xl, line.xl);
            b[t].xr = max(b[t].xr, line.xr);
            b[t].centx = static_cast<int>((b[t].xl + b[t].xr) / 2.0);
        } else {
            b.push_back(line);
            ++t;
        }
    }

    return {b, t + 1}; // 返回聚类后的直线和数量
}



void cluster_and_set_threshold(vector<Line>& b, vector<int>& thresholds) {
    // 计算每条线的宽度并转换为Mat格式
    Mat data(b.size(), 1, CV_32F);
    for (size_t i = 0; i < b.size(); ++i) {
        int width = b[i].xr - b[i].xl;  // 计算线的宽度
        data.at<float>(i) = static_cast<float>(width);
    }

    // 使用K-means聚类算法对线的宽度进行聚类
    Mat labels, centers;
    kmeans(data, 4, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 4, KMEANS_PP_CENTERS, centers);

    // 将聚类中心点的宽度转换为vector<int>格式，并按从小到大的顺序排序
    vector<int> centers_vec(4);
    for (int i = 0; i < 4; ++i) {
        centers_vec[i] = static_cast<int>(centers.at<float>(i));
    }
    sort(centers_vec.begin(), centers_vec.end());

    // 设置阈值为相邻两个聚类中心点的宽度的平均值
    thresholds.resize(3);
    thresholds[0] = (centers_vec[0] + centers_vec[1]) / 2;
    thresholds[1] = (centers_vec[1] + centers_vec[2]) / 2;
    thresholds[2] = (centers_vec[2] + centers_vec[3]) / 2;
}
string code_rec(const Rect& roi, const vector<Line>& b, int cls_number, Mat& src1, Mat& src2) {
    string head;
    // 遍历所有的聚类直线
    vector<int> thresholds;
    cluster_and_set_threshold(const_cast<vector<Line>&>(b), thresholds);
    for (int i = 0; i < cls_number; ++i) {
        line(src1, Point(b[i].xl, b[i].centy), Point(b[i].xr, b[i].centy), Scalar(0, 255, 255), 3, LINE_AA);
        circle(src1, Point(b[i].centx, b[i].centy), 3, Scalar(0, 0, 255), 3);
        int width = b[i].xr - b[i].xl;

        if (b[i].centy > 120) {
            if (width < thresholds[0]) {
                if (b[i].centx < 110) {
                    put_code(roi, "1", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '1';
                } else if (b[i].centx < 200) {
                    put_code(roi, "2", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '2';
                } else if (b[i].centx < 290) {
                    put_code(roi, "3", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '3';
                } else {
                    put_code(roi, "4", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '4';
                }
            } else if (width < thresholds[1]) {
                if (b[i].centx < 170) {
                    put_code(roi, "5", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '5';
                } else if (b[i].centx < 250) {
                    put_code(roi, "6", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '6';
                } else {
                    put_code(roi, "7", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '7';
                }
            } else if (width < thresholds[2]) {
                if (b[i].centx < 220) {
                    put_code(roi, "8", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '8';
                } else {
                    put_code(roi, "9", Point(b[i].xl, b[i].centy), src1, src2);
                    head += '9';
                }
            } else {
                put_code(roi, "0", Point(b[i].xl, b[i].centy), src1, src2);
                head += '0';
            }
        }
    }
    return head;
}

int locate(const int x_cut, const string& head) {
    const string pi = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865";
    int location = -1;
    if (head.length() >= 5) {
        for (size_t i = 0; i <= head.length() - 5; ++i) {
            size_t found = pi.find(head.substr(i, 5));
            if (found != string::npos) {
                location = static_cast<int>(found);
                // 找到匹配后，向两边扩展搜索
                int left = i, right = i + 4;
                while (left > 0 && pi[found - 1] == head[left - 1]) {
                    --found;
                    --left;
                }
                while (right < head.length() - 1 && pi[found + 5] == head[right + 1]) {
                    ++right;
                }
                break;
            }
        }
    }
    return location - x_cut;
}
Rect roi_track(const Rect& roi, vector<Line>& lines) {
    // 获取ROI的四个边界
    int x_min = roi.x;
    int x_max = roi.x + roi.width;
    int y_min = roi.y;
    int y_max = roi.y + roi.height;

    // 获取线段的x坐标的最小值和最大值
    auto x_comp = [](const Line& a, const Line& b) { return a.xl < b.xl; };
    int x_min_line = min_element(lines.begin(), lines.end(), x_comp)->xl;
    int x_max_line = max_element(lines.begin(), lines.end(), x_comp)->xr;

    // 获取线段的y坐标的最小值和最大值
    auto y_comp = [](const Line& a, const Line& b) { return a.centy < b.centy; };
    int y_min_line = min_element(lines.begin(), lines.end(), y_comp)->centy;
    int y_max_line = max_element(lines.begin(), lines.end(), y_comp)->centy;

    // 设置阈值，当线段距离ROI边界的距离小于或大于这个阈值时，调整ROI
    int threshold = 5;  // 可以根据实际情况调整

    // 根据线段的坐标调整ROI的边界
    if (x_min_line < threshold || x_min_line > threshold * 2) {
        x_min = std::max(0, x_min + x_min_line - threshold);
    }
    if (roi.width - x_max_line < threshold || roi.width - x_max_line > threshold * 2) {
        x_max = x_min + x_max_line + threshold;
    }
    if (y_min_line < threshold || y_min_line > threshold * 2) {
        y_min = std::max(0, y_min + y_min_line - threshold);
    }
    if (roi.height - y_max_line < threshold || roi.height - y_max_line > threshold * 2) {
        y_max = y_min + y_max_line + threshold;
    }

    return Rect(x_min, y_min, x_max - x_min, y_max - y_min);
}
int main() {
    // 视频准备
    VideoCapture cap("test1.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    Mat frame;
    //计算frame的长宽
    Rect roi(40, 220, 400, 320);

    // 设置ROI为frame的下半区域
    int frameCount = 1;
    while (cap.read(frame)) {
        auto start = chrono::high_resolution_clock::now();
        // 预处理
        cv::Mat rec_img_ROI ;
        rec_img_ROI = pre_proc(roi, frame);

        // 霍夫变换
        vector<Vec4i> lines;
        HoughLinesP(rec_img_ROI, lines, 1, CV_PI / 180, 50, 50, 10);

        // 聚类
        pair<vector<Line>, int> clustering = cluster(lines);
        vector<Line>& cls_lines = clustering.first;
        int cls_num = clustering.second;

        // 识别
        string rec_code = code_rec(roi, cls_lines, cls_num, rec_img_ROI,frame);

        // 定位
        int location = locate(0, rec_code);
        detect_and_draw_orb_features(frame);
        // 输出结果
        cout << "----------------------------------" << endl;
        cout << "Frame: " << frameCount << endl;
        cout << "The number of detected lines: " << lines.size() << endl;
        cout << "The number of clustered lines: " << cls_num << endl;
        cout << "Recognized code: " << rec_code.substr(0, 8) << endl;
        cout << "Location: " << (location < 0 ? "None" : to_string(location)) << endl;

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Recognition time (sec): " << elapsed.count() / 1000 << endl;

        // 显示结果
        imshow("rec_img", rec_img_ROI);
        imshow("Frame", frame);
        char key = static_cast<char>(waitKey(20));
        if (key == 27) break; // ESC to exit

        // 更新ROI
        //获取输入图像resize_img的宽度和高度
       
        //roi = roi_track(roi, cls_lines);
        //等待5s
        waitKey(500);
        frameCount++;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
