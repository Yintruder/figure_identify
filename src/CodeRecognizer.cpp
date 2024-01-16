#include "CodeRecognizer.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <map>

using namespace cv;
using namespace std;

CodeRecognizer::CodeRecognizer() {
    // 初始化ROI和位置映射
    roi = Rect(430, 400, 200, 520); 
    create_location_map();
    rightmost_line_x = 0;
    location=0;


}

void CodeRecognizer::create_location_map() {
    // 从1-40是从(9,175)到(424,153)的直线
    for (int i = 1; i <= 40; ++i) {
        float x = 9.0f + (i - 1) * (415.0f / 39.0f);  // Linear interpolation
        float y = 175.0f - (i - 1) * (22.0f / 39.0f);  // Linear interpolation
        location_map[i] = cv::Point2f(x, y);
    }

    // 从40-55是从(424,153)到(470,118)的圆曲线
    for (int i = 41; i <= 55; ++i) {
        float x = 424.0f + (i - 40) * (46.0f / 14.0f);  // Linear interpolation for x
        float y = 153.0f - (i - 40)*(i-40) *0.16;  // Sinusoidal interpolation for y
        location_map[i] = cv::Point2f(x, y);
    }

    // 从55-76是从(470,118)到(418,3)的直线
    for (int i = 56; i <= 76; ++i) {
        float x = 470.0f - (i - 55) * (52.0f / 20.0f);  // Linear interpolation
        float y = 118.0f - (i - 55) * (115.0f / 20.0f);  // Linear interpolation
        location_map[i] = cv::Point2f(x, y);
    }
}

pair<Mat, Mat> CodeRecognizer::pre_proc(const Rect& roi, const Mat& src) {
    // 将原始图像转换为灰度图像
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    
    // 复制gray到pre_img
    Mat pre_img = gray.clone();
    
    // 画出ROI区域
    rectangle(pre_img, roi, Scalar(0, 255, 0), 2);
    
    // 反色处理
    //bitwise_not(pre_img, pre_img);
    
    // 裁剪ROI并缩放
    Mat img_roi = pre_img(roi);
    Mat code;
    cv::resize(img_roi, code, Size(400, 320), 0, 0, INTER_LINEAR);
    
    // 二值化
    threshold(code, code, 150, 255, THRESH_BINARY);
    
    // 降噪处理
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat h_kernel = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1));
    Mat rec;
    erode(code, rec, h_kernel);
    erode(rec, rec, kernel);
    
    return {pre_img, rec};
}

void CodeRecognizer::put_code(const Rect& roi, const string& num, const Point& loc, Mat& src1, Mat& src2) {
     // 计算在src1上的文本位置
    Point p1(loc.x - 20, loc.y);
    putText(src1, num, p1, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);

    // 计算在src2上的文本位置
    Point p2(static_cast<int>(roi.x + loc.x * (roi.width / 400.0) - 20), loc.y + roi.y);
    putText(src2, num, p2, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
}

void CodeRecognizer::detect_and_draw_orb_features(Mat& img) {
    // 创建ORB特征检测器
    Ptr<ORB> orb = ORB::create();

    // 检测ORB特征
    vector<KeyPoint> keypoints;
    orb->detect(img, keypoints);

    // 在图像上绘制特征点
    drawKeypoints(img, keypoints, img, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);
}


// 比较函数，用于sort
bool CodeRecognizer::compareByCentY(const Line& a, const Line& b) {
    return a.centy > b.centy;
}

pair<vector<CodeRecognizer::Line>, int> CodeRecognizer::cluster(const std::vector<cv::Vec4i>& lines) {
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
    sort(a.begin(), a.end(),[this](const Line& a, const Line& b) { return compareByCentY(a, b); });

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


string CodeRecognizer::code_rec(const Rect& roi, const vector<Line>& b, int cls_number, Mat& src1, Mat& src2) {
    string head;
    // 遍历所有的聚类直线
    for (int i = 0; i < cls_number; ++i) {
        line(src1, Point(b[i].xl, b[i].centy), Point(b[i].xr, b[i].centy), Scalar(0, 255, 255), 3, LINE_AA);
        circle(src1, Point(b[i].centx, b[i].centy), 3, Scalar(0, 0, 255), 3);
        int width = b[i].xr - b[i].xl;

        if (b[i].centy > 120) {
            if (width < 90) {
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
            } else if (width < 170) {
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
            } else if (width < 270) {
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

int CodeRecognizer::locate(const int x_cut, const string& head) {
    const string pi = "1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679821480865";
    int location = -1;
    if (head.length() >= 5) {
        for (size_t i = 0; i <= head.length() - 5; ++i) {
            size_t found = pi.find(head.substr(i, 5));
            if (found != string::npos) {
                location = static_cast<int>(found);
                break;
            }
        }
    }
    return location - x_cut;
}
Rect CodeRecognizer::roi_track(const Rect& roi, vector<Line>& lines) {
    int x_min = roi.x;
    int x_max = roi.x + roi.width;
    int y_min = roi.y;
    int y_max = roi.y + roi.height;

    // ROI提取并自动调整
    sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) { return a.xl < b.xl; });
    int x_min_nx = x_min;
    if (x_min == 40) {
        x_min_nx = round(lines[0].xl * (x_max - x_min) / 400) + x_min - 20;
    } else {
        if (lines[0].xl < 20) {
            x_min_nx = x_min - 10;
        } else if (lines[0].xl > 40) {
            x_min_nx = x_min + 10;
        }
    }
    if (x_min_nx < 0) {
        x_min_nx = 0;
    }

    sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) { return a.xr > b.xr; });
    int x_max_nx = x_max;
    if (x_min == 40) {
        x_max_nx = round(lines[0].xr * (x_max - x_min) / 400) + x_min + 20;
    } else {
        if (lines[0].xr > 380) {
            x_max_nx = x_max + 10;
        } else if (lines[0].xr < 360) {
            x_max_nx = x_max - 10;
        }
    }
    if (x_max_nx > 480) {
        x_max_nx = 480;
    }

    sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) { return a.centy > b.centy; });
    int y_max_nx = y_max;
    if (x_min == 40) {
        y_max_nx = round(lines[0].centy * (y_max - y_min) / 320) + y_min + 15;
    } else {
        if (lines[0].centy > 310) {
            y_max_nx = y_max + 10;
        } else if (lines[0].centy < 300) {
            y_max_nx = y_max - 10;
        }
    }
    if (y_max_nx > 640) {
        y_max_nx = 640;
    }
    int y_min_nx = y_max_nx - 320;
    if (y_min_nx < 0) {
        y_min_nx = 0;
    }

    return Rect(x_min_nx, y_min_nx, x_max_nx - x_min_nx, y_max_nx - y_min_nx);
}


Rect CodeRecognizer::processFrame(const Mat& frame) {
    auto start = chrono::high_resolution_clock::now();

    // 预处理
    pair<Mat, Mat> processed = pre_proc(roi, frame);
    Mat& resize_img = processed.first;
    Mat& rec_img_bw = processed.second;
    //在原图上绘制ROI
    cv::rectangle(resize_img, roi, cv::Scalar(0, 255, 0), 2);
    // 霍夫变换
    vector<Vec4i> hough_lines;
    HoughLinesP(rec_img_bw, hough_lines, 1, CV_PI / 180, 50, 50, 10);

    // 聚类
    pair<vector<Line>, int> clustering = cluster(hough_lines);
    lines = clustering.first;
    int cls_num = clustering.second;

    // 识别
    rec_code = code_rec(roi, lines, cls_num, rec_img_bw, resize_img);

    // 定位：返回在地图中的位置
    location = locate(0, rec_code);

    // 更新最右侧直线的x坐标
    if (!lines.empty()) {
        sort(lines.begin(), lines.end(), [](const Line& a, const Line& b) { return a.xr > b.xr; });
        rightmost_line_x = lines.front().xr;
    }

    // 更新ROI
    //效果很差，我先关闭了，也许可以靠yolo识别来弄

    //roi = roi_track(roi, lines);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    // 可以在这里输出识别和处理信息
    // ...
    //可视化检查
    imshow("rec_img", rec_img_bw);
    imshow("resize_img",resize_img);

    return roi;
}

Point2f CodeRecognizer::getFeaturePoint() const {
    // 基于最原始图像的位置计算特征点
    if (rightmost_line_x > 0 && !lines.empty()) {
        // 计算最右侧特征点的坐标
        float x = static_cast<float>(roi.x + rightmost_line_x * (roi.width / 400.0));
        float y = static_cast<float>(roi.y + lines.front().centy * (roi.height / 320.0));
        return Point2f(x, y);
    }
    return Point2f(-1, -1); // 如果没有找到合适的直线，则返回无效坐标
}


