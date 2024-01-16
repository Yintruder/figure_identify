#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <map>
using namespace cv;
using namespace std;

// 结构体用于替代Python中的字典
struct Line {
    int centx, centy, xl, xr;
};

// 预处理函数
pair<Mat, Mat> pre_proc(const Rect& roi, const Mat& src) {
    // 将原始图像转换为灰度图像
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    
    // 复制gray到pre_img
    Mat pre_img = gray.clone();
    
    // 画出ROI区域
    rectangle(pre_img, roi, Scalar(0, 255, 0), 2);
    
    // 反色处理
    bitwise_not(pre_img, pre_img);
    
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
void put_code(const Rect& roi, const string& num, const Point& loc, Mat& src1, Mat& src2) {
    // 计算在src1上的文本位置
    Point p1(loc.x - 20, loc.y);
    putText(src1, num, p1, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);

    // 计算在src2上的文本位置
    Point p2(static_cast<int>(roi.x + loc.x * (roi.width / 400.0) - 20), loc.y + roi.y);
    putText(src2, num, p2, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 255, 255), 1, LINE_AA);
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


string code_rec(const Rect& roi, const vector<Line>& b, int cls_number, Mat& src1, Mat& src2) {
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

int locate(const int x_cut, const string& head) {
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
// 创建一个从代号到坐标的映射

void create_location_map(std::map<int, cv::Point2f>& location_map) {
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
Rect roi_track(const Rect& roi, vector<Line>& lines) {
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
int main() {
    // 视频准备
    VideoCapture cap("test1.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    cv::Mat frame;
    Rect roi(40, 220, 400, 320); // 初始ROI
    cv::Mat map_img = cv::imread("map.png");  // 读取地图图片
    int frameCount = 1;
    //创建初始空地图
    std::map<int, cv::Point2f> location_map;
    create_location_map(location_map);
    while (cap.read(frame)) {
        auto start = chrono::high_resolution_clock::now();

        // 预处理
        pair<Mat, Mat> processed = pre_proc(roi, frame);
        Mat& resize_img = processed.first;
        Mat& rec_img_bw = processed.second;

        // 霍夫变换
        vector<Vec4i> lines;
        HoughLinesP(rec_img_bw, lines, 1, CV_PI / 180, 50, 50, 10);

        // 聚类
        pair<vector<Line>, int> clustering = cluster(lines);
        vector<Line>& cls_lines = clustering.first;
        int cls_num = clustering.second;
        //detect_and_draw_orb_features(rec_img_bw);
        // 识别
        string rec_code = code_rec(roi, cls_lines, cls_num, rec_img_bw,resize_img);

        // 定位
        int location = locate(0, rec_code);
        //detect_and_draw_orb_features(resize_img);
        //detect_and_draw_orb_features(frame);
        
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
        //imshow("Resize Image", resize_img);
        imshow("rec_img", rec_img_bw);
        std::string filename = "./figure/" + std::to_string(frameCount) + ".png";
        cv::imwrite(filename, rec_img_bw);

        //imshow("Frame", frame);
        char key = static_cast<char>(waitKey(20));
        if (key == 27) break; // ESC to exit
        // 检查代号是否在映射中
        if (location_map.count(location) > 0) {
            // 在图像上绘制点
            cv::circle(map_img, location_map[location], 5, cv::Scalar(0, 0, 255), -1);
        }

        // 显示图像
        cv::imshow("Location", map_img);
        std::string filename_map = "./map/" + std::to_string(frameCount) + ".png";
        cv::imwrite(filename_map, map_img);
        // 更新ROI
        roi = roi_track(roi, cls_lines);
        
        //绘制地图点
        frameCount++;
        //方便操作暂停一会
       
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
