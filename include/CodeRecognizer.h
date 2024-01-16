#ifndef CODERECOGNIZER_H
#define CODERECOGNIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>

class CodeRecognizer {
public:
    CodeRecognizer();
    cv::Rect processFrame(const cv::Mat& frame);
    cv::Point2f getFeaturePoint() const;
    std::string getRecCode() const { return rec_code; }
    int getLocation() const { return location; }

private:
    struct Line {
        int centx, centy, xl, xr;
    };

    cv::Rect roi;
    std::map<int, cv::Point2f> location_map;
    std::vector<Line> lines;
    int rightmost_line_x;

    std::pair<cv::Mat, cv::Mat> pre_proc(const cv::Rect& roi, const cv::Mat& src);
    void put_code(const cv::Rect& roi, const std::string& num, const cv::Point& loc, cv::Mat& src1, cv::Mat& src2);
    void detect_and_draw_orb_features(cv::Mat& img);
    std::pair<std::vector<Line>,int> cluster(const std::vector<cv::Vec4i>& lines);
    std::string code_rec(const cv::Rect& roi, const std::vector<Line>& b, int cls_number, cv::Mat& src1, cv::Mat& src2);
    int locate(const int x_cut, const std::string& head);
    void create_location_map();
    cv::Rect roi_track(const cv::Rect& roi, std::vector<Line>& lines);
    bool compareByCentY(const Line& a, const Line& b);
    std::string rec_code;
    int location;
};

#endif // CODERECOGNIZER_H
