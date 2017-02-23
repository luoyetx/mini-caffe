#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <caffe/profiler.hpp>
#include "landmark.hpp"

using namespace cv;
using namespace std;
using namespace caffe;

/*! \brief Timer */
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /*! \brief start or restart timer */
  inline void Tic() {
    start_ = Clock::now();
  }
  /*! \brief stop timer */
  inline void Toc() {
    end_ = Clock::now();
  }
  /*! \brief return time in ms */
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }

private:
  Clock::time_point start_, end_;
};

void showLandmarks(Mat &image, Rect &bbox, vector<Point2f> &landmarks) {
  Mat img;
  image.copyTo(img);
  rectangle(img, bbox, Scalar(0, 0, 255), 2);
  for (int i = 0; i < landmarks.size(); i++) {
    Point2f &point = landmarks[i];
    circle(img, point, 2, Scalar(0, 255, 0), -1);
  }
  imshow("landmark", img);
  waitKey(0);
}

int main(int argc, char *argv[]) {
  FaceDetector fd;
  Landmarker lder;
  fd.LoadXML("../deeplandmark/haarcascade_frontalface_alt.xml");
  lder.LoadModel("../models/deeplandmark");

  Mat image;
  Mat gray;
  image = imread("../deeplandmark/test.jpg");
  if (image.data == NULL) return -1;
  cvtColor(image, gray, CV_BGR2GRAY);

  vector<Rect> bboxes;
  fd.DetectFace(gray, bboxes);

  vector<Point2f> landmarks;
  for (int i = 0; i < bboxes.size(); i++) {
    BBox bbox_ = BBox(bboxes[i]).subBBox(0.1, 0.9, 0.2, 1);
    const int kTestN = 1000;
    Timer timer;
    double time = 0;
    for (int j = 0; j < kTestN; j++) {
      timer.Tic();
      landmarks = lder.DetectLandmark(gray, bbox_);
      timer.Toc();
      time += timer.Elasped();
    }
    cout << "costs " << time / kTestN << " ms" << endl;
    showLandmarks(image, bbox_.rect, landmarks);
  }
  return 0;
}
