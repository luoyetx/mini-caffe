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

void showLandmarks(Mat &image, Rect &bbox, vector<Point2f> &landmarks) {
  Mat img;
  image.copyTo(img);
  rectangle(img, bbox, Scalar(0, 0, 255), 2);
  for (int i = 0; i < landmarks.size(); i++) {
    Point2f &point = landmarks[i];
    circle(img, point, 2, Scalar(0, 255, 0), -1);
  }
  imwrite("./deeplandmark-result.jpg", image);
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

  Profiler* profiler = Profiler::Get();

  vector<Point2f> landmarks;
  for (int i = 0; i < bboxes.size(); i++) {
    BBox bbox_ = BBox(bboxes[i]).subBBox(0.1, 0.9, 0.2, 1);
    const int kTestN = 1000;
    double time = 0;
    for (int j = 0; j < kTestN; j++) {
      auto tic = profiler->Now();
      landmarks = lder.DetectLandmark(gray, bbox_);
      auto toc = profiler->Now();
      time += double(toc - tic) / 1000;
    }
    cout << "costs " << time / kTestN << " ms" << endl;
    showLandmarks(image, bbox_.rect, landmarks);
  }
  return 0;
}
