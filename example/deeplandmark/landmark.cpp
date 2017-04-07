#include <cassert>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "landmark.hpp"

using namespace cv;
using namespace std;
using namespace caffe;


BBox::BBox(int x, int y, int w, int h) {
  this->x = x; this->y = y;
  this->width = w; this->height = h;
  this->rect = Rect(x, y, w, h);
}

BBox::BBox(const Rect &rect) {
  this->x = rect.x; this->y = rect.y;
  this->width = rect.width; this->height = rect.height;
  this->rect = rect;
}

void BBox::Project(const vector<Point2f> &absLandmark, vector<Point2f> &relLandmark) const {
  assert(absLandmark.size() == relLandmark.size());
  for (int i = 0; i < absLandmark.size(); i++) {
    const Point2f &point1 = absLandmark[i];
    Point2f &point2 = relLandmark[i];
    point2.x = (point1.x - this->x) / this->width;
    point2.y = (point1.y - this->y) / this->height;
  }
}

void BBox::ReProject(const vector<Point2f> &relLandmark, vector<Point2f> &absLandmark) const {
  assert(relLandmark.size() == absLandmark.size());
  for (int i = 0; i < relLandmark.size(); i++) {
    const Point2f &point1 = relLandmark[i];
    Point2f &point2 = absLandmark[i];
    point2.x = point1.x*this->width + this->x;
    point2.y = point1.y*this->height + this->y;
  }
}

BBox BBox::subBBox(float left, float right, float top, float bottom) const {
  assert(right>left && bottom>top);
  float x, y, w, h;
  x = this->x + left*this->width;
  y = this->y + top*this->height;
  w = this->width*(right - left);
  h = this->height*(bottom - top);
  return BBox(x, y, w, h);
}


CNN::CNN(const string &network, const string &model) {
  cnn = new Net(network);
  assert(cnn);
  cnn->CopyTrainedLayersFrom(model);
}

vector<Point2f> CNN::forward(const Mat &data, const string &layer) {
  shared_ptr<Blob> blob = cnn->blob_by_name("data");
  blob->Reshape(1, 1, data.rows, data.cols);
  float *blob_data = blob->mutable_cpu_data();
  const float *ptr = NULL;
  for (int i = 0; i < data.rows; i++) {
    ptr = data.ptr<float>(i);
    for (int j = 0; j < data.cols; j++) {
      blob_data[i*data.cols + j] = ptr[j];
    }
  }

  cnn->Forward();

  shared_ptr<caffe::Blob> landmarks = cnn->blob_by_name(layer);
  vector<Point2f> points(landmarks->count() / 2);
  for (int i = 0; i < points.size(); i++) {
    Point2f &point = points[i];
    point.x = landmarks->data_at(0, 2 * i, 0, 0);
    point.y = landmarks->data_at(0, 2 * i + 1, 0, 0);
  }
  return points;
}


void FaceDetector::LoadXML(const string &path) {
  bool res = cc.load(path);
  assert(res);
}

int FaceDetector::DetectFace(const Mat &img, vector<Rect> &rects) {
  assert(img.type() == CV_8UC1);
  Mat gray(img.rows, img.cols, CV_8UC1);
  img.copyTo(gray);

  cc.detectMultiScale(gray, rects, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, \
    Size(30, 30), Size(gray.cols, gray.rows));

  return rects.size();
}

void Landmarker::LoadModel(const string &path) {
  string network = path + "/1_F.prototxt";
  string model = path + "/1_F.caffemodel";
  F = new CNN(network, model);
  string networks[10] = { "/2_LE1.prototxt", "/2_LE2.prototxt", "/2_RE1.prototxt", "/2_RE2.prototxt", \
                          "/2_N1.prototxt", "/2_N2.prototxt", "/2_LM1.prototxt", "/2_LM2.prototxt", \
                          "/2_RM1.prototxt", "/2_RM2.prototxt" };
  string models[10] = { "/2_LE1.caffemodel", "/2_LE2.caffemodel", "/2_RE1.caffemodel", "/2_RE2.caffemodel", \
                        "/2_N1.caffemodel", "/2_N2.caffemodel", "/2_LM1.caffemodel", "/2_LM2.caffemodel", \
                        "/2_RM1.caffemodel", "/2_RM2.caffemodel" };
  for (int i = 0; i < 5; i++) {
    network = path + networks[2 * i];
    model = path + models[2 * i];
    level2[2 * i] = new CNN(network, model);
    network = path + networks[2 * i + 1];
    model = path + models[2 * i + 1];
    level2[2 * i + 1] = new CNN(network, model);
  }
}

Mat GetPatch(const Mat &img, const BBox &bbox, const Point2f &point, \
                         double padding, BBox& patch_bbox) {
  double x = bbox.x + point.x*bbox.width;
  double y = bbox.y + point.y*bbox.height;
  Rect roi;
  roi.x = x - bbox.width*padding;
  roi.y = y - bbox.height*padding;
  roi.width = 2 * padding*bbox.width;
  roi.height = 2 * padding*bbox.height;
  patch_bbox = BBox(roi);
  return img(roi).clone();
}

Mat process(const Mat &img, Size size) {
  Mat data;
  resize(img, data, size);
  data.convertTo(data, CV_32FC1);
  Scalar meanScalar, stdScalar;
  cv::meanStdDev(data, meanScalar, stdScalar);
  float mean = meanScalar.val[0];
  float std = stdScalar.val[0];
  data = (data - mean) / std;
  return data;
}

vector<Point2f> Landmarker::DetectLandmark(const Mat &img, const BBox &bbox){
  assert(img.type() == CV_8UC1);
  Mat data = process(img(bbox.rect), Size(39, 39));
  // level 1
  vector<Point2f> landmarks = F->forward(data, "fc2");
  // level 2
  for (int i = 0; i < 5; i++) {
    Point2f point = landmarks[i];
    BBox patch_bbox(0, 0, 0, 0);

    Mat roi = GetPatch(img, bbox, point, 0.16, patch_bbox);
    data = process(roi, Size(15, 15));
    CNN *net = level2[2 * i];
    vector<Point2f> res = net->forward(data, "fc2");
    patch_bbox.ReProject(res, res);
    bbox.Project(res, res);
    Point2f p1 = res[0];

    roi = GetPatch(img, bbox, point, 0.18, patch_bbox);
    data = process(roi, Size(15, 15));
    net = level2[2 * i + 1];
    res = net->forward(data, "fc2");
    patch_bbox.ReProject(res, res);
    bbox.Project(res, res);
    Point2f p2 = res[0];

    landmarks[i] = Point2f((p1.x + p2.x) / 2., (p1.y + p2.y) / 2.);
  }
  bbox.ReProject(landmarks, landmarks);
  return landmarks;
}
