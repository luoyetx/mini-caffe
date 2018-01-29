#include <vector>
#include <algorithm>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace caffe;

struct BBox {
  float x1, y1, x2, y2, score;
};

vector<int> NonMaximumSuppression(const vector<BBox>& bboxes,
                                  const float nms_th) {
  typedef std::multimap<float, int> ScoreMapper;
  ScoreMapper sm;
  const int n = bboxes.size();
  vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = (bboxes[i].x2 - bboxes[i].x1 + 1)*(bboxes[i].y2 - bboxes[i].y1 + 1);
    sm.insert(ScoreMapper::value_type(bboxes[i].score, i));
  }
  vector<int> picked;
  while (!sm.empty()) {
    int last_idx = sm.rbegin()->second;
    picked.push_back(last_idx);
    const BBox& last = bboxes[last_idx];
    for (ScoreMapper::iterator it = sm.begin(); it != sm.end();) {
      int idx = it->second;
      const BBox& curr = bboxes[idx];
      float x1 = std::max(curr.x1, last.x1);
      float y1 = std::max(curr.y1, last.y1);
      float x2 = std::min(curr.x2, last.x2);
      float y2 = std::min(curr.y2, last.y2);
      float w = std::max(0.f, x2 - x1 + 1);
      float h = std::max(0.f, y2 - y1 + 1);
      float ov = (w*h) / (areas[idx] + areas[last_idx] - w*h);
      if (ov > nms_th) {
        ScoreMapper::iterator it_ = it;
        it_++;
        sm.erase(it);
        it = it_;
      }
      else {
        it++;
      }
    }
  }
  return picked;
}

std::vector<BBox> ForwardNet(Net& net, const Mat& img, float scale_factor, bool keep_m3=true, float th = 0.3f) {
  // prepare input data
  vector<Mat> bgr;
  cv::split(img, bgr);
  bgr[0].convertTo(bgr[0], CV_32F, 1.f, -102.9801f);
  bgr[1].convertTo(bgr[1], CV_32F, 1.f, -115.9465f);
  bgr[2].convertTo(bgr[2], CV_32F, 1.f, -122.7717f);
  shared_ptr<Blob> data = net.blob_by_name("data");
  data->Reshape(1, 3, img.rows, img.cols);
  const int bias = data->offset(0, 1, 0, 0);
  const int bytes = bias*sizeof(float);
  memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
  memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
  memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
  shared_ptr<Blob> im_info = net.blob_by_name("im_info");
  im_info->mutable_cpu_data()[0] = img.rows;
  im_info->mutable_cpu_data()[1] = img.cols;
  im_info->mutable_cpu_data()[2] = scale_factor;
  // forward
  net.Forward();
  // post process
  std::vector<BBox> result;
  char tmp[32];
  int level = (keep_m3 ? 3 : 2);
  for (int i = 1; i <= level; i++) {
    sprintf(tmp, "m%d@ssh_boxes", i);
    shared_ptr<Blob> bboxes = net.blob_by_name(tmp);
    sprintf(tmp, "m%d@ssh_cls_prob", i);
    shared_ptr<Blob> probs = net.blob_by_name(tmp);
    int num_rois = bboxes->num();
    vector<BBox> rois;
    rois.reserve(num_rois);
    for (int j = 0; j < num_rois; j++) {
      float score = probs->data_at(j, 0, 0, 0);
      if (score > th) {
        BBox bbox;
        bbox.x1 = bboxes->data_at(j, 1, 0, 0) / scale_factor;
        bbox.y1 = bboxes->data_at(j, 2, 0, 0) / scale_factor;
        bbox.x2 = bboxes->data_at(j, 3, 0, 0) / scale_factor;
        bbox.y2 = bboxes->data_at(j, 4, 0, 0) / scale_factor;
        bbox.score = score;
        rois.push_back(bbox);
      }
    }
    result.insert(result.end(), rois.begin(), rois.end());
  }
  return result;
}

float ComputeScaleFactor(int width, int height, int target_size, int max_size) {
  int mmin = min(width, height);
  int mmax = max(width, height);
  float scale_factor = static_cast<float>(target_size) / mmin;
  if (scale_factor * mmax > max_size) {
    scale_factor = static_cast<float>(max_size) / mmax;
  }
  return scale_factor;
}

int main(int argc, char* argv[]) {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }
  else{
    LOG(DFATAL) << "This example must run with GPU, or your PC memory will blow up";
    return 0;
  }
  Net net("../models/ssh/test_ssh.prototxt");
  net.CopyTrainedLayersFrom("../models/ssh/SSH.caffemodel");
  Mat img = imread("../ssh/demo.jpg");

  Profiler* profiler = Profiler::Get();
  profiler->TurnON();
  uint64_t tic = profiler->Now();

  //vector<float> scales{ 1200 };
  vector<float> scales{ 500, 800, 1200, 1600 };
  int max_size = 1600;
  int pyramid_min_size = 800;
  int pyramid_max_size = 1200;
  bool use_pyramid = (scales.size() != 1);

  vector<BBox> rois;
  if (!use_pyramid) {
    int width = img.cols;
    int height = img.rows;
    float scale_factor = ComputeScaleFactor(width, height, scales[0], max_size);
    Mat data;
    cv::resize(img, data, cv::Size(0, 0), scale_factor, scale_factor);
    rois = ForwardNet(net, data, scale_factor);
  }
  else {
    int width = img.cols;
    int height = img.rows;
    float base_scale_factor = ComputeScaleFactor(width, height, pyramid_min_size, pyramid_max_size);
    for (int i = 0; i < scales.size(); i++) {
      float scale_factor = scales[i] / pyramid_min_size * base_scale_factor;
      Mat data;
      cv::resize(img, data, cv::Size(0, 0), scale_factor, scale_factor);
      bool keep_m3 = (i < scales.size() - 1 ? true : false);
      vector<BBox> this_rois = ForwardNet(net, data, scale_factor, keep_m3);
      rois.insert(rois.end(), this_rois.begin(), this_rois.end());
    }
  }
  float nms_thresh = 0.3f;
  vector<int> keep = NonMaximumSuppression(rois, nms_thresh);

  uint64_t toc = profiler->Now();
  profiler->TurnOFF();
  profiler->DumpProfile("./ssh-profile.json");
  LOG(INFO) << "Time cost " << double(toc - tic) / 1000 << " ms";

  for (int i = 0; i < keep.size(); i++) {
    BBox& bbox = rois[keep[i]];
    cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
    cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
    char buff[32];
    sprintf(buff, "%.2f", bbox.score);
    cv::putText(img, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
  }
  cv::imwrite("./ssh-result.jpg", img);
  cv::imshow("result", img);
  cv::waitKey(0);
  return 0;
}
