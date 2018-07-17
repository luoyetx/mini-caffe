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
  float x1, y1, x2, y2;
};

static inline void TransforBBox(BBox& bbox,
                                const float dx, const float dy,
                                const float d_log_w, const float d_log_h,
                                const float img_width, const float img_height) {
  const float w = bbox.x2 - bbox.x1 + 1;
  const float h = bbox.y2 - bbox.y1 + 1;
  const float ctr_x = bbox.x1 + 0.5f*w;
  const float ctr_y = bbox.y1 + 0.5f*h;
  const float pred_ctr_x = dx*w + ctr_x;
  const float pred_ctr_y = dy*h + ctr_y;
  const float pred_w = exp(d_log_w)*w;
  const float pred_h = exp(d_log_h)*h;
  bbox.x1 = pred_ctr_x - 0.5f*pred_w;
  bbox.y1 = pred_ctr_y - 0.5f*pred_h;
  bbox.x2 = pred_ctr_x + 0.5f*pred_w;
  bbox.y2 = pred_ctr_y + 0.5f*pred_h;
  bbox.x1 = std::max(0.f, std::min(bbox.x1, img_width - 1));
  bbox.y1 = std::max(0.f, std::min(bbox.y1, img_height - 1));
  bbox.x2 = std::max(0.f, std::min(bbox.x2, img_width - 1));
  bbox.y2 = std::max(0.f, std::min(bbox.y2, img_height - 1));
}

static vector<int> NonMaximumSuppression(const vector<float>& score,
                                         const vector<BBox>& bboxes,
                                         const float nms_th);

int main(int argc, char* argv[]) {
  if (caffe::GPUAvailable()) {
    caffe::SetMode(caffe::GPU, 0);
  }
  Net net("../models/r-fcn/test_agnostic.prototxt");
  net.CopyTrainedLayersFrom("../models/r-fcn/resnet50_rfcn_final.caffemodel");
  net.MarkOutputs({ "rois" });

  Mat img = imread("../r-fcn/004545.jpg");

  caffe::Profiler* profiler = caffe::Profiler::Get();
  profiler->TurnON();
  uint64_t tic = profiler->Now();

  int height = img.rows;
  int width = img.cols;
  const int kSizeMin = 600;
  const int kSizeMax = 1000;
  const float kScoreThreshold = 0.8f;
  const char* kClassNames[] = { "__background__", "aeroplane", "bicycle", "bird", "boat",
                                "bottle", "bus", "car", "cat", "chair",
                                "cow", "diningtable", "dog", "horse",
                                "motorbike", "person", "pottedplant",
                                "sheep", "sofa", "train", "tvmonitor" };

  float smin = min(height, width);
  float smax = max(height, width);
  float scale_factor = kSizeMin / smin;
  if (smax * scale_factor > kSizeMax) {
    scale_factor = kSizeMax / smax;
  }
  Mat imgResized;
  cv::resize(img, imgResized, Size(0, 0), scale_factor, scale_factor);

  vector<Mat> bgr;
  cv::split(imgResized, bgr);
  bgr[0].convertTo(bgr[0], CV_32F, 1.f, -102.9801f);
  bgr[1].convertTo(bgr[1], CV_32F, 1.f, -115.9465f);
  bgr[2].convertTo(bgr[2], CV_32F, 1.f, -122.7717f);

  shared_ptr<Blob> data = net.blob_by_name("data");
  data->Reshape(1, 3, imgResized.rows, imgResized.cols);
  const int bias = data->offset(0, 1, 0, 0);
  const int bytes = bias*sizeof(float);
  memcpy(data->mutable_cpu_data() + 0 * bias, bgr[0].data, bytes);
  memcpy(data->mutable_cpu_data() + 1 * bias, bgr[1].data, bytes);
  memcpy(data->mutable_cpu_data() + 2 * bias, bgr[2].data, bytes);
  shared_ptr<Blob> im_info = net.blob_by_name("im_info");
  im_info->mutable_cpu_data()[0] = imgResized.rows;
  im_info->mutable_cpu_data()[1] = imgResized.cols;
  im_info->mutable_cpu_data()[2] = scale_factor;

  net.Forward();

  shared_ptr<Blob> rois = net.blob_by_name("rois");
  shared_ptr<Blob> cls_prob = net.blob_by_name("cls_prob");
  shared_ptr<Blob> bbox_pred = net.blob_by_name("bbox_pred");

  const int num_rois = rois->num();
  // every class
  std::vector<float> scores;
  std::vector<BBox> bboxes;
  for (int c = 1; c <= 20; c++) {
    scores.clear();
    bboxes.clear();
    for (int i = 0; i < num_rois; i++) {
      const float score = cls_prob->data_at(i, c, 0, 0);
      if (score > kScoreThreshold) {
        scores.push_back(score);
        BBox bbox;
        bbox.x1 = rois->data_at(i, 1, 0, 0);
        bbox.y1 = rois->data_at(i, 2, 0, 0);
        bbox.x2 = rois->data_at(i, 3, 0, 0);
        bbox.y2 = rois->data_at(i, 4, 0, 0);
        const float dx = bbox_pred->data_at(i, 4 + 0, 0, 0);
        const float dy = bbox_pred->data_at(i, 4 + 1, 0, 0);
        const float d_log_w = bbox_pred->data_at(i, 4 + 2, 0, 0);
        const float d_log_h = bbox_pred->data_at(i, 4 + 3, 0, 0);
        TransforBBox(bbox, dx, dy, d_log_w, d_log_h, imgResized.cols, imgResized.rows);
        bbox.x1 /= scale_factor;
        bbox.y1 /= scale_factor;
        bbox.x2 /= scale_factor;
        bbox.y2 /= scale_factor;
        bboxes.push_back(bbox);
      }
    }
    vector<int> picked = NonMaximumSuppression(scores, bboxes, 0.3);
    // draw
    const int num_picked = picked.size();
    for (int i = 0; i < num_picked; i++) {
      BBox& bbox = bboxes[picked[i]];
      cv::Rect rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1 + 1, bbox.y2 - bbox.y1 + 1);
      cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
      char buff[300];
      sprintf(buff, "%s: %.2f", kClassNames[c], scores[i]);
      cv::putText(img, buff, cv::Point(bbox.x1, bbox.y1), FONT_HERSHEY_PLAIN, 1, Scalar(0, 255, 0));
    }
  }

  uint64_t toc = profiler->Now();
  profiler->TurnOFF();
  profiler->DumpProfile("./rfcn-profile.json");

  LOG(INFO) << "Costs " << (toc - tic) / 1000.f << " ms";

  MemPoolState st = caffe::MemPoolGetState();
  auto __Calc__ = [](int size) -> double {
    return std::round(static_cast<double>(size) / (1024 * 1024) * 100) / 100;
  };
  LOG(INFO) << "[CPU] Hold " << __Calc__(st.cpu_mem) << " M, Not Uses " << __Calc__(st.unused_cpu_mem) << " M";
  LOG(INFO) << "[GPU] Hold " << __Calc__(st.gpu_mem) << " M, Not Uses " << __Calc__(st.unused_gpu_mem) << " M";
  cv::imwrite("./rfcn-result.jpg", img);
  cv::imshow("result", img);
  cv::waitKey(0);
  return 0;
}

vector<int> NonMaximumSuppression(const vector<float>& scores,
                                  const vector<BBox>& bboxes,
                                  const float nms_th) {
  typedef std::multimap<float, int> ScoreMapper;
  ScoreMapper sm;
  const int n = scores.size();
  vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = (bboxes[i].x2 - bboxes[i].x1 + 1)*(bboxes[i].y2 - bboxes[i].y1 + 1);
    sm.insert(ScoreMapper::value_type(scores[i], i));
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
