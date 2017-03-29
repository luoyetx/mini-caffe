#include "./proposal_layer.hpp"
#include "../util/nms.hpp"

namespace caffe {

/*!
 * \brief transform bbox
 * \param box bbox
 * \param dx, dy, d_log_w, d_log_h bbox offset
 * \param img_width, img_height
 * \param min_bbox_size minimum size of box
 * \return 1 if bbox is valid (> min_bbox_size), 0 if not
 */
static int TransformBBox(real_t* box,
                         const real_t dx, const real_t dy,
                         const real_t d_log_w, const real_t d_log_h,
                         const real_t img_width, const real_t img_height,
                         const real_t min_bbox_size) {
  // width & height of box
  const real_t w = box[2] - box[0] + 1;
  const real_t h = box[3] - box[1] + 1;
  // center location of box
  const real_t ctr_x = box[0] + 0.5f * w;
  const real_t ctr_y = box[1] + 0.5f * h;

  // new center location according to offset (dx, dy)
  const real_t pred_ctr_x = dx * w + ctr_x;
  const real_t pred_ctr_y = dy * h + ctr_y;
  // new width & height according to offset d(log w), d(log h)
  const real_t pred_w = exp(d_log_w) * w;
  const real_t pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - 0.5f * pred_w;
  box[1] = pred_ctr_y - 0.5f * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + 0.5f * pred_w;
  box[3] = pred_ctr_y + 0.5f * pred_h;

  // adjust new corner locations to be within the image region
  // clip bbox
  box[0] = std::max(static_cast<real_t>(0), std::min(box[0], img_width - 1));
  box[1] = std::max(static_cast<real_t>(0), std::min(box[1], img_height - 1));
  box[2] = std::max(static_cast<real_t>(0), std::min(box[2], img_width - 1));
  box[3] = std::max(static_cast<real_t>(0), std::min(box[3], img_height - 1));

  // recompute new width & height
  const real_t box_w = box[2] - box[0] + 1;
  const real_t box_h = box[3] - box[1] + 1;

  // check if new box's size >= threshold
  return (box_w >= min_bbox_size) && (box_h >= min_bbox_size);
}

/*! \brief sort rois by score */
static void SortBox(real_t* rois, const int left, const int right,
                    const int num_top) {
  const real_t pivot_score = rois[(left + right) / 2 * 5 + 4];
  int i, j;
  i = left; j = right;
  do {
    while (rois[5 * i + 4] > pivot_score) i++;
    while (rois[5 * j + 4] < pivot_score) j--;
    if (i <= j) {
      for (int t = 0; t < 5; t++) {
        std::swap(rois[5 * i + t], rois[5 * j + t]);
      }
      i++;
      j--;
    }
  } while (i <= j);
  // sort [left, j]
  if (left < j) SortBox(rois, left, j, num_top);
  // sort [i, right], if i > num_top, no need to sort
  if (i < right && i <= num_top) SortBox(rois, i, right, num_top);
}

/*! \brief generate base anchors */
static void GenerateAnchors(int base_size,
                             const vector<real_t> ratios,
                             const vector<real_t> scales,
                             Blob& anchors_) {
  // base box's width & height & center location
  const real_t base_area = static_cast<real_t>(base_size * base_size);
  const real_t center = static_cast<real_t>(0.5 * (base_size - 1));
  // enumerate all transformed boxes
  real_t *anchors = anchors_.mutable_cpu_data();
  for (int i = 0; i < ratios.size(); ++i) {
    // transformed width & height for given ratio factors
    const real_t ratio_w = static_cast<real_t>(std::round(std::sqrt(base_area / ratios[i])));
    const real_t ratio_h = static_cast<real_t>(std::round(ratio_w * ratios[i]));
    for (int j = 0; j < scales.size(); ++j) {
      // transformed width & height for given scale factors
      const real_t scale_w = 0.5 * static_cast<real_t>(ratio_w * scales[j] - 1);
      const real_t scale_h = 0.5 * static_cast<real_t>(ratio_h * scales[j] - 1);
      // (x1, y1, x2, y2) for transformed box
      const real_t x1 = center - scale_w;
      const real_t x2 = center + scale_w;
      const real_t y1 = center - scale_h;
      const real_t y2 = center + scale_h;
      anchors[0] = x1;
      anchors[1] = y1;
      anchors[2] = x2;
      anchors[3] = y2;
      anchors += 4;
    }
  }
}
static void sort_box(real_t list_cpu[], const int start, const int end,
                     const int num_top) {
  const real_t pivot_score = list_cpu[start * 5 + 4];
  int left = start + 1, right = end;
  real_t temp[5];
  while (left <= right)
  {
    while (left <= end && list_cpu[left * 5 + 4] >= pivot_score)
      ++left;
    while (right > start && list_cpu[right * 5 + 4] <= pivot_score)
      --right;
    if (left <= right)
    {
      for (int i = 0; i < 5; ++i)
      {
        temp[i] = list_cpu[left * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[left * 5 + i] = list_cpu[right * 5 + i];
      }
      for (int i = 0; i < 5; ++i)
      {
        list_cpu[right * 5 + i] = temp[i];
      }
      ++left;
      --right;
    }
  }

  if (right > start)
  {
    for (int i = 0; i < 5; ++i)
    {
      temp[i] = list_cpu[start * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[start * 5 + i] = list_cpu[right * 5 + i];
    }
    for (int i = 0; i < 5; ++i)
    {
      list_cpu[right * 5 + i] = temp[i];
    }
  }

  if (start < right - 1)
  {
    sort_box(list_cpu, start, right - 1, num_top);
  }
  if (right + 1 < num_top && right + 1 < end)
  {
    sort_box(list_cpu, right + 1, end, num_top);
  }
}

/*! \brief generate proposal bboxes based on base anchors and feature map */
static void GenerateProposalsCPU(const real_t* score_map,
                                 const real_t* bbox_map,
                                 const real_t* anchors,
                                 real_t* proposals,
                                 const int num_anchors,
                                 const int fm_height, const int fm_width,
                                 const real_t img_height, const real_t img_width,
                                 const real_t min_bbox_size, const int feat_stride) {
  const int fm_stride = fm_height * fm_width;
  real_t* proposal = proposals;

  for (int h = 0; h < fm_height; ++h) {
    for (int w = 0; w < fm_width; ++w) {
      const real_t x = w * feat_stride;
      const real_t y = h * feat_stride;
      const real_t* box = bbox_map + h * fm_width + w;
      const real_t *score = score_map + h * fm_width + w;
      for (int k = 0; k < num_anchors; ++k) {
        const real_t dx = box[(k * 4 + 0) * fm_stride];
        const real_t dy = box[(k * 4 + 1) * fm_stride];
        const real_t d_log_w = box[(k * 4 + 2) * fm_stride];
        const real_t d_log_h = box[(k * 4 + 3) * fm_stride];

        proposal[0] = x + anchors[k * 4 + 0];
        proposal[1] = y + anchors[k * 4 + 1];
        proposal[2] = x + anchors[k * 4 + 2];
        proposal[3] = y + anchors[k * 4 + 3];
        proposal[4] = TransformBBox(proposal, dx, dy, d_log_w, d_log_h,
                                    img_width, img_height, min_bbox_size) *
                          score[k * fm_stride];
        proposal += 5;
      }
    }
  }
}

static void NonMaximumSuppressionCPU(const int num_proposals,
                                     const real_t* proposals,
                                     int* rois_indices,
                                     int& num_rois,
                                     const real_t nms_th,
                                     const int max_num_rois) {
  typedef std::multimap<real_t, int> ScoreMapper;
  ScoreMapper sm;
  vector<float> areas(num_proposals);
  const real_t* proposal = proposals;
  for (int i = 0; i < num_proposals; i++) {
    areas[i] = (proposal[2] - proposal[0] + 1)*(proposal[3] - proposal[1] + 1);
    sm.insert(ScoreMapper::value_type(proposal[4], i));
    proposal += 5;
  }
  int counter = 0;
  while (!sm.empty()) {
    int last_idx = sm.rbegin()->second;
    rois_indices[counter++] = last_idx;
    if (counter == max_num_rois) break;
    const real_t* last = proposals + last_idx * 5;
    for (ScoreMapper::iterator it = sm.begin(); it != sm.end();) {
      int idx = it->second;
      const real_t* curr = proposals + idx * 5;
      float x1 = std::max(curr[0], last[0]);
      float y1 = std::max(curr[1], last[1]);
      float x2 = std::min(curr[2], last[2]);
      float y2 = std::min(curr[3], last[3]);
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
  num_rois = counter;
}

static void RetrieveRoisCPU(const int num_rois,
                            const real_t* proposals,
                            const int* roi_indices,
                            real_t* rois,
                            real_t* roi_scores) {
  for (int i = 0; i < num_rois; ++i) {
    const real_t* proposal = proposals + roi_indices[i] * 5;
    rois[i * 5 + 0] = 0;
    rois[i * 5 + 1] = proposal[0];
    rois[i * 5 + 2] = proposal[1];
    rois[i * 5 + 3] = proposal[2];
    rois[i * 5 + 4] = proposal[3];
    if (roi_scores) {
      roi_scores[i] = proposal[4];
    }
  }
}

void ProposalLayer::LayerSetUp(const vector<Blob*> &bottom,
                               const vector<Blob*> &top) {
  ProposalParameter param = this->layer_param_.proposal_param();
  base_size_ = param.base_size();
  feat_stride_ = param.feat_stride();
  pre_nms_topn_ = param.pre_nms_topn();
  post_nms_topn_ = param.post_nms_topn();
  nms_thresh_ = param.nms_thresh();
  min_size_ = param.min_size();

  vector<real_t> ratios(param.ratio_size());
  for (int i = 0; i < param.ratio_size(); ++i) {
    ratios[i] = param.ratio(i);
  }
  vector<real_t> scales(param.scale_size());
  for (int i = 0; i < param.scale_size(); ++i) {
    scales[i] = param.scale(i);
  }

  vector<int> anchors_shape(2);
  anchors_shape[0] = ratios.size() * scales.size();
  anchors_shape[1] = 4;
  anchors_.Reshape(anchors_shape);
  GenerateAnchors(base_size_, ratios, scales, anchors_);

  vector<int> roi_indices_shape(1);
  roi_indices_shape[0] = post_nms_topn_;
  roi_indices_.Reshape(roi_indices_shape);

  // rois blob : holds R regions of interest, each is a 5 - tuple
  // (n, x1, y1, x2, y2) specifying an image batch index n and a
  // rectangle(x1, y1, x2, y2)
  vector<int> top_shape(2);
  top_shape[0] = bottom[0]->shape(0) * post_nms_topn_;
  top_shape[1] = 5;
  top[0]->Reshape(top_shape);

  // scores blob : holds scores for R regions of interest
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}

void ProposalLayer::Forward_cpu(const vector<Blob*>& bottom,
                                const vector<Blob*>& top) {
  const real_t *anchors_score_map = bottom[0]->cpu_data();
  const real_t *anchors_bbox_map = bottom[1]->cpu_data();
  const real_t *im_info = bottom[2]->cpu_data();
  real_t *rois = top[0]->mutable_cpu_data();
  real_t *rois_score = (top.size() > 1) ? top[1]->mutable_cpu_data() : nullptr;

  CHECK_EQ(bottom[0]->shape(0), 1) << "Only support single scale.";

  // bottom shape: (2 x num_anchors) x H x W
  const int fm_height = bottom[0]->height();
  const int fm_width = bottom[0]->width();
  // input image height & width
  const real_t img_height = im_info[0];
  const real_t img_width = im_info[1];
  // scale factor for height & width
  const real_t scale_factor = im_info[2];
  // minimum box width & height
  const real_t min_bbox_size = min_size_ * scale_factor;
  // number of all proposals = num_anchors * H * W
  const int num_proposals = anchors_.shape(0) * fm_height * fm_width;
  // number of top-n proposals before NMS
  const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
  // number of final RoIs
  int num_rois = 0;

  // enumerate all proposals
  //   num_proposals = num_anchors * H * W
  //   (x1, y1, x2, y2, score) for each proposal
  // NOTE: for bottom, only foreground scores are passed
  // also clip bbox inside bbox boundary and filter bbox with min_bbox_size
  vector<int> proposals_shape{num_proposals, 5};
  proposals_.Reshape(proposals_shape);
  GenerateProposalsCPU(anchors_score_map,
                       anchors_bbox_map,
                       anchors_.cpu_data(),
                       proposals_.mutable_cpu_data(),
                       anchors_.shape(0),
                       fm_height, fm_width,
                       img_height, img_width,
                       min_bbox_size, feat_stride_);

  //SortBox(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);
  sort_box(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);

  //NonMaximumSuppressionCPU(pre_nms_topn, proposals_.cpu_data(),
  //                         roi_indices_.mutable_cpu_data(), num_rois,
  //                         nms_thresh_, post_nms_topn_);
  nms_cpu(pre_nms_topn, proposals_.cpu_data(),
          roi_indices_.mutable_cpu_data(), &num_rois,
          0, nms_thresh_, post_nms_topn_);

  RetrieveRoisCPU(num_rois, proposals_.cpu_data(), roi_indices_.cpu_data(),
                  rois, rois_score);

  // reshape if num_rois < post_nms_topn_
  vector<int> top_shape{num_rois, 5};
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}

#ifndef USE_CUDA
STUB_GPU(ProposalLayer);
#endif

REGISTER_LAYER_CLASS(Proposal);

} // namespace caffe
