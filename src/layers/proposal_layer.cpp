#include "./proposal_layer.hpp"
#include "../util/nms.hpp"

using std::max;
using std::min;

namespace caffe {

static int transform_box(real_t box[],
                         const real_t dx, const real_t dy,
                         const real_t d_log_w, const real_t d_log_h,
                         const real_t img_W, const real_t img_H,
                         const real_t min_box_W, const real_t min_box_H) {
  // width & height of box
  const real_t w = box[2] - box[0] + (real_t)1;
  const real_t h = box[3] - box[1] + (real_t)1;
  // center location of box
  const real_t ctr_x = box[0] + (real_t)0.5 * w;
  const real_t ctr_y = box[1] + (real_t)0.5 * h;

  // new center location according to gradient (dx, dy)
  const real_t pred_ctr_x = dx * w + ctr_x;
  const real_t pred_ctr_y = dy * h + ctr_y;
  // new width & height according to gradient d(log w), d(log h)
  const real_t pred_w = exp(d_log_w) * w;
  const real_t pred_h = exp(d_log_h) * h;

  // update upper-left corner location
  box[0] = pred_ctr_x - (real_t)0.5 * pred_w;
  box[1] = pred_ctr_y - (real_t)0.5 * pred_h;
  // update lower-right corner location
  box[2] = pred_ctr_x + (real_t)0.5 * pred_w;
  box[3] = pred_ctr_y + (real_t)0.5 * pred_h;

  // adjust new corner locations to be within the image region,
  box[0] = std::max((real_t)0, std::min(box[0], img_W - (real_t)1));
  box[1] = std::max((real_t)0, std::min(box[1], img_H - (real_t)1));
  box[2] = std::max((real_t)0, std::min(box[2], img_W - (real_t)1));
  box[3] = std::max((real_t)0, std::min(box[3], img_H - (real_t)1));

  // recompute new width & height
  const real_t box_w = box[2] - box[0] + (real_t)1;
  const real_t box_h = box[3] - box[1] + (real_t)1;

  // check if new box's size >= threshold
  return (box_w >= min_box_W) * (box_h >= min_box_H);
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

static void generate_anchors(int base_size,
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

static void enumerate_proposals_cpu(const real_t bottom4d[],
                                    const real_t d_anchor4d[],
                                    const real_t anchors[],
                                    real_t proposals[],
                                    const int num_anchors,
                                    const int bottom_H, const int bottom_W,
                                    const real_t img_H, const real_t img_W,
                                    const real_t min_box_H, const real_t min_box_W,
                                    const int feat_stride) {
  real_t *p_proposal = proposals;
  const int bottom_area = bottom_H * bottom_W;

  for (int h = 0; h < bottom_H; ++h) {
    for (int w = 0; w < bottom_W; ++w) {
      const real_t x = w * feat_stride;
      const real_t y = h * feat_stride;
      const real_t *p_box = d_anchor4d + h * bottom_W + w;
      const real_t *p_score = bottom4d + h * bottom_W + w;
      for (int k = 0; k < num_anchors; ++k) {
        const real_t dx = p_box[(k * 4 + 0) * bottom_area];
        const real_t dy = p_box[(k * 4 + 1) * bottom_area];
        const real_t d_log_w = p_box[(k * 4 + 2) * bottom_area];
        const real_t d_log_h = p_box[(k * 4 + 3) * bottom_area];

        p_proposal[0] = x + anchors[k * 4 + 0];
        p_proposal[1] = y + anchors[k * 4 + 1];
        p_proposal[2] = x + anchors[k * 4 + 2];
        p_proposal[3] = y + anchors[k * 4 + 3];
        p_proposal[4] = transform_box(p_proposal,
                                      dx, dy, d_log_w, d_log_h,
                                      img_W, img_H, min_box_W, min_box_H) *
                        p_score[k * bottom_area];
        p_proposal += 5;
      } // endfor k
    }   // endfor w
  }     // endfor h
}

static void retrieve_rois_cpu(const int num_rois,
                              const int item_index,
                              const real_t proposals[],
                              const int roi_indices[],
                              real_t rois[],
                              real_t roi_scores[]) {
  for (int i = 0; i < num_rois; ++i) {
    const real_t *const proposals_index = proposals + roi_indices[i] * 5;
    rois[i * 5 + 0] = item_index;
    rois[i * 5 + 1] = proposals_index[0];
    rois[i * 5 + 2] = proposals_index[1];
    rois[i * 5 + 3] = proposals_index[2];
    rois[i * 5 + 4] = proposals_index[3];
    if (roi_scores) {
      roi_scores[i] = proposals_index[4];
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
  generate_anchors(base_size_, ratios, scales, anchors_);

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

void ProposalLayer::Forward_cpu(const vector<Blob*> &bottom,
                                const vector<Blob*> &top) {
  const real_t *p_bottom_item = bottom[0]->cpu_data();
  const real_t *p_d_anchor_item = bottom[1]->cpu_data();
  const real_t *p_img_info_cpu = bottom[2]->cpu_data();
  real_t *p_roi_item = top[0]->mutable_cpu_data();
  real_t *p_score_item = (top.size() > 1) ? top[1]->mutable_cpu_data() : NULL;

  vector<int> proposals_shape(2);
  vector<int> top_shape(2);
  proposals_shape[0] = 0;
  proposals_shape[1] = 5;
  top_shape[0] = 0;
  top_shape[1] = 5;

  for (int n = 0; n < bottom[0]->shape(0); ++n) {
    // bottom shape: (2 x num_anchors) x H x W
    const int bottom_H = bottom[0]->height();
    const int bottom_W = bottom[0]->width();
    // input image height & width
    const real_t img_H = p_img_info_cpu[0];
    const real_t img_W = p_img_info_cpu[1];
    // scale factor for height & width
    const real_t scale_H = p_img_info_cpu[2];
    const real_t scale_W = p_img_info_cpu[3];
    // minimum box width & height
    const real_t min_box_H = min_size_ * scale_H;
    const real_t min_box_W = min_size_ * scale_W;
    // number of all proposals = num_anchors * H * W
    const int num_proposals = anchors_.shape(0) * bottom_H * bottom_W;
    // number of top-n proposals before NMS
    const int pre_nms_topn = std::min(num_proposals, pre_nms_topn_);
    // number of final RoIs
    int num_rois = 0;

    // enumerate all proposals
    //   num_proposals = num_anchors * H * W
    //   (x1, y1, x2, y2, score) for each proposal
    // NOTE: for bottom, only foreground scores are passed
    proposals_shape[0] = num_proposals;
    proposals_.Reshape(proposals_shape);
    enumerate_proposals_cpu(
        p_bottom_item + num_proposals, p_d_anchor_item,
        anchors_.cpu_data(), proposals_.mutable_cpu_data(), anchors_.shape(0),
        bottom_H, bottom_W, img_H, img_W, min_box_H, min_box_W,
        feat_stride_);

    sort_box(proposals_.mutable_cpu_data(), 0, num_proposals - 1, pre_nms_topn_);

    nms_cpu(pre_nms_topn, proposals_.cpu_data(),
            roi_indices_.mutable_cpu_data(), &num_rois,
            0, nms_thresh_, post_nms_topn_);

    retrieve_rois_cpu(
        num_rois, n, proposals_.cpu_data(), roi_indices_.cpu_data(),
        p_roi_item, p_score_item);

    top_shape[0] += num_rois;

    p_bottom_item += bottom[0]->offset(1);
    p_d_anchor_item += bottom[1]->offset(1);
    p_roi_item += num_rois * 5;
    p_score_item += num_rois * 1;
  }

  top[0]->Reshape(top_shape);
  if (top.size() > 1)
  {
    top_shape.pop_back();
    top[1]->Reshape(top_shape);
  }
}

#ifndef USE_CUDA
STUB_GPU(ProposalLayer);
#endif

REGISTER_LAYER_CLASS(Proposal);

} // namespace caffe
