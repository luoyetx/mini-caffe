#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <map>
#include <string>

#include "./io.hpp"
#include "./upgrade_proto.hpp"
#include "../proto/caffe.pb.h"

namespace caffe {

bool NetNeedsUpgrade(const NetParameter& net_param) {
  return NetNeedsV1ToV2Upgrade(net_param) || NetNeedsInputUpgrade(net_param);
}

bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param) {
  bool success = true;
  if (NetNeedsV1ToV2Upgrade(*param)) {
    DLOG(INFO) << "Attempting to upgrade input file specified using deprecated "
              << "V1LayerParameter: " << param_file;
    NetParameter original_param(*param);
    if (!UpgradeV1Net(original_param, param)) {
      success = false;
      LOG(ERROR) << "Warning: had one or more problems upgrading "
                 << "V1LayerParameter (see above); continuing anyway.";
    } else {
      DLOG(INFO) << "Successfully upgraded file specified using deprecated "
                << "V1LayerParameter";
    }
  }
  // NetParameter uses old style input fields; try to upgrade it.
  if (NetNeedsInputUpgrade(*param)) {
    DLOG(INFO) << "Attempting to upgrade input file specified using deprecated "
              << "input fields: " << param_file;
    UpgradeNetInput(param);
    DLOG(INFO) << "Successfully upgraded file specified using deprecated "
              << "input fields.";
    DLOG(WARNING) << "Note that future Caffe releases will only support "
                 << "input layers and not input fields.";
  }
  return success;
}

void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param) {
  CHECK(ReadProtoFromTextFile(param_file, param))
      << "Failed to parse NetParameter file: " << param_file;
  UpgradeNetAsNeeded(param_file, param);
}

void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param) {
  CHECK(ReadProtoFromBinaryFile(param_file, param))
      << "Failed to parse NetParameter file: " << param_file;
  UpgradeNetAsNeeded(param_file, param);
}

bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param) {
  return net_param.layers_size() > 0;
}

bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param) {
  if (v1_net_param.layer_size() > 0) {
    LOG(FATAL) << "Refusing to upgrade inconsistent NetParameter input; "
        << "the definition includes both 'layer' and 'layers' fields. "
        << "The current format defines 'layer' fields with string type like "
        << "layer { type: 'Layer' ... } and not layers { type: LAYER ... }. "
        << "Manually switch the definition to 'layer' format to continue.";
  }
  bool is_fully_compatible = true;
  net_param->CopyFrom(v1_net_param);
  net_param->clear_layers();
  net_param->clear_layer();
  for (int i = 0; i < v1_net_param.layers_size(); ++i) {
    if (!UpgradeV1LayerParameter(v1_net_param.layers(i),
                                 net_param->add_layer())) {
      LOG(ERROR) << "Upgrade of input layer " << i << " failed.";
      is_fully_compatible = false;
    }
  }
  return is_fully_compatible;
}

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param) {
  layer_param->Clear();
  bool is_fully_compatible = true;
  for (int i = 0; i < v1_layer_param.bottom_size(); ++i) {
    layer_param->add_bottom(v1_layer_param.bottom(i));
  }
  for (int i = 0; i < v1_layer_param.top_size(); ++i) {
    layer_param->add_top(v1_layer_param.top(i));
  }
  if (v1_layer_param.has_name()) {
    layer_param->set_name(v1_layer_param.name());
  }
  for (int i = 0; i < v1_layer_param.include_size(); ++i) {
    layer_param->add_include()->CopyFrom(v1_layer_param.include(i));
  }
  for (int i = 0; i < v1_layer_param.exclude_size(); ++i) {
    layer_param->add_exclude()->CopyFrom(v1_layer_param.exclude(i));
  }
  if (v1_layer_param.has_type()) {
    layer_param->set_type(UpgradeV1LayerType(v1_layer_param.type()));
  }
  for (int i = 0; i < v1_layer_param.blobs_size(); ++i) {
    layer_param->add_blobs()->CopyFrom(v1_layer_param.blobs(i));
  }
  for (int i = 0; i < v1_layer_param.param_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_name(v1_layer_param.param(i));
  }
  ParamSpec_DimCheckMode mode;
  for (int i = 0; i < v1_layer_param.blob_share_mode_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    switch (v1_layer_param.blob_share_mode(i)) {
    case V1LayerParameter_DimCheckMode_STRICT:
      mode = ParamSpec_DimCheckMode_STRICT;
      break;
    case V1LayerParameter_DimCheckMode_PERMISSIVE:
      mode = ParamSpec_DimCheckMode_PERMISSIVE;
      break;
    default:
      LOG(FATAL) << "Unknown blob_share_mode: "
                 << v1_layer_param.blob_share_mode(i);
      break;
    }
    layer_param->mutable_param(i)->set_share_mode(mode);
  }
  for (int i = 0; i < v1_layer_param.blobs_lr_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_lr_mult(v1_layer_param.blobs_lr(i));
  }
  for (int i = 0; i < v1_layer_param.weight_decay_size(); ++i) {
    while (layer_param->param_size() <= i) { layer_param->add_param(); }
    layer_param->mutable_param(i)->set_decay_mult(
        v1_layer_param.weight_decay(i));
  }
  for (int i = 0; i < v1_layer_param.loss_weight_size(); ++i) {
    layer_param->add_loss_weight(v1_layer_param.loss_weight(i));
  }
  if (v1_layer_param.has_concat_param()) {
    layer_param->mutable_concat_param()->CopyFrom(
        v1_layer_param.concat_param());
  }
  if (v1_layer_param.has_convolution_param()) {
    layer_param->mutable_convolution_param()->CopyFrom(
        v1_layer_param.convolution_param());
  }
  if (v1_layer_param.has_dropout_param()) {
    layer_param->mutable_dropout_param()->CopyFrom(
        v1_layer_param.dropout_param());
  }
  if (v1_layer_param.has_eltwise_param()) {
    layer_param->mutable_eltwise_param()->CopyFrom(
        v1_layer_param.eltwise_param());
  }
  if (v1_layer_param.has_exp_param()) {
    layer_param->mutable_exp_param()->CopyFrom(
        v1_layer_param.exp_param());
  }
  if (v1_layer_param.has_inner_product_param()) {
    layer_param->mutable_inner_product_param()->CopyFrom(
        v1_layer_param.inner_product_param());
  }
  if (v1_layer_param.has_lrn_param()) {
    layer_param->mutable_lrn_param()->CopyFrom(
        v1_layer_param.lrn_param());
  }
  if (v1_layer_param.has_mvn_param()) {
    layer_param->mutable_mvn_param()->CopyFrom(
        v1_layer_param.mvn_param());
  }
  if (v1_layer_param.has_pooling_param()) {
    layer_param->mutable_pooling_param()->CopyFrom(
        v1_layer_param.pooling_param());
  }
  if (v1_layer_param.has_power_param()) {
    layer_param->mutable_power_param()->CopyFrom(
        v1_layer_param.power_param());
  }
  if (v1_layer_param.has_relu_param()) {
    layer_param->mutable_relu_param()->CopyFrom(
        v1_layer_param.relu_param());
  }
  if (v1_layer_param.has_sigmoid_param()) {
    layer_param->mutable_sigmoid_param()->CopyFrom(
        v1_layer_param.sigmoid_param());
  }
  if (v1_layer_param.has_softmax_param()) {
    layer_param->mutable_softmax_param()->CopyFrom(
        v1_layer_param.softmax_param());
  }
  if (v1_layer_param.has_slice_param()) {
    layer_param->mutable_slice_param()->CopyFrom(
        v1_layer_param.slice_param());
  }
  if (v1_layer_param.has_tanh_param()) {
    layer_param->mutable_tanh_param()->CopyFrom(
        v1_layer_param.tanh_param());
  }
  if (v1_layer_param.has_threshold_param()) {
    layer_param->mutable_threshold_param()->CopyFrom(
        v1_layer_param.threshold_param());
  }
  if (v1_layer_param.has_loss_param()) {
    layer_param->mutable_loss_param()->CopyFrom(
        v1_layer_param.loss_param());
  }
  return is_fully_compatible;
}

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type) {
  switch (type) {
  case V1LayerParameter_LayerType_NONE:
    return "";
  case V1LayerParameter_LayerType_ABSVAL:
    return "AbsVal";
  case V1LayerParameter_LayerType_BNLL:
    return "BNLL";
  case V1LayerParameter_LayerType_CONCAT:
    return "Concat";
  case V1LayerParameter_LayerType_CONVOLUTION:
    return "Convolution";
  case V1LayerParameter_LayerType_DECONVOLUTION:
    return "Deconvolution";
  case V1LayerParameter_LayerType_DROPOUT:
    return "Dropout";
  case V1LayerParameter_LayerType_ELTWISE:
    return "Eltwise";
  case V1LayerParameter_LayerType_EXP:
    return "Exp";
  case V1LayerParameter_LayerType_FLATTEN:
    return "Flatten";
  case V1LayerParameter_LayerType_INNER_PRODUCT:
    return "InnerProduct";
  case V1LayerParameter_LayerType_LRN:
    return "LRN";
  case V1LayerParameter_LayerType_MVN:
    return "MVN";
  case V1LayerParameter_LayerType_POOLING:
    return "Pooling";
  case V1LayerParameter_LayerType_POWER:
    return "Power";
  case V1LayerParameter_LayerType_RELU:
    return "ReLU";
  case V1LayerParameter_LayerType_SIGMOID:
    return "Sigmoid";
  case V1LayerParameter_LayerType_SOFTMAX:
    return "Softmax";
  case V1LayerParameter_LayerType_SPLIT:
    return "Split";
  case V1LayerParameter_LayerType_SLICE:
    return "Slice";
  case V1LayerParameter_LayerType_TANH:
    return "TanH";
  case V1LayerParameter_LayerType_THRESHOLD:
    return "Threshold";
  default:
    LOG(FATAL) << "Unknown V1LayerParameter layer type: " << type;
    return "";
  }
}

bool NetNeedsInputUpgrade(const NetParameter& net_param) {
  return net_param.input_size() > 0;
}

void UpgradeNetInput(NetParameter* net_param) {
  // Collect inputs and convert to Input layer definitions.
  // If the NetParameter holds an input alone, without shape/dim, then
  // it's a legacy caffemodel and simply stripping the input field is enough.
  bool has_shape = net_param->input_shape_size() > 0;
  bool has_dim = net_param->input_dim_size() > 0;
  if (has_shape || has_dim) {
    LayerParameter* layer_param = net_param->add_layer();
    layer_param->set_name("input");
    layer_param->set_type("Input");
    InputParameter* input_param = layer_param->mutable_input_param();
    // Convert input fields into a layer.
    for (int i = 0; i < net_param->input_size(); ++i) {
      layer_param->add_top(net_param->input(i));
      if (has_shape) {
        input_param->add_shape()->CopyFrom(net_param->input_shape(i));
      } else {
        // Turn legacy input dimensions into shape.
        BlobShape* shape = input_param->add_shape();
        int first_dim = i*4;
        int last_dim = first_dim + 4;
        for (int j = first_dim; j < last_dim; j++) {
          shape->add_dim(net_param->input_dim(j));
        }
      }
    }
    // Swap input layer to beginning of net to satisfy layer dependencies.
    for (int i = net_param->layer_size() - 1; i > 0; --i) {
      net_param->mutable_layer(i-1)->Swap(net_param->mutable_layer(i));
    }
  }
  // Clear inputs.
  net_param->clear_input();
  net_param->clear_input_shape();
  net_param->clear_input_dim();
}

}  // namespace caffe
