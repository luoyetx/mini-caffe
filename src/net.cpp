#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>

#include "caffe/net.hpp"
#include "caffe/profiler.hpp"
#include "./layer.hpp"
#include "./util/math_functions.hpp"
#include "./util/upgrade_proto.hpp"
#include "./proto/caffe.pb.h"

namespace caffe {

Net::Net(const string& param_file) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

void Net::Init(const NetParameter& param) {
  // Basically, build all the layers and set up their connections.
  name_ = param.name();
  std::map<string, int> blob_name_to_idx;
  std::set<string> available_blobs;
  // For each layer, set up its input and output
  bottom_vecs_.resize(param.layer_size());
  top_vecs_.resize(param.layer_size());
  bottom_id_vecs_.resize(param.layer_size());
  top_id_vecs_.resize(param.layer_size());
  param_id_vecs_.resize(param.layer_size());
  for (int layer_id = 0; layer_id < param.layer_size(); ++layer_id) {
    // Setup layer.
    const LayerParameter& layer_param = param.layer(layer_id);
    layers_.push_back(LayerRegistry::CreateLayer(layer_param));
    layer_names_.push_back(layer_param.name());
    // Figure out this layer's input and output
    const int num_bottom = layer_param.bottom_size();
    for (int bottom_id = 0; bottom_id < num_bottom; ++bottom_id) {
      AppendBottom(param, layer_id, bottom_id, &available_blobs, &blob_name_to_idx);
    }
    const int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
    }
    // After this layer is connected, set it up.
    layers_[layer_id]->SetUp(bottom_vecs_[layer_id], top_vecs_[layer_id]);
    // Layer Parameters
    const int num_param_blobs = layers_[layer_id]->blobs().size();
    for (int param_id = 0; param_id < num_param_blobs; ++param_id) {
      AppendParam(param, layer_id, param_id);
    }
  }
  for (size_t blob_id = 0; blob_id < blob_names_.size(); ++blob_id) {
    blob_names_index_[blob_names_[blob_id]] = blob_id;
  }
  for (size_t layer_id = 0; layer_id < layer_names_.size(); ++layer_id) {
    layer_names_index_[layer_names_[layer_id]] = layer_id;
  }
}

// Helper for Net::Init: add a new top blob to the net.
void Net::AppendTop(const NetParameter& param, const int layer_id,
                    const int top_id, std::set<string>* available_blobs,
                    std::map<string, int>* blob_name_to_idx) {
  shared_ptr<LayerParameter> layer_param(
      new LayerParameter(param.layer(layer_id)));
  const string& blob_name = (layer_param->top_size() > top_id) ?
      layer_param->top(top_id) : "(automatic)";
  // Check if we are doing in-place computation
  if (blob_name_to_idx && layer_param->bottom_size() > top_id &&
      blob_name == layer_param->bottom(top_id)) {
    // In-place computation
    top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
    top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
  } else if (blob_name_to_idx &&
             blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()) {
    // If we are not doing in-place computation but have duplicated blobs,
    // raise an error.
    LOG(FATAL) << "Top blob '" << blob_name
               << "' produced by multiple sources.";
  } else {
    // Normal output.
    shared_ptr<Blob> blob_pointer(new Blob);
    const int blob_id = blobs_.size();
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    if (blob_name_to_idx) { (*blob_name_to_idx)[blob_name] = blob_id; }
    top_id_vecs_[layer_id].push_back(blob_id);
    top_vecs_[layer_id].push_back(blob_pointer.get());
  }
  if (available_blobs) { available_blobs->insert(blob_name); }
}

// Helper for Net::Init: add a new bottom blob to the net.
int Net::AppendBottom(const NetParameter& param, const int layer_id,
                      const int bottom_id, std::set<string>* available_blobs,
                      std::map<string, int>* blob_name_to_idx) {
  const LayerParameter& layer_param = param.layer(layer_id);
  const string& blob_name = layer_param.bottom(bottom_id);
  if (available_blobs->find(blob_name) == available_blobs->end()) {
    LOG(FATAL) << "Unknown bottom blob '" << blob_name << "' (layer '"
               << layer_param.name() << "', bottom index " << bottom_id << ")";
  }
  const int blob_id = (*blob_name_to_idx)[blob_name];
  bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
  bottom_id_vecs_[layer_id].push_back(blob_id);
  return blob_id;
}

void Net::AppendParam(const NetParameter& param, const int layer_id,
                      const int param_id) {
  const LayerParameter& layer_param = layers_[layer_id]->layer_param();
  const int param_size = layer_param.param_size();
  string param_name =
    (param_size > param_id) ? layer_param.param(param_id).name() : "";
  if (param_name.size()) {
    param_display_names_.push_back(param_name);
  }
  else {
    std::ostringstream param_display_name;
    param_display_name << layer_param.name() << "_" << param_id;
    param_display_names_.push_back(param_display_name.str());
  }
  const int net_param_id = params_.size();
  params_.push_back(layers_[layer_id]->blobs()[param_id]);
  param_id_vecs_[layer_id].push_back(net_param_id);
}

real_t Net::MemSize() const {
  size_t memory_used_ = 0;
  for (auto blob : this->blobs_) {
    memory_used_ += blob->count()*sizeof(real_t);
  }
  for (auto layer : this->layers_) {
    for (auto param : layer->blobs()) {
      memory_used_ += param->count()*sizeof(real_t);
    }
  }
  return static_cast<real_t>(memory_used_) / (1024 * 1024);
}

void Net::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Profiler *profiler = Profiler::Get();
  for (int i = start; i <= end; ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    profiler->ScopeStart(layers_[i]->type());
    layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    profiler->ScopeEnd();
  }
}

void Net::ForwardFrom(int start) {
  return ForwardFromTo(start, layers_.size() - 1);
}

void Net::ForwardTo(int end) {
  return ForwardFromTo(0, end);
}

void Net::Reshape() {
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Reshape(bottom_vecs_[i], top_vecs_[i]);
  }
}

void Net::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      continue;
    }
    vector<shared_ptr<Blob> >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}

void Net::CopyTrainedLayersFrom(const string& trained_filename) {
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

void Net::ToProto(NetParameter* param) const {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < layers_.size(); ++i) {
    LayerParameter* layer_param = param->add_layer();
    layers_[i]->ToProto(layer_param);
  }
}

bool Net::has_blob(const string& blob_name) const {
  return blob_names_index_.find(blob_name) != blob_names_index_.end();
}

const shared_ptr<Blob> Net::blob_by_name(const string& blob_name) const {
  shared_ptr<Blob> blob_ptr;
  CHECK(has_blob(blob_name)) << "Unknown blob name " << blob_name;
  blob_ptr = blobs_[blob_names_index_.find(blob_name)->second];
  return blob_ptr;
}

bool Net::has_layer(const string& layer_name) const {
  return layer_names_index_.find(layer_name) != layer_names_index_.end();
}

const shared_ptr<Layer> Net::layer_by_name(const string& layer_name) const {
  shared_ptr<Layer> layer_ptr;
  CHECK(has_layer(layer_name)) << "Unknown layer name " << layer_name;
  layer_ptr = layers_[layer_names_index_.find(layer_name)->second];
  return layer_ptr;
}

shared_ptr<NetParameter> ReadTextNetParameterFromFile(const string& file) {
  shared_ptr<NetParameter> np(new NetParameter);
  ReadNetParamsFromTextFileOrDie(file, np.get());
  return np;
}

shared_ptr<NetParameter> ReadTextNetParameterFromBuffer(const char* buffer, int buffer_len) {
  shared_ptr<NetParameter> np(new NetParameter);
  CHECK(google::protobuf::TextFormat::ParseFromString(std::string(buffer, buffer_len), np.get()))
    << "Parse Text NetParameter from Buffer failed";
  return np;
}

shared_ptr<NetParameter> ReadBinaryNetParameterFromFile(const string& file) {
  shared_ptr<NetParameter> np(new NetParameter);
  ReadNetParamsFromBinaryFileOrDie(file, np.get());
  return np;
}

shared_ptr<NetParameter> ReadBinaryNetParameterFromBuffer(const char* buffer, int buffer_len) {
  using google::protobuf::uint8;
  shared_ptr<NetParameter> np(new NetParameter);
  google::protobuf::io::CodedInputStream ci(reinterpret_cast<const uint8*>(buffer), buffer_len);
  CHECK(np->ParseFromCodedStream(&ci)) << "Parse Binary NetParameter from Buffer failed";
  return np;
}

}  // namespace caffe
