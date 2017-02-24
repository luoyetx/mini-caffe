/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers could be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>
#include <functional>

#include "caffe/base.hpp"
#include "./proto/caffe.pb.h"

namespace caffe {

class Layer;

class LayerRegistry {
 public:
  using Creator = std::function<shared_ptr<Layer>(const LayerParameter&)>;
  using CreatorRegistry = std::map<string, Creator>;

  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer> CreateLayer(const LayerParameter& param) {
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (auto iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // static variables.
  LayerRegistry() = default;
  DISABLE_COPY_AND_ASSIGN(LayerRegistry);

  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (auto iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};

class LayerRegister {
 public:
  LayerRegister(const string& type, LayerRegistry::Creator creator) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry::AddCreator(type, creator);
  }
};

#define REGISTER_LAYER_CREATOR(type, creator)                               \
  static LayerRegister layer_register(#type, creator)

#define REGISTER_LAYER_CLASS(type)                                          \
  static shared_ptr<Layer> CreateLayer(const LayerParameter& param)         \
  {                                                                         \
    return shared_ptr<Layer>(new type##Layer(param));                       \
  }                                                                         \
  REGISTER_LAYER_CREATOR(type, CreateLayer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
