#include <vector>

#include "./silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
