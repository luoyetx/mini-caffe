#include <vector>

#include "../util/math_functions.hpp"
#include "./silence_layer.hpp"

namespace caffe {

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);

}  // namespace caffe
