#include "caffe/graph/graph.hpp"

namespace caffe {
namespace graph {

/*! \brief Graph Implement */
class GraphImpl : public Graph {
 public:
  GraphImpl(const caffe::Net& net);
  GraphImpl(const caffe::NetParameter& prototxt,
            const caffe::NetParameter& caffemodel);
};

}  // namespace graph
}  // namespace caffe
