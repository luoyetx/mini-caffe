#ifndef CAFFE_GRAPH_GRAPH_HPP_
#define CAFFE_GRAPH_GRAPH_HPP_

#include "caffe/net.hpp"

namespace caffe {
namespace graph {

/*!
 * \brief Graph represent of Caffe model
 */
class Graph {
 public:
  virtual ~Graph() {}
  /*!
   * \brief create Graph
   * \param net caffe::Net object
   * \param prototxt, caffemodel protobuf data of the model
   */
  static Graph* Create(const caffe::Net& net);
  static Graph* Create(const caffe::NetParameter& prototxt,
                       const caffe::NetParameter& caffemodel);

  virtual void Forward() = 0;
  virtual void MarkInputs() = 0;
  virtual void MarkOutputs() = 0;
};

}  // namespace graph
}  // namespace caffe

#endif  // CAFFE_GRAPH_GRAPH_HPP_
