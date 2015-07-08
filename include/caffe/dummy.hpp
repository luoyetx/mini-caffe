#ifndef CAFFE_DUMMY_EXTERN_HPP_
#define CAFFE_DUMMY_EXTERN_HPP_

#include <caffe/layer_factory.hpp>
#include <caffe/neuron_layers.hpp>
#include <caffe/loss_layers.hpp>
#include <caffe/common.hpp>
#include <caffe/vision_layers.hpp>
#include <caffe/data_layers.hpp>

namespace caffe {

/**
 * This header file tries to solve static vars initialization in static libraries issue with msvc.
 * All vars below will never should never ever be used.
 */

extern AbsValLayer<float> *DUMMY_f_AbsValLayer;
extern AccuracyLayer<float> *DUMMY_f_AccuracyLayer;
extern ArgMaxLayer<float> *DUMMY_f_ArgMaxLayer;
extern BNLLLayer<float> *DUMMY_f_BNLLLayer;
extern ConcatLayer<float> *DUMMY_f_ConcatLayer;
extern ContrastiveLossLayer<float> *DUMMY_f_ContrastiveLossLayer;
extern DataLayer<float> *DUMMY_f_DataLayer;
extern DeconvolutionLayer<float> *DUMMY_f_DeconvolutionLayer;
extern DropoutLayer<float> *DUMMY_f_DropoutLayer;
extern DummyDataLayer<float> *DUMMY_f_DummyDataLayer;
extern EltwiseLayer<float> *DUMMY_f_EltwiseLayer;
extern EuclideanLossLayer<float> *DUMMY_f_EuclideanLossLayer;
extern ExpLayer<float> *DUMMY_f_ExpLayer;
extern FilterLayer<float> *DUMMY_f_FilterLayer;
extern FlattenLayer<float> *DUMMY_f_FlattenLayer;
extern HingeLossLayer<float> *DUMMY_f_HingeLossLayer;
extern Im2colLayer<float> *DUMMY_f_Im2colLayer;
extern ImageDataLayer<float> *DUMMY_f_ImageDataLayer;
extern InfogainLossLayer<float> *DUMMY_f_InfogainLossLayer;
extern InnerProductLayer<float> *DUMMY_f_InnerProductLayer;
extern LogLayer<float> *DUMMY_f_LogLayer;
extern LRNLayer<float> *DUMMY_f_LRNLayer;
extern MemoryDataLayer<float> *DUMMY_f_MemoryDataLayer;
extern MultinomialLogisticLossLayer<float> *DUMMY_f_MultinomialLogisticLossLayer;
extern MVNLayer<float> *DUMMY_f_MVNLayer;
extern PowerLayer<float> *DUMMY_f_PowerLayer;
extern PReLULayer<float> *DUMMY_f_PReLULayer;
extern ReductionLayer<float> *DUMMY_f_ReductionLayer;
extern ReshapeLayer<float> *DUMMY_f_ReshapeLayer;
extern SigmoidCrossEntropyLossLayer<float> *DUMMY_f_SigmoidCrossEntropyLossLayer;
extern SilenceLayer<float> *DUMMY_f_SilenceLayer;
extern SliceLayer<float> *DUMMY_f_SliceLayer;
extern SoftmaxWithLossLayer<float> *DUMMY_f_SoftmaxWithLossLayer;
extern SplitLayer<float> *DUMMY_f_SplitLayer;
extern SPPLayer<float> *DUMMY_f_SPPLayer;
extern ThresholdLayer<float> *DUMMY_f_ThresholdLayer;
extern WindowDataLayer<float> *DUMMY_f_WindowDataLayer;

extern AbsValLayer<double> *DUMMY_d_AbsValLayer;
extern AccuracyLayer<double> *DUMMY_d_AccuracyLayer;
extern ArgMaxLayer<double> *DUMMY_d_ArgMaxLayer;
extern BNLLLayer<double> *DUMMY_d_BNLLLayer;
extern ConcatLayer<double> *DUMMY_d_ConcatLayer;
extern ContrastiveLossLayer<double> *DUMMY_d_ContrastiveLossLayer;
extern DataLayer<double> *DUMMY_d_DataLayer;
extern DeconvolutionLayer<double> *DUMMY_d_DeconvolutionLayer;
extern DropoutLayer<double> *DUMMY_d_DropoutLayer;
extern DummyDataLayer<double> *DUMMY_d_DummyDataLayer;
extern EltwiseLayer<double> *DUMMY_d_EltwiseLayer;
extern EuclideanLossLayer<double> *DUMMY_d_EuclideanLossLayer;
extern ExpLayer<double> *DUMMY_d_ExpLayer;
extern FilterLayer<double> *DUMMY_d_FilterLayer;
extern FlattenLayer<double> *DUMMY_d_FlattenLayer;
extern HingeLossLayer<double> *DUMMY_d_HingeLossLayer;
extern Im2colLayer<double> *DUMMY_d_Im2colLayer;
extern ImageDataLayer<double> *DUMMY_d_ImageDataLayer;
extern InfogainLossLayer<double> *DUMMY_d_InfogainLossLayer;
extern InnerProductLayer<double> *DUMMY_d_InnerProductLayer;
extern LogLayer<double> *DUMMY_d_LogLayer;
extern LRNLayer<double> *DUMMY_d_LRNLayer;
extern MemoryDataLayer<double> *DUMMY_d_MemoryDataLayer;
extern MultinomialLogisticLossLayer<double> *DUMMY_d_MultinomialLogisticLossLayer;
extern MVNLayer<double> *DUMMY_d_MVNLayer;
extern PowerLayer<double> *DUMMY_d_PowerLayer;
extern PReLULayer<double> *DUMMY_d_PReLULayer;
extern ReductionLayer<double> *DUMMY_d_ReductionLayer;
extern ReshapeLayer<double> *DUMMY_d_ReshapeLayer;
extern SigmoidCrossEntropyLossLayer<double> *DUMMY_d_SigmoidCrossEntropyLossLayer;
extern SilenceLayer<double> *DUMMY_d_SilenceLayer;
extern SliceLayer<double> *DUMMY_d_SliceLayer;
extern SoftmaxWithLossLayer<double> *DUMMY_d_SoftmaxWithLossLayer;
extern SplitLayer<double> *DUMMY_d_SplitLayer;
extern SPPLayer<double> *DUMMY_d_SPPLayer;
extern ThresholdLayer<double> *DUMMY_d_ThresholdLayer;
extern WindowDataLayer<double> *DUMMY_d_WindowDataLayer;

} // namespace

#endif // CAFFE_DUMMY_EXTERN_HPP_
