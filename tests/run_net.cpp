#include <random>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <caffe/net.hpp>

using namespace std;
using namespace caffe;

struct Mat {
  int rows_, cols_;
  float *data_;

  Mat()
    : rows_(0), cols_(0), data_(nullptr) {
  }
  Mat(int rows, int cols)
    : rows_(rows), cols_(cols), data_(new float[rows*cols]) {
  }
  Mat(const Mat &m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    int size = rows_ * cols_;
    data_ = new float[size];
    memcpy(data_, m.data_, size * sizeof(float));
  }
  Mat(Mat &&m) 
    : data_(m.data_) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    m.data_ = nullptr;
  }
  Mat &operator=(const Mat &m) {
    if (this == &m) return *this;
    rows_ = m.rows_;
    cols_ = m.cols_;
    int size = rows_ * cols_;
    if (data_) delete[] data_;
    data_ = new float[size];
    memcpy(data_, m.data_, size * sizeof(float));
    return *this;
  }
  Mat &operator=(Mat &&m) {
    rows_ = m.rows_;
    cols_ = m.cols_;
    std::swap(data_, m.data_);
    return *this;
  }
  ~Mat() {
    if (data_) delete[] data_;
  }

  int rows() { return rows_; }
  int cols() { return cols_; }
  float *data() { return data_; }

  static Mat Random(int rows, int cols, float scale=1) {
    Mat m(rows, cols);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<float> urd(-scale, scale);
    std::generate(m.data_, m.data_ + m.rows_*m.cols_, [&]() { return urd(dre); });
    return m;
  }
};

/*! \brief Timer */
class Timer {
  using Clock = std::chrono::high_resolution_clock;
public:
  /*! \brief start or restart timer */
  inline void Tic() {
    start_ = Clock::now();
  }
  /*! \brief stop timer */
  inline void Toc() {
    end_ = Clock::now();
  }
  /*! \brief return time in ms */
  inline double Elasped() {
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
    return duration.count();
  }
private:
  Clock::time_point start_, end_;
};

struct TestFunctor {
  std::shared_ptr<caffe::Net<float> > net;

  TestFunctor(const std::string &prototxt, const std::string &caffemodel)
    : net(new caffe::Net<float>(prototxt)) {
    net->CopyTrainedLayersFrom(caffemodel);
  }
  /*!
   * \brief generate input data
   * \return feature maps
   */
  virtual std::vector<Mat> Generate() = 0;
  /*!
   * \brief forward network
   * \param img input image
   */
  void Forward() {
    // transform input image to feature map
    auto data = this->Generate();
    std::shared_ptr<Blob<float> > input = this->net->blob_by_name("data");
    CHECK_GT(data.size(), 1);
    CHECK_EQ(1, input->num());
    CHECK_EQ(data.size(), input->channels());
    CHECK_EQ(data[0].rows(), input->height());
    CHECK_EQ(data[0].cols(), input->width());
    // copy to network input buffer
    const int bias = input->offset(0, 1);
    const int bytes = bias * sizeof(float);
    for (int k = 0; k < data.size(); k++) {
      memcpy(input->mutable_cpu_data() + 0 * bias, data[k].data(), bytes);
    }
    // forward network
    this->net->Forward();
  }
};

struct NIN : public TestFunctor {
  NIN(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

struct GoogLeNet : public TestFunctor {
  GoogLeNet(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

struct ResNet : public TestFunctor {
  ResNet(const std::string &prototxt, const std::string &caffemodel)
    : TestFunctor(prototxt, caffemodel) {}

  virtual std::vector<Mat> Generate() override {
    std::vector<Mat> fm;
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    fm.push_back(Mat::Random(224, 224, 128));
    return fm;
  }
};

int main(int argc, char *argv[]) {
#ifdef USE_CUDA
  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(0);
#else
  Caffe::set_mode(Caffe::CPU);
#endif  // USE_CUDA
  Timer timer;
  // test nin, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#network-in-network-model
  {
    LOG(INFO) << "Test NIN";
    auto test = NIN("model/nin.prototxt", "model/nin.caffemodel");
    timer.Tic();
    test.Forward();
    timer.Toc();
    LOG(INFO) << "Forward NIN costs " << timer.Elasped() << " ms";
  }
  // test googlenet, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing
  {
    LOG(INFO) << "Test GoogLeNet";
    auto test = GoogLeNet("model/googlenet.prototxt", "model/googlenet.caffemodel");
    timer.Tic();
    test.Forward();
    timer.Toc();
    LOG(INFO) << "Forward GoogLeNet costs " << timer.Elasped() << " ms";
  }
  // test resnet, model from https://github.com/BVLC/caffe/wiki/Model-Zoo#imagenet-pre-trained-models-with-batch-normalization
  {
    LOG(INFO) << "Test ResNet";
    auto test = ResNet("model/resnet.prototxt", "model/resnet.caffemodel");
    timer.Tic();
    test.Forward();
    timer.Toc();
    LOG(INFO) << "Forward ResNet costs " << timer.Elasped() << " ms";
  }
  return 0;
}
