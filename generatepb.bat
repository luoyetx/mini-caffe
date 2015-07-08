"./3rdparty/bin/protoc" -I="./src/caffe/proto" --cpp_out="./src/caffe/proto" "./src/caffe/proto/caffe.proto"
copy .\src\caffe\proto\caffe.pb.h .\include\caffe\proto\caffe.pb.h