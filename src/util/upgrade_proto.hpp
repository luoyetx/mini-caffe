#ifndef CAFFE_UTIL_UPGRADE_PROTO_H_
#define CAFFE_UTIL_UPGRADE_PROTO_H_

#include <string>

#include "../proto/caffe.pb.h"

namespace caffe {

// Return true iff the net is not the current version.
bool NetNeedsUpgrade(const NetParameter& net_param);

// Check for deprecations and upgrade the NetParameter as needed.
bool UpgradeNetAsNeeded(const string& param_file, NetParameter* param);

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const string& param_file,
                                    NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const string& param_file,
                                      NetParameter* param);

// Return true iff the Net contains any layers specified as V1LayerParameters.
bool NetNeedsV1ToV2Upgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade a NetParameter with
// deprecated V1LayerParameters.
bool UpgradeV1Net(const NetParameter& v1_net_param, NetParameter* net_param);

bool UpgradeV1LayerParameter(const V1LayerParameter& v1_layer_param,
                             LayerParameter* layer_param);

const char* UpgradeV1LayerType(const V1LayerParameter_LayerType type);

// Return true iff the Net contains input fields.
bool NetNeedsInputUpgrade(const NetParameter& net_param);

// Perform all necessary transformations to upgrade input fields into layers.
void UpgradeNetInput(NetParameter* net_param);

}  // namespace caffe

#endif   // CAFFE_UTIL_UPGRADE_PROTO_H_
