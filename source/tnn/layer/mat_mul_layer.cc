// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <cmath>

#include "tnn/layer/base_layer.h"
#include "tnn/utils/dims_vector_utils.h"

namespace TNN_NS {
DECLARE_LAYER(MatMul, LAYER_MATMUL);

Status MatMulLayer::InferOutputDataType() {
    return BaseLayer::InferOutputDataType();
}

Status MatMulLayer::InferOutputShape() {
    ASSERT(input_blobs_.size() == 2);
    auto& output_dim = output_blobs_[0]->GetBlobDesc().dims;
    auto dim_a   = input_blobs_[0]->GetBlobDesc().dims;
    auto dim_b   = input_blobs_[1]->GetBlobDesc().dims;
    int pad_axis = dim_a.size();
    for (int i = dim_a.size() - 1; i >= 0; ++i) {
        if (dim_a[i] == dim_b[i] && dim_a[i] == 1) {
            pad_axis -= 1;
        }
    }
	// input0:[1, 32, 64, 1]
	// input1:[1, 64, 96, 1]
	//                    |
	//                  pad _axis
    switch (pad_axis) {
        case 4:{
            ASSERT(dim_a[0] == dim_b[0]);
            ASSERT(dim_a[1] == dim_b[1]);
            ASSERT(dim_a[3] == dim_b[2]);
            output_dim[0] = dim_a[0];
            output_dim[1] = dim_a[1];
            output_dim[2] = dim_a[2];
            output_dim[3] = dim_b[3];
            break;
        }
        case 3: {
            ASSERT(dim_a[0] == dim_b[0]);
            ASSERT(dim_a[2] == dim_b[1]);
            output_dim[0] = dim_a[0];
            output_dim[1] = dim_a[1];
            output_dim[2] = dim_b[2];
            output_dim[3] = 1;
            break;
        }
        case 2: {
            ASSERT(dim_a[1] == dim_b[0]);
            output_dim[0] = dim_a[0];
            output_dim[1] = dim_b[1];
            output_dim[2] = 1;
            output_dim[3] = 1;
            break;
        }
        default:{
            LOGE("MatMul get wrong input dims, please check input's dims\n");
            return TNNERR_PARAM_ERR;
        }
    }
    return TNN_OK;
}

REGISTER_LAYER(MatMul, LAYER_MATMUL);

}  // namespace TNN_NS
