// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"


namespace ov {
namespace detail {
namespace {
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape, const std::vector<int>& dims);
ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis);
std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                          ov::Output<ov::Node> past_seqlen,
                                          std::shared_ptr<ov::Node> current_seqlen,
                                          std::shared_ptr<ov::Node> cos_cache,
                                          std::shared_ptr<ov::Node> sin_cache,
                                          std::shared_ptr<ov::Node> dim_head_size,
                                          bool interleaved);
}  // namespace
}  // namespace detail
}  // namespace ov

ov::pass::GroupQueryAttentionDecomposition::GroupQueryAttentionDecomposition() {
    MATCHER_SCOPE(GroupQeuryAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v15::GroupQueryAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node =
            ov::as_type_ptr<ov::op::v15::GroupQueryAttention>(pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = decompose(node);
        ov::replace_node(node, new_output_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

ov::OutputVector ov::pass::GroupQueryAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::v15::GroupQueryAttention> node) {
    using namespace ov::op;

    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();
    // TODO: add softcap support

    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto past_key = node->input_value(3);
    auto past_value = node->input_value(4);
    auto seqlens_k = node->input_value(5);
    auto cos_cache = node->input_value(6);
    auto sin_cache = node->input_value(7);

    // The length of all tokens (past + current) is `seqlens_k` + 1, current = Q.shape[2], past = `seqlens_k` + 1 - current

    const auto T = Q.get_element_type();
    const auto q_shape = std::make_shared<v3::ShapeOf>(Q);
    const auto current_sequence_length = detail::get_dimensions(q_shape, {2});
    auto head_size_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {node->get_head_size()});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto one_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto seqlens_elemi64 = std::make_shared<v0::Convert>(seqlens_k, ov::element::i64);
    auto real_seqlens = std::make_shared<v1::Add>(seqlens_elemi64, one);

    // Only consider batch is 1
    auto seqlens_1d = std::make_shared<v1::Reshape>(real_seqlens, one, false); // 12 or 13, ..
    auto past_sequence_length = std::make_shared<v1::Subtract>(seqlens_1d, current_sequence_length);
    auto current_seqlen_scalar = std::make_shared<v1::Reshape>(current_sequence_length, one_without_shape, false); // 12 or 1

    if (do_rotary) {
        Q = detail::rotaryEmbedding(Q,
                            past_sequence_length,
                            current_seqlen_scalar,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
        K = detail::rotaryEmbedding(K,
                            past_sequence_length,
                            current_seqlen_scalar,
                            cos_cache.get_node_shared_ptr(),
                            sin_cache.get_node_shared_ptr(),
                            head_size_node,
                            rotary_interleaved);
    }
    // present = concat(K, V) if 'past' input is unavailable
    // or
    // present = concat(past, K, V)
    auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
        return std::make_shared<v0::Concat>(ov::OutputVector{past, current}, 2);
    };

    // 1x32x2048x96 (last 12 or 13 make sense)
    K = construct_kv_cache(past_key, K);
    V = construct_kv_cache(past_value, V);

    const auto K_node_shape = std::make_shared<v3::ShapeOf>(K);
    const auto concat_kv_sequence_length = detail::get_dimensions(K_node_shape, {2});  // 2036+12 or 2047+1 = 2048
    auto concat_kv_sequence_length_scalar = std::make_shared<v1::Reshape>(concat_kv_sequence_length, one_without_shape, false); // 2048

    // Gather [1:last] as present
    // 1:2048
    // results a 2047 length position, no matter prefill or kvcache
    ov::Output<ov::Node> position = std::make_shared<v4::Range>(one_without_shape, concat_kv_sequence_length_scalar, one_without_shape, ov::element::i32);
    auto present_k = std::make_shared<v8::Gather>(K, position, two); // 1x32x2047x96
    auto present_v = std::make_shared<v8::Gather>(V, position, two); // 1x32x2047x96

    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
        const auto kv_shape = std::make_shared<v3::ShapeOf>(K);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto kv_shape_prev_2 = detail::get_dimensions(kv_shape, {0, 1});
        const auto kv_shape_last_2 = detail::get_dimensions(kv_shape, {2, 3});
        auto new_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, new_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, new_kv_shape, false);
        K = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = std::make_shared<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        auto q_shape = std::make_shared<v3::ShapeOf>(Q);
        // (batch_size, num_heads, sequence_len, head_size)
        const auto q_shape_prev_2 = detail::get_dimensions(q_shape, {0, 1});
        auto extended_kv_shape = std::make_shared<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
        K = std::make_shared<v1::Reshape>(K, extended_kv_shape, false);
        V = std::make_shared<v1::Reshape>(V, extended_kv_shape, false);
    }

    // consider the mask before softmax op
    // need to apply low-triangle mask to attention score.
    // two steps

    // 1st step
    // construct the current_seqlen x max_seqlen mask, and apply triangle compare
    std::shared_ptr<ov::Node> col_range =
        std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                    concat_kv_sequence_length_scalar,
                                    one_without_shape,
                                    ov::element::i64); // [0,1,2,...,2047]
    auto hori_range = std::make_shared<v0::Unsqueeze>(col_range, zero);  // 1x2048

    std::shared_ptr<ov::Node> row_range =
         std::make_shared<v4::Range>(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                    current_seqlen_scalar,
                                    one_without_shape,
                                    ov::element::i64); // [0,1,2,...,] or [0]
    auto vert_range = std::make_shared<v0::Unsqueeze>(row_range, one);   // 12x1 or 1x1
    const auto pastK_node_shape = std::make_shared<v3::ShapeOf>(past_key);
    const auto pastk_sequence_length = detail::get_dimensions(pastK_node_shape, {2});  // 2036 or 2047
    auto vert_range_offset = std::make_shared<v1::Add>(vert_range, pastk_sequence_length);  // [2036,2037,...,2047], or [2047]

    // triangle compare
    // below is 0, upper is -inf.
    auto triu = std::make_shared<v1::Greater>(hori_range, vert_range_offset);            // 12x2048 or 1x2048, ...
    auto typed_zero = v0::Constant::create(T, ov::Shape{}, {0});
    std::shared_ptr<ov::Node> minus_inf = nullptr;
    if (T == ov::element::f32)
        minus_inf = ov::op::v0::Constant::create(T, ov::Shape{}, {-std::numeric_limits<float>::infinity()});
    else if (T == ov::element::f16)
        minus_inf = ov::op::v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()});
    auto atten_mask = std::make_shared<v1::Select>(triu, minus_inf, typed_zero);  // 12x2048 or 1x2048, ...

    // 2nd step
    // col index compare, 0-threshold col index, mask is -inf
    auto row_index_shape = std::make_shared<v0::Concat>(ov::NodeVector{current_sequence_length, one}, 0);
    auto empty_sequence_length = std::make_shared<v1::Subtract>(concat_kv_sequence_length, seqlens_1d); // 2036 or 2035, 2034...
    auto row_index = std::make_shared<v3::Broadcast>(empty_sequence_length, row_index_shape); // shape = 12x1 or 1x1, data = [2036, ...] or [2035, ...] or [2034, ...]
    auto row_compare_range_offset = std::make_shared<v1::Subtract>(row_index, one); // shape = 12x1 or 1x1, data = [2035, ...] or [2034, ...] or [2033, ...]

    auto compare_result = std::make_shared<v1::Greater>(hori_range, row_compare_range_offset); // 12x2048 or 1x2048, ... last 12/13/14/... columns data, will be True
    // the final atten mask is [-inf, -inf, -inf, ..., -inf, triangle(lower=0, upper=-inf)]
    auto final_atten_mask = std::make_shared<v1::Select>(compare_result, atten_mask, minus_inf); // 12x2048 or 1x2048, mask

    std::shared_ptr<ov::Node> softmax_input = std::make_shared<v0::MatMul>(Q, K, false, true);
    std::shared_ptr<ov::Node> alpha;
    if (scale == 0.0f) {
        alpha = std::make_shared<v0::Sqrt>(head_size_node);
        alpha = std::make_shared<v1::Divide>(one, alpha);
    } else {
        alpha = v0::Constant::create(T, ov::Shape{}, {scale});
    }
    softmax_input = std::make_shared<v1::Multiply>(softmax_input, alpha); // [1,32,12,2048] or [1,32,1,2048], ...

    std::shared_ptr<ov::Node> softmax_input_added = std::make_shared<v1::Add>(softmax_input, final_atten_mask);
    const auto softmax = std::make_shared<v8::Softmax>(softmax_input_added, 3);

    // softmax((Q x K' + mask) / sqrt(head_size)) x V
    std::shared_ptr<ov::Node> qga_output = std::make_shared<v0::MatMul>(softmax, V);
    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads * head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto qga_output_transposed = std::make_shared<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    auto output = std::make_shared<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

    return {output, present_k, present_v};
}


namespace ov {
namespace detail {
namespace {
// make split functions is a copy-past from ONNX FE. TODO: move it to one place
ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis) {
    using namespace ov::op;
    const auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    const auto split = std::make_shared<v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims) {
    using namespace ov::op;
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<ov::op::v3::ShapeOf>(node), dims);
}

std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                          ov::Output<ov::Node> past_seqlen,
                                          std::shared_ptr<ov::Node> current_seqlen,
                                          std::shared_ptr<ov::Node> cos_cache,
                                          std::shared_ptr<ov::Node> sin_cache,
                                          std::shared_ptr<ov::Node> dim_head_size,
                                          bool interleaved) {
    using namespace ov::op;
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto zero_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one_without_shape = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    ov::Output<ov::Node> position_ids = std::make_shared<v4::Range>(zero_without_shape, current_seqlen, one_without_shape, ov::element::i64); // 0-12, 0-1
    position_ids = std::make_shared<v1::Add>(past_seqlen, position_ids); // 0-12, 12-13

    auto cos = std::make_shared<v8::Gather>(cos_cache, position_ids, zero); // 12x48, 1x48
    auto sin = std::make_shared<v8::Gather>(sin_cache, position_ids, zero); // 12x48, 1x48

    if (interleaved) {
        auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

        auto cache_shape = std::make_shared<v3::ShapeOf>(cos_cache);
        auto cache_last_dim = get_dimensions(cos_cache, {-1});

        auto input_shape = std::make_shared<v3::ShapeOf>(input);

        auto dim_bns = get_dimensions(input_shape, {0, 1, 2});
        std::shared_ptr<ov::Node> half_last_dim = cache_last_dim;

        auto negtive_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, two}, 0);
        auto reshaped_input = std::make_shared<v1::Reshape>(input, split_input_shape, false);

        auto in_split = make_split(reshaped_input, 2, -1);
        split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim}, 0);
        auto in_split_0 = std::make_shared<v1::Reshape>(in_split[0], split_input_shape, false);
        auto in_split_1 = std::make_shared<v1::Reshape>(in_split[1], split_input_shape, false);

        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split_0, cos),
                                                    std::make_shared<v1::Multiply>(in_split_1, sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split_0, sin),
                                               std::make_shared<v1::Multiply>(in_split_1, cos));

        split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_last_dim, one}, 0);
        auto res_0_5d = std::make_shared<v1::Reshape>(res_0, split_input_shape, false);
        auto res_1_5d = std::make_shared<v1::Reshape>(res_1, split_input_shape, false);

        auto concat_ret = std::make_shared<v0::Concat>(ov::NodeVector{res_0_5d, res_1_5d}, -1);
        return std::make_shared<v1::Reshape>(concat_ret, input_shape, false);
    } else {
        auto in_split = make_split(input, 2, -1);
        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split[0], cos),
                                                    std::make_shared<v1::Multiply>(in_split[1], sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split[0], sin),
                                               std::make_shared<v1::Multiply>(in_split[1], cos));

        return std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);
    }
}
}  // namespace
}  // namespace detail
}  // namespace ov