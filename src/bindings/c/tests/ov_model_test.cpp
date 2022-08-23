// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_model, ov_model_const_outputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_list_t output_ports;
    output_ports.ports = nullptr;
    OV_ASSERT_OK(ov_model_const_outputs(model, &output_ports));
    ASSERT_NE(nullptr, output_ports.ports);

    ov_output_const_node_list_free(&output_ports);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_inputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_list_t input_ports;
    input_ports.ports = nullptr;
    OV_ASSERT_OK(ov_model_const_inputs(model, &input_ports));
    ASSERT_NE(nullptr, input_ports.ports);

    ov_output_const_node_list_free(&input_ports);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_outputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t output_ports;
    output_ports.ports = nullptr;
    OV_ASSERT_OK(ov_model_outputs(model, &output_ports));
    ASSERT_NE(nullptr, output_ports.ports);

    ov_output_node_list_free(&output_ports);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_inputs) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_list_t input_ports;
    input_ports.ports = nullptr;
    OV_ASSERT_OK(ov_model_inputs(model, &input_ports));
    ASSERT_NE(nullptr, input_ports.ports);

    ov_output_node_list_free(&input_ports);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_input) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_const_input(model, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_output_const_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_const_input_by_name(model, "data", &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_node_get_shape(input_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_const_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_input_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_const_input_by_index(model, 0, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_node_get_shape(input_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_const_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_input) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_input(model, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_output_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_input_by_name(model, "data", &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(input_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_input_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* input_port = nullptr;
    OV_ASSERT_OK(ov_model_input_by_index(model, 0, &input_port));
    ASSERT_NE(nullptr, input_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(input_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_node_free(input_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_output) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_const_output(model, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_output_const_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_output_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_const_output_by_index(model, 0, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_node_get_shape(output_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_const_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_const_output_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_const_output_by_name(model, "fc_out", &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_const_node_get_shape(output_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_const_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_output) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_output(model, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_output_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_output_by_index) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_output_by_index(model, 0, &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(output_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_output_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_node_t* output_port = nullptr;
    OV_ASSERT_OK(ov_model_output_by_name(model, "fc_out", &output_port));
    ASSERT_NE(nullptr, output_port);

    ov_shape_t shape;
    OV_ASSERT_OK(ov_node_get_shape(output_port, &shape));
    ov_shape_deinit(&shape);

    ov_output_node_free(output_port);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_is_dynamic) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ASSERT_NO_THROW(ov_model_is_dynamic(model));

    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_reshape_input_by_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    ov_output_const_node_list_t input_ports_1;
    input_ports_1.ports = nullptr;
    OV_ASSERT_OK(ov_model_const_inputs(model, &input_ports_1));
    ASSERT_NE(nullptr, input_ports_1.ports);

    char* tensor_name = nullptr;
    OV_ASSERT_OK(ov_node_list_get_any_name_by_index(&input_ports_1, 0, &tensor_name));

    ov_shape_t shape = {0, nullptr};
    OV_ASSERT_OK(ov_shape_init(&shape, 4));
    shape.dims[0] = 1;
    shape.dims[1] = 3;
    shape.dims[2] = 896;
    shape.dims[3] = 896;

    ov_partial_shape_t* partial_shape = nullptr;
    OV_ASSERT_OK(ov_shape_to_partial_shape(&shape, &partial_shape));
    OV_ASSERT_OK(ov_model_reshape_input_by_name(model, tensor_name, partial_shape));

    ov_output_const_node_list_t input_ports_2;
    input_ports_2.ports = nullptr;
    OV_ASSERT_OK(ov_model_const_inputs(model, &input_ports_2));
    ASSERT_NE(nullptr, input_ports_2.ports);

    EXPECT_NE(input_ports_1.ports, input_ports_2.ports);

    ov_shape_deinit(&shape);
    ov_partial_shape_free(partial_shape);
    ov_free(tensor_name);
    ov_output_const_node_list_free(&input_ports_1);
    ov_output_const_node_list_free(&input_ports_2);
    ov_model_free(model);
    ov_core_free(core);
}

TEST(ov_model, ov_model_get_friendly_name) {
    ov_core_t* core = nullptr;
    OV_ASSERT_OK(ov_core_create(&core));
    ASSERT_NE(nullptr, core);

    ov_model_t* model = nullptr;
    OV_ASSERT_OK(ov_core_read_model(core, xml, bin, &model));
    ASSERT_NE(nullptr, model);

    char* friendly_name = nullptr;
    OV_ASSERT_OK(ov_model_get_friendly_name(model, &friendly_name));
    ASSERT_NE(nullptr, friendly_name);

    ov_free(friendly_name);
    ov_model_free(model);
    ov_core_free(core);
}