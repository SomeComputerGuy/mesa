/*
 * Copyright 2022 Red Hat.
 *
 * SPDX-License-Identifier: MIT
 */

#include "nir.h"

struct rusticl_lower_state {
    size_t base_global_invoc_id_loc;
    size_t base_workgroup_id_loc;
    size_t const_buf_loc;
    size_t global_size_loc;
    size_t printf_buf_loc;
    size_t format_arr_loc;
    size_t order_arr_loc;
    size_t work_dim_loc;
    size_t num_workgroups_loc;
    size_t shared_size_loc;
};

bool rusticl_lower_intrinsics(nir_shader *nir, struct rusticl_lower_state *state);
bool rusticl_lower_inputs(nir_shader *nir);
bool rusticl_insert_libclc_config(nir_shader *nir);
bool rusticl_has_mode_checks(nir_shader *nir);
bool rusticl_lower_shared_scratch_vars_to_global(nir_shader *nir);
bool rusticl_lower_local_args_to_global(nir_shader *nir, const bool *is_local, unsigned count);
bool rusticl_lower_global_to_shared_scratch(nir_shader *nir);
