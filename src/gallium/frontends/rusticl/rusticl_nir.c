/*
 * Copyright 2022 Red Hat.
 *
 * SPDX-License-Identifier: MIT
 */

#include "CL/cl.h"

#include "nir.h"
#include "nir_builder.h"

#include "rusticl_nir.h"

static bool
rusticl_lower_intrinsics_filter(const nir_instr* instr, const void* state)
{
    return instr->type == nir_instr_type_intrinsic;
}

static nir_def*
rusticl_lower_intrinsics_instr(
    nir_builder *b,
    nir_instr *instr,
    void* _state
) {
    nir_intrinsic_instr *intrins = nir_instr_as_intrinsic(instr);
    struct rusticl_lower_state *state = _state;

    switch (intrins->intrinsic) {
    case nir_intrinsic_image_deref_format:
    case nir_intrinsic_image_deref_order: {
        int32_t offset;
        nir_deref_instr *deref;
        nir_def *val;
        nir_variable *var;

        if (intrins->intrinsic == nir_intrinsic_image_deref_format) {
            offset = CL_SNORM_INT8;
            var = nir_find_variable_with_location(b->shader, nir_var_uniform, state->format_arr_loc);
        } else {
            offset = CL_R;
            var = nir_find_variable_with_location(b->shader, nir_var_uniform, state->order_arr_loc);
        }

        val = intrins->src[0].ssa;

        if (nir_def_is_deref(val)) {
            nir_deref_instr *deref = nir_def_as_deref(val);
            nir_variable *var = nir_deref_instr_get_variable(deref);
            assert(var);
            val = nir_imm_intN_t(b, var->data.binding, val->bit_size);
        }

        // we put write images after read images
        if (glsl_type_is_image(var->type)) {
            val = nir_iadd_imm(b, val, b->shader->info.num_textures);
        }

        deref = nir_build_deref_var(b, var);
        deref = nir_build_deref_array(b, deref, val);
        val = nir_u2uN(b, nir_load_deref(b, deref), 32);

        // we have to fix up the value base
        val = nir_iadd_imm(b, val, -offset);

        return val;
    }
    case nir_intrinsic_load_global_invocation_id:
        if (intrins->def.bit_size == 64)
            return nir_u2u64(b, nir_load_global_invocation_id(b, 32));
        return NULL;
    case nir_intrinsic_load_base_global_invocation_id:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->base_global_invoc_id_loc));
    case nir_intrinsic_load_base_workgroup_id:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->base_workgroup_id_loc));
    case nir_intrinsic_load_global_size:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->global_size_loc));
    case nir_intrinsic_load_num_workgroups:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->num_workgroups_loc));
    case nir_intrinsic_load_constant_base_ptr:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->const_buf_loc));
    case nir_intrinsic_load_printf_buffer_address:
        return nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->printf_buf_loc));
    case nir_intrinsic_addr_mode_is: {
        /* Flat generic pointers: shared and scratch memory are windows of the 64-bit address
         * space starting at load_shared_base_ptr / load_scratch_base_ptr. */
        nir_variable_mode mode = nir_intrinsic_memory_modes(intrins);
        nir_def *addr = intrins->src[0].ssa;
        nir_variable *size_var =
            nir_find_variable_with_location(b->shader, nir_var_uniform, state->shared_size_loc);
        assert(size_var);

        nir_def *shared_size = nir_u2u64(b, nir_load_var(b, size_var));
        nir_def *in_shared =
            nir_ult(b, nir_isub(b, addr, nir_load_shared_base_ptr(b, 1, 64)), shared_size);
        nir_def *in_scratch =
            nir_ult(b, nir_isub(b, addr, nir_load_scratch_base_ptr(b, 1, 64, 1)),
                    nir_imm_int64(b, b->shader->scratch_size));

        if (mode & nir_var_mem_shared)
            return in_shared;
        if (mode & (nir_var_function_temp | nir_var_shader_temp))
            return in_scratch;
        assert(mode & nir_var_mem_global);
        return nir_inot(b, nir_ior(b, in_shared, in_scratch));
    }
    case nir_intrinsic_load_work_dim:
        assert(nir_find_variable_with_location(b->shader, nir_var_uniform, state->work_dim_loc));
        return nir_u2uN(b, nir_load_var(b, nir_find_variable_with_location(b->shader, nir_var_uniform, state->work_dim_loc)),
                        intrins->def.bit_size);
    default:
        return NULL;
    }
}

bool
rusticl_lower_intrinsics(nir_shader *nir, struct rusticl_lower_state* state)
{
    return nir_shader_lower_instructions(
        nir,
        rusticl_lower_intrinsics_filter,
        rusticl_lower_intrinsics_instr,
        state
    );
}

static nir_def*
rusticl_lower_input_instr(struct nir_builder *b, nir_instr *instr, void *_)
{
   nir_intrinsic_instr *intrins = nir_instr_as_intrinsic(instr);
   if (intrins->intrinsic != nir_intrinsic_load_kernel_input)
      return NULL;

   nir_def *ubo_idx = nir_imm_int(b, 0);
   nir_def *uniform_offset = intrins->src[0].ssa;

   assert(intrins->def.bit_size >= 8);
   nir_def *load_result =
      nir_load_ubo(b, intrins->num_components, intrins->def.bit_size,
                   ubo_idx, nir_iadd_imm(b, uniform_offset, nir_intrinsic_base(intrins)));

   nir_intrinsic_instr *load = nir_def_as_intrinsic(load_result);

   nir_intrinsic_set_align_mul(load, nir_intrinsic_align_mul(intrins));
   nir_intrinsic_set_align_offset(load, nir_intrinsic_align_offset(intrins));
   nir_intrinsic_set_range_base(load, nir_intrinsic_base(intrins));
   nir_intrinsic_set_range(load, nir_intrinsic_range(intrins));

   return load_result;
}

bool
rusticl_lower_inputs(nir_shader *shader)
{
   bool progress = false;

   assert(!shader->info.first_ubo_is_default_ubo);

   progress = nir_shader_lower_instructions(
      shader,
      rusticl_lower_intrinsics_filter,
      rusticl_lower_input_instr,
      NULL
   );

   nir_foreach_variable_with_modes(var, shader, nir_var_mem_ubo) {
      var->data.binding++;
      var->data.driver_location++;
   }
   shader->info.num_ubos++;

   if (shader->num_uniforms > 0) {
      const struct glsl_type *type = glsl_array_type(glsl_uint8_t_type(), shader->num_uniforms, 1);
      nir_variable *ubo = nir_variable_create(shader, nir_var_mem_ubo, type, "kernel_input");
      ubo->data.binding = 0;
      ubo->data.explicit_binding = 1;
   }

   shader->info.first_ubo_is_default_ubo = true;
   return progress;
}

static void
create_libclc_config_func_impl(nir_shader *shader, const char *name, bool value)
{
    nir_function *fma = nir_function_create(shader, name);

    fma->num_params = 1;
    fma->params = ralloc_array(fma, nir_parameter, 1);
    fma->params[0] = (struct nir_parameter) {
        .num_components = 1,
        .bit_size = shader->info.cs.ptr_size,
        .is_return = true,
        .type = glsl_uintN_t_type(shader->info.cs.ptr_size),
    };

    nir_function_impl_create(fma);
    nir_builder b = nir_builder_at(nir_before_impl(fma->impl));
    nir_def *param = nir_load_param(&b, 0);
    nir_deref_instr *deref = nir_build_deref_cast(&b, param, nir_var_function_temp, glsl_bool_type(), 0);
    nir_store_deref(&b, deref, nir_imm_bool(&b, value), 0x1);
}

bool
rusticl_insert_libclc_config(nir_shader *shader)
{
    nir_foreach_function_impl(impl, shader)
        nir_no_progress(impl);

    bool has_fma = nir_has_ffma(shader, 32);
    bool daz16 = nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode, 16);
    bool daz32 = nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode, 32);
    bool daz64 = nir_is_denorm_flush_to_zero(shader->info.float_controls_execution_mode, 64);
    bool dp16 = nir_is_denorm_preserve(shader->info.float_controls_execution_mode, 16);
    bool dp32 = nir_is_denorm_preserve(shader->info.float_controls_execution_mode, 32);
    bool dp64 = nir_is_denorm_preserve(shader->info.float_controls_execution_mode, 64);

    create_libclc_config_func_impl(shader, "__clc_runtime_has_hw_fma32", has_fma);

    create_libclc_config_func_impl(shader, "__clc_fp16_subnormals_supported", dp16);
    create_libclc_config_func_impl(shader, "__clc_fp32_subnormals_supported", dp32);
    create_libclc_config_func_impl(shader, "__clc_fp64_subnormals_supported", dp64);

    /* LLVM-23 renamed the denorm ones and reversed their meaning */
    create_libclc_config_func_impl(shader, "__clc_denormals_are_zero_fp16", daz16);
    create_libclc_config_func_impl(shader, "__clc_denormals_are_zero_fp32", daz32);
    create_libclc_config_func_impl(shader, "__clc_denormals_are_zero_fp64", daz64);

    return true;
}

static bool
has_mode_check(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *found = data;
   if (intr->intrinsic == nir_intrinsic_deref_mode_is ||
       intr->intrinsic == nir_intrinsic_addr_mode_is)
      *found = true;
   return false;
}

/* Whether the shader checks the address space of a generic pointer at run time
 * (to_global/to_local/to_private, OpGenericCastToPtrExplicit).
 */
bool
rusticl_has_mode_checks(nir_shader *nir)
{
   bool found = false;
   nir_shader_intrinsics_pass(nir, has_mode_check, nir_metadata_all, &found);
   return found;
}

struct local_args {
   const bool *is_local;
   unsigned count;
};

static bool
lower_local_arg(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct local_args *la = data;
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (deref->deref_type != nir_deref_type_var ||
       deref->var->data.mode != nir_var_uniform)
      return false;

   int loc = deref->var->data.location;
   if (loc < 0 || loc >= (int)la->count || !la->is_local[loc])
      return false;

   assert(intr->def.num_components == 1 && intr->def.bit_size == 64);
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *addr = nir_iadd(b, nir_load_shared_base_ptr(b, 1, 64), &intr->def);
   nir_def_rewrite_uses_after(&intr->def, addr);
   return true;
}

/* With flat generic pointers a __local pointer is a 64-bit address, but the
 * value of a __local kernel argument is an offset into the work-group's local
 * memory: add the base address of that memory to it.
 */
bool
rusticl_lower_local_args_to_global(nir_shader *nir, const bool *is_local, unsigned count)
{
   struct local_args la = { is_local, count };
   return nir_shader_intrinsics_pass(nir, lower_local_arg, nir_metadata_control_flow, &la);
}

/* If addr is base_op + offset (through a chain of scalar iadds), build and
 * return the offset, otherwise return NULL without building anything.
 */
static nir_def *
strip_base(nir_builder *b, nir_def *addr, nir_intrinsic_op base_op, unsigned depth)
{
   if (depth > 16 || addr->num_components != 1)
      return NULL;

   nir_instr *instr = nir_def_instr(addr);
   if (instr->type == nir_instr_type_intrinsic &&
       nir_instr_as_intrinsic(instr)->intrinsic == base_op)
      return nir_imm_intN_t(b, 0, addr->bit_size);

   if (instr->type != nir_instr_type_alu)
      return NULL;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_iadd)
      return NULL;

   for (unsigned i = 0; i < 2; i++) {
      if (alu->src[i].swizzle[0] != 0)
         continue;
      nir_def *off = strip_base(b, alu->src[i].src.ssa, base_op, depth + 1);
      if (off)
         return nir_iadd(b, off, nir_ssa_for_alu_src(b, alu, 1 - i));
   }

   return NULL;
}

static bool
lower_global_to_shared_scratch(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned addr_src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      addr_src = 0;
      break;
   case nir_intrinsic_store_global:
      addr_src = 1;
      break;
   default:
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *addr = intr->src[addr_src].ssa;
   bool shared = true;
   nir_def *off = strip_base(b, addr, nir_intrinsic_load_shared_base_ptr, 0);
   if (!off && intr->intrinsic != nir_intrinsic_global_atomic &&
       intr->intrinsic != nir_intrinsic_global_atomic_swap) {
      off = strip_base(b, addr, nir_intrinsic_load_scratch_base_ptr, 0);
      shared = false;
   }
   if (!off)
      return false;

   off = nir_u2u32(b, off);
   nir_def *res = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_global:
      if (shared)
         res = nir_load_shared(b, intr->def.num_components, intr->def.bit_size, off,
                               .access = nir_intrinsic_access(intr),
                               .align_mul = nir_intrinsic_align_mul(intr),
                               .align_offset = nir_intrinsic_align_offset(intr));
      else
         res = nir_load_scratch(b, intr->def.num_components, intr->def.bit_size, off,
                                .access = nir_intrinsic_access(intr),
                                .align_mul = nir_intrinsic_align_mul(intr),
                                .align_offset = nir_intrinsic_align_offset(intr));
      break;
   case nir_intrinsic_store_global:
      if (shared)
         nir_store_shared(b, intr->src[0].ssa, off,
                          .write_mask = nir_intrinsic_write_mask(intr),
                          .access = nir_intrinsic_access(intr),
                          .align_mul = nir_intrinsic_align_mul(intr),
                          .align_offset = nir_intrinsic_align_offset(intr));
      else
         nir_store_scratch(b, intr->src[0].ssa, off,
                           .write_mask = nir_intrinsic_write_mask(intr),
                           .align_mul = nir_intrinsic_align_mul(intr),
                           .align_offset = nir_intrinsic_align_offset(intr));
      break;
   case nir_intrinsic_global_atomic:
      res = nir_shared_atomic(b, intr->def.bit_size, off, intr->src[1].ssa,
                              .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   case nir_intrinsic_global_atomic_swap:
      res = nir_shared_atomic_swap(b, intr->def.bit_size, off, intr->src[1].ssa,
                                   intr->src[2].ssa,
                                   .atomic_op = nir_intrinsic_atomic_op(intr));
      break;
   default:
      UNREACHABLE("filtered above");
   }

   if (res)
      nir_def_rewrite_uses(&intr->def, res);
   nir_instr_remove(&intr->instr);
   return true;
}

/* With flat generic pointers every shared and function memory access is first
 * lowered to a global access at load_shared_base_ptr/load_scratch_base_ptr +
 * offset. Turn those whose address space is known back into shared and scratch
 * accesses; only real generic accesses stay global.
 */
bool
rusticl_lower_global_to_shared_scratch(nir_shader *nir)
{
   return nir_shader_intrinsics_pass(nir, lower_global_to_shared_scratch,
                                     nir_metadata_control_flow, NULL);
}

static bool
lower_var_to_global(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);

   /* pointers to shared and function memory already hold flat addresses */
   if (deref->deref_type == nir_deref_type_cast) {
      if (!(deref->modes & ~(nir_var_mem_shared | nir_var_function_temp))) {
         deref->modes = nir_var_mem_global;
         return true;
      }
      return false;
   }

   if (deref->deref_type != nir_deref_type_var)
      return false;

   nir_variable *var = deref->var;
   nir_intrinsic_op base_op;
   if (var->data.mode == nir_var_mem_shared)
      base_op = nir_intrinsic_load_shared_base_ptr;
   else if (var->data.mode == nir_var_function_temp)
      base_op = nir_intrinsic_load_scratch_base_ptr;
   else
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *base = base_op == nir_intrinsic_load_shared_base_ptr
                      ? nir_load_shared_base_ptr(b, 1, 64)
                      : nir_load_scratch_base_ptr(b, 1, 64, 1);
   nir_def *addr = nir_iadd_imm(b, base, var->data.driver_location);

   /* the local and thread-local storage bases are at least 16-byte aligned */
   nir_deref_instr *cast =
      nir_build_deref_cast_with_alignment(b, addr, nir_var_mem_global, deref->type, 0, 16,
                                          var->data.driver_location % 16);
   nir_def_replace(&deref->def, &cast->def);
   return true;
}

/* With flat generic pointers, shared and function variables are addressed
 * like global memory at load_shared_base_ptr/load_scratch_base_ptr + their
 * offset, so that their pointers can be cast to and from generic ones. Run
 * after nir_lower_vars_to_explicit_types assigned the offsets.
 */
bool
rusticl_lower_shared_scratch_vars_to_global(nir_shader *nir)
{
   bool progress = nir_shader_instructions_pass(nir, lower_var_to_global,
                                                nir_metadata_control_flow, NULL);
   if (progress)
      nir_fixup_deref_modes(nir);
   return progress;
}
