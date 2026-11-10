/*
 * Copyright 2026 Arun Iyengar
 * SPDX-License-Identifier: MIT
 *
 * Geometry and tessellation shaders on Bifrost (job manager): lower the
 * programs with libpoly to compute jobs plus a hardware vertex shader for the
 * rasterization draw, as the asahi gallium driver does.
 */

#include "compiler/nir/nir_builder.h"
#include "poly/cl/libpoly.h"
#include "poly/geometry.h"
#include "poly/nir/poly_nir.h"
#include "pan_context.h"
#include "pan_nir.h"

/* libpoly helpers use these intrinsics; give them their panfrost meaning. */
static bool
lower_poly_misc(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   b->cursor = nir_before_instr(&intr->instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_ro_to_rw_poly:
      /* Read-only and read-write mappings are the same on Mali */
      nir_def_replace(&intr->def, intr->src[0].ssa);
      return true;

   case nir_intrinsic_load_rasterization_stream:
      /* GLES has a single vertex stream */
      nir_def_replace(&intr->def, nir_imm_int(b, 0));
      return true;

   default:
      return false;
   }
}

bool
panfrost_nir_lower_poly_misc(nir_shader *nir)
{
   bool progress = false;

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, lower_poly_misc,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, nir, poly_nir_lower_sysvals);
   return progress;
}

/*
 * The memory layout between stages has one 32-bit word per component. Widen
 * 16-bit outputs (mediump) before they are stored to memory.
 */
static bool
widen_store_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output ||
       nir_src_bit_size(intr->src[0]) != 16)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_alu_type T = nir_alu_type_get_base_type(nir_intrinsic_src_type(intr));
   nir_def *v = intr->src[0].ssa;

   if (T == nir_type_float)
      v = nir_f2f32(b, v);
   else if (T == nir_type_int)
      v = nir_i2i32(b, v);
   else
      v = nir_u2u32(b, v);

   nir_src_rewrite(&intr->src[0], v);
   nir_intrinsic_set_src_type(intr, T | 32);
   return true;
}

/* ... and narrow 16-bit per-vertex inputs after loading them as 32-bit. */
static bool
widen_load_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_per_vertex_input ||
       intr->def.bit_size != 16)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_alu_type T = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr));

   intr->def.bit_size = 32;
   nir_intrinsic_set_dest_type(intr, T | 32);

   nir_def *v;
   if (T == nir_type_float)
      v = nir_f2f16(b, &intr->def);
   else
      v = nir_u2u16(b, &intr->def);

   nir_def_rewrite_uses_after(&intr->def, v);
   return true;
}

bool
panfrost_nir_widen_poly_io(nir_shader *nir)
{
   bool progress = false;

   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, widen_store_output,
            nir_metadata_control_flow, NULL);
   NIR_PASS(progress, nir, nir_shader_intrinsics_pass, widen_load_input,
            nir_metadata_control_flow, NULL);
   return progress;
}

/* Vertex shader ran as a compute job: hardware vertex IDs are zero-based. */
static bool
lower_vertex_id_zero_base(nir_builder *b, nir_intrinsic_instr *intr,
                          void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vertex_id_zero_base)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_load_raw_vertex_id(b));
   return true;
}

/*
 * Lower a vertex shader feeding a geometry or tessellation control shader. It
 * runs as a compute job over the vertex range of the draw, like the transform
 * feedback program, and stores its outputs to the poly vertex output buffer.
 */
bool
panfrost_nir_lower_vs_to_memory(nir_shader *nir)
{
   NIR_PASS(_, nir, panfrost_nir_widen_poly_io);
   NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_vertex_id_zero_base,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, panfrost_nir_lower_poly_misc);

   nir->info.has_transform_feedback_varyings = false;
   nir->xfb_info = NULL;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return true;
}

/*
 * Wrap the whole shader in "if (global_invocation_id.x < limit)" so a compute
 * program can run with a workgroup wider than one invocation. The limit is a
 * 32-bit word in the geometry parameters.
 */
static void
guard_global_id_x(nir_shader *nir, unsigned gp_offset)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));

   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *gp = nir_load_geometry_param_buffer_poly(&b);
   nir_def *limit = nir_load_global_constant(
      &b, 1, 32, nir_iadd_imm(&b, gp, gp_offset), .align_mul = 4);
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_push_if(&b, nir_ult(&b, x, limit));
   {
      b.cursor = nir_cf_reinsert(&body, b.cursor);
   }
   nir_pop_if(&b, NULL);

   nir_progress(true, impl, nir_metadata_none);
}

static void
make_compute(nir_shader *nir, unsigned wg_x)
{
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = wg_x;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->info.outputs_written = 0;
   nir->info.inputs_read = 0;
   nir->xfb_info = NULL;

   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out, NULL);
   NIR_PASS(_, nir, panfrost_nir_lower_poly_misc);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Width of the workgroups of the main geometry program */
#define PAN_GS_WG_SIZE 16

/*
 * Lower a geometry shader (I/O already lowered) into its programs. Takes
 * ownership of nothing; the returned shaders are allocated against NULL.
 */
void
panfrost_nir_lower_gs(const nir_shader *nir, struct poly_gs_info *info,
                      nir_shader **main_out, nir_shader **count_out,
                      nir_shader **pre_gs_out, nir_shader **rast_out,
                      unsigned *wg_size, unsigned *vertex_stride)
{
   nir_shader *gs = nir_shader_clone(NULL, nir);

   /* TODO: transform feedback of geometry shader outputs */
   gs->xfb_info = NULL;
   gs->info.has_transform_feedback_varyings = false;

   NIR_PASS(_, gs, panfrost_nir_widen_poly_io);

   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   poly_nir_lower_gs(gs, &count, &rast, &pre_gs, info);

   /* Index values of dynamic topologies: (primitive << log2) * stride + vertex */
   *vertex_stride = util_next_power_of_two(gs->info.gs.vertices_out);

   /* Main program: (primitive, instance) -> global invocation (x, y) */
   guard_global_id_x(gs, offsetof(struct poly_geometry_params, grid[0]));
   make_compute(gs, PAN_GS_WG_SIZE);
   *wg_size = PAN_GS_WG_SIZE;

   if (count) {
      guard_global_id_x(count, offsetof(struct poly_geometry_params, grid[0]));
      make_compute(count, PAN_GS_WG_SIZE);
   }

   make_compute(pre_gs, 1);

   NIR_PASS(_, rast, panfrost_nir_lower_poly_misc);
   NIR_PASS(_, rast, nir_remove_dead_variables, nir_var_shader_in, NULL);
   rast->info.inputs_read = 0;

   /* The varying layout indexes outputs by base: make them dense */
   NIR_PASS(_, rast, nir_recompute_io_bases, nir_var_shader_out);
   nir_shader_gather_info(rast, nir_shader_get_entrypoint(rast));

   *main_out = gs;
   *count_out = count;
   *pre_gs_out = pre_gs;
   *rast_out = rast;
}

/*
 * Lower a tessellation control shader (I/O already lowered) to a compute
 * program: one workgroup per patch, one invocation per output vertex, storing
 * its outputs to the poly TCS output buffer.
 */
/* libpoly assumes a patch fits in a subgroup, which Mali subgroups (4 or 8
 * invocations) do not guarantee: synchronise the whole workgroup instead.
 */
static bool
widen_subgroup_barrier(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_barrier)
      return false;

   bool progress = false;

   if (nir_intrinsic_execution_scope(intr) == SCOPE_SUBGROUP) {
      nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
      progress = true;
   }

   if (nir_intrinsic_memory_scope(intr) == SCOPE_SUBGROUP) {
      nir_intrinsic_set_memory_scope(intr, SCOPE_WORKGROUP);
      progress = true;
   }

   return progress;
}

nir_shader *
panfrost_nir_lower_tcs(const nir_shader *nir)
{
   nir_shader *tcs = nir_shader_clone(NULL, nir);
   unsigned wg = tcs->info.tess.tcs_vertices_out;

   NIR_PASS(_, tcs, panfrost_nir_widen_poly_io);
   NIR_PASS(_, tcs, poly_nir_lower_tcs, false);
   NIR_PASS(_, tcs, nir_shader_intrinsics_pass, widen_subgroup_barrier,
            nir_metadata_all, NULL);
   make_compute(tcs, wg);
   return tcs;
}

/*
 * Lower a tessellation evaluation shader (I/O already lowered) to the
 * hardware vertex shader of the rasterization draw, one invocation per domain
 * point.
 */
nir_shader *
panfrost_nir_lower_tes(const nir_shader *nir)
{
   nir_shader *tes = nir_shader_clone(NULL, nir);

   NIR_PASS(_, tes, panfrost_nir_widen_poly_io);
   NIR_PASS(_, tes, poly_nir_lower_tes, true);
   NIR_PASS(_, tes, panfrost_nir_lower_poly_misc);
   NIR_PASS(_, tes, nir_remove_dead_variables, nir_var_shader_in, NULL);
   tes->info.inputs_read = 0;
   tes->info.patch_inputs_read = 0;
   NIR_PASS(_, tes, nir_recompute_io_bases, nir_var_shader_out);
   nir_shader_gather_info(tes, nir_shader_get_entrypoint(tes));
   return tes;
}
